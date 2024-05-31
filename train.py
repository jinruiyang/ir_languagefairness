import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from model import DPRModel
import argparse
import logging
from tqdm import tqdm
import json
import os
import torch.nn.functional as F
from collections import defaultdict

class MultieupDataset(Dataset):
    def __init__(self, queries, passages, qrels, tokenizer, max_query_length, max_passage_length):
        self.queries = queries
        self.passages = passages
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id = list(self.queries.keys())[idx]
        query = self.queries[query_id]
        positive_passage_id = self.qrels[query_id][0]
        positive_passage = self.passages[positive_passage_id]

        # Tokenize the queries and passages
        tokenized_query = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_query_length, return_tensors='pt')
        tokenized_passage = self.tokenizer(positive_passage, padding='max_length', truncation=True, max_length=self.max_passage_length, return_tensors='pt')

        return {k: v.squeeze(0) for k, v in tokenized_query.items()}, {k: v.squeeze(0) for k, v in tokenized_passage.items()}

def collate_fn(batch):
    queries = {key: torch.stack([item[0][key] for item in batch]) for key in batch[0][0]}
    passages = {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1]}
    return queries, passages

def compute_loss(query_embeddings, passage_embeddings):
    cosine_sim = torch.mm(query_embeddings, passage_embeddings.T)
    target = torch.arange(0, cosine_sim.shape[0]).to(cosine_sim.device)
    loss = F.cross_entropy(cosine_sim, target)
    return loss

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def log_data_info(queries, passages, qrels, logger):
    num_queries = len(queries)
    num_passages = len(passages)
    relevance_counts = [len(qrel) for qrel in qrels.values()]
    avg_relevance = sum(relevance_counts) / len(relevance_counts) if relevance_counts else 0

    logger.info(f"Number of queries: {num_queries}")
    logger.info(f"Number of passages: {num_passages}")
    logger.info(f"Average number of positive relevance per query: {avg_relevance:.2f}")

def main(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Print out configuration
    logger.info("Configuration:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    logger.info("Loading and preprocessing data...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    queries = load_json(args.query_file)
    passages = load_json(args.passage_file)
    qrels = load_json(args.qrels_file)

    log_data_info(queries, passages, qrels, logger)

    dataset = MultieupDataset(queries, passages, qrels, tokenizer, args.max_query_length, args.max_passage_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPRModel(model_name=args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_training_steps = args.epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint file {args.checkpoint} not found. Starting from scratch.")

    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        for step, (queries, passages) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            queries = {key: value.to(device) for key, value in queries.items()}
            passages = {key: value.to(device) for key, value in passages.items()}

            optimizer.zero_grad()

            # Remove token_type_ids if they exist in the queries and passages
            if 'token_type_ids' in queries:
                queries.pop('token_type_ids')
            if 'token_type_ids' in passages:
                passages.pop('token_type_ids')

            query_embeddings = model.encode_queries(**queries)
            passage_embeddings = model.encode_passages(**passages)
            loss = compute_loss(query_embeddings, passage_embeddings)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % args.log_steps == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    logger.info("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DPR model on the multieup dataset")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query file (JSON)")
    parser.add_argument("--passage_file", type=str, required=True, help="Path to the passage file (JSON)")
    parser.add_argument("--qrels_file", type=str, required=True, help="Path to the qrels file (JSON)")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-uncased", help="Model name for the encoder")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--log_steps", type=int, default=10, help="Number of steps between logging")
    parser.add_argument("--max_query_length", type=int, default=64, help="Maximum length of query tokens")
    parser.add_argument("--max_passage_length", type=int, default=180, help="Maximum length of passage tokens")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the checkpoints and other outputs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file to resume training")
    args = parser.parse_args()
    main(args)
