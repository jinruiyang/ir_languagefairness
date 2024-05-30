import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer
from model import DPRModel
import argparse
import logging
from tqdm import tqdm
import json
import os
import torch.nn.functional as F
from collections import defaultdict


class MultieupDataset(Dataset):
    def __init__(self, queries, passages, qrels, alignment_file, tokenizer, max_query_length, max_passage_length,
                 use_cache, cache_dir):
        self.queries = queries
        self.passages = passages
        self.qrels = qrels
        self.alignment_file = alignment_file
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.alignment_queries = self.load_alignment_queries_lazy()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id = list(self.queries.keys())[idx]
        query = self.queries[query_id]
        positive_passage_id = self.qrels[query_id][0]
        positive_passage = self.passages[positive_passage_id]

        # Tokenize the queries and passages
        tokenized_query = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_query_length,
                                         return_tensors='pt')
        tokenized_passage = self.tokenizer(positive_passage, padding='max_length', truncation=True,
                                           max_length=self.max_passage_length, return_tensors='pt')

        return {k: v.squeeze(0) for k, v in tokenized_query.items()}, {k: v.squeeze(0) for k, v in
                                                                       tokenized_passage.items()}, query_id

    def load_alignment_queries_lazy(self):
        alignment_cache_path = os.path.join(self.cache_dir, 'tokenized_alignment_queries.json')
        if self.use_cache and os.path.exists(alignment_cache_path):
            with open(alignment_cache_path, 'r', encoding='utf-8') as f:
                alignment_data = json.load(f)
            alignment_queries = defaultdict(dict)
            for query_id, content in alignment_data.items():
                alignment_queries[query_id] = {
                    'query': {k: torch.tensor(v) for k, v in content['query'].items()},
                    'parallel_queries': {k: {k2: torch.tensor(v2) for k2, v2 in v.items()} for k, v in
                                         content['parallel_queries'].items()}
                }
            return alignment_queries
        else:
            with open(self.alignment_file, 'r', encoding='utf-8') as f:
                alignment_data = json.load(f)
            alignment_queries = defaultdict(dict)
            for query_id, content in alignment_data.items():
                # Ensure the correct structure
                if isinstance(content, dict) and 'query' in content and 'parallel_queries' in content:
                    alignment_queries[query_id] = {
                        'query': content['query'],
                        'parallel_queries': content['parallel_queries']
                    }
                else:
                    print(f"Skipping misformatted entry for query_id: {query_id}")
            return alignment_queries

    def get_alignment_query(self, query_id):
        if query_id in self.alignment_queries and 'query' not in self.alignment_queries[query_id]:
            query = self.alignment_queries[query_id]['query']
            parallel_queries = self.alignment_queries[query_id]['parallel_queries']
            self.alignment_queries[query_id] = {
                'query': self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_query_length,
                                        return_tensors='pt'),
                'parallel_queries': [
                    self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_query_length,
                                   return_tensors='pt')
                    for _, text in parallel_queries
                ]
            }
        return self.alignment_queries[query_id]


def collate_fn(batch):
    queries = {key: torch.stack([item[0][key] for item in batch]) for key in batch[0][0]}
    passages = {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1]}
    query_ids = [item[2] for item in batch]
    return queries, passages, query_ids


def compute_contrastive_loss(query_embeddings, temperature=0.1):
    """
    Computes the contrastive loss for query embeddings using cross-entropy.
    Args:
        query_embeddings (torch.Tensor): Tensor of shape (2N, d) containing the embeddings of 2N queries.
        temperature (float): Temperature parameter for scaling the similarities.
    Returns:
        torch.Tensor: Scalar tensor containing the contrastive loss.
    """
    # Number of query pairs
    N = query_embeddings.size(0) // 2

    # Compute similarity matrix
    similarity_matrix = torch.mm(query_embeddings, query_embeddings.t()) / temperature

    # Create targets
    target = torch.arange(0, 2 * N).to(query_embeddings.device)

    # Mask out the diagonal (self-similarity)
    mask = torch.eye(2 * N, dtype=torch.bool).to(query_embeddings.device)
    similarity_matrix.masked_fill_(mask, float('-inf'))

    # Compute cross-entropy loss
    loss_i = F.cross_entropy(similarity_matrix[:N], target[:N])
    loss_j = F.cross_entropy(similarity_matrix[N:], target[N:])

    # Combine the losses
    contrastive_loss = (loss_i + loss_j) / 2

    return contrastive_loss


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f):
        return json.load(f)


def compute_parallel_loss_loss(dataset, query_ids, model, device, args):
    parallel_loss = 0
    for query_id in query_ids:
        # Get the alignment query for the current query_id
        alignment_query = dataset.get_alignment_query(query_id)
        # Get N pairs of parallel queries
        parallel_queries = alignment_query['parallel_queries']
        if len(parallel_queries) < args.N:
            print(f"Not enough parallel queries for query_id: {query_id}")
            continue
        random_parallel_queries = random.sample(parallel_queries, args.N)

        # Extract the parallel query IDs
        parallel_query_ids = {pq[0] for pq in random_parallel_queries}

        # Get N-1 non-parallel queries (excluding the parallel queries)
        all_query_ids = set(dataset.queries.keys())
        non_parallel_query_ids = list(all_query_ids - parallel_query_ids)
        if len(non_parallel_query_ids) < args.N - 1:
            print(f"Not enough non-parallel queries available for query_id: {query_id}")
            continue
        random_non_parallel_query_ids = random.sample(non_parallel_query_ids, args.N - 1)
        non_parallel_queries = [dataset.queries[qid] for qid in random_non_parallel_query_ids]

        # Tokenize parallel queries
        tokenized_parallel_queries = [dataset.tokenizer(pq[1], padding='max_length', truncation=True,
                                                        max_length=dataset.max_query_length, return_tensors='pt') for pq
                                      in random_parallel_queries]
        tokenized_parallel_queries = [{k: v.to(device) for k, v in tpq.items()} for tpq in tokenized_parallel_queries]

        # Tokenize non-parallel queries
        tokenized_non_parallel_queries = [dataset.tokenizer(npq, padding='max_length', truncation=True,
                                                            max_length=dataset.max_query_length, return_tensors='pt')
                                          for npq in non_parallel_queries]
        tokenized_non_parallel_queries = [{k: v.to(device) for k, v in tnpq.items()} for tnpq in
                                          tokenized_non_parallel_queries]

        # Encode parallel and non-parallel queries
        parallel_query_embeddings = [model.encode_queries(**tpq) for tpq in tokenized_parallel_queries]
        non_parallel_query_embeddings = [model.encode_queries(**tnpq) for tnpq in tokenized_non_parallel_queries]

        # Concatenate embeddings
        all_query_embeddings = torch.cat(parallel_query_embeddings + non_parallel_query_embeddings, dim=0)

        # Compute contrastive loss with 2N queries
        parallel_loss += compute_contrastive_loss(all_query_embeddings, temperature=0.1)

    return parallel_loss / len(query_ids)


def log_data_info(queries, passages, qrels, logger):
    num_queries = len(queries)
    num_passages = len(passages)
    relevance_counts = [len(qrel) for qrel in qrels.values()]
    avg_relevance = sum(relevance_counts) / len(relevance_counts) if relevance_counts else 0

    logger.info(f"Number of queries: {num_queries}")
    logger.info(f"Number of passages: {num_passages}")
    logger.info(f"Average number of positive relevance per query: {avg_relevance:.2f}")


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Configuration:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    logger.info("Loading and preprocessing data...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    queries = load_json(args.query_file)
    passages = load_json(args.passage_file)
    qrels = load_json(args.qrels_file)

    log_data_info(queries, passages, qrels, logger)

    dataset = MultieupDataset(queries, passages, qrels, args.alignment_file, tokenizer, args.max_query_length,
                              args.max_passage_length, args.use_cache, args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPRModel(model_name=args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_training_steps = args.epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

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
        epoch_dpr_loss = 0
        epoch_parallel_loss = 0

        for step, (queries, passages, query_ids) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            queries = {key: value.to(device) for key, value in queries.items()}
            passages = {key: value.to(device) for key, value in passages.items()}

            optimizer.zero_grad()

            query_embeddings = model.encode_queries(**queries)
            passage_embeddings = model.encode_passages(**passages)
            dpr_loss = compute_loss(query_embeddings, passage_embeddings)

            parallel_loss = compute_parallel_loss_loss(dataset, query_ids, model, device, args)

            loss = dpr_loss + parallel_loss * args.alignment_loss_weight

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_dpr_loss += dpr_loss.item()
            epoch_parallel_loss += parallel_loss.item()

            if step % args.log_steps == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{args.epochs}], Step [{step}/{len(dataloader)}], DPR Loss: {dpr_loss.item():.4f}, Alignment Loss: {parallel_loss.item():.4f}, Combined Loss: {loss.item():.4f}")

        avg_dpr_loss = epoch_dpr_loss / len(dataloader)
        avg_parallel_loss = epoch_parallel_loss / len(dataloader)
        avg_loss = (epoch_dpr_loss + epoch_parallel_loss) / len(dataloader)
        logger.info(
            f"Epoch [{epoch + 1}/{args.epochs}] completed. Average DPR Loss: {avg_dpr_loss:.4f}, Average Alignment Loss: {avg_parallel_loss:.4f}, Average Combined Loss: {avg_loss:.4f}")

        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
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
    parser = argparse.ArgumentParser(description="Train a DPR model on the multieup dataset with alignment loss")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query file (JSON)")
    parser.add_argument("--passage_file", type=str, required=True, help="Path to the passage file (JSON)")
    parser.add_argument("--qrels_file", type=str, required=True, help="Path to the qrels file (JSON)")
    parser.add_argument("--alignment_file", type=str, required=True, help="Path to the alignment file (JSON)")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Model name for the encoder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--log_steps", type=int, default=10, help="Number of steps between logging")
    parser.add_argument("--alignment_loss_weight", type=float, default=0.1, help="Weight for the alignment loss")
    parser.add_argument("--max_query_length", type=int, default=64, help="Maximum length of query tokens")
    parser.add_argument("--max_passage_length", type=int, default=180, help="Maximum length of passage tokens")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the checkpoints and other outputs")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to save the cached tokenized data")
    parser.add_argument("--use_cache", action='store_true', help="Flag to use cached tokenized data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file to resume training")
    parser.add_argument("--N", type=int, default=1, help="Number of parallel and non-parallel queries to use")
    args = parser.parse_args()
    main(args)
