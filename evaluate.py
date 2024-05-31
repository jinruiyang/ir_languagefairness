import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model import DPRModel
import argparse
import logging
import json
from tqdm import tqdm
import faiss
import numpy as np
import os
import ir_measures
from ir_measures import *

class MultieupDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_id, text = list(self.data.items())[idx]
        return item_id, text

    def collate_fn(self, batch):
        ids, texts = zip(*batch)
        tokenized_batch = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True,
                                         max_length=self.max_length)
        return ids, tokenized_batch

def encode_data(model, dataloader, device):
    encoded_data = {}
    model.eval()
    with torch.no_grad():
        for ids, tokenized_batch in tqdm(dataloader, desc="Encoding data"):
            tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
            outputs = model(**tokenized_batch).last_hidden_state.mean(dim=1).cpu().numpy()
            for id, output in zip(ids, outputs):
                encoded_data[id] = output.reshape(1, -1)  # Ensure each embedding is 2D
    return encoded_data

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

def evaluate(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Loading and preprocessing data...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    queries = load_json(args.query_file)
    passages = load_json(args.passage_file)
    qrels = load_json(args.qrels_file)

    log_data_info(queries, passages, qrels, logger)

    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPRModel(model_name=args.model_name).to(device)

    # Load the checkpoint
    if os.path.isfile(args.model_checkpoint):
        logger.info(f"Loading model checkpoint from {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model checkpoint loaded successfully.")
    else:
        logger.error(f"Model checkpoint file {args.model_checkpoint} not found.")
        return

    model.eval()

    logger.info("Encoding queries and passages...")
    query_dataset = MultieupDataset(queries, tokenizer, args.max_query_length)
    passage_dataset = MultieupDataset(passages, tokenizer, args.max_passage_length)
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, collate_fn=query_dataset.collate_fn)
    passage_dataloader = DataLoader(passage_dataset, batch_size=args.batch_size, collate_fn=passage_dataset.collate_fn)

    encoded_queries = encode_data(model.query_encoder, query_dataloader, device)
    encoded_passages = encode_data(model.passage_encoder, passage_dataloader, device)

    # Debugging: Check if encoded_passages is correctly populated
    if not encoded_passages:
        logger.error("Encoded passages are empty. Please check the data and encoding process.")
        return

    # Debugging: Print the shape of the first passage embedding
    first_passage_embedding = next(iter(encoded_passages.values()))
    logger.info(f"Shape of the first passage embedding: {first_passage_embedding.shape}")

    dimension = first_passage_embedding.shape[1]
    logger.info(f"Passage embedding dimension: {dimension}")

    logger.info("Building FAISS index...")
    try:
        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.IndexFlatL2(dimension)  # build a flat (CPU) index
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # copy to GPU
        ids, embeddings = zip(*encoded_passages.items())
        embeddings = np.vstack(embeddings)
        gpu_index.add(embeddings)
        logger.info("FAISS index built using GPU.")
    except Exception as e:
        logger.warning(f"Failed to build FAISS index on GPU: {e}. Falling back to CPU.")
        index_flat = faiss.IndexFlatL2(dimension)  # build a flat (CPU) index
        ids, embeddings = zip(*encoded_passages.items())
        embeddings = np.vstack(embeddings)
        index_flat.add(embeddings)
        gpu_index = index_flat
        logger.info("FAISS index built using CPU.")

    logger.info("Evaluating...")
    rankings = {}
    run = []
    qrels_list = []
    ranking_tsv_lines = []
    for qid, query_embedding in tqdm(encoded_queries.items(), desc="Evaluating"):
        query_embedding = query_embedding.reshape(1, -1)  # Ensure query_embedding is 2D
        D, I = gpu_index.search(query_embedding, args.top_k)
        retrieved_passages = [ids[i] for i in I[0]]
        relevance_scores = [1 if pid in qrels.get(qid, []) else 0 for pid in retrieved_passages]
        rankings[qid] = {'rank': retrieved_passages, 'relevance': relevance_scores}

        # Debugging: Print relevance scores for each query
        logger.debug(f"Query ID: {qid}, Relevance Scores: {relevance_scores}")

        # Prepare run and qrels for ir_measures
        for rank, pid in enumerate(retrieved_passages):
            score = 1 / (rank + 1)  # Convert rank to score
            run.append((qid, pid, score))
            ranking_tsv_lines.append(f"{qid}\t{pid}\t{rank + 1}\t{relevance_scores[rank]}")
        for pid in qrels.get(qid, []):
            qrels_list.append((qid, pid, 1))

    # Debugging: Print the first few elements of run and qrels_list
    logger.debug(f"First few elements of run: {run[:10]}")
    logger.debug(f"First few elements of qrels_list: {qrels_list[:10]}")

    # Save ranking.tsv.annotated
    ranking_tsv_path = os.path.join(args.output_dir, "ranking.tsv.annotated")
    with open(ranking_tsv_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(ranking_tsv_lines))
    logger.info(f"Rankings saved to {ranking_tsv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DPR model on the multieup dataset")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query file (JSON)")
    parser.add_argument("--passage_file", type=str, required=True, help="Path to the passage file (JSON)")
    parser.add_argument("--qrels_file", type=str, required=True, help="Path to the qrels file (JSON)")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-uncased", help="Model name for the encoder")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--top_k", type=int, default=100, help="Top K passages to retrieve for each query")
    parser.add_argument("--max_query_length", type=int, default=64, help="Maximum length of query tokens")
    parser.add_argument("--max_passage_length", type=int, default=256, help="Maximum length of passage tokens")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the rankings and other outputs")
    args = parser.parse_args()
    evaluate(args)
