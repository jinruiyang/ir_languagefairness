import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from model import DPRModel  # Ensure you have a model.py file with the DPRModel class
import argparse
import logging
from tqdm import tqdm
import json
import os
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np
import ir_measures
from ir_measures import *
from peer_measure import PEER
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import matplotlib.pyplot as plt


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
        self.query_ids = list(self.queries.keys())

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        query = self.queries[query_id]
        positive_passage_id = self.qrels[query_id][0]
        positive_passage = self.passages[positive_passage_id]

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

            # self.validate_alignment_queries(alignment_queries)
            return alignment_queries
        else:
            with open(self.alignment_file, 'r', encoding='utf-8') as f:
                alignment_data = json.load(f)
            alignment_queries = defaultdict(dict)
            for query_id, content in alignment_data.items():
                if isinstance(content, dict) and 'query' in content and 'parallel_queries' in content:
                    alignment_queries[query_id] = {
                        'query': content['query'],
                        'parallel_queries': content['parallel_queries']
                    }
                else:
                    print(f"Skipping misformatted entry for query_id: {query_id}")
            # self.validate_alignment_queries(alignment_queries)
            return alignment_queries

    def validate_alignment_queries(self, alignment_queries):
        for id, parallel_query in enumerate(content['parallel_queries']):
            query_text = parallel_query[1]
            parallel_qid = parallel_query[0]
            if isinstance(query_text, str) and any(
                    term in query_text.lower() for term in ["nan", "null", "none"]):  # Corrected line
                    print(f'skiping invalid parallel query {parallel_qid}')
                    content['parallel_queries'][id][1] = ''

    def get_alignment_query(self, query_id):
        if query_id not in self.alignment_queries:
            print(f"Error: query_id {query_id} not found in alignment_queries")
            print(f"Available keys: {list(self.alignment_queries.keys())[:10]}")  # Print a sample of available keys
            return None

        if 'query' not in self.alignment_queries[query_id]:
            query = self.alignment_queries[query_id].get('query', None)
            parallel_queries = self.alignment_queries[query_id].get('parallel_queries', [])

            if not query:
                print(f"Error: Missing 'query' for query_id {query_id}")
                return None
            if not parallel_queries:
                print(f"Error: Missing 'parallel_queries' for query_id {query_id}")
                return None

            # Tokenize and store the query and parallel queries
            self.alignment_queries[query_id] = {
                'query': self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_query_length,
                                        return_tensors='pt'),
                'parallel_queries': [
                    self.tokenizer(query_text, padding='max_length', truncation=True, max_length=self.max_query_length,
                                   return_tensors='pt')
                    for _, query_text in parallel_queries
                ]
            }

        if 'query' not in self.alignment_queries[query_id]:
            print(f"Warning: 'query' key is missing for query_id {query_id} after tokenization")

        return self.alignment_queries[query_id]


def collate_fn(batch):
    queries = {key: torch.stack([item[0][key] for item in batch]) for key in batch[0][0]}
    passages = {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1]}
    query_ids = [item[2] for item in batch]
    return queries, passages, query_ids

# def compute_loss(query_embeddings, passage_embeddings):
#     cosine_sim = torch.mm(query_embeddings, passage_embeddings.T)
#     target = torch.arange(0, cosine_sim.shape[0]).to(cosine_sim.device)
#     loss = F.cross_entropy(cosine_sim, target)
#     return loss

def compute_loss(query_embeddings, passage_embeddings):
    assert query_embeddings.dim() == 2 and passage_embeddings.dim() == 2, "Embeddings must be 2D tensors"
    cosine_sim = torch.mm(query_embeddings, passage_embeddings.T)
    target = torch.arange(0, cosine_sim.shape[0]).to(cosine_sim.device)
    loss = F.cross_entropy(cosine_sim, target)
    return loss


def compute_parallel_loss(dataset, query_ids, query_embeddings, model, device, passage_embeddings):
    total_parallel_loss = 0
    total_query_similarity = 0
    epsilon = 1e-8  # Small epsilon value to avoid log(0)

    # Create a dictionary to map query_id to its corresponding embedding
    query_id_to_embedding = {query_id: query_embeddings[idx].unsqueeze(0) for idx, query_id in enumerate(query_ids)}

    for query_id in query_ids:
        # Get the alignment query for the current query_id
        alignment_query = dataset.get_alignment_query(query_id)

        # Check if 'parallel_queries' key exists
        if 'parallel_queries' not in alignment_query:
            continue

        # Sample a parallel query (second element in each parallel_queries list is the query text)
        parallel_query = random.choice(alignment_query['parallel_queries'])
        parallel_query_text = parallel_query[1]
        parallel_query_qid = parallel_query[0]

        # Tokenize and encode the parallel query
        tokenized_parallel_query = dataset.tokenizer(parallel_query_text, padding='max_length', truncation=True,
                                                     max_length=dataset.max_query_length, return_tensors='pt')
        tokenized_parallel_query = {k: v.to(device) for k, v in tokenized_parallel_query.items()}

        # Remove token_type_ids if they exist in the queries and passages
        if 'token_type_ids' in tokenized_parallel_query:
            tokenized_parallel_query.pop('token_type_ids')

        parallel_query_embedding = model.encode_queries(**tokenized_parallel_query)

        original_query_embedding = query_id_to_embedding[query_id]
        query_similarity = F.cosine_similarity(original_query_embedding, parallel_query_embedding)
        # print(f"Cosine similarity between original query {query_id} and parallel query {parallel_query_qid}: {query_similarity.item()}")
        total_query_similarity += query_similarity.item()
        # Compute similarities for the parallel query
        similarities_parallel = torch.mm(parallel_query_embedding, passage_embeddings.T)
        similarity_distribution_parallel = F.softmax(similarities_parallel, dim=-1)

        # Use the precomputed embedding for the original query
        original_query_embedding = query_id_to_embedding[query_id]
        similarities_original = torch.mm(original_query_embedding, passage_embeddings.T)
        similarity_distribution_original = F.softmax(similarities_original, dim=-1)

        # Compute KL Divergence loss
        kl_loss = F.kl_div((similarity_distribution_parallel + epsilon).log(),
                           (similarity_distribution_original + epsilon).detach(), reduction='batchmean')

        total_parallel_loss += kl_loss
    # print(f"Average cosine similarity between original and parallel query in this batch {total_query_similarity / len(query_ids)}")
    # Return the average parallel loss over all queries
    return total_parallel_loss / len(query_ids), total_query_similarity / len(query_ids)


def normalize_embeddings(embeddings):
    return F.normalize(embeddings, p=2, dim=-1)


class EvaluationQueryDataset(Dataset):
    def __init__(self, queries, tokenizer, max_length):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_ids = list(self.queries.keys())

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        query = self.queries[query_id]
        tokenized_query = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_length,
                                         return_tensors='pt')
        return query_id, {k: v.squeeze(0) for k, v in tokenized_query.items()}


class EvaluationPassageDataset(Dataset):
    def __init__(self, passages, tokenizer, max_length):
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.passage_ids = list(self.passages.keys())

    def __len__(self):
        return len(self.passage_ids)

    def __getitem__(self, idx):
        passage_id = self.passage_ids[idx]
        passage = self.passages[passage_id]
        tokenized_passage = self.tokenizer(passage, padding='max_length', truncation=True, max_length=self.max_length,
                                           return_tensors='pt')
        return passage_id, {k: v.squeeze(0) for k, v in tokenized_passage.items()}


def eval_collate_fn(batch):
    ids, texts = zip(*batch)
    tokenized_batch = {k: torch.stack([text[k] for text in texts]) for k in texts[0]}
    return ids, tokenized_batch


def encode_data(model, dataloader, device):
    encoded_data = {}
    model.eval()
    with torch.no_grad():
        for ids, tokenized_batch in tqdm(dataloader, desc="Encoding data"):
            tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
            outputs = model(**tokenized_batch).last_hidden_state.mean(dim=1).cpu().numpy()
            for id, output in zip(ids, outputs):
                encoded_data[id] = output.reshape(1, -1)
    return encoded_data


def log_data_info(queries, passages, qrels, logger):
    num_queries = len(queries)
    num_passages = len(passages)
    relevance_counts = [len(qrel) for qrel in qrels.values()]
    avg_relevance = sum(relevance_counts) / len(relevance_counts) if relevance_counts else 0

    logger.info(f"Number of queries: {num_queries}")
    logger.info(f"Number of passages: {num_passages}")
    logger.info(f"Average number of positive relevance per query: {avg_relevance:.2f}")


def reorder_rankings(base_directory, lang, sorted_qid_df, logger):
    file_path = os.path.join(base_directory, 'ranking.tsv.annotated')
    data = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'did', 'rank', 'relevance'])

    sorted_qids = sorted_qid_df[lang].dropna().tolist()
    if not sorted_qids:
        logger.info(f"No sorted QIDs for language: {lang}")
        return

    data = data[data['qid'].isin(sorted_qids)].copy()
    data['qid'] = pd.Categorical(data['qid'], categories=sorted_qids, ordered=True)
    data.sort_values(['qid', 'rank'], inplace=True)

    new_tsv_path = os.path.join(base_directory, 'parallel.ranking.tsv.annotated.tsv')
    data.to_csv(new_tsv_path, sep='\t', index=False, header=False)

    output_dict = {}
    for _, row in data.iterrows():
        qid = row['qid']
        if qid not in output_dict:
            output_dict[qid] = {'did': [], 'rank': [], 'relevance': []}
        output_dict[qid]['did'].append(row['did'])
        output_dict[qid]['rank'].append(row['rank'])
        output_dict[qid]['relevance'].append(row['relevance'])

    json_path = os.path.join(base_directory, 'parallel.ranking.json')
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    logger.info(f"Rankings saved and reordered for {lang}")


def compute_metrics(output_path, lang, epoch, logger, results_df):
    file_path = os.path.join(output_path, f'epoch_{epoch}_results/{lang}/parallel.ranking.tsv.annotated.tsv')
    try:
        data = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'did', 'rank', 'relevance'])

        qrels = {}
        run = {}
        lang_mapping = {}

        for idx, row in data.iterrows():
            qid = row['qid']
            did = row['did']
            score = 1 / row['rank']
            rel = row['relevance']
            lang_id = did.split('#')[1]

            if qid not in qrels:
                qrels[qid] = {}
            if qid not in run:
                run[qid] = {}

            qrels[qid][did] = rel
            run[qid][did] = score
            lang_mapping[did] = lang_id

        standard_results = ir_measures.calc_aggregate([MAP @ 100, MRR @ 100, nDCG @ 100, Recall @ 100], qrels, run)
        measure = PEER(weights={0: 0, 1: 1}, lang_mapping=lang_mapping) @ 1000
        peer_results = ir_measures.calc_aggregate([measure], qrels, run)

        results_df.loc[lang] = {
            'MAP@100': standard_results[MAP @ 100],
            'MRR@100': standard_results[MRR @ 100],
            'nDCG@100': standard_results[nDCG @ 100],
            'Recall@100': standard_results[Recall @ 100],
            'PEER@1000': peer_results[measure]
        }

        print(f"Metrics for {lang}:")
        print(results_df.loc[lang])

    except FileNotFoundError:
        logger.error(f"File not found for language {lang}: {file_path}")
    except Exception as e:
        logger.error(f"Error processing language {lang}: {str(e)}")


def compute_correlation(output_path, lang, epoch, logger, correlation_results, spearman_values, kendall_values):
    cutoff_k = 5
    rank_lists_by_lang = {lang: []}

    json_path = os.path.join(output_path, f'epoch_{epoch}_results/{lang}/parallel.ranking.json')
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
            for qid, contents in data.items():
                filtered_dids = [did for did, rank in zip(contents['did'], contents['rank']) if rank <= cutoff_k]
                rank_lists_by_lang[lang].append(filtered_dids)
    except FileNotFoundError:
        logger.error(f"File not found for language {lang}: {json_path}")
    except Exception as e:
        logger.error(f"Error processing language {lang}: {str(e)}")

    num_queries = len(next(iter(rank_lists_by_lang.values())))

    for index in range(num_queries):
        for i, lang1 in enumerate([lang]):
            for lang2 in [lang1]:
                list1 = rank_lists_by_lang[lang1][index]
                list2 = rank_lists_by_lang[lang2][index]
                spearman_corr, _ = spearmanr(list1, list2, nan_policy='omit')
                kendall_corr, _ = kendalltau(list1, list2, nan_policy='omit')
                correlation_results.append({
                    "lang_pair": f"{lang1}-{lang2} Index {index}",
                    "spearmanr": spearman_corr,
                    "kendalltau": kendall_corr
                })
                if spearman_corr is not None:
                    spearman_values.append(spearman_corr)
                if kendall_corr is not None:
                    kendall_values.append(kendall_corr)

    print(f'Correlation results for {lang}:')
    print(correlation_results)


def plot_losses(dpr_losses, alignment_losses, output_dir):
    plt.figure()
    plt.plot(dpr_losses, label='DPR Loss')
    plt.plot(alignment_losses, label='Alignment Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_losses.png'))
    plt.show()


def plot_eval_metrics(eval_metrics, output_dir):
    epochs = list(range(1, len(eval_metrics['MRR@100']) + 1))
    plt.figure()
    plt.plot(epochs, eval_metrics['MRR@100'], label='MRR@100', color='b')
    plt.plot(epochs, eval_metrics['PEER@1000'], label='PEER@1000', color='g')
    plt.plot(epochs, eval_metrics['spearman_mean'], label='Spearman Mean', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Evaluation Metrics')
    plt.legend()

    for i, epoch in enumerate(epochs):
        plt.text(epoch, eval_metrics['MRR@100'][i], f'{eval_metrics["MRR@100"][i]:.3f}', ha='right')
        plt.text(epoch, eval_metrics['PEER@1000'][i], f'{eval_metrics["PEER@1000"][i]:.3f}', ha='right')
        plt.text(epoch, eval_metrics['spearman_mean'][i], f'{eval_metrics["spearman_mean"][i]:.3f}', ha='right')

    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'))
    plt.show()


def evaluate_and_save_results(args, model, epoch, device, logger, tokenizer, sorted_qid_df, optimizer=None,
                              scheduler=None):
    evaluate_model(
        args=args,
        model=model,
        epoch=epoch,
        device=device,
        logger=logger,
        languages=args.languages,
        tokenizer=tokenizer,
        sorted_qid_df=sorted_qid_df,
        alignment_file=args.alignment_file,
        max_query_length=args.max_query_length,
        max_passage_length=args.max_passage_length,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir
    )




def evaluate_model(args, model, epoch, device, logger, languages, tokenizer, sorted_qid_df, alignment_file,
                   max_query_length, max_passage_length, use_cache, cache_dir):
    results_dir = os.path.join(args.output_dir, f'epoch_{epoch}_results')
    os.makedirs(results_dir, exist_ok=True)
    top_k = getattr(args, 'top_k', 100)

    results_df = pd.DataFrame(index=languages, columns=['MAP@100', 'MRR@100', 'nDCG@100', 'Recall@100', 'PEER@1000'])
    correlation_results = []
    spearman_values = []
    kendall_values = []

    for lang in languages:
        query_file = os.path.join(args.dev_input_dir, f'queries_by_language/dev_queries_{lang}.json')
        qrels_file = os.path.join(args.dev_input_dir, f'qrels_by_language/dev_qrels_{lang}.json')
        query_data = load_json(query_file)
        passage_data = load_json(args.dev_passages)
        qrels_data = load_json(qrels_file)

        log_data_info(query_data, passage_data, qrels_data, logger)

        query_dataset = EvaluationQueryDataset(query_data, tokenizer, max_query_length)
        query_dataloader = DataLoader(query_dataset, batch_size=min(args.batch_size, len(query_data)),
                                      collate_fn=eval_collate_fn)

        passage_dataset = EvaluationPassageDataset(passage_data, tokenizer, max_passage_length)
        passage_dataloader = DataLoader(passage_dataset, batch_size=args.batch_size, collate_fn=eval_collate_fn)

        print(
            f"Eval batch size: {args.batch_size}, Number of queries: {len(query_data)}, Number of passages: {len(passage_data)}")

        encoded_queries = encode_data(model.query_encoder, query_dataloader, device)
        encoded_passages = encode_data(model.passage_encoder, passage_dataloader, device)

        # Verify that all passages are encoded
        print(f"Number of encoded passages: {len(encoded_passages)}")
        print(f"Sample encoded passages: {list(encoded_passages.keys())[:10]}")

        dimension = next(iter(encoded_passages.values())).shape[1]
        index_flat = faiss.IndexFlatL2(dimension)
        ids, embeddings = zip(*encoded_passages.items())
        embeddings = np.vstack(embeddings)
        index_flat.add(embeddings)

        rankings = defaultdict(list)
        run = []
        qrels_list = []
        ranking_tsv_lines = []

        for qid, query_embedding in tqdm(encoded_queries.items(), desc=f"Evaluating {lang}"):
            query_embedding = query_embedding.reshape(1, -1)
            D, I = index_flat.search(query_embedding, top_k)
            retrieved_passages = [ids[i] for i in I[0]]
            relevance_scores = [1 if pid in qrels_data.get(qid, []) else 0 for pid in retrieved_passages]

            for rank, pid in enumerate(retrieved_passages):
                score = 1 / (rank + 1)
                run.append((qid, pid, score))
                ranking_tsv_lines.append(f"{qid}\t{pid}\t{rank + 1}\t{relevance_scores[rank]}")
                rankings[qid].append((qid, pid, rank + 1, relevance_scores[rank]))

            for pid in qrels_data.get(qid, []):
                qrels_list.append((qid, pid, 1))

        base_directory = os.path.join(results_dir, lang)
        os.makedirs(base_directory, exist_ok=True)
        ranking_tsv_path = os.path.join(base_directory, 'ranking.tsv.annotated')
        with open(ranking_tsv_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(ranking_tsv_lines))

        reorder_rankings(base_directory, lang, sorted_qid_df, logger)
        logger.info(f"Rankings saved and reordered for {lang}")

        compute_metrics(args.output_dir, lang, epoch, logger, results_df)
        compute_correlation(args.output_dir, lang, epoch, logger, correlation_results, spearman_values, kendall_values)

    print(f"Overall Metrics for epoch {epoch}:")
    print(results_df)

    avg_row = results_df.mean().rename('Average')
    var_row = results_df.var().rename('Variance')
    results_df = pd.concat([results_df, avg_row.to_frame().T, var_row.to_frame().T])

    print(results_df)
    results_df.to_csv(f'{args.output_dir}/epoch_{epoch}_results/ir_metrics_mdpr_ep{epoch}.csv', float_format='%.3f')

    spearman_mean = sum(spearman_values) / len(spearman_values) if spearman_values else None
    kendall_mean = sum(kendall_values) / len(kendall_values) if kendall_values else None

    analysis_path = os.path.join(args.output_dir, f'epoch_{epoch}_results/Analysis')
    os.makedirs(analysis_path, exist_ok=True)

    output_file = os.path.join(analysis_path, 'correlation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            "correlation_results": correlation_results,
            "spearman_mean": spearman_mean,
            "kendall_mean": kendall_mean
        }, f, indent=4)

    averages_file = os.path.join(analysis_path, f'average_correlations_epoch_{epoch}.json')
    averages_data = {
        "epoch": epoch,
        "spearman_mean": spearman_mean,
        "kendall_mean": kendall_mean
    }
    with open(averages_file, 'w') as f:
        json.dump(averages_data, f, indent=4)

    print(f'Average Spearman correlation for epoch {epoch}: {spearman_mean}')
    print(f'Average Kendall correlation for epoch {epoch}: {kendall_mean}')





def main(args):
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

    dataset = MultieupDataset(queries, passages, qrels, args.alignment_file, tokenizer, args.max_query_length, args.max_passage_length, args.use_cache, args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPRModel(model_name=args.model_name).to(device)
    total_training_steps = args.epochs * len(dataloader)
    sorted_qid_df = pd.read_csv(args.sorted_qid)

    optimizer, scheduler = None, None
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

    if args.mode in ["train", "train+evaluate"]:

        logger.info("Starting training...")
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=total_training_steps)

        alignment_loss_threshold = 2e-5
        stop_alignment_loss = False

        dpr_losses = []
        alignment_losses = []
        eval_metrics = {
            'MRR@100': [],
            'PEER@1000': [],
            'spearman_mean': []
        }

        for epoch in range(start_epoch, args.epochs):
            dataset = MultieupDataset(queries, passages, qrels, args.alignment_file, tokenizer, args.max_query_length,
                                      args.max_passage_length, args.use_cache, args.cache_dir)
            model.train()
            epoch_dpr_loss = 0
            epoch_parallel_loss = 0

            for step, (queries, passages, query_ids) in enumerate(
                    tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
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
                dpr_loss = compute_loss(query_embeddings, passage_embeddings)

                if not stop_alignment_loss:
                    parallel_loss, query_similarity = compute_parallel_loss(dataset, query_ids, query_embeddings, model, device,
                                                          passage_embeddings)

                    if parallel_loss.item() < alignment_loss_threshold:
                        stop_alignment_loss = True
                else:
                    parallel_loss = torch.tensor(0.0).to(device)

                # loss = dpr_loss + parallel_loss * args.alignment_loss_weight
                loss = dpr_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                dpr_losses.append(dpr_loss.item())
                alignment_losses.append(parallel_loss.item())

                epoch_dpr_loss += dpr_loss.item()
                if not stop_alignment_loss:
                    epoch_parallel_loss += parallel_loss.item()

                if step % args.log_steps == 0:
                    logger.info(
                        f"Epoch [{epoch + 1}/{args.epochs}], Step [{step}/{len(dataloader)}], DPR Loss: {dpr_loss.item():.4f}, Alignment Loss: {parallel_loss.item():.4f}, Combined Loss: {loss.item():.4f}, Query Similarity: {query_similarity:.4f}")

            if args.mode == "train+evaluate":
                checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                try:
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }
                    if optimizer and scheduler:
                        checkpoint_data.update({
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        })
                    torch.save(checkpoint_data, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")

                logger.info(f"Evaluating epoch {epoch + 1}...")
                evaluate_and_save_results(args, model, epoch, device, logger, tokenizer, sorted_qid_df, optimizer,
                                          scheduler)
                # compute_metrics(args.output_dir, args.languages, epoch, logger)
                # compute_correlation(args.output_dir, args.languages, epoch, logger)

        if args.mode == "train+evaluate":
            plot_losses(dpr_losses, alignment_losses, args.output_dir)
            plot_eval_metrics(eval_metrics, args.output_dir)
            logger.info("Training completed.")

    if args.mode in ["evaluate"]:
        logger.info("Starting evaluation...")
        if args.checkpoint and os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            evaluate_and_save_results(args, model, args.epochs, device, logger, tokenizer, sorted_qid_df)
        # compute_metrics(args.output_dir, args.languages, args.epochs, logger)
        # compute_correlation(args.output_dir, args.languages, args.epochs, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DPR model on the multieup dataset with alignment loss")
    parser.add_argument("--mode", type=str, default="train+evaluate", choices=["train", "evaluate", "train+evaluate"],
                        help="Mode to run the script in")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query file (JSON)")
    parser.add_argument("--passage_file", type=str, required=True, help="Path to the passage file (JSON)")
    parser.add_argument("--qrels_file", type=str, required=True, help="Path to the qrels file (JSON)")
    parser.add_argument("--alignment_file", type=str, required=True, help="Path to the alignment file (JSON)")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Model name for the encoder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--log_steps", type=int, default=10, help="Number of steps between logging")
    parser.add_argument("--alignment_loss_weight", type=float, default=0.5, help="Weight for the alignment loss")
    parser.add_argument("--max_query_length", type=int, default=64, help="Maximum length of query tokens")
    parser.add_argument("--max_passage_length", type=int, default=180, help="Maximum length of passage tokens")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the checkpoints and other outputs")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to save the cached tokenized data")
    parser.add_argument("--use_cache", action='store_true', help="Flag to use cached tokenized data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file to resume training")
    parser.add_argument("--dev_input_dir", type=str, required=True,
                        help="Directory containing development data (queries, qrels)")
    parser.add_argument("--dev_passages", type=str, required=True, help="Path to the development passages file (JSON)")
    parser.add_argument("--languages", type=list,
                        default=["EN", "DA", "DE", "NL", "SV", "MT", "RO", "ES", "FR", "IT", "PT", "PL", "BG", "CS",
                                 "SK", "SL", "HR", "HU", "FI", "ET", "LV", "LT", "EL", "GA"],
                        help="List of languages for evaluation")
    parser.add_argument("--sorted_qid", type=str, required=True, help="Path to the sorted QID JSON file")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top passages to retrieve for each query")
    args = parser.parse_args()
    main(args)
