# ir_languagefairness

## Step 1 Download data
https://drive.google.com/drive/folders/1UFjEx0lUzW25WDgCVBQ3kdiMrAM9TXoZ?usp=sharing

## Step 2 Install environment 

## Train with Evaluate

CUDA_VISIBLE_DEVICES=0 python src/trainCL_with_eva.py \
    --query_file {your own path}/train_queries_shuffled.json \
    --passage_file {your own path}/train_passages.json \
    --qrels_file {your own path}/train_qrels_shuffled.json \
    --alignment_file {your own path}/alignment_queries.json \
    --model_name xlm-roberta-base \
    --batch_size 96 \
    --learning_rate 1e-5 \
    --epochs 3 \
    --log_steps 10 \
    --max_query_length 64 \
    --max_passage_length 180 \
    --alignment_loss_weight 1 \
    --output_dir {your own path to save the model checkpoints}/checkpoints/cl \
    --dev_input_dir {your own path to save the dev data}/dev \
    --dev_passages {your own path to save the dev data}/dev_passages.json \
    --sorted_qid {your own path to save the dev data}/sorted_qid_df.csv \
    --mode train+evaluate
