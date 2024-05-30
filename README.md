# ir_languagefairness


CUDA_VISIBLE_DEVICES=0 python src/trainCL_with_eva.py \
    --query_file /data/gpfs/projects/punim0478/jinruiy/EP/multi-EP/mDPR/dpr-multieup/data/raw/train/train_queries_shuffled.json \
    --passage_file /data/gpfs/projects/punim0478/jinruiy/EP/multi-EP/mDPR/dpr-multieup/data/raw/train/train_passages.json \
    --qrels_file /data/gpfs/projects/punim0478/jinruiy/EP/multi-EP/mDPR/dpr-multieup/data/raw/train/train_qrels_shuffled.json \
    --alignment_file /data/gpfs/projects/punim0478/jinruiy/EP/multi-EP/mDPR/dpr-multieup/data/alignment_queries.json \
    --model_name xlm-roberta-base \
    --batch_size 96 \
    --learning_rate 1e-5 \
    --epochs 3 \
    --log_steps 10 \
    --max_query_length 64 \
    --max_passage_length 180 \
    --alignment_loss_weight 1 \
    --output_dir /data/gpfs/projects/punim0029/jinruiy/multi-EP/mdpr/checkpoints/cl \
    --dev_input_dir /data/gpfs/projects/punim0478/jinruiy/EP/multi-EP/mDPR/dpr-multieup/data/raw/dev \
    --dev_passages /data/gpfs/projects/punim0478/jinruiy/EP/multi-EP/mDPR/dpr-multieup/data/raw/dev/dev_passages.json \
    --sorted_qid /data/gpfs/projects/punim0478/jinruiy/EP/multi-EP/mDPR/dpr-multieup/data/raw/dev/sorted_qid_df.csv \
    --mode train+evaluate
