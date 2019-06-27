#!/bin/sh

export BERT_BASE_DIR=/nethome/zyu336/uncased_L-12_H-768_A-12

python create_pretraining_data.py \
    --input_file=/nethome/zyu336/emotionX2019/emotionpush_eval.txt \
    --output_file=./tmp/tf_examples.tfrecord \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5

python run_pretraining.py \
    --input_file=./tmp/tf_examples.tfrecord \
    --output_dir=./tmp/pretraining_output \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=10000 \
    --num_warmup_steps=10 \
    --learning_rate=2e-5

python extract_features.py \
    --input_file=/nethome/zyu336/emotionX2019/emotionpush_eval.txt \
    --output_file=/nethome/zyu336/emotionX2019/data/EmotionPush_bert/emotionpush_eval_embedding.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=./tmp/pretraining_output/model.ckpt-10000 \
    --layers=-1 \
    --max_seq_length=128 \
    --batch_size=8 \

rm -rf ./tmp/*

python create_pretraining_data.py \
    --input_file=/nethome/zyu336/emotionX2019/friends_eval.txt \
    --output_file=./tmp/tf_examples.tfrecord \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5

python run_pretraining.py \
    --input_file=./tmp/tf_examples.tfrecord \
    --output_dir=./tmp/pretraining_output \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=10000 \
    --num_warmup_steps=10 \
    --learning_rate=2e-5

python extract_features.py \
    --input_file=/nethome/zyu336/emotionX2019/friends_eval.txt \
    --output_file=/nethome/zyu336/emotionX2019/friends_eval_embedding.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=./tmp/pretraining_output/model.ckpt-10000 \
    --layers=-1 \
    --max_seq_length=128 \
    --batch_size=8 \

rm -rf ./tmp/*

