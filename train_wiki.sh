#!/bin/bash

export BERT_DIR=gs://bert-th/bert_multi_case
export WIKI_DIR=gs://bert-th/wiki-dataset
export TPU_NAME=grpc://10.26.200.90:8470
export OUTPUT_DIR=output

python run_wiki.py \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$WIKI_DIR/wiki_train.json \
  --do_predict=True \
  --predict_file=$WIKI_DIR/wiki_dev.json \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=4.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR \
  --use_tpu=True \
  --tpu_name=$TPU_NAME