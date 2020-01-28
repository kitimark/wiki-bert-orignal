#!/bin/bash

source venv/bin/activate

export WIKI_DIR=gs://bert-th/wiki-dataset
export BERT_BASE_DIR=gs://bert-th/bert_multi_case

python run_classifier.py \
  --task_name=wiki_question_type \
  --do_train=true \
  --do_eval=true \
  --data_dir=$WIKI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --use_tpu=true \
  --tpu_name=grpc://10.111.210.26:8470 \
  --output_dir=gs://bert-th/output-wiki_question_type_multi-1