#!/usr/bin/env bash

export BERT_BASE_DIR=/home/gluo/uncased_L-12_H-768_A-12
export GLUE_DIR=/home/gluo/Dataset

name=$1
# rm -rf tmp/bert_output_scorer_${name}/*

# python
python3 run_scorer.py \
  --task_name=mutation-${name} \
  --do_train=false \
  --do_eval=false \
  --do_test=true \
  --data_dir=$GLUE_DIR/cnndm-30k \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=4.0 \
  --output_dir=./tmp/bert_output_scorer_${name}/ \
  --save_checkpoints_steps=200


