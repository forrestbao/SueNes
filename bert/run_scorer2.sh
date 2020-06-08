#!/usr/bin/env bash

export BERT_BASE_DIR=/work/Data/uncased_L-12_H-768_A-12
export GLUE_DIR=/work/Data

rm -rf tmp/bert_output_scorer/*

# python
/usr/bin/python3.6 run_scorer.py \
  --task_name=sts \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/cnndm-30k \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=50 \
  --output_dir=./tmp/bert_output_scorer2/ \
  --save_checkpoints_steps=200


