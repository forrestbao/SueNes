#!/usr/bin/env bash

export BERT_BASE_DIR=/work/Data/uncased_L-12_H-768_A-12
#/work//bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/work/Data/

# python
/usr/bin/python3.6 run_classifier.py \
  --task_name=cnndm \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/cnndm \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=10 \
  --learning_rate=2e-5 \
  --num_train_epochs=4.0 \
  --output_dir=./tmp/bert_output/

