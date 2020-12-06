export BERT_BASE_DIR=./bert_tiny
export DATA_DIR=../data/

for dataset in cnn_dailymail 
do 
  for method in cross 
  do 

    python3 run_classifier.py \
      --task_name=$method  \
      --do_train=False \
      --do_eval=true \
      --do_lower_case=true \
      --data_dir=$DATA_DIR/$dataset/$method/ \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=./result/$dataset/$method/

  done
done 