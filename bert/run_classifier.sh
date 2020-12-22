export BERT_BASE_DIR=./bert_tiny
export DATA_DIR=../data/

exp_type=basic  # or TAC or newsroom 

for dataset in cnn_dailymail billsum scientic_papers
do 
  for method in cross add replace delete 
  do 

    python3 run_classifier.py -W ignore \
      --task_name=$exp_type \
      --do_train=True \
      --do_eval=true \
      --do_lower_case=true \
      --do_predict=True \
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