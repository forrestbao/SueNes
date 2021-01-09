export BERT_BASE_DIR=/home/forrest/bert_models/bert_tiny
export DATA_DIR=../data/

exp_type=basic  # or TAC or newsroom 

for dataset in cnn_dailymail
do 
  for method in mix # cross add replace delete 
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
      --init_checkpoint=./result/$dataset/$method/model.ckpt-53800 \
      --max_seq_length=512 \
      --train_batch_size=64 \
      --learning_rate=1e-5 \
      --num_train_epochs=3.0 \
      --output_dir=./result/$dataset/$method/ \
      --convert_batch=100000 \
      --do_tfrecord=False 
  done
done 

# learning rete was 2e-5 for bert_tiny, cross/add/delete/replace
# changed to 1e-5 for bert_tiny, mix
#       --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
