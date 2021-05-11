export BERT_BASE_DIR=$HOME/bert_models/bert_base_uncased
export DATA_DIR=/mnt/12T/data/NLP/anti-rogue/data
export RESULT_DIR=/mnt/12T/data/NLP/anti-rogue/result_base_sent

export NVIDIA_VISIBLE_DEVICES=0 # set to none to use CPU only
export CUDA_VISIBLE_DEVICES=0  # set to none to use CPU only

exp_type=basic  # or TAC or newsroom 

for dataset in cnn_dailymail billsum scientific_papers big_patent
do 
  for method in delete # cross delete replace add mix 
  do 

    cat $DATA_DIR/$dataset/$method/train_*.tsv > $DATA_DIR/$dataset/$method/train.tsv
    cat $DATA_DIR/$dataset/$method/test_*.tsv > $DATA_DIR/$dataset/$method/test.tsv

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
      --max_seq_length=512 \
      --train_batch_size=14 \
      --learning_rate=1e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$RESULT_DIR/$dataset/$method/ \
      --convert_batch=1000000 \
      --do_tfrecord=True
  done
done 
