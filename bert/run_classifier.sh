export BERT_BASE_DIR=$HOME/bert_models/bert_base
export DATA_DIR=$HOME/anti-rouge/data/
export RESULT_DIR=$HOME/anti-rouge/bert/result_base

export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

exp_type=basic  # or TAC or newsroom 

for dataset in scientific_papers # billsum cnn_dailymail
do 
  for method in cross add replace delete mix
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
      --max_seq_length=400 \
      --train_batch_size=14 \
      --learning_rate=1e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$RESULT_DIR/$dataset/$method/ \
      --convert_batch=1000000 \
      --do_tfrecord=True
  done
done 