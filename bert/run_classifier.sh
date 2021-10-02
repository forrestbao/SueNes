
## System configuration, GPUs, paths, etc. 
# Configuration on Bao's computer 
export NVIDIA_VISIBLE_DEVICES=0 # set to none to use CPU only
export CUDA_VISIBLE_DEVICES=0  # set to none to use CPU only

EXP_DIR=../exp/
BERT_MODEL=bert_base_uncased
BERT_DIR=$HOME/bert_models/$BERT_MODEL
# End of Bao's configuration 
# Please do not delete

## Experiments related configurations
training_sets="billsum scientific_papers cnn_dailymail big_patent"
methods="sent_delete_char sent_replace_char word_delete word_replace cross word_add"


# We have three test sets, tac, realsumm, and newsroom. 
# Only realsumm is tested with training 
# The othe two are tested when training is off 
for dataset in $training_sets;
do 
  for method in $methods
  do
  human_eval_dataset=realsumm
  python3 run_classifier.py -W ignore \
    --task_name=basic \
    --do_lower_case=true \
    --do_train=True \
    --do_eval=True \
    --do_predict=True \
    --data_dir=$EXP_DIR/data/$dataset/$method/ \
    --output_dir=$EXP_DIR/result\_$BERT_MODEL/$dataset/$method/ \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --max_seq_length=512 \
    --train_batch_size=14 \
    --learning_rate=1e-5 \
    --num_train_epochs=1.0 \
    --convert_batch=1000000 \
    --do_tfrecord=True \
    --human_eval_dataset=$human_eval_dataset
  done
done 

# Now test in newsroom and tac 
for dataset in $training_sets;
do 
  for method in $methods
  do
    for human_eval_dataset in newsroom tac
    do 
    python3 run_classifier.py -W ignore \
      --task_name=basic \
      --do_lower_case=true \
      --do_train=False \
      --do_eval=False \
      --do_predict=True \
      --data_dir=$EXP_DIR/data/$dataset/$method/ \
      --output_dir=$EXP_DIR/result\_$BERT_MODEL/$dataset/$method/ \
      --vocab_file=$BERT_DIR/vocab.txt \
      --bert_config_file=$BERT_DIR/bert_config.json \
      --init_checkpoint=$BERT_DIR/bert_model.ckpt \
      --max_seq_length=512 \
      --train_batch_size=14 \
      --learning_rate=1e-5 \
      --num_train_epochs=1.0 \
      --convert_batch=1000000 \
      --do_tfrecord=True \
      --human_eval_dataset=$human_eval_dataset
    done 
  done
done

