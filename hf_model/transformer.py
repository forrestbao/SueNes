from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import RobertaTokenizerFast
from transformers import EncoderDecoderModel
from transformers import EncoderDecoderModel
from transformers import TrainingArguments
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

# from seq2seq_trainer import Seq2SeqTrainer
from dataclasses import dataclass, field
from typing import Optional
import datasets
import transformers
import pandas as pd
from datasets import Dataset


TRAIN_TSV = 'exp/data/cnn_dailymail/sent_delete_sent/train.tsv'

# tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
# model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
# text = "By. Ashley Collman for MailOnline. Donald Trump made the best case that his hair is real by taking the ALS ice-bucket challenge on Wednesday. The real-estate mogul was nominated by Homer Simpson, Vince McMahon and Mike Tyson  and decided to accept the challenge in style. From the top of Trump Tower, the Donald had the beautiful Miss USA and Miss Universe dump ice-cold bottled water over his head to raise awareness of ALS- also known as Lou Gherig's disease. Scroll down for video. Challenge accepted: Billionaire real-estate mogul Donald Trump took the ALS ice bucket challenge on Thursday, accepting nominations from Homer Simpson, Vince McMahon and Mike Tyson. In the video he even made fun of the ambiguity surrounding his well-coiffed hair. 'Everybody is going crazy over this thing,' he said. 'I guess they want to see whether or not it's my real hair - which it is.' The two beauty-pageant queens then dump the two buckets of water over his head, soaking his bespoke suit. 'What a mess,' Trump says. Doing it in style: Sparing no cost, Trump had only the finest Trump bottled water poured over his head in the challenge. You have 24 hours: The Donald nominated President Obama, as well as his sons Eric and Donald Jr to take the challenge next. Assistants: Miss Universe (left) and Miss USA (right) helped Trump take the challenge. Trump owns the beauty pageant organizations. Lots of glue? The Donald's hair held up well to the challenge, despite a long-standing rumor that he wears a toupée. 'What a mess,' Trump said after the two beauty queens dumped the water over his head. The ALS ice-bucket challenge has been sweeping the internet. Those who are challenged can either opt out by donating to the cause, or film themselves dumping ice-water over their heads. Many have chosen to both donate and post a video. While Trump did not post about donating to the charity, the billionaire likely wrote out a check for the good cause. Since July, the ice-bucket challenge has helped raise over $90million to go towards ALS research. Trump went on to nominate President Obama, as well as his sons Eric and Donald Trump Jr. Momentous: The ALS ice-bucket challenge has helped raise over $90million in donations since July."
# ref_summary = "The real estate mogul nominated President Obama as well as his sons Eric and Donald Jr to take the challenge next."
# text_token_ids = tokenizer(text)
# text_token_ids = tokenizer.convert_ids_to_tokens(text_tokens.input_ids)
# print(text_token_ids)

def read_train_data():
    '''Read train.csv, find the max scored reference summary
    and create dataframes.'''
    data = []
    with open(TRAIN_TSV) as file:
        for item in file:
            items = item.split('\t')
            best_ref = items[1]
            best_score = items[2]
            for i in range(4, len(items)):
                if items[i] > best_score:
                    best_ref = items[i-1]
                    best_score = items[i]
            data_item = [items[0], best_ref, best_score]
            data.append(data_item)
    cols = ['text', 'summary', 'score']
    # df = pd.read_csv(TRAIN_TSV, delimiter='\t', header=None, names=cols)
    df = pd.DataFrame(data, columns=cols)
    return df

df = read_train_data()
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
# df.head(1)

train_data = Dataset.from_pandas(df[:27000])
val_data = Dataset.from_pandas(df[27000:28211])
test_data = Dataset.from_pandas(df[28211:])

train_data = Dataset.from_pandas(df[:101])
val_data = Dataset.from_pandas(df[101:151])
test_data = Dataset.from_pandas(df[151:171])

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
batch_size=256
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]

    return batch

#processing training data
train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "summary"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

#processing validation data
val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "summary"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "roberta-base", "roberta-base", tie_encoder_decoder=True)

# set special tokens
roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
roberta_shared.config.eos_token_id = tokenizer.eos_token_id

# sensible parameters for beam search
# set decoding params                               
roberta_shared.config.max_length = 512
roberta_shared.config.early_stopping = True
roberta_shared.config.no_repeat_ngram_size = 3
roberta_shared.config.length_penalty = 2.0
roberta_shared.config.num_beams = 4
roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

rouge = datasets.load_metric("rouge")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

    # evaluate_during_training=True,
    # fp16=True, 
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    do_train=True,
    do_eval=True,
    logging_steps=2, 
    save_steps=16, 
    eval_steps=500, 
    warmup_steps=500, 
    overwrite_output_dir=True,
    save_total_limit=1,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()
print('Training done!')

print('Test started!')
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = EncoderDecoderModel.from_pretrained("./checkpoint-6432")
model.to("cuda")
batch_size = 1024

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred"] = output_str
    return batch

results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["text"])
pred_str = results["pred"]
label_str = results["summary"]
print('Test done!')