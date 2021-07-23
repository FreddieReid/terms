

#download packages

!pip install datasets
!pip install transformers
!pip install rouge_score
!pip install --upgrade fsspec
import transformers
import datasets
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed)
import rouge_score
import nltk

nltk.download("punkt", quiet=True)
import logging
import pandas as pd
import numpy as np
import logging
from datasets import Dataset
from sklearn.model_selection import train_test_split

# calculate readability Level

!pip install py-readability-metrics
!python -m nltk.downloader punkt

# install packages for bluert


logger = logging.getLogger(__name__)

#!pip install --upgrade pip # ensures that pip is current
!pip install git+https://github.com/google-research/bleurt.git

# ensure it is using the cpu to save ram on GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# load bleurt

bleurt = load_metric("bleurt")

#load model and training arguments
class model_args:
    model_name = 'sshleifer/distill-pegasus-cnn-16-4'
    use_fast_tokenizer = True

class data_args:
  pad_to_max_length = True
  ignore_pad_token_for_loss = True


output_dir = 'saved_models'
!mkdir "saved_models/"

training_args = Seq2SeqTrainingArguments(predict_with_generate=True, output_dir= output_dir, per_device_train_batch_size=1,
                                         per_device_eval_batch_size=1, num_train_epochs=4, evaluation_strategy = "epoch",weight_decay=0.01,learning_rate=2e-5,
                                         logging_steps = 2, gradient_accumulation_steps = 36, skip_memory_metrics = False,  run_name = 'pegasus_summarization',
                                        load_best_model_at_end=True)

# get own annotated data

cols1 = ['document_ref', 'original_text', 'abtractive_summary']

own_data = pd.read_csv('../input/summaryc/terms_summarisations.csv', usecols = cols1).reset_index(drop = True)

own_data.isnull().sum()


# get professionally annotated data

cols = ["document_ref", "original_text", "labels", "abtractive_summary"]

data = pd.read_csv('../input/summaryf/manor_li_data.csv', usecols = cols)

total_data = pd.concat([data, own_data])


#extract 'important data'
own_data_imp = own_data[own_data["abtractive_summary"] != "not important"]

# split data

training_data = data
validation_data = own_data_imp[(own_data_imp["document_ref"] == "barc_pc") | (own_data_imp["document_ref"] == "sant_cc_tc")]
test_data = own_data_imp[(own_data_imp["document_ref"] != "barc_pc") & (own_data_imp["document_ref"] != "sant_cc_tc")]



#create dataset

training_dataset = Dataset.from_pandas(training_data)
validation_dataset = Dataset.from_pandas(validation_data)
test_dataset = Dataset.from_pandas(test_data)

#load model

model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_fast=True)

#define preprocessing

def preprocess_function(examples):
    inputs = examples["original_text"]
    targets = examples["abtractive_summary"]
    model_inputs = tokenizer(inputs, max_length= 1024, padding= "max_length", truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding= "max_length", truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#perform preprocessing

column_names= ["document_ref"]
column_names1 = ["document_ref", "__index_level_0__"]

training_dataset = training_dataset.map(preprocess_function,batched=True,remove_columns=column_names)

eval_dataset = validation_dataset.map(preprocess_function,batched=True,remove_columns=column_names1)

test_dataset = test_dataset.map(preprocess_function,batched=True,remove_columns=column_names1)

cols = ["abtractive_summary", "original_text"]

training_dataset = training_dataset.remove_columns(cols)
eval_dataset = eval_dataset.remove_columns(cols)
test_dataset = test_dataset.remove_columns(cols)

# data collator

label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
      tokenizer,
      model=model,
      label_pad_token_id=label_pad_token_id,
      pad_to_multiple_of=8 if training_args.fp16 else None,
  )

metric = load_metric("rouge")

# define compute metrics

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


#load the trainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# train the model

%%time

train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics

#55351584650c1776ec7b4204c92ad0049d629818

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#get predictions against validtion and test set

pred_results = trainer.predict(test_dataset=eval_dataset, metric_key_prefix="predict")

pred_results.metrics

predictions = tokenizer.batch_decode(pred_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)

# get BLEURT scores

bleurt_scores = bleurt.compute(predictions=predictions, references=test_data["abtractive_summary"])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )

# get readability scores

text = pd.DataFrame(predictions).to_string(header = False, index = False, index_names = False)

from readability import Readability

r = Readability(text)
fk = r.flesch_kincaid()
print(fk.grade_level)

r = Readability(text)
dc = r.dale_chall()
print(dc.score)

r = Readability(text)
cl = r.coleman_liau()
print(cl.grade_level)
