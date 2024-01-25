from datasets import Dataset

import pandas as pd

from datasets import Audio

import gc
from transformers import WhisperProcessor
from transformers import WhisperTokenizer

from transformers import WhisperFeatureExtractor
import torch
import packaging
from dataclasses import dataclass

from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration

import evaluate
import pandas as pd
df_shuffled = pd.read_csv("~/finetuning/finetuneeee.csv")
from tensorflow.python.client import device_lib
#df_shuffled = pd.read_csv("~/finetuning/parityFinetune.csv")
#df_shuffled = pd.read_csv("~/finetuning/finetuneeee.csv")
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_available_gpus()
#train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.75)])
#test_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.75):int(len(df_shuffled)*0.9)])
#val_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.9):])

train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.7)])
test_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.7):])
#val_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.9):])


train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.7)])
test_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.7):])

df_shuffled[:int(len(df_shuffled)*0.75)]
len(train_dataset)
train_dataset = train_dataset.cast_column('myaudio', Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column('myaudio', Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
def prepare_dataset(examples):

    # compute log-Mel input features from input audio array

    audio = examples['myaudio']

    examples['input_features'] = feature_extractor(

        audio['array'], sampling_rate=16000).input_features[0]
 
    del examples['myaudio']

    sentences = examples['transcription']

    # encode target text to label ids

    examples['labels'] = tokenizer(sentences).input_ids

    del examples['transcription']

    return examples
train_dataset = train_dataset.map(prepare_dataset, num_proc=1)

test_dataset = test_dataset.map(prepare_dataset, num_proc=1)



@dataclass

class DataCollatorSpeechSeq2SeqWithPadding:

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        # split inputs and labels since they have to be of different lengths and need different padding methods

        # first treat the audio inputs by simply returning torch tensors

        input_features = [{'input_features': feature['input_features']} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        # get the tokenized label sequences

        label_features = [{'input_ids': feature['labels']} for feature in features]

        # pad the labels to max length

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')

        # replace padding with -100 to ignore loss correctly

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,

        # cut bos token here as it’s append later anyways

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():

            labels = labels[:, 1:]

        batch['labels'] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load('wer')
def compute_metrics(pred):

    pred_ids = pred.predictions

    label_ids = pred.label_ids

    # replace -100 with the pad_token_id

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {'wer': wer}
# Load a Pre-Trained Checkpoint


model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(

    output_dir='./finetuning/models/smallBeWhisperSpeechIndependent_lr-5_max4000_warm500',  # change to a repo name of your choice

    per_device_train_batch_size=8,

    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size

    learning_rate=1e-5,

    warmup_steps=500,

    # max_steps=15000,
    max_steps=4000,


    gradient_checkpointing=True,

    fp16=True,

    evaluation_strategy='steps',

    per_device_eval_batch_size=1,

    predict_with_generate=True,

    generation_max_length=225,

    save_steps=500,

    eval_steps=250,

    # logging_steps=25,

    report_to=['tensorboard'],

    load_best_model_at_end=True,

    metric_for_best_model='wer',

    greater_is_better=False,

    push_to_hub=False,

)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(

    args=training_args,

    model=model,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    tokenizer=processor.feature_extractor,

)
trainer.train()
