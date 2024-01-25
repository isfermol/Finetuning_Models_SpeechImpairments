from datasets import Dataset
from datasets import load_metric
import pandas as pd
from transformers import Wav2Vec2ForCTC
from datasets import Audio
import random
import gc
import torch
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, List, Union
import evaluate
import pandas as pd
import re

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer

# from transformers import Wav2Vec2CTCTokenizer
# from transformers import Wav2Vec2FeatureExtractor
# from transformers import Wav2Vec2Processor


from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from datasets import load_dataset
from transformers import Wav2Vec2Processor
#df_shuffled = pd.read_csv("./finetuning/parityFinetune.csv")
df_shuffled = pd.read_csv("./finetuning/finetuneeee.csv")

train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.75)])

#test_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.75):int(len(df_shuffled)*0.9)])
#val_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.9):])train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.75)])
#train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.75)])train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.75)])train_dataset = Dataset.from_pandas(df_shuffled[>


train = df_shuffled[:int(len(df_shuffled)*0.75)]

test = df_shuffled[int(len(df_shuffled)*0.75):int(len(df_shuffled)*0.9)]
val = df_shuffled[int(len(df_shuffled)*0.9):]
len(train_dataset)

#model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
#feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h", sequence_length=12)
#processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#train_encodings = tokenizer(train['myaudio'], truncation=True, padding=True)
#test_encodings = tokenizer(test['myaudio'], truncation=True, padding=True)
 # test_encodings = tokenizer(twTest, truncation=True, padding=True)
class classDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, transcription):
          self.myaudio = encodings
          self.transcription = transcription

      def __getitem__(self, idx):
          item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
          item['transcription'] = torch.tensor(self.transctiption)
          return item

      def __len__(self):
          return len(self.transcription)

#train_dataset = classDataset(train_encodings, train['transcription'])
#test_dataset = classDataset(test_encodings, test['transcription'])
#
test_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.75):int(len(df_shuffled)*0.9)])
val_dataset = Dataset.from_pandas(df_shuffled[int(len(df_shuffled)*0.9):])
train_dataset = Dataset.from_pandas(df_shuffled[:int(len(df_shuffled)*0.75)])



train_dataset = train_dataset.cast_column('myaudio', Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column('myaudio', Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column('myaudio', Audio(sampling_rate=16000))
chars_to_ignore_regex = "[\,\?\.\!\-\;\:\']"

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower() + " "
    return batch

train_dataset = train_dataset.map(remove_special_characters)
test_dataset = test_dataset.map(remove_special_characters)

def extract_all_chars(batch):
#ITERAR EN LAS FILAS

 #for index, row in batch.iterrows():
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

#vocabs = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True)
#vocab_test = test_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True)
print('train_dataset ------- ', train_dataset)
print('train_dataset[0] ------- ', train_dataset[0])

vocabs = train_dataset.map(extract_all_chars)
vocab_test = test_dataset.map(extract_all_chars)
#array of arrays
print('voc -------------- ', vocabs)
myvoc = []
testVoc = []
for el in vocabs["vocab"]:
	for letter in el[0]:
		if letter not in myvoc:
			myvoc.append(letter)
print('vocab -------------- ', myvoc, ' length vocab-> ', len(myvoc) )
for el in vocab_test["vocab"]:
        for letter in el[0]:
                if letter not in testVoc:
                        testVoc.append(letter)

#vocab_list = list(set(vocabs["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_list = list(set(myvoc) | set(testVoc))



vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print('vdict--------- ', vocab_dict)


vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
#feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, sequence_length=12, padding_value=0.0, do_normalize=True, return_attention_mask=False)
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
rand_int = random.randint(0, len(train_dataset["transcription"]))

print("Target text:", len(train_dataset[rand_int]["transcription"]))
print("Input array shape:", np.asarray(train_dataset[rand_int]["myaudio"]["array"]).shape)
print("Sampling rate:", train_dataset[rand_int]["myaudio"]["sampling_rate"])

#def prepare_dataset(batch):
 #   audio = batch["myaudio"]

    # batched output is "un-batched" to ensure mapping is correct
  #  batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
   # print(batch["input_values"].shape)
    
    #with processor.as_target_processor():
     #   batch["labels"] = processor(batch["transcription"]).input_ids
   # return batch


#-----------------------------------------------------------------------
def tokens(batch):
    batch[transcriptio] = tokenizer(batch, truncation=True, padding=True)
    return batch
#train_dataset = train_dataset.map(tokens)
#test_dataset = test_dataset.map(tokens)

#train_dataset = train_dataset.map(prepare_dataset, num_proc = 4)
#def prepare_dataset(batch, event=None):
 #   print('b1 ',batch)
  #  audio = batch["myaudio"]

    # batched output is "un-batched" to ensure mapping is correct
   # batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
   # print('b21 ',batch)

   # batch["input_length"] = len(batch["input_values"])
   # print('b2 ',batch)


   # with processor.as_target_processor():
    #    batch["labels"] = processor(batch["transcription"]).input_ids
    
   # print('b3 ',batch)

   # return batch


def prepare_dataset(batch, event=None):
 #   print('b1 ',batch)
    audio = batch["myaudio"]

    # batched output is "un-batched" to ensure mapping is correct
   # batch["input_values"] = processor(audio["array"], sequence_length=12, sampling_rate=16000).input_values[0]
 
    batch["input_values"] = processor(audio["array"],  sampling_rate=16000).input_values[0]
#   print('b21 ',batch)

   # batch["input_length"] = len(batch["input_values"])
#    print('b2 ',batch)

   
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    #------------------------------------------------test------------------------------
    print('b3 ',batch["labels"])
   # mask_length = 8  # Por ejemplo
    #for i in range(len(batch["labels"])):
     #  batch["labels"][i] = batch["labels"][i][:mask_length]
    
     #-------------------------------------------------------------------------------------

    return batch


train_dataset = train_dataset.map(prepare_dataset, num_proc=1)

test_dataset = test_dataset.map(prepare_dataset, num_proc=1)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
           # padding=self.padding,
#            max_length=self.max_length,
            max_length= 12,
            padding = True,
        
#   padding = "max_length",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                #padding=self.padding,
                padding = True,
                max_length= 12,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

# Load configuration from .json file
#config = Wav2Vec2Config.from_pretrained("wav2vec2-base-960h-config.json")

# Update the max_length parameter
#config.max_length = 6  # Replace 'new_max_length' with your desired value

# Instantiate the 

#model using the modified configuration
#model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-base-960h-config.json", config=config)


from transformers import Wav2Vec2Config

#config = Wav2Vec2Config()

#config.mask_length = 8
#config.sequence_length = 9

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
 #   config = config,
    mask_time_length = 2,
    mask_feature_length = 2,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
#config=config
)

#model.config.set_parameters(mask_length=9, sequence_length=10)

model.freeze_feature_extractor()
#train_dataset = tokenizer.encode(train_dataset, add_special_tokens=True, return_tensors="pt")
#test_dataset = tokenizer.encode(test_dataset, add_special_tokens=True, return_tensors="pt")


training_args = TrainingArguments(
  output_dir='./finetuning/models/model23test_wav2veccontexTut/',
  group_by_length=True,
  per_device_train_batch_size=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,

#    tokenizer=processor.feature_extractor,
)

history = trainer.train()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
