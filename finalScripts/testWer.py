import pandas as pd

import nltk
nltk.download('punkt')
from transformers import pipeline
from transformers import WhisperFeatureExtractor

from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from huggingsound import SpeechRecognitionModel

transcriptions_whisper = []
transcriptions_whisper_finetuned = []
transcriptions_w2v = []
real_transcription = []


df_shuffled = pd.read_csv("/media/isabel/EXTERNAL_USB/tfm/EnglishTalkbank/DOCS/csvs/parityFinetune.csv")
data_test = df_shuffled[int(len(df_shuffled)*0.9):]
for index, row in data_test.iterrows():
  try:

    print("An exception occurred") 
    print('LINEA- ',row)
    real_transcription.append(row['transcription'])
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="English", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", tokenizer = tokenizer, feature_extractor=feature_extractor)
    transcript = pipe(row['audio'],generate_kwargs = {"task":"transcribe", "language":"<|en|>"})
    transcriptions_whisper.append(transcript['text'])
    print(f"whisper: -> {transcript['text']}")

    model = WhisperForConditionalGeneration.from_pretrained("/home/isabel/Documentos/tfm/myModels/models/baseWhisper/checkpoint-3000")
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer = tokenizer, feature_extractor=feature_extractor)
    transcript = pipe(row['audio'],generate_kwargs = {"task":"transcribe", "language":"<|en|>"})
    print(f"finetuned whisper: -> {transcript['text']}")
    transcriptions_whisper_finetuned.append(transcript['text'])

  except: 
    print("An exception occurred")

str_transcript = ''
str_whisper = ''
str_finewhisper = ''
str_whispers2t = ''
fine = ""
trans = ""
whis = ""
array = data_test['transcription'].values
for i in range(len(transcriptions_whisper_finetuned)):
  trans = array[i].lower() + " "
  str_transcript += trans.replace(".", "")
  fine = transcriptions_whisper_finetuned[i].lower() + " "
  str_finewhisper += fine.replace(".", "")
  whis = transcriptions_whisper[i].lower() + " "
  str_whisper += whis.replace(".", "")
#   str_whispers2t += transcriptions_s2t[i].lower() + " "

  print(transcriptions_whisper_finetuned[i])


def calculate_wer(reference, hypothesis):
    ref_words = nltk.word_tokenize(reference.lower())
    hyp_words = nltk.word_tokenize(hypothesis.lower())

    # Calculate the WER using nltk.edit_distance()
    wer = nltk.edit_distance(ref_words, hyp_words) / len(ref_words)

    return wer

reference = str_transcript
hypothesis = str_finewhisper

wer = calculate_wer(reference, hypothesis)
print("Word Error Rate finetuned:", wer)

reference = str_transcript
hypothesis = str_whisper

wer = calculate_wer(reference, hypothesis)
print("Word Error Rate Whisper:", wer)


#
# base con validation partition
# Word Error Rate finetuned: 0.25359444719881014
# Word Error Rate Whisper: 1.0825483391175013


