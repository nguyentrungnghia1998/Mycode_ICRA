# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd

df_T6 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_T6/final_label.csv", delimiter=";")
df_T9 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_T9/final_label.csv", delimiter=";")
df_training = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_training/final_label.csv", delimiter=";")
df_training.rename(columns={'audio': 'folder'}, inplace=True)
# df_day_10_05 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_voice_daily/month_10/day_05/final_label.csv", delimiter=";")
# df_day_10_19 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_voice_daily/month_10/day_19/final_label.csv", delimiter=";")
# df_day_12_02 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_voice_daily/month_12/day_02/final_label.csv", delimiter=";")
# df_day_12_09 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_voice_daily/month_12/day_09/final_label.csv", delimiter=";")
# df_day_12_10 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_voice_daily/month_12/day_10/final_label.csv", delimiter=";")
# df_day_12_12 = pd.read_csv("/home/jovyan/ai-core/speech_to_text/data/ghtk_voice_daily/month_12/day_12/final_label.csv", delimiter=";")

# +
# len(df_T6), len(df_T9), len(df_training), len(df_day_10_05), len(df_day_10_19), len(df_day_12_02), len(df_day_12_09), len(df_day_12_10), len(df_day_12_12)
# -

df_full = pd.concat([df_T6, df_T9, df_training], axis=0, ignore_index=True)
df_full.drop(["start", "end"], axis =1, inplace=True)

df_T6

all_sentences = list(set(df_full['text'].tolist()))
len(all_sentences)

# +
# import re
# from tqdm import tqdm
# chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

# with open('text.txt', 'w') as f:
#     for sentence in tqdm(all_sentences):
#         f.write(sentence)
#         f.write('\n')

# +
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

# Load processor
processor = Wav2Vec2Processor.from_pretrained("custom_model")

# Get the vocabulary dictionary
vocab_dict = processor.tokenizer.get_vocab()

# Remove <s> and </s> tokens if they exist
# vocab_dict = {k: v for k, v in vocab_dict.items() if k not in tokens_to_remove}

# Sort the vocabulary dictionary by the values (IDs)
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

# Output the sorted vocabulary dictionary
print(sorted_vocab_dict)

# +
# from transformers import Wav2Vec2CTCTokenizer

# # Save the modified vocabulary to a temporary file
# vocab_file = "updated_vocab.json"
# with open(vocab_file, "w") as f:
#     import json
#     json.dump(vocab_dict, f)

# # Reload tokenizer with updated vocab
# tokenizer = Wav2Vec2CTCTokenizer(vocab_file)

# # Replace the tokenizer in the processor
# processor.tokenizer = tokenizer

# +
from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path="5gram_correct.arpa",
)
# -

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)


# +
# processor_with_lm.save_pretrained("GHTK_wav2vec2")

# +
class CFG:
    my_model_name = 'custom_model'
    processor_name = 'GHTK_wav2vec2'
    processor_without_LM = 'custom_model'
    
from transformers import Wav2Vec2ProcessorWithLM,pipeline

# processor = Wav2Vec2ProcessorWithLM.from_pretrained(CFG.processor_name)


asr_w_LM = pipeline("automatic-speech-recognition", model=CFG.my_model_name ,feature_extractor=processor_with_lm.feature_extractor, tokenizer=processor_with_lm.tokenizer,decoder=processor_with_lm.decoder)
asr_wo_LM = pipeline("automatic-speech-recognition", model = CFG.my_model_name, feature_extractor=processor.feature_extractor, tokenizer=processor.tokenizer)

# +
import librosa
def infer(audio_path):
    speech, sr = librosa.load(audio_path, sr=processor_with_lm.feature_extractor.sampling_rate)
    my_LM_prediction = asr_w_LM(
                speech
            )

    return my_LM_prediction["text"]

def infer_wo_LM(audio_path):
    speech, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    my_LM_prediction = asr_wo_LM(
                speech
            )

    return my_LM_prediction['text']


# -

from IPython.display import Audio

sample = df_full.sample()
path = sample["folder"].values[0]
print("With LM: ", infer(path))
print("Without LM: ", infer_wo_LM(path))
print("Label: ",sample["text"].values[0])
Audio(path)


