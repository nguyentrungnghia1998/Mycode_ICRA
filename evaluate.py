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

# +
import librosa
import torch
import os
import argparse

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from tqdm import tqdm
from pyctcdecode import build_ctcdecoder


# -

class Inferencer:
    def __init__(self, device, huggingface_folder, w2v_model_path, kenlm_model_path, alpha = 1.0):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(huggingface_folder)
        vocab_dict = self.processor.tokenizer.get_vocab()
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1]) if k not in ["<s>", "</s>"]}
        self.decoder = build_ctcdecoder(
            labels=list(sorted_vocab_dict.keys()),
            kenlm_model_path=kenlm_model_path,
            alpha=alpha
        )
        # self.processor_with_lm = Wav2Vec2ProcessorWithLM(
        #     feature_extractor=self.processor.feature_extractor,
        #     tokenizer=self.processor.tokenizer,
        #     decoder=self.decoder
        # )
        self.model = Wav2Vec2ForCTC.from_pretrained(huggingface_folder).to(self.device)
        if w2v_model_path is not None:
            self.preload_model(w2v_model_path)


    def preload_model(self, model_path) -> None:
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.
        Args:
            model_path: The file path of the *.tar file
        """
        assert os.path.exists(model_path), f"The file {model_path} is not exist. please check path."
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict = True)
        print(f"Model preloaded successfully from {model_path}.")


    def transcribe(self, wav) -> str:
        input_values = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_values
        logits = self.model(input_values.to(self.device)).logits

        # Sử dụng KenLM với beam search decoding
        pred_transcript = self.decoder.decode(logits.cpu().detach().numpy()[0])  # Chuyển logits sang numpy và giải mã
        return pred_transcript

    def run(self, test_filepath):
        filename = test_filepath.split('/')[-1].split('.')[0]
        filetype = test_filepath.split('.')[1]
        if filetype == 'txt':
            f = open(test_filepath, 'r')
            lines = f.read().splitlines()
            f.close()

            f = open(test_filepath.replace(filename, 'transcript_'+filename), 'w+')
            for line in tqdm(lines):
                wav, _ = librosa.load(line, sr = 16000)
                transcript = self.transcribe(wav)
                f.write(line + ' ' + transcript + '\n')
            f.close()

        else:
            wav, _ = librosa.load(test_filepath, sr = 16000)
            print(f"transcript: {self.transcribe(wav)}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR INFERENCE ARGS')
    args.add_argument('-f', '--test_filepath', type=str, required = True,
                      help='It can be either the path to your audio file (.wav, .mp3) or a text file (.txt) containing a list of audio file paths.')
    args.add_argument('-s', '--huggingface_folder', type=str, default = 'huggingface-hub',
                      help='The folder where you stored the huggingface files. Check the <local_dir> argument of [huggingface.args] in config.toml. Default value: "huggingface-hub".')
    args.add_argument('-m', '--w2v_model_path', type=str, default = None,
                      help='Path to the model (.tar file) in saved/<project_name>/checkpoints. If not provided, default uses the pytorch_model.bin in the <HUGGINGFACE_FOLDER>')
    args.add_argument('-k', '--kenlm_model_path', type=str, default = None)
    args.add_argument('-d', '--device_id', type=int, default = 0,
                      help='The device you want to test your model on if CUDA is available. Otherwise, CPU is used. Default value: 0')
    args = args.parse_args()
    
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    inferencer = Inferencer(
        device = device, 
        huggingface_folder = args.huggingface_folder, 
        w2v_model_path = args.w2v_model_path,
        kenlm_model_path = args.kenlm_model_path)

    inferencer.run(args.test_filepath)




