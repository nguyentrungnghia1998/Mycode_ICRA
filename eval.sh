python evaluate.py \
    -f 00000.wav \
    -s custom_model \
    -m /home/jovyan/ai-core/speech_to_text/nghia_semi_supervised/icra/model/wav2vec2-finetuned-vietnamese_2/checkpoint-20000/pytorch_model.bin \
    -k 5gram_correct.arpa \
    -d 0