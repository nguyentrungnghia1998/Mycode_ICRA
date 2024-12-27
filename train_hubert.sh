CUDA_VISIBLE_DEVICES=0 python contentvec/fairseq/fairseq_cli/hydra_train.py \
  --config-dir config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/home/jovyan/ai-core/speech_to_text/nghia_semi_supervised/icra/data_mau/public_data \
  task.label_dir=/home/jovyan/ai-core/speech_to_text/nghia_semi_supervised/icra/data_mau/public_data/km_label \
  task.labels='["km"]' \
  model.label_rate=100