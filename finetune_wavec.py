import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer)
import torch
import re

# +
# Đọc DataFrame của bạn (nếu chưa có)
# Đọc DataFrame của bạn (nếu chưa có)
df_train = pd.read_csv("filtered_train_data.csv")
df_test = pd.read_csv("final_label_test.csv")

train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))
# +
# Tải processor
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

# Định nghĩa tập hợp các ký tự cần loại bỏ
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"“%\‘\’\']'

def remove_special_characters(batch):
    batch["labeled_text"] = batch["labeled_text"].lower()
    batch["labeled_text"] = re.sub(chars_to_ignore_regex, '', batch["labeled_text"])
    return batch

def speech_file_to_array(batch):
    speech_array, sampling_rate = librosa.load(batch["audio"], sr=16000)
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    return batch

def prepare_dataset(batch):
    # Xử lý âm thanh
    batch = speech_file_to_array(batch)
    # Chuẩn bị input_values
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0]
    # Chuẩn bị labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["labeled_text"]).input_ids
    return batch

# Áp dụng tiền xử lý
train_dataset = train_dataset.map(remove_special_characters)
test_dataset = test_dataset.map(remove_special_characters)

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

# +
import dataclasses
from typing import Any, Dict, List, Union

@dataclasses.dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Thay thế giá trị padding trong labels bằng -100 để bỏ qua trong quá trình tính loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# +
model = Wav2Vec2ForCTC.from_pretrained(
    "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
model.config.ctc_zero_infinity = True

# Đảm bảo mô hình sử dụng đúng số lượng lớp âm vị (vocab size)
model.config.vocab_size = len(processor.tokenizer)

# model.freeze_feature_encoder()

# Định nghĩa tham số huấn luyện
training_args = TrainingArguments(
    output_dir="model/wav2vec2-finetuned-vietnamese_4",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=torch.cuda.is_available(),  # Sử dụng FP16 nếu có GPU hỗ trợ
    save_steps=2500,
    eval_steps=2500,
    logging_steps=50,
    max_steps=50000,
    learning_rate=5e-5,
    warmup_steps=5000,
    save_total_limit=5,
)


import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_annealing_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles, lr_min):
    """
    Tạo scheduler với learning rate giảm theo hàm cosine từ lr_max đến lr_min, với giai đoạn warmup.

    Args:
        optimizer: Optimizer cần điều chỉnh learning rate.
        num_warmup_steps: Số bước cho giai đoạn warmup.
        num_training_steps: Tổng số bước huấn luyện.
        num_cycles: Số chu kỳ cosine annealing.
        lr_min: Learning rate tối thiểu.

    Returns:
        Một scheduler dạng LambdaLR.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Giai đoạn warmup: tăng linearly từ 0 đến lr_max
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Giai đoạn cosine annealing
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * ((progress % (1/num_cycles)) * num_cycles)))
            return lr_min / training_args.learning_rate + (1 - lr_min / training_args.learning_rate) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

from transformers import Trainer, AdamW

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps):
        # Tạo optimizer nếu chưa có
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=training_args.learning_rate,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01
            )
        # Tạo scheduler tùy chỉnh
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_annealing_scheduler_with_warmup(
                self.optimizer,
                num_warmup_steps=training_args.warmup_steps,
                num_training_steps=training_args.max_steps,
                num_cycles=int((training_args.max_steps - training_args.warmup_steps) / 10000),
                lr_min=5e-6  # lr_min
            )


# + endofcell="--"
from jiwer import wer as wer_metric

# wer_metric = evaluate.load("wer")

# -

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = processor.batch_decode(pred_ids)
    # Xóa các ký tự đặc biệt trong văn bản dự đoán
    pred_str = [re.sub(chars_to_ignore_regex, '', s).lower() for s in pred_str]

    # Xử lý labels
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    label_str = [re.sub(chars_to_ignore_regex, '', s).lower() for s in label_str]

    wer = wer_metric(label_str, pred_str)
    return {"wer": wer}


# from transformers import TrainerCallback

# class LogGradNormCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         """
#         Hàm này được gọi sau khi Trainer hoàn thành 1 step (có/không có gradient accumulation).
#         """
#         trainer = kwargs["trainer"]
#         model = trainer.model

#         # Tính gnorm
#         parameters = [p for p in model.parameters() if p.grad is not None]
#         if len(parameters) > 0:
#             total_norm = 0.0
#             norm_type = 2.0
#             for p in parameters:
#                 param_norm = p.grad.data.norm(norm_type)
#                 total_norm += param_norm.item() ** norm_type
#             total_norm = total_norm ** (1.0 / norm_type)

#             # Log ra bảng console / tensorboard:
#             # logs mặc định của HF dùng dictionary "logs"
#             logs = {"gnorm": total_norm}
#             trainer.log(logs)  # Hoặc self.log, tuỳ version HF

#         return control


# # +
trainer = CustomTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor
    # callbacks=[LogGradNormCallback()]  # Thêm callback
)

trainer.train()

# trainer.save_model("model/wav2vec2-finetuned-vietnamese_2")
# processor.save_pretrained("model/wav2vec2-finetuned-vietnamese_2")


