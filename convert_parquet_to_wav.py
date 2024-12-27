import pandas as pd
import soundfile as sf
import io
from tqdm import tqdm
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Convert parquet audio bytes to wav files.")
    parser.add_argument(
        "--input_parquet_folder", 
        required=True, 
        type=str, 
        help="Path to the folder containing input parquet files."
    )
    parser.add_argument(
        "--output_wav_folder", 
        required=True, 
        type=str, 
        help="Path to the folder to save output wav files."
    )
    return parser


def convert_bytes_to_wav(audio, output_root):
    # Đọc thông tin từ audio["bytes"]
    audio_bytes = audio["bytes"]
    output_path = audio["path"]

    # Chuyển bytes thành buffer
    audio_buffer = io.BytesIO(audio_bytes)

    # Đọc dữ liệu từ buffer
    data, samplerate = sf.read(audio_buffer, dtype='int16')

    # Lưu dữ liệu vào file WAV
    sf.write(os.path.join(output_root, output_path), data, samplerate)


def main(args):
    input_parquet_folder = args.input_parquet_folder
    output_wav_folder = args.output_wav_folder

    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_wav_folder):
        os.makedirs(output_wav_folder)

    list_parquet = os.listdir(input_parquet_folder)

    stt = 0
    for parquet_file in list_parquet:
        print(f"Processing {parquet_file}")
        df = pd.read_parquet(os.path.join(input_parquet_folder,parquet_file))
        audios = df["audio"].tolist()
        for audio in tqdm(audios):
            if audio["path"] is None:
                audio["path"] = f"{stt:06d}.wav"
            convert_bytes_to_wav(audio=audio, output_root=output_wav_folder)
            stt+=1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)