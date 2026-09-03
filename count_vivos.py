import os
from pathlib import Path


def count_records_in_vivos(base_dir="data/vivos"):
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Thư mục '{base_dir}' không tồn tại.")
        return

    # In tiêu đề bảng
    print(f"{'Thư mục':<20} | {'Tổng số file':<15} | {'Số file âm thanh (.wav)':<25}")
    print("-" * 68)

    # Duyệt qua các thư mục con trực tiếp (vd: train, test, mels, pitch...)
    for folder in base_path.iterdir():
        if folder.is_dir():
            # Đếm tổng số lượng file trong thư mục (đệ quy cả thư mục con)
            total_files = sum(1 for _ in folder.rglob("*") if _.is_file())

            # Đếm riêng các file .wav
            wav_files = sum(1 for _ in folder.rglob("*.wav"))

            print(f"{folder.name:<20} | {total_files:<15} | {wav_files:<25}")


if __name__ == "__main__":
    count_records_in_vivos("data/vivos")
