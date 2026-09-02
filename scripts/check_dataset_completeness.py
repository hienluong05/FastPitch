import argparse
import os
from pathlib import Path

def check_completeness(filelists, data_dir):
    missing_files = {
        'pitch': [],
        'mels': [],
        'alignment_priors': []
    }
    
    total_records = 0
    
    for filelist in filelists:
        if not os.path.exists(filelist):
            print(f"Filelist not found: {filelist}")
            continue
            
        with open(filelist, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                total_records += 1
                # Format có thể là 2 cột (audio|text) hoặc 3 cột (audio|pitch|text)
                audio_path = line.split('|')[0]
                
                # Lấy tên file gốc (ví dụ: VIVOSDEV01_R002)
                base_name = Path(audio_path).stem
                pt_filename = f"{base_name}.pt"
                
                # Đường dẫn cần kiểm tra
                pitch_path = Path(data_dir) / 'pitch' / pt_filename
                mel_path = Path(data_dir) / 'mels' / pt_filename
                prior_path = Path(data_dir) / 'alignment_priors' / pt_filename
                
                if not pitch_path.exists():
                    missing_files['pitch'].append(pt_filename)
                if not mel_path.exists():
                    missing_files['mels'].append(pt_filename)
                if not prior_path.exists():
                    missing_files['alignment_priors'].append(pt_filename)

    # In kết quả báo cáo
    print(f"Tổng số bản ghi đã kiểm tra: {total_records}")
    print("-" * 40)
    
    is_complete = True
    for category, missing in missing_files.items():
        if missing:
            is_complete = False
            print(f"[{category.upper()}] THIẾU {len(missing)} files.")
            print(f"   Ví dụ một số file thiếu: {missing[:5]}")
        else:
            print(f"[{category.upper()}] Đầy đủ! (0 file thiếu)")
            
    print("-" * 40)
    if is_complete:
        print("TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC CHUẨN BỊ ĐẦY ĐỦ!")
    else:
        print("CẢNH BÁO: Dữ liệu chưa đầy đủ, bạn có thể cần chạy lại prepare_dataset.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kiểm tra tính đầy đủ của file mels, pitch, alignment_priors")
    parser.add_argument('--filelists', nargs='+', default=['filelists/vivos_train.txt', 'filelists/vivos_test.txt'],
                        help='Các file filelist cần kiểm tra')
    parser.add_argument('--data-dir', default='data/vivos',
                        help='Thư mục chứa data (có các thư mục con pitch, mels, alignment_priors)')
    
    args = parser.parse_args()
    check_completeness(args.filelists, args.data_dir)
