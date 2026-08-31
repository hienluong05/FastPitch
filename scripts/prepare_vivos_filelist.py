import os
import argparse
from pathlib import Path

def process_vivos(vivos_dir, output_dir):
    vivos_dir = Path(vivos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        prompts_file = vivos_dir / split / 'prompts.txt'
        waves_dir = vivos_dir / split / 'waves'
        
        if not prompts_file.exists():
            print(f"Warning: {prompts_file} not found.")
            continue
            
        out_lines = []
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                file_id = parts[0]
                text = parts[1]
                
                speaker_id = file_id.split('_')[0]
                wav_path = waves_dir / speaker_id / f"{file_id}.wav"
                
                # FastPitch format: audio_path|text
                out_lines.append(f"{wav_path.absolute()}|{text}")
                
        out_file = output_dir / f"vivos_{split}.txt"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines))
        print(f"Created {out_file} with {len(out_lines)} samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vivos-dir', type=str, required=True, help='Path to Vivos dataset')
    parser.add_argument('--out-dir', type=str, default='filelists', help='Output directory for filelists')
    args = parser.parse_args()
    process_vivos(args.vivos_dir, args.out_dir)
