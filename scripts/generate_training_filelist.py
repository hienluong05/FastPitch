"""
Script to generate correct 3-column filelists for FastPitch training.
Format: mels/<name>.pt|pitch/<name>.pt|<text>

Paths are RELATIVE to dataset_path (e.g., data/vivos), because
load_filepaths_and_text() in common/utils.py joins dataset_path + path.

When load_mel_from_disk=True, column 1 must point to the mel .pt file,
NOT the .wav file.
"""
import argparse
from pathlib import Path


def generate_filelist(input_file, output_file):
    updated_lines = []
    skipped = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')
            
            # Get the text (always the last column)
            text = parts[-1]
            
            # Get the audio path (always the first column)
            audio_path = parts[0]
            
            # Derive the base filename (e.g., VIVOSSPK01_R001)
            base_name = Path(audio_path).stem
            
            # Create relative paths (relative to dataset_path)
            mel_path = f'mels/{base_name}.pt'
            pitch_path = f'pitch/{base_name}.pt'
            
            updated_lines.append(f'{mel_path}|{pitch_path}|{text}')

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')

    print(f'Generated {len(updated_lines)} lines')
    print(f'Saved to {output_file}')
    if updated_lines:
        print(f'Example: {updated_lines[0][:100]}...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 3-column filelist for FastPitch training with load_mel_from_disk=True')
    parser.add_argument('--input', required=True,
                        help='Input filelist (2 or 3 column, any path format)')
    parser.add_argument('--output', required=True,
                        help='Output filelist (3-column with relative mel/pitch paths)')
    args = parser.parse_args()

    generate_filelist(args.input, args.output)
