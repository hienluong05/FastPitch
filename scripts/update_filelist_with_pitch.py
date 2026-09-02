"""
Script to update filelist from 2-column format (audio|text) 
to 3-column format (audio|pitch|text) for FastPitch training.
"""
import argparse
from pathlib import Path


def update_filelist(input_file, output_file, pitch_dir):
    updated_lines = []
    skipped = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')
            if len(parts) == 2:
                audio_path, text = parts
                # Derive pitch filename from audio filename
                pitch_fname = Path(audio_path).with_suffix('.pt').name
                pitch_path = f'{pitch_dir}/{pitch_fname}'
                updated_lines.append(f'{audio_path}|{pitch_path}|{text}')
            elif len(parts) == 3:
                # Already 3 columns, keep as-is
                updated_lines.append(line)
            else:
                print(f'WARNING: Skipping malformed line: {line}')
                skipped += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')

    print(f'Updated {len(updated_lines)} lines, skipped {skipped}')
    print(f'Saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input filelist (2-column)')
    parser.add_argument('--output', required=True, help='Output filelist (3-column)')
    parser.add_argument('--pitch-dir', default='data/vivos/pitch',
                        help='Relative path to pitch directory (default: data/vivos/pitch)')
    args = parser.parse_args()

    update_filelist(args.input, args.output, args.pitch_dir)
