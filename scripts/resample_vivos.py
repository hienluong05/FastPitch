import os
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import multiprocessing
from tqdm import tqdm

def resample_file(args):
    in_file, out_file, target_sr = args
    if not os.path.exists(out_file):
        try:
            # Load with librosa, which resamples automatically if target_sr is provided
            y, _ = librosa.load(in_file, sr=target_sr)
            
            # Save using soundfile
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            sf.write(out_file, y, target_sr, subtype='PCM_16')
        except Exception as e:
            print(f"Error processing {in_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Resample Vivos Dataset")
    parser.add_argument('--vivos-dir', type=str, required=True, help="Path to raw vivos data")
    parser.add_argument('--target-sr', type=int, default=22050, help="Target sampling rate")
    args = parser.parse_args()

    vivos_dir = Path(args.vivos_dir)
    if not vivos_dir.exists():
        print(f"Directory {vivos_dir} does not exist.")
        return

    # Collect all wav files
    wav_files = []
    for split in ['train', 'test']:
        waves_dir = vivos_dir / split / 'waves'
        if waves_dir.exists():
            for spk_dir in waves_dir.iterdir():
                if spk_dir.is_dir():
                    for wav_path in spk_dir.glob('*.wav'):
                        wav_files.append(wav_path)

    if not wav_files:
        print("No wav files found.")
        return

    print(f"Found {len(wav_files)} wav files. Resampling to {args.target_sr} Hz...")
    
    # We will overwrite them in place to keep the structure simple
    # But to be safe, we write to a temporary file first, then replace
    tasks = []
    for w in wav_files:
        tmp_path = w.with_suffix('.resampled.wav')
        tasks.append((str(w), str(tmp_path), args.target_sr))

    # Process in parallel
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(resample_file, tasks), total=len(tasks)):
        pass
    pool.close()
    pool.join()

    # Replace original files
    print("Replacing original files...")
    for in_file, tmp_file, _ in tasks:
        if os.path.exists(tmp_file):
            os.replace(tmp_file, in_file)

    print("Done resampling!")

if __name__ == "__main__":
    main()
