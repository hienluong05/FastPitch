import argparse
import io
import torch
import numpy as np
from scipy.io.wavfile import write
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

import models
from common.text.text_processing import get_text_processing
from hifigan.models import Denoiser

app = FastAPI(title="FastPitch TTS API", description="API cho việc sinh giọng nói từ văn bản (TTS) sử dụng FastPitch")

class TTSRequest(BaseModel):
    text: str
    lang: str = "vi"

# Global variables to store models
generator = None
vocoder = None
denoiser = None
device = None
text_cleaners = ['vietnamese_cleaners'] # You can change this based on your model
symbol_set = 'vivos' # change to your symbol set

def load_models(fastpitch_path, hifigan_path, use_cuda=True):
    global generator, vocoder, denoiser, device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    print(f"Loading FastPitch from {fastpitch_path}...")
    generator, _, _ = models.load_and_setup_model(
        'FastPitch', None, fastpitch_path, False, device, 
        forward_is_infer=True, jitable=False
    )
    generator.eval()

    print(f"Loading HiFi-GAN from {hifigan_path}...")
    vocoder, _, _ = models.load_and_setup_model(
        'HiFi-GAN', None, hifigan_path, False, device,
        forward_is_infer=True, jitable=False
    )
    vocoder.eval()
    denoiser = Denoiser(vocoder, win_length=1024).to(device)
    print("Models loaded successfully!")

@app.on_event("startup")
async def startup_event():
    # Provide the default paths to your trained checkpoints here
    fastpitch_ckpt = "pretrained_models/fastpitch.pt" # Cập nhật đường dẫn này
    hifigan_ckpt = "pretrained_models/hifigan.pt" # Cập nhật đường dẫn này
    
    try:
        # Uncomment this line and set the correct paths when models are available
        # load_models(fastpitch_ckpt, hifigan_ckpt)
        pass
    except Exception as e:
        print(f"Warning: Could not load models at startup: {e}")

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    if generator is None or vocoder is None:
        raise HTTPException(status_code=503, detail="Mô hình TTS chưa được tải.")
    
    if request.lang != "vi":
        raise HTTPException(status_code=400, detail="Hiện tại chỉ hỗ trợ ngôn ngữ tiếng Việt (lang='vi').")

    text = request.text
    
    # Process text
    tp = get_text_processing(symbol_set, text_cleaners, p_arpabet=0.0)
    sequence = tp.encode_text(text)
    sequence = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        # Generate mel spectrogram
        mel, mel_lens, *_ = generator(sequence)
        
        # Generate audio from mel
        audio = vocoder(mel).float()
        audio = denoiser(audio.squeeze(1), 0.0)
        audio = audio.squeeze(1) * 32768.0
        audio = audio.cpu().numpy().astype(np.int16)
        
    # Write to memory buffer
    buffer = io.BytesIO()
    write(buffer, 22050, audio[0])
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
