import torch
import torchaudio
import os
from denoiser import pretrained
from denoiser.dsp import convert_audio

def DEMUCS(input_dir, output_dir):
    """Applied DEMUCS denoising algorithm. 

    Args:
        input_dir (str): Path to folder with audios in wav format and any sample rate.
        output_dir (str): Path to the output folder of denoised audios.
    """

    model = pretrained.dns64().cpu()
    for filename in os.listdir(input_dir):

        wav, sr = torchaudio.load(filename)
        wav = convert_audio(wav.cpu(), sr, model.sample_rate, model.chin)

        with torch.no_grad():
            denoised_tensor = model(wav[None])[0]

        torchaudio.save(os.path.join(output_dir, filename), denoised_tensor, model.sample_rate)
