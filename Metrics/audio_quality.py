from torchaudio.pipelines import SQUIM_OBJECTIVE
import numpy as np
import os
import torchaudio.functional as F
import torchaudio
from tqdm import tqdm

def calculate_pesq_sisdr(folder_path):
    STOI = []
    PESQ = []
    SISDR = []
    model = SQUIM_OBJECTIVE.get_model()

    for folders in tqdm(os.listdir(folder_path)):
        for audio_path in os.listdir(os.path.join(folder_path, folders)):
            WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load(os.path.join(folder_path, folders, audio_path))
            if SAMPLE_RATE_SPEECH != 16000:
                WAVEFORM_SPEECH = F.resample(WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH, 16000)

            stoi_hyp, pesq_hyp, si_sdr_hyp = model(WAVEFORM_SPEECH[0:1, :])
            
            STOI.append(float(stoi_hyp[0]))
            PESQ.append(float(pesq_hyp[0]))
            SISDR.append(float(si_sdr_hyp[0]))

    return PESQ, SISDR


