from STT.whisper import stt_whisper
from AudioAnalyzer.NISQA import run_audio_predict
from Denoising.deep_net import denoise_deep_net
from VAD.VAD import vad_audio_splitter 

import os
import shutil

# Env Var
TEST = True
VERBOSE = True

# From: Audio_to_Process -> Audios_Raw
## Normalizando nomenclatura de los datos y verificar que metadata este actualizado
process_path = "Datos/Audio_to_Process"
raw_path = "Datos/Audios_Raw"

audios_to_process = os.listdir(process_path)
for audio in audios_to_process:

    if TEST:
        shutil.move(os.path.join(process_path, audio), os.path.join(raw_path, audio))

# From: Audios_Raw -> Audios_Denoise
## Agrega denoise a todo el audio
raw_path = os.path.join("Datos", "Audios_Raw")
denoise_path = os.path.join("Datos", "Audios_Denoise") 
denoise_deep_net(raw_path, denoise_path)

# From: Audios_Denoise -> Audios VAD
## Separa los audios largos en chunks
denoise_path = os.path.join("Datos", "Audios_Denoise")
vad_path = os.path.join("Datos", "Audios_VAD") 

audios_to_split = os.listdir(denoise_path)
for audio in audios_to_split:
    name = audio.split(".")[0]
    folder_ouput_vad = os.path.join(vad_path, name)
    os.mkdir(folder_ouput_vad)
    vad_audio_splitter(os.path.join(denoise_path, audio), folder_ouput_vad)

# From: Audios_VAD -> Audios_Clean
## Elimina audios nos deseados
vad_path = os.path.join("Datos", "Audios_VAD") 
clean_audios_path = os.path.join("Datos", "Audios_Clean") 

folders_vad = os.listdir(vad_path)
for folder in folders_vad:

    chunk_audios = os.listdir(os.path.join(vad_path, folder))
    os.makedirs(os.path.join(clean_audios_path, folder), exist_ok=True)  # Crea la subcarpeta en el destino

    for path_chunk in chunk_audios:
        ## Si el AudioAnalyzer considera que el audio es aceptable
        if run_audio_predict(path_chunk):
            folder_input = os.path.join(vad_path, folder, path_chunk)
            folder_dest = os.path.join(clean_audios_path, folder, path_chunk)
            shutil.move(folder_input, folder_dest)
        else:
            # Agregar a la carpeta de remove
            if VERBOSE:
                print(f"Se descartó el audio {path_chunk}")

# From: Audios_Clean -> Audios_Transcript