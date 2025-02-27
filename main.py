from main_sections import *

import os

# Config data flow (si se aplica o no estos procesos en la cadena)
config_flow = {"denoising": True, "cleaning": True}

def automatic_dataset_generator(config):
    # Siempre empieza en Audios to Process 
    process_path = os.path.join("Datos", "Audio_to_Process") 
    audios_to_process = os.listdir(process_path)
    
    # Para codear esto bien tendría que plantear la lógica de los if como un árbol (no tengo ganas)
    audio_processing(audios_to_process)
    
    # Brancheo para poder hacer diferentes configuraciones
    if config["denoising"]:
        audio_denoise(audios_to_process)
        audio_vad(audios_to_process, "Audios_Denoise")
        if config["cleaning"]:
            audio_clean(audios_to_process)
            audio_transcript(audios_to_process, "Audios_Clean")
            audio_transcript_to_dataset(audios_to_process)
        else:
            audio_transcript(audios_to_process, "Audios_VAD")
            audio_transcript_to_dataset(audios_to_process)
    else:
        audio_vad(audios_to_process, "Audios_Raw")
        if config["cleaning"]:
            audio_clean(audios_to_process)
            audio_transcript(audios_to_process, "Audios_Clean")
            audio_transcript_to_dataset(audios_to_process)
        else:
            audio_transcript(audios_to_process, "Audios_VAD")
            audio_transcript_to_dataset(audios_to_process)

def simple_direct_implementation():
    process_path = os.path.join("Datos", "Audio_to_Process") 
    audios_to_process = os.listdir(process_path)
    
    audio_processing(audios_to_process)
    audio_denoise(audios_to_process)
    audio_vad(audios_to_process, "Audios_Denoise")
    audio_clean(audios_to_process)
    audio_transcript(audios_to_process, "Audios_Clean")
    audio_transcript_to_dataset(audios_to_process)

if __name__ == "__main__":
    automatic_dataset_generator(config_flow)