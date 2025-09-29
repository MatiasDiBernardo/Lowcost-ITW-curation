from main_sections import *


import os

# Config data flow (whether these processes are applied in the chain)
config_flow = {"denoising": True, "cleaning": True, "STT": False}

def automatic_dataset_generator(config):
    # Always starts in Audios to Process
    process_path = os.path.join("Datos", "Audio_to_Process")
    audios_to_process = os.listdir(process_path)

    # To code this well, I should set up the logic of the ifs as a tree (I don't feel like it)
    audios_to_process = audio_processing(audios_to_process)

    # Branching to allow different configurations
    if config["denoising"]:
        audio_denoise(audios_to_process)
        audio_vad(audios_to_process, "Audios_Denoise")
        if config["cleaning"]:
            if config['STT']:
                audio_clean(audios_to_process)
                audio_transcript(audios_to_process, "Audios_Clean")
            else:
                audio_clean(audios_to_process)
        else:
            audio_transcript(audios_to_process, "Audios_VAD")
    else:
        audio_vad(audios_to_process, "Audios_Raw")
        if config["cleaning"]:
            if config['STT']:
                audio_clean(audios_to_process)
                audio_transcript(audios_to_process, "Audios_Clean")
            else:
                audio_clean(audios_to_process)
        else:
            audio_transcript(audios_to_process, "Audios_VAD")

def simple_direct_implementation():
    process_path = os.path.join("Datos", "Audios_Raw")
    audios_to_process = os.listdir(process_path)

    audio_processing(audios_to_process)
    audio_denoise(audios_to_process)
    audio_vad(audios_to_process, "Audios_Denoise")
    audio_clean(audios_to_process)
    audio_transcript(audios_to_process, "Audios_Clean")

if __name__ == "__main__":
    automatic_dataset_generator(config_flow)