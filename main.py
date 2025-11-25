from main_sections import *

import os

def low_resources_pipeline():
    # Start with raw audios in Audios to Process
    process_path = os.path.join("Data", "Audio_to_Process")
    audios_to_process = os.listdir(process_path)

    # Normalize audios (format, rename)
    audios_to_process = audio_processing(audios_to_process)
    
    # Applied voice activity detection (VAD)
    audio_vad(audios_to_process)
    
    # Applied denoising
    audio_denoise(audios_to_process)
    
    # Applied Speaker & Overlap Filtering
    audio_speaker_filt(audios_to_process)
    
    # Applied MOS filtering
    audio_filt(audios_to_process)
    
    # Applied transcriptions
    audio_transcript(audios_to_process)
    
    # Final data formatting
    # audio_dataset() -> final function to normalize dataset
    
    # Remove repetition
    # audio_clear_stages() -> removes intermediate stages

if __name__ == "__main__":
    low_resources_pipeline()