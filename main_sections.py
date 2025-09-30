import os
import yaml
import shutil
import warnings

from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

VAD_config = config["VAD"]
mean_duration=VAD_config["mean_duration"]
std_desv=VAD_config["std_desv"]

TEST = config["test"]
VERBOSE = config["verbose"]

if not VERBOSE:
    warnings.simplefilter("ignore", UserWarning)

from QualityPredition.NISQA import run_audio_predict
from Denoising.deep_net import denoise_deep_net
from VAD.VAD import vad_audio_splitter 
from STT.whisper import stt_whisper


def get_id():
    """Checks the processed audios folder and gets the last index
    """
    folder_data = "Datos/Audios_VAD"
    data = os.listdir(folder_data)
    if len(data) == 0:
        return 0

    ids = [int(folder.split("_")[1]) for folder in data]

    return max(ids)

def audio_processing(path_audios):
    """This function moves the audios from the section to process to the raw audios folder.
    Verifies that the metadata of the audios to add exists and normalizes the naming.

    Args:
        path_audios (str): List of audio names in Audios_to_Process
    Return:
        (list[str]): List of paths with normalized names
    """

    process_path = os.path.join("Datos", "Audio_to_Process")
    raw_path = os.path.join("Datos", "Audios_Raw")
    path_list_normalized = []
    id_audio = get_id()


    for audio in tqdm(path_audios, desc="Processing audios"):
        path = os.path.join(process_path, audio)
        name = audio.split(".")[0]
        id_audio += 1

    # Apply format (convert to wav just in case for compatibility)
        if path.endswith("mp3"):
            audio = AudioSegment.from_mp3(path)
            path = os.path.join(process_path, name + ".wav")

            # Export as WAV
            audio.export(path, format="wav")
            os.remove(os.path.join(process_path, name + ".mp3"))

    # Normalize name
        name_normalizado = f"audio_{str(id_audio)}.wav"
        path_normalizado = os.path.join(process_path, name_normalizado)
        os.rename(path, path_normalizado)
        
    # Move to the next folder
        shutil.move(path_normalizado, os.path.join(raw_path, name_normalizado))
        path_list_normalized.append(name_normalizado)

    return path_list_normalized

def audio_denoise(path_audios):
    """Applies denoising to the audios and moves them to the Audios_Denoise folder

    Args:
        path_audios (str): List of audio names in Audios_to_Process
    """
    raw_path = os.path.join("Datos", "Audios_Raw")
    denoise_path = os.path.join("Datos", "Audios_Denoise") 

    for audio in tqdm(path_audios, desc="Denoising audios"):
        denoise_deep_net(os.path.join(raw_path, audio), os.path.join(denoise_path, audio))

def audio_vad(path_audios, path_before):
    """Splits the audios to process into chunks and puts them in the Audios_VAD folder

    Args:
        path_audios (str): List of audio names in Audios_to_Process
        path_before (str): Name of the folder to get the data for VAD. E.g., Audios_Denoise
    """

    denoise_path = os.path.join("Datos", path_before)
    vad_path = os.path.join("Datos", "Audios_VAD") 

    for audio in tqdm(path_audios, desc="Splitting audios"):
        name = audio.split(".")[0]
        folder_ouput_vad = os.path.join(vad_path, name)
        os.mkdir(folder_ouput_vad)
        vad_audio_splitter(os.path.join(denoise_path, audio), folder_ouput_vad,mean_duration,std_desv)

def audio_clean(path_audios):
    """Filters out audio files that don't meet the AudioAnalyzer criteria.

    Args:
        path_audios (str): Audio files names from audio VAD.
    """
    vad_path = os.path.join("Datos", "Audios_VAD") 
    clean_audios_path = os.path.join("Datos", "Audios_Clean") 

    name_folders = [name.split(".")[0] for name in path_audios]

    for folder in tqdm(name_folders, desc="Cleaning audios"):

        chunk_audios = os.listdir(os.path.join(vad_path, folder))
        os.makedirs(os.path.join(clean_audios_path, folder), exist_ok=True)

        for path_chunk in chunk_audios:
            # If AudioAnalyzer considers the audio acceptable
            segment_audio = os.path.join(vad_path, folder, path_chunk)
            if run_audio_predict(segment_audio):
                folder_dest = os.path.join(clean_audios_path, folder, path_chunk)
                shutil.copy(segment_audio, folder_dest)
            else:
                folder_dest = os.path.join(clean_audios_path, "removed", path_chunk)
                shutil.copy(segment_audio, folder_dest)
                if VERBOSE:
                    print(f"Se descartó el audio {path_chunk}")

def audio_transcript(path_audios, path_before):
    """Create transcription of the audios and moves them to the Audios_Transcript folder.

    Args:
        path_audios (str): Name audio list from Audios_to_Process
        path_before (str): Names of the folder from where to transcript. Ej, Audios_Clean
    """

    clean_path = os.path.join("Datos", path_before)
    transcript_path = os.path.join("Datos", "Audios_Transcript") 
    
    name_folders = [name.split(".")[0] for name in path_audios]

    for folder in tqdm(name_folders, desc="Transcribiendo audios"):

        chunk_audios = os.listdir(os.path.join(clean_path, folder))
        os.makedirs(os.path.join(transcript_path, folder), exist_ok=True)

        data = []
        for path_chunk in chunk_audios:
            # Make transcription and pair it with the audio name
            segment_audio = os.path.join(clean_path, folder, path_chunk)
            transcript = stt_whisper(segment_audio)
            name = path_chunk.split(".")[0]
            data.append((name, transcript))
            # Move the audio to the transcript folder
            shutil.copy(os.path.join(clean_path, folder, path_chunk), os.path.join(transcript_path, folder, path_chunk))
        
        df = pd.DataFrame(data, columns=["Audio Path", "Transcription"])
        name_csv_file = os.path.join(transcript_path, "transcripts", f"{folder}.csv")
        df.to_csv(name_csv_file, index=False, encoding="utf-8")
