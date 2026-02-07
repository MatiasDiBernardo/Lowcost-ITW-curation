import os
import yaml
import shutil
import warnings

from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Loads configuration from config.yaml file
filt_type = config["quality_prediction"]["type"]

mean_duration = config["VAD"]["mean_duration"]
std_desv = config["VAD"]["std_desv"]

den_type = config["denoising"]["type"]

apply_osd = config["speaker_filtering"]["apply_osd"]
apply_diarization = config["speaker_filtering"]["apply_diarization"]
min_overlap = config["speaker_filtering"]["min_overlap"]
hf_token = os.environ.get("HF_TOKEN")
    
if not hf_token and (apply_osd or apply_diarization):
    print("\n[WARNING] 'HF_TOKEN' environment variable not found.")
    print("Speaker filtering models (OSD/Diarization) may fail or require a manual login via CLI.")
    print("To fix: export HF_TOKEN='your_token_here'\n")

stt_type = config["STT"]["type"]

TEST = config["test"]
VERBOSE = config["verbose"]

if not VERBOSE:
    warnings.simplefilter("ignore", UserWarning)

from QualityPrediction.NISQA import filter_audios_by_nisqa
from QualityPrediction.DNSMOS import filter_audios_by_dnsmos
from Denoising.deep_net import denoise_deep_net
from Denoising.demucs import denoise_demucs

from VAD.VAD import vad_audio_splitter 
from SpeakerDiarization.osd import load_osd_pipeline, detect_overlapped_speech
from SpeakerDiarization.speaker_verification import load_embedding_model, verify_multi_speaker_by_clustering
from STT.whisper import stt_whisper

def get_id():
    """Checks the processed audios folder and gets the last index.
    This renaming pretends to avoids duplicates and naming errors.
    """
    folder_data = "Data/Audios_VAD"
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

    process_path = os.path.join("Data", "Audio_to_Process")
    raw_path = os.path.join("Data", "Audios_Raw")
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


def audio_vad(path_audios):
    """Splits the audios to process into chunks and puts them in the Audios_VAD folder

    Args:
        path_audios (str): List of audio names in Audios_to_Process
    """
    prev_path = os.path.join("Data", "Audios_Raw")
    vad_path = os.path.join("Data", "Audios_VAD") 

    for audio in tqdm(path_audios, desc="Splitting audios"):
        name = audio.split(".")[0]
        folder_ouput_vad = os.path.join(vad_path, name)
        os.mkdir(folder_ouput_vad)
        vad_audio_splitter(os.path.join(prev_path, audio), folder_ouput_vad, mean_duration, std_desv)


def audio_denoise(path_audios):
    """Applies denoising to the audios and moves them to the Audios_Denoise folder

    Args:
        path_audios (str): List of audio names in Audios_to_Process
    """
    prev_path = os.path.join("Data", "Audios_VAD")
    denoise_path = os.path.join("Data", "Audios_Denoise") 

    name_folders = [name.split(".")[0] for name in path_audios]

    for folder in tqdm(name_folders, desc="Denoising audios"):

        chunk_audios = os.listdir(os.path.join(prev_path, folder))
        os.makedirs(os.path.join(denoise_path, folder), exist_ok=True)

        for audio in chunk_audios:
            input_audio = os.path.join(prev_path, folder, audio)
            output_audio = os.path.join(denoise_path, folder, audio)
            if den_type == "DeepFilterNet":
                denoise_deep_net(input_audio, output_audio)
            
            if den_type == "Demucs":
                denoise_demucs(input_audio, output_audio)
            
            if den_type == "No denoising":
                shutil.copy(input_audio, output_audio)

def audio_speaker_filt(path_audios):
    """
    Filters out audio files that contain overlapped speech or multiple speakers.
    It uses Pyannote models which require a valid Hugging Face token set in the
    'HF_TOKEN' environment variable.

    Args:
        path_audios (list[str]): List of audio names processed in the previous stage.
    """
    
    prev_path = os.path.join("Data", "Audios_Denoise")
    filt_path = os.path.join("Data", "Audios_Speaker_Filt") 
    
    osd_pipeline = load_osd_pipeline(hf_token) if apply_osd else None
    emb_inference = load_embedding_model(hf_token) if apply_diarization else None

    name_folders = [name.split(".")[0] for name in path_audios]

    for folder in tqdm(name_folders, desc="Filtering by Speaker/Overlap"):
        
        chunk_audios = os.listdir(os.path.join(prev_path, folder))
        os.makedirs(os.path.join(filt_path, folder), exist_ok=True)

        for audio in chunk_audios:
            input_audio = os.path.join(prev_path, folder, audio)
            output_audio = os.path.join(filt_path, folder, audio)
            
            is_rejected = False

            # 1. Check Multi-Speaker
            if apply_diarization and not is_rejected:
                if verify_multi_speaker_by_clustering(input_audio, emb_inference): 
                    is_rejected = True

            # 2. Check Overlap
            if apply_osd and not is_rejected:
                if detect_overlapped_speech(input_audio, osd_pipeline, min_overlap):
                    is_rejected = True

            if not is_rejected:
                shutil.copy(input_audio, output_audio)
            else:
                pass

def audio_filt(path_audios):
    """Filters out audio files that don't meet threshold criteria (NISQA or DNSMOS).

    Args:
        path_audios (str): List of audio names in Audios_to_Process
    """
    prev_path = os.path.join("Data", "Audios_Speaker_Filt") 
    filt_path = os.path.join("Data", "Audios_Clean") 

    name_folders = [name.split(".")[0] for name in path_audios]

    for folder_name in tqdm(name_folders, desc="Cleaning audios"):

        folder_path = os.path.join(prev_path, folder_name)
        os.makedirs(os.path.join(filt_path, folder_name), exist_ok=True)

        filtered_audios = []

        if filt_type == "NISQA":
            filtered_audios = filter_audios_by_nisqa(folder_path)
                
        elif filt_type == "DNSMOS":
            filtered_audios = filter_audios_by_dnsmos(folder_path)
        
        elif filt_type == "NO FILT":
            filtered_audios = os.listdir(folder_path)

        for audio in filtered_audios:
            # audio can be either the name or the path, depending on the filter function. Using basename to be sure to get the name in both cases.
            input_audio = os.path.join(folder_path, os.path.basename(audio)) 
            output_audio = os.path.join(filt_path, folder_name, os.path.basename(audio)) 
            shutil.copy(input_audio, output_audio)

def audio_transcript(path_audios):
    """Create transcription of the audios and moves them to the Audios_Transcript folder.

    Args:
        path_audios (str): Name audio list from Audios_to_Process
    """

    prev_path = os.path.join("Data", "Audios_Denoise")
    transcript_path = os.path.join("Data", "Audios_Transcript") 
    
    name_folders = [name.split(".")[0] for name in path_audios]

    for folder in tqdm(name_folders, desc="Transcribing audios"):

        chunk_audios = os.listdir(os.path.join(prev_path, folder))
        os.makedirs(os.path.join(transcript_path, folder), exist_ok=True)

        data = []
        for audio in chunk_audios:
            input_audio = os.path.join(prev_path, folder, audio)
            output_audio = os.path.join(transcript_path, folder, audio)

            # Make transcription and pair it with the audio name
            if stt_type == "Whisper":
                transcript = stt_whisper(input_audio)
                name = audio.split(".")[0]
                data.append((name, transcript))
                shutil.copy(input_audio, output_audio)
            
            if stt_type == "No STT":
                shutil.copy(input_audio, output_audio)
        
        if len(data) != 0:
            df = pd.DataFrame(data, columns=["Audio Path", "Transcription"])
            name_csv_file = os.path.join(transcript_path, f"transcript_{folder}.csv")
            df.to_csv(name_csv_file, index=False, encoding="utf-8")


def audio_dataset():
    """Compacts all audios into one single audio_data folder and a single transcripts.csv
    """
    return None

def clear_audio_stages():
    """Deletes all data in the intermediate stages
    """
    return None
