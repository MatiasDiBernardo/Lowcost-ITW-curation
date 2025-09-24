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

def metadata_verification(csv_path: str, target: str):
    """
    Se fija si el nombre del audio en "Audio_to_Process" esta en 
    el archivo de metadatos.
    """
    # Load with header trimming spaces
    df = pd.read_csv(csv_path, encoding='utf-8')

    # Filter exact matches
    matches = df[df['Estimulo'] == target]

    # Evaluate match results
    if matches.empty:
        raise ValueError(f"No match found for Estimulo: '{target}'")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for Estimulo: '{target}'")

    return int(matches['ID'].iloc[0])

def audio_processing(path_audios):
    """Esta función pasa los audio de la sección a procesar a la carpeta de audios raw.
    Verifica que exista la metadata de los audios a agregar y normaliza la nomenclatura.

    Args:
        path_audios (str): Lista de nombres de los audios en Audios_to_Process
    Return:
        (list[str]): Lista de paths con los nombres normalizados
    """

    process_path = os.path.join("Datos", "Audio_to_Process")
    raw_path = os.path.join("Datos", "Audios_Raw")
    path_list_normalized = []

    for audio in tqdm(path_audios, desc="Processing audios"):
        path = os.path.join(process_path, audio)
        name = audio.split(".")[0]

        if TEST and name[:-1] == "Test":
            # En TEST solo mueve los audios de prueba (no se verifica metadata)
            shutil.move(os.path.join(process_path, audio), os.path.join(raw_path, audio))

        if not TEST and name[:-1] != "Test":
            # Verificar que audio esté en csv (verificar que el nombre del archivo este en el CSV)
            id_audio = metadata_verification("metadata.csv", name)

            # Normalizar Amplitud

            # Aplicar formato (lo paso a wav por las dudas para que funcione)
            if path.endswith("mp3"):
                audio = AudioSegment.from_mp3(path)
                path = os.path.join(process_path, name + ".wav")

                # Export as WAV
                audio.export(path, format="wav")
                os.remove(os.path.join(process_path, name + ".mp3"))

            # Normaliza nombre
            name_normalizado = f"audio_{str(id_audio)}.wav"
            path_normalizado = os.path.join(process_path, name_normalizado)
            os.rename(path, path_normalizado)
            
            # Mueve a la siguiente carpeta
            shutil.move(path_normalizado, os.path.join(raw_path, name_normalizado))
            path_list_normalized.append(name_normalizado)

    return path_list_normalized


def audio_denoise(path_audios):
    """ Aplica denoising a los audios y los mueve a la carpeta Audios_Denoise

    Args:
        path_audios (str): Lista de nombres de los audios en Audios_to_Process
    """
    raw_path = os.path.join("Datos", "Audios_Raw")
    denoise_path = os.path.join("Datos", "Audios_Denoise") 

    for audio in tqdm(path_audios, desc="Denoising audios"):
        denoise_deep_net(os.path.join(raw_path, audio), os.path.join(denoise_path, audio))

def audio_vad(path_audios, path_before):
    """Separa los audios a procesar en chunks y los pone en la carpeta de Audios_VAD

    Args:
        path_audios (str): Lista de nombres de los audios en Audios_to_Process
        path_before (str): Nombre de la carpeta donde sacar los datos para el vad. Ej, Audios_Denoise
    """

    denoise_path = os.path.join("Datos", path_before)
    vad_path = os.path.join("Datos", "Audios_VAD") 

    for audio in tqdm(path_audios, desc="Splitting audios"):
        name = audio.split(".")[0]
        folder_ouput_vad = os.path.join(vad_path, name)
        os.mkdir(folder_ouput_vad)
        vad_audio_splitter(os.path.join(denoise_path, audio), folder_ouput_vad,mean_duration,std_desv)

def audio_clean(path_audios):
    """Filtra los audios que no cumplen los criterios del AudioAnalyzer, Filler Detection 
    y best STT.

    Args:
        path_audios (str): Lista de nombres de los audios en Audios_to_Process
    """
    vad_path = os.path.join("Datos", "Audios_VAD") 
    clean_audios_path = os.path.join("Datos", "Audios_Clean") 

    name_folders = [name.split(".")[0] for name in path_audios]

    for folder in tqdm(name_folders, desc="Cleaning audios"):

        chunk_audios = os.listdir(os.path.join(vad_path, folder))
        os.makedirs(os.path.join(clean_audios_path, folder), exist_ok=True)  # Crea la subcarpeta en el destino

        for path_chunk in chunk_audios:
            ## Si el AudioAnalyzer considera que el audio es aceptable
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
    """Agrega transcripción (por ahora sin descarte)

    Args:
        path_audios (str): Lista de nombres de los audios en Audios_to_Process
        path_before (str): Nombre de la carpeta donde sacar los datos para transcribir. Ej, Audios_Clean
    """

    clean_path = os.path.join("Datos", path_before)
    transcript_path = os.path.join("Datos", "Audios_Transcript") 
    
    name_folders = [name.split(".")[0] for name in path_audios]

    for folder in tqdm(name_folders, desc="Transcribiendo audios"):

        chunk_audios = os.listdir(os.path.join(clean_path, folder))
        os.makedirs(os.path.join(transcript_path, folder), exist_ok=True)  # Crea la subcarpeta en el destino

        data = []
        for path_chunk in chunk_audios:
            # Genera la transcripción y la agrega en conjunto con el nombre del archivo
            segment_audio = os.path.join(clean_path, folder, path_chunk)
            transcript = stt_whisper(segment_audio)
            name = path_chunk.split(".")[0]
            data.append((name, transcript))
            # Mueve el audio a la carpeta transcripción
            shutil.copy(os.path.join(clean_path, folder, path_chunk), os.path.join(transcript_path, folder, path_chunk))
        
        df = pd.DataFrame(data, columns=["Audio Path", "Transcription"])
        name_csv_file = os.path.join(transcript_path, "transcripts", f"{folder}.csv")
        df.to_csv(name_csv_file, index=False, encoding="utf-8")

def audio_transcript_to_dataset(path_audios):
    print("Final")