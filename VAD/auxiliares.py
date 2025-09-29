import os
import yaml
import shutil
import numpy as np
from pydub import AudioSegment

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TEST = config["test"]
VERBOSE = config["verbose"]

def load_to_wav(audio_path,output_path = "converted_audio.wav"):
    # Convert path to absolute to avoid problems with relative paths
    audio_path = os.path.abspath(audio_path)

    # Check if the file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    # Determine format and load audio
    if audio_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_path)
    elif audio_path.endswith('.wav'):
        audio = AudioSegment.from_wav(audio_path)
    else:
        raise ValueError("Unsupported file format. Use .mp3 or .wav.")

    # Convert to mono and adjust sample rate
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    # Export converted file
    audio.export(output_path, format="wav")
    #print(f"File successfully converted: {output_path}")


def split_audio_chunks(audio_file, timestamps, output_folder, mean_duration=20, std_duration=5, gap=0, offset=0, extend_silence=False, use_normal_distribution=False):
    """
    Splits an audio file into fragments based on the given times and adjusts their duration.
    Can follow a normal distribution of target duration if the option is enabled.
    
    Args:
        audio_file (str): Path to the original audio file.
        timestamps (list): List of dictionaries with start and end times in seconds.
        output_folder (str): Folder where the fragments will be saved.
        mean_duration (int): Desired mean duration for the fragments in seconds.
        std_duration (int): Standard deviation of the fragment duration in seconds.
        gap (int): Additional time in milliseconds to add before and after each fragment.
        offset (int): Initial number for naming the fragments.
        extend_silence (bool): If True, distributes silence between consecutive segments.
        use_normal_distribution (bool): If True, uses a normal distribution to define the duration of the chunks.
    """

    formato = audio_file[-3:].lower()
    nombre = os.path.basename(audio_file)[:-4]

    # Cargar el archivo de audio original
    if formato == "wav":
        audio = AudioSegment.from_wav(audio_file)
    elif formato == "mp3":
        audio = AudioSegment.from_mp3(audio_file)
    else:
        raise ValueError("Formato de audio no soportado. Usa WAV o MP3.")

    # Delete and recreate the output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    adjusted_timestamps = []

    MAX_ADJUSTMENT = 5  # Maximum silence time that can be added to a chunk if extend_silence is active

    if extend_silence:
        for i in range(len(timestamps)):
            start = timestamps[i]['start']
            end = timestamps[i]['end']
            
            if i > 0:
                prev_end = timestamps[i - 1]['end']
                silence_gap = start - prev_end
                if silence_gap > 0:
                    adjustment = min(silence_gap / 2, MAX_ADJUSTMENT)
                    start -= adjustment
                    adjusted_timestamps[-1]['end'] += adjustment
            
            adjusted_timestamps.append({'start': start, 'end': end})
    else:
        adjusted_timestamps = timestamps

    combined_chunks = []
    if use_normal_distribution:
        current_chunk = AudioSegment.silent(duration=0)
        current_duration = 0
        minimum_duration_accepted = 10  # Seconds
        if TEST:
            np.random.seed(7)

        target_duration = max(minimum_duration_accepted, np.random.normal(mean_duration, std_duration))  # Generate the first target duration

        for ts in adjusted_timestamps:
            start_ms = int(ts['start'] * 1000)
            end_ms = int(ts['end'] * 1000)
            segment = audio[start_ms - gap:end_ms + gap]
            segment_duration = len(segment) / 1000

            current_chunk += segment
            current_duration += segment_duration

            if current_duration >= target_duration:
                combined_chunks.append(current_chunk)
                current_chunk = AudioSegment.silent(duration=0)
                current_duration = 0
                target_duration = max(1, np.random.normal(mean_duration, std_duration))

        if current_duration > 0:
            combined_chunks.append(current_chunk)
    else:
        for ts in adjusted_timestamps:
            start_ms = int(ts['start'] * 1000)
            end_ms = int(ts['end'] * 1000)
            segment = audio[start_ms - gap:end_ms + gap]
            combined_chunks.append(segment)

    # Exportar los fragmentos
    for i, chunk in enumerate(combined_chunks):
        chunk_number = f"{i + 1 + offset:04}"
        chunk_name = f"{nombre}_{chunk_number}.wav"
        chunk_path = os.path.join(output_folder, chunk_name)
        chunk.export(chunk_path, format="wav")

    if VERBOSE:
        print(f"✅ Se guardaron {len(combined_chunks)} fragmentos en {output_folder}.")

def extract_and_sort_timestamps(audio_dict):
    """
    Extrae y ordena las marcas temporales de un diccionario de segmentos de audio.

    :param audio_dict: Diccionario con claves de audio y valores como marcas temporales.
    :return: Lista de diccionarios con marcas temporales ordenadas.
    """
    timestamps = []

    # Recorrer cada entrada del diccionario
    for audio_key in sorted(audio_dict.keys()):  # Ordenar por claves de audio (audio_01, audio_02, etc.)
        segments = audio_dict[audio_key]
        
        # Si el valor es un único segmento, lo convertimos en una lista
        if isinstance(segments, dict):
            segments = [segments]
        
        # Agregar los segmentos a la lista de timestamps
        timestamps.extend(segments)

    return timestamps
