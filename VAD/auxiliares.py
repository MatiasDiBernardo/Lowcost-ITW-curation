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
    # Convertir ruta a absoluta para evitar problemas con rutas relativas
    audio_path = os.path.abspath(audio_path)

    # Verificar si el archivo existe
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"No se encontró el archivo: {audio_path}")

    # Determinar formato y cargar audio
    if audio_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_path)
    elif audio_path.endswith('.wav'):
        audio = AudioSegment.from_wav(audio_path)
    else:
        raise ValueError("Formato de archivo no soportado. Usa .mp3 o .wav.")

    # Convertir a mono y ajustar la tasa de muestreo
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    # Exportar archivo convertido
    audio.export(output_path, format="wav")
    #print(f"Archivo convertido con éxito: {output_path}")


def split_audio_chunks(audio_file, timestamps, output_folder, mean_duration=20, std_duration=5, gap=0, offset=0, extend_silence=False, use_normal_distribution=False):
    """
    Divide un archivo de audio en fragmentos basados en los tiempos dados y ajusta su duración.
    Puede seguir una distribución normal de duración objetivo si se habilita la opción.
    
    Args:
        audio_file (str): Ruta del archivo de audio original.
        timestamps (list): Lista de diccionarios con los tiempos de inicio y fin en segundos.
        output_folder (str): Carpeta donde se guardarán los fragmentos.
        mean_duration (int): Duración media deseada para los fragmentos en segundos.
        std_duration (int): Desviación estándar de la duración de los fragmentos en segundos.
        gap (int): Tiempo adicional en milisegundos a agregar antes y después de cada fragmento.
        offset (int): Número inicial para nombrar los fragmentos.
        extend_silence (bool): Si es True, distribuye el silencio entre los segmentos consecutivos.
        use_normal_distribution (bool): Si es True, usa una distribución normal para definir la duración de los chunks.
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

    # Eliminar y recrear la carpeta de salida
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    adjusted_timestamps = []

    if extend_silence:
        for i in range(len(timestamps)):
            start = timestamps[i]['start']
            end = timestamps[i]['end']
            
            if i > 0:
                prev_end = timestamps[i - 1]['end']
                silence_gap = start - prev_end
                if silence_gap > 0:
                    adjustment = silence_gap / 2
                    start -= adjustment
                    adjusted_timestamps[-1]['end'] += adjustment
            
            adjusted_timestamps.append({'start': start, 'end': end})
    else:
        adjusted_timestamps = timestamps

    combined_chunks = []
    if use_normal_distribution:
        current_chunk = AudioSegment.silent(duration=0)
        current_duration = 0
        minimum_duration_accepted = 10  # Segundos
        if TEST:
            np.random.seed(7)

        target_duration = max(minimum_duration_accepted, np.random.normal(mean_duration, std_duration))  # Generar la primera duración objetivo

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
