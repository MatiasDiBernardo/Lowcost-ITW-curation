import os
from pydub import AudioSegment
import shutil


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
    print(f"Archivo convertido con éxito: {output_path}")


import shutil

def split_audio_chunks(audio_file, timestamps, output_folder, tmin=0, tmax=9999, gap=0, offset=0):
    """
    Divide un archivo de audio en fragmentos basados en los tiempos dados y ajusta su duración
    combinando fragmentos cortos hasta alcanzar tmin y asegurando que no superen tmax.

    Args:
        audio_file (str): Ruta del archivo de audio original.
        timestamps (list): Lista de diccionarios con los tiempos de inicio y fin en segundos.
        output_folder (str): Carpeta donde se guardarán los fragmentos.
        tmin (int): Duración mínima en segundos de un fragmento resultante.
        tmax (int | None): Duración máxima en segundos de un fragmento resultante. Si es None, no hay límite.
        gap (int): Tiempo adicional en milisegundos a agregar antes y después de cada fragmento.
        offset (int): Número inicial para nombrar los fragmentos.
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

    combined_chunks = []
    current_chunk = AudioSegment.silent(duration=0)  # Inicialmente vacío
    current_duration = 0

    for ts in timestamps:
        start_ms = int(ts['start'] * 1000)  # Convertir segundos a milisegundos
        end_ms = int(ts['end'] * 1000)      # Convertir segundos a milisegundos
        segment = audio[start_ms - gap:end_ms + gap]
        segment_duration = len(segment) / 1000  # Convertir a segundos

        # Agregar el fragmento al chunk actual
        current_chunk += segment
        current_duration += segment_duration

        # Verificar si ya superamos tmin
        if current_duration >= tmin:
            # Si tmax está definido y lo superamos, guardamos el fragmento actual y empezamos uno nuevo
            if tmax is not None and current_duration > tmax:
                print(f"⚠️ Advertencia: Un segmento combinado alcanzó {current_duration:.2f}s, mayor a tmax={tmax}s. Se guarda sin recortar.")
            combined_chunks.append(current_chunk)

            # Reiniciar el acumulador
            current_chunk = AudioSegment.silent(duration=0)
            current_duration = 0

    # Agregar el último fragmento si quedó algo sin guardar
    if current_duration > 0:
        combined_chunks.append(current_chunk)

    # Exportar los fragmentos
    for i, chunk in enumerate(combined_chunks):
        chunk_number = f"{i + 1 + offset:04}"
        chunk_name = f"{nombre}_{chunk_number}.wav"
        chunk_path = os.path.join(output_folder, chunk_name)
        chunk.export(chunk_path, format="wav")

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
