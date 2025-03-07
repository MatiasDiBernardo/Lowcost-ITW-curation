from pydub import AudioSegment
from silero_vad import load_silero_vad, read_audio
import whisper
import librosa
import optuna
from pydub.utils import which
import os
import shutil
from VAD.auxiliares import load_to_wav, split_audio_chunks, extract_and_sort_timestamps
from VAD.clasificador_wps import categorize_and_filter_segments,classify_segments_by_speed,get_lowest_speed_category
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import optuna
import logging

optuna.logging.set_verbosity(logging.WARNING)  # Oculta INFO y DEBUG


# Variables globales para almacenar los mejores resultados
best_speech_timestamps = None
best_score = -float("inf")  # Inicializar con un valor negativo infinito

# Función objetivo para Optuna
def objective(trial, param_config,wav):
    
    global best_speech_timestamps, best_score  # Para actualizar la variable global

    # Generar los hiperparámetros dinámicamente
    params = {}
    for param_name, config in param_config.items():
        if config["type"] == "int":
            params[param_name] = trial.suggest_int(
                param_name, config["low"], config["high"], step=config["step"]
            )
        elif config["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, config["choices"]
            )
        elif config["type"] == "float":
            params[param_name] = trial.suggest_float(
                param_name, config["low"], config["high"]
            )
        else:
            raise ValueError(f"Tipo desconocido para {param_name}: {config['type']}")

    # Ejecutar el VAD con los parámetros generados
    speech_timestamps = get_speech_timestamps(
        wav,
        silero,
        return_seconds=True,
        threshold=params["threshold"],
        min_speech_duration_ms=params["min_speech_duration_ms"],
        min_silence_duration_ms=params["min_silence_duration_ms"],
    )

    # Métrica: Maximizar el número de segmentos detectados
    score = len(speech_timestamps)

    # Si este trial es el mejor hasta ahora, actualizamos los resultados
    if score > best_score:
        best_score = score
        best_speech_timestamps = speech_timestamps

    return score


# Función para seleccionar el sampler (método de optimización)
def get_sampler(method,search_space={}):
    if method == "random":
        return optuna.samplers.RandomSampler()
    elif method == "tpe":
        return optuna.samplers.TPESampler()
    elif method == "grid":
        return optuna.samplers.GridSampler(search_space)
    else:
        raise ValueError(f"Método de optimización desconocido: {method}")
    

def parametros(vel):
    param_config={}
    if vel == "Fast":
        param_config = {
        "threshold": {"type": "float", "low": 0.7, "high": 0.9},
        "min_speech_duration_ms":  {"type": "float", "low": 50 ,"high":200},
        "min_silence_duration_ms":  {"type": "float", "low": 50, "high":150},
        }
    elif vel == "Normal":
        param_config = {
        "threshold": {"type": "float", "low": 0.7, "high": 0.9},
        "min_speech_duration_ms":  {"type": "float", "low": 250, "high":500},
        "min_silence_duration_ms":  {"type": "float", "low": 50, "high":200},
        }
    elif vel == "Slow":
        param_config = {
        "threshold": {"type": "float", "low": 0.4, "high": 0.8},  
        "min_speech_duration_ms":  {"type": "float", "low": 400, "high":800},
        "min_silence_duration_ms":  {"type": "float", "low": 100, "high":300},
        }
    return param_config


def process_audio_chunks(audio_dict,audio_file,n_trials,sampler):
    """
    Procesa los audios según las categorías y guarda los resultados en un diccionario,
    ajustando los límites temporales para que respeten el audio original.
    
    :param audio_dict: Diccionario con las categorías como claves y los índices de los audios como valores.
    :return: Diccionario con los resultados optimizados para cada audio, incluyendo los segmentos no optimizados.
    """
    global best_speech_timestamps, best_score, original_best_speech_timestamps


    nombre = os.path.basename(audio_file)[:-4]
    results = {}  # Diccionario para almacenar los resultados
    total_segments = len(original_best_speech_timestamps)  # Número total de segmentos detectados inicialmente

    # Identificar todos los índices existentes
    all_indices = set(range(total_segments))

    # Identificar los índices que no están en el diccionario de entrada
    input_indices = {index for indices in audio_dict.values() for index in indices}
    non_optimized_indices = all_indices - input_indices

    # Guardar los segmentos no optimizados directamente en el diccionario de resultados
    for index in non_optimized_indices:
        audio_key = f"audio_{index + 1:02d}"  # Formato del nombre del segmento
        results[audio_key] = original_best_speech_timestamps[index]  # Guardar los límites originales directamente

    # Iterar sobre cada categoría en el diccionario
    for category, indices in audio_dict.items():
        for index in indices:
            # Obtener el nombre del archivo
            audio_number = index + 1  # Sumar 1 porque los índices comienzan en 0
            audio_filename = f"{nombre}_{audio_number:04d}.wav"  # Formato del nombre del archivo
            audio_path = f"audio_chunks_parciales/{audio_filename}"

            # Verificar que el índice sea válido en `original_best_speech_timestamps`
            #if index >= len(original_best_speech_timestamps):
            #    print(f"Advertencia: El índice {index} excede el rango de `original_best_speech_timestamps`. Saltando...")
            #    continue

            # Obtener el tiempo inicial y final del segmento en el audio original
            original_segment = original_best_speech_timestamps[index]
            original_start_time = original_segment["start"]

            # Cargar el archivo de audio
            load_to_wav(audio_path)
            wav = read_audio(r'converted_audio.wav')

            # Configurar los parámetros para la categoría
            param_config = parametros(category)

            # Reiniciar las variables locales (ya no globales)
            best_score = -float("inf")
            best_speech_timestamps = None


            #print("---------------------------------------------------------------------------------------")
            #print("Optimizando audio", audio_filename, "en velocidad", category)
            #print(audio_path)
            
            # Crear el objeto optuna study
            study = optuna.create_study(directions=["maximize"], sampler=sampler)

            # Ejecutar la optimización
            # n_trials=10
            study.optimize(lambda trial: objective(trial, param_config, wav), n_trials)

            #print("Marcas de tiempo del los segmentos")
            #print(best_speech_timestamps)

            # Ajustar los límites temporales para que respeten el tiempo original
            adjusted_timestamps = [
                {
                    "start": start + original_start_time,
                    "end": end + original_start_time
                }
                for segment in best_speech_timestamps
                for start, end in [segment.values()]
            ]

            #print("Marcas de tiempo relativas al Audio original")
            #print(adjusted_timestamps)

            # Guardar el resultado ajustado en el diccionario
            audio_key = f"audio_{audio_number:02d}"  # Clave del resultado
            results[audio_key] = adjusted_timestamps

    return results


AudioSegment.converter = which("ffmpeg") or r"C:\ruta\completa\a\ffmpeg\bin\ffmpeg.exe"

silero = load_silero_vad()

model = whisper.load_model("tiny")


def vad_audio_splitter(path, path_folder_out, min_duracion=15, max_duracion=30):

    global best_speech_timestamps, best_score, original_best_speech_timestamps
    """Separa un audio en segmentos de duración entre una duración mínima y máxima de acuerdo
    a la detección de actividad de voces del VAD.

    Args:
        path (str): Path del audio de entrada a separar
        path_folder_out (str): Path de la carpeta de salida donde se van a colocar los subcarpetas
        min_duracion (int): Cantidad mínima de duración de los segmentos en segundos. Default 15
        max_duracion (int): Cantidad máxima de duración de los segmentos en segundos. Default 30
    """

    #--------------------------- PARAMETROS CONFIGURABLES ------------------------------------------------
    # Escoge el método de optimización (puedes cambiar esta variable rápidamente)
    method = "tpe"  # Cambia a "random", "grid", "tpe".

    # Si se selecciona grid se debe cargar los valores para la grilla
    grid={ "threshold": [0.3, 0.5, 0.7], 
        "min_speech_duration_ms": [50, 100, 200],
        "min_silence_duration_ms": [50, 100, 200],
    }

    n_trials = 10  # Cambia el número de pruebas

    #-------------------------------------------------------------------------------------------------------


    audio, sr = librosa.load(path, sr=None)  # Carga el audio con su frecuencia de muestreo original
    total_time = len(audio) / sr  # Duración total del audio en segundos
    result = model.transcribe(path, language="es", task="transcribe")

    audio_categorizado=classify_segments_by_speed(result)

    menor_speed=get_lowest_speed_category(audio_categorizado)

    load_to_wav(path)
    wav = read_audio(r'converted_audio.wav')

    # Selecciona los parametros de acuerdo con lo ingresado
    param_config=parametros(menor_speed)

    # Selecciona el sampler
    sampler = get_sampler(method, grid)

    study = optuna.create_study(directions=["maximize"], sampler=sampler)

    # Optimizar usando el diccionario de configuración
    study.optimize(lambda trial: objective(trial, param_config,wav), n_trials)


    #print(f"Método de optimización utilizado: {method}")

    split_audio_chunks(path,best_speech_timestamps,output_folder="audio_chunks_parciales")

    seg_re_clasificar=categorize_and_filter_segments(best_speech_timestamps,audio_categorizado["segments"],menor_speed)

    original_best_speech_timestamps = best_speech_timestamps.copy()

    reusltados_re_clasificados=process_audio_chunks(seg_re_clasificar,path,n_trials,sampler)

    resultados_finales=extract_and_sort_timestamps(reusltados_re_clasificados)

    # borro los chunks parciales
    if os.path.exists("audio_chunks_parciales"):
        shutil.rmtree("audio_chunks_parciales")

    # borro el converted audio
    if os.path.exists("converted_audio.wav"):
        os.remove("converted_audio.wav")

    split_audio_chunks(path,resultados_finales,path_folder_out,gap=0,offset=0,tmax=max_duracion,tmin=min_duracion,extend_silence=True)


