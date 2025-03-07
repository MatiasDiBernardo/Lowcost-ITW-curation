import pytest
import os
import librosa

def tiempo_chunks(path_audio):
    time = 0
    for audio in os.listdir(path_audio):
        y, sr = librosa.load(os.path.join(path_audio, audio), mono=True)
        time += len(y)/sr

    return time

def test_chunks_track1():
    name_track = "Test1"  # Sin file extension
    chunk_target = 7  # Cantidad de chunks esperada

    chunk_real = len(os.listdir(os.path.join("Datos", "Audios_VAD", name_track)))  # Cantidad de chunks real
    
    assert chunk_target == chunk_real, f"Los segmentos esperados para {name_track} eran {chunk_target} pero tenemos {chunk_real} segmentos"

def test_time_track1():
    name_track = "Test1.wav"  # Con file extension
    max_time_diff = 70  # Máxima diferencia de tiempo entre audio original y VAD en segundos 

    # Duración de los segmentos similar a la original 
    path_base = os.path.join("Datos", "Audios_VAD", name_track.split(".")[0])
    time_vad = tiempo_chunks(path_base)
    
    y, sr = librosa.load(os.path.join("Datos", "Audios_Raw", name_track), mono=True)
    time_original = len(y)/sr
    
    assert abs(time_vad - time_original) < max_time_diff, f"El tiempo del audio original es de {str(round(time_original, 2))} mientras que la suma del VAD da {str(round(time_vad))}"

def test_chunks_track2():
    name_track = "Test2"  # Sin file extension
    chunk_target = 2  # Cantidad de chunks esperada

    chunk_real = len(os.listdir(os.path.join("Datos", "Audios_VAD", name_track)))  # Cantidad de chunks real
    
    assert chunk_target == chunk_real, f"Los segmentos esperados para {name_track} eran {chunk_target} pero tenemos {chunk_real} segmentos"
    
def test_time_track2():
    name_track = "Test2.wav"  # Con file extension
    max_time_diff = 30  # Máxima diferencia de tiempo entre audio original y VAD en segundos 

    # Duración de los segmentos similar a la original 
    path_base = os.path.join("Datos", "Audios_VAD", name_track.split(".")[0])
    time_vad = tiempo_chunks(path_base)
    
    y, sr = librosa.load(os.path.join("Datos", "Audios_Raw", name_track), mono=True)
    time_original = len(y)/sr
    
    assert abs(time_vad - time_original) < max_time_diff, f"El tiempo del audio original es de {round(time_original, 2)} mientras que la suma del VAD da {round(time_vad)}"
