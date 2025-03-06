import pytest
import os

def test_vad_track1():
    name_track = "Test1"
    # Chunks esperados
    chunk_target = 7  # Cantidad de chunks esperada
    chunk_real = len(os.listdir(os.path.join("Datos", "Audios_VAD", name_track)))  # Cantidad de chunks real
    
    assert chunk_target == chunk_real, f"Los segmentos esperados para {name_track} eran {chunk_target} pero tenemos {chunk_real} segmentos"

    # Duración de los chunks cerca de la media
    
    # Duración de los segmentos similar a la original 
    
def test_vad_track2():
    name_track = "Test2"
    # Chunks esperados
    chunk_target = 2  # Cantidad de chunks esperada
    lista_chunks = os.listdir(os.path.join("Datos", "Audios_VAD", name_track))  # Cantidad de chunks real
    chunk_real =  len(lista_chunks)
    
    assert chunk_target == chunk_real, f"Los segmentos esperados para {name_track} eran {chunk_target} pero tenemos {chunk_real} segmentos"
