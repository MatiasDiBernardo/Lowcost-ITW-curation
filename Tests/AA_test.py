import pytest
import os

# Estos test buscan ser independientes del resultado del VAD

def test_detection_track1():
    name_track = "Test1"

    ammount_vad_chunks = len(os.listdir(os.path.join("Datos", "Audios_VAD", name_track)))
    # En este caso pasan todos los chunks
    chunks_survive = ammount_vad_chunks # De la cantidad de chunks que genera el VAD cuantos tiene que pasar
    actual_survival_chunks = len(os.listdir(os.path.join("Datos", "Audios_Clean", name_track)))  # Cantidad real de chunks que pasaron el filtro
    
    assert chunks_survive == actual_survival_chunks, f"Se esperaban que pasen {str(chunks_survive)} audios, pero pasaron {str(actual_survival_chunks)} audios."
    
def test_detection_track2():
    name_track = "Test2"
    ammount_vad_chunks = len(os.listdir(os.path.join("Datos", "Audios_VAD", name_track)))
    # En este caso no pasa ningún chunk
    chunks_survive = 0 # De la cantidad de chunks que genera el VAD cuantos tiene que pasar
    actual_survival_chunks = len(os.listdir(os.path.join("Datos", "Audios_Clean", name_track)))  # Cantidad real de chunks que pasaron el filtro
    
    assert chunks_survive == actual_survival_chunks, f"Se esperaban que pasen {str(chunks_survive)} audios, pero pasaron {str(actual_survival_chunks)} audios."
    