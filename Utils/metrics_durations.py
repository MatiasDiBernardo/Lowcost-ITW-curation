import os
from pydub import AudioSegment

# Script para analizar los tiempos del dataset específicado por IDs
IDs_to_analyze = [6]

total_time = 0
for id in IDs_to_analyze:
    audio_path = os.path.join("Datos", "Audios_Raw", f"audio_{str(id)}.mp3")
    audio = AudioSegment.from_file(audio_path)
    minutes = len(audio) / 1000 / 60
    total_time += minutes

time_after_chain = 0
for id in IDs_to_analyze:
    folder_audios = os.path.join("Datos", "Audios_Transcript", f"audio_{str(id)}")
    for chunk in os.listdir(folder_audios):
        audio_path = os.path.join("Datos", "Audios_Transcript", f"audio_{str(id)}", chunk)
        audio = AudioSegment.from_file(audio_path)
        minutes = len(audio) / 1000 / 60
        time_after_chain += minutes

print("Los resultados de las dureciones son los siguientes:")
print("Duración original: ", total_time)
print("Duración after pipeline: ", time_after_chain)
