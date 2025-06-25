from STT.whisper import stt_whisper
from STT.whisper_fast import stt_transcribe_faster
from STT.whisper_groq import STT_Groq
import time

audio = "Datos/Audios_VAD/audio_4/audio_4_0001.mp3"

time_0 = time.time()
rta_w1 = stt_whisper(audio)
time_1= time.time()
time_w1 = time_1 - time_0

rta_w2 = stt_transcribe_faster(audio)
time_2 = time.time()
time_w2 = time_2 - time_1

rta_w3 = STT_Groq(audio)
time_3 = time.time()
time_w3= time_3 - time_2

print("Transcripción Whisper Común:")
print(rta_w1)
print(" ")
print("Transcripción Fast Whisper:")
print(rta_w2)
print(" ")
print("Transcripción Whisper Groq:")
print(rta_w3)
print(" ")

print("Tiempos")
print("Whisper común (CPU): ", time_w1)
print("Faster Whisper (CPU): ", time_w2)
print("Whipser Grok: (GPU)", time_w3)
