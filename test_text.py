from STT.whisper import stt_whisper
from STT.whisper_fast import stt_transcribe_faster
import time

audio = "Datos/Audios_VAD/audio_4/audio_4_0001.mp3"

time_start = time.time()
rta_w1 = stt_whisper(audio)
time_finish = time.time()
time_w1 = time_finish - time_start

rta_w2 = stt_transcribe_faster(audio)
time_w2 = time.time() - time_finish

print("Transcripción Large Comun:")
print(rta_w1)
print("Transcripción Large Fast:")
print(rta_w2)

print("Tiempos")
print("Whisper común: ", time_w1)
print("Whipser Fast: ", time_w2)
