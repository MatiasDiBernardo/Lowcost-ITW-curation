import os
from pydub import AudioSegment

def convert_mp3(audio_path, main_root):
    if audio_path.endswith("wav"):
        path_wav = os.path.join(main_root, audio_path)
        audio = AudioSegment.from_wav(path_wav)

        name = audio_path.split(".")[0]
        path_mp3 = os.path.join(main_root, name + ".mp3")

        # Export as MP3
        audio.export(path_mp3, format="mp3")
        os.remove(path_wav)

def all_mp3():
    # Audios Raw
    path_root = os.path.join("Datos", "Audios_Raw")
    audios = os.listdir(path_root)
    for audio in audios:
        convert_mp3(audio, path_root)

    # Audios Denoise
    path_root = os.path.join("Datos", "Audios_Denoise")
    audios = os.listdir(path_root)
    for audio in audios:
        convert_mp3(audio, path_root)
    
    # Audios VAD
    path_root = os.path.join("Datos", "Audios_VAD")
    path_folders = os.listdir(path_root)
    for folder in path_folders:
        path_complete = os.path.join(path_root, folder)
        audios = os.listdir(path_complete)
        for audio in audios:
            convert_mp3(audio, path_complete)

    # Audios Clean
    path_root = os.path.join("Datos", "Audios_Clean")
    path_folders = os.listdir(path_root)
    for folder in path_folders:
        path_complete = os.path.join(path_root, folder)
        audios = os.listdir(path_complete)
        for audio in audios:
            convert_mp3(audio, path_complete)

    # Audios Transcript
    path_root = os.path.join("Datos", "Audios_Transcript")
    path_folders = os.listdir(path_root)
    for folder in path_folders:
        path_complete = os.path.join(path_root, folder)
        audios = os.listdir(path_complete)
        for audio in audios:
            convert_mp3(audio, path_complete)
    
    print("All audios are on MP3 Format")
    
all_mp3()
