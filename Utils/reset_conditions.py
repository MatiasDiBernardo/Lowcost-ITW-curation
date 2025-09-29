import os
import shutil


# To use when you want to return to the original state (here put the audios you want to test with)
audios_to_recover = ["Test1.wav", "Test2.wav", "Test3.wav", "Test4.wav", "Test5.wav", "Test6.wav", "Test7.wav"]


# Returns to the process folder
process_path = os.path.join("Datos", "Audio_to_Process")
raw_path = os.path.join("Datos", "Audios_Raw")
for audio in audios_to_recover:
    shutil.move(os.path.join(raw_path, audio), os.path.join(process_path, audio))


# Removes audios in denoised
denoise_path = os.path.join("Datos", "Audios_Denoise") 
for audio in audios_to_recover:
    path_to_remove = os.path.join(denoise_path, audio)
    if os.path.exists(path_to_remove):
        os.remove(path_to_remove)


# Removes folders in vad, clean and transcript
vad_path = os.path.join("Datos", "Audios_VAD") 
clean_audios_path = os.path.join("Datos", "Audios_Clean") 
transcript_path = os.path.join("Datos", "Audios_Transcript") 

name_folders = [name.split(".")[0] for name in audios_to_recover]

for folder in name_folders:
    shutil.rmtree(os.path.join(vad_path, folder), ignore_errors=True)
    shutil.rmtree(os.path.join(clean_audios_path, folder), ignore_errors=True)
    shutil.rmtree(os.path.join(transcript_path, folder), ignore_errors=True)
    
print("Returned to the initial state correctly")