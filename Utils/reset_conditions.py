import os
import shutil

# Returns to the process folder
process_path = os.path.join("Data", "Audio_to_Process")
raw_path = os.path.join("Data", "Audios_Raw")

audios_to_recover = os.listdir(raw_path)
for audio in audios_to_recover:
    shutil.move(os.path.join(raw_path, audio), os.path.join(process_path, audio))

# Removes folders in vad, clean and transcript
vad_path = os.path.join("Data", "Audios_VAD") 
denoise_path = os.path.join("Data", "Audios_Denoise") 
speaker_filt_path = os.path.join("Data", "Audios_Speaker_Filt")
clean_audios_path = os.path.join("Data", "Audios_Clean") 
transcript_path = os.path.join("Data", "Audios_Transcript") 

name_folders = [name.split(".")[0] for name in audios_to_recover]

for folder in name_folders:
    shutil.rmtree(os.path.join(vad_path, folder), ignore_errors=True)
    shutil.rmtree(os.path.join(denoise_path, folder), ignore_errors=True)
    shutil.rmtree(os.path.join(speaker_filt_path, folder), ignore_errors=True)
    shutil.rmtree(os.path.join(clean_audios_path, folder), ignore_errors=True)
    shutil.rmtree(os.path.join(transcript_path, folder), ignore_errors=True)
    
print("Returned to the initial state correctly")