import os

# Crea la carpetas principales
root_folder = "Datos"
name_folders_data = ["Audio_to_Process", "Audios_Raw", "Audios_Denoise", "Audios_VAD", "Audios_Clean", "Audios_Transcript", "Dataset"]

for folder in name_folders_data:
    os.makedirs(os.path.join(root_folder, folder))


# Crear subcarpetas para archivos específicos
os.makedirs(os.path.join(root_folder, "Audios_Clean", "removed"))
os.makedirs(os.path.join(root_folder, "Audios_Transcript", "transcrips"))
