from QualityPrediction.nisqa.nisqa.NISQA_model import nisqaModel, set_verbose

import os
import librosa
import yaml
from glob import glob

# Carga configuración
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

nisqa_config = config["quality_prediction"]

OUTPUT_DIR = ''  # Si se le asigna valor, graba un csv con los resultados de NISQA en la ruta que se le pase.
THRESHOLD = nisqa_config["threshold"]
MAX_SECONDS = nisqa_config["max_seconds"]
MIN_SECONDS = nisqa_config["min_seconds"]
NUM_WORKERS = nisqa_config["num_workers"]
BATCH_SIZE = nisqa_config["batch_size"]

def run_folder_predict(input_dir: str, output_dir: str = OUTPUT_DIR, num_workers: int = NUM_WORKERS, batch_size: str = BATCH_SIZE, ms_channel: str = None, verbose = False):
    ''' Evalúa si la calidad de los audios (.wav o .mp3) de una carpeta superan el umbral, usando el modelo NISQA. Retorna un array de booleanos.
    
        Recibe los parámetros que recibiría NISQA por consola y los mete en un dict, tomado del archivo 'run_predict.py' de NISQA.
        Inicializa el modelo con los parámetros y ejecuta la predicción. Esta devuelve un DataFrame, del que se obtiene la predicción de MOS.
        Con estos valores se forma el array de booleanos que indica si los archivos superaron el umbral.
        NOTA: Tener cuidado con el orden de los archivos, NISQA los ordena alfabeticamente, incluidos los numeros. Ej: aa_12.mp3 va a aparecer antes que aa_5.mp3. '''

    set_verbose(verbose)    
    if verbose:
        print('Running run_folder_predict')
    args = {'mode': 'predict_dir', 'output_dir': output_dir, 'pretrained_model': 'weights/nisqa.tar', 'data_dir': input_dir, 'num_workers': num_workers, 'bs': batch_size, 'ms_channel': ms_channel }

    nisqa = nisqaModel(args)
    df = nisqa.predict()

    out = []
    wavs = glob(os.path.join(input_dir, '*.wav'))
    mp3s = glob(os.path.join(input_dir, '*.mp3'))
    files = wavs + mp3s
    files = [os.path.basename(file) for file in files]
    for file in files:
        mos = df.loc[df['deg'] == file, 'mos_pred'].values[0]
        out.append(mos >= THRESHOLD)

    return out

def filtering_nisqa(audio_path: str, output_dir: str = OUTPUT_DIR, num_workers: int = NUM_WORKERS, batch_size: str = BATCH_SIZE, ms_channel: str = None, verbose = False):
    """Evaluate and returns a boolean for approved or discarded audio based on quality MOS predicted by NISQA. Threhosld setup in config.yaml

    Args:
        audio_path (path): Path to audio to evaluate.
        output_dir (str, optional): Defaults to OUTPUT_DIR.
        num_workers (int, optional): Defaults to NUM_WORKERS.
        batch_size (str, optional): Defaults to BATCH_SIZE.
        ms_channel (str, optional): Defaults to None.
        verbose (bool, optional): Defaults to False.

    Returns:
        boolean: True if audio is over MOS threshold. Otherwise False
    """

    set_verbose(verbose)
    if verbose:
        print('Running run_audio_predict')
    args = {'mode': 'predict_file', 'output_dir': output_dir, 'pretrained_model': 'weights/nisqa.tar', 'deg': audio_path, 'num_workers': num_workers, 'bs': batch_size, 'ms_channel': ms_channel }
    
    # Audio is not valid if the duration is outside the limits (configured by the user)
    y, sr = librosa.load(audio_path, mono=True)
    audio_duration = len(y)/sr
    if audio_duration > MAX_SECONDS or audio_duration < MIN_SECONDS:
        return False

    nisqa = nisqaModel(args)
    df = nisqa.predict()
    mos = df.loc[0, 'mos_pred']

    return mos >= THRESHOLD