from AudioAnalyzer.nisqa.nisqa.NISQA_model import nisqaModel, set_verbose
import os
from glob import glob
import pandas as pd

OUTPUT_DIR = ''  # Si se le asigna valor, graba un csv con los resultados de NISQA en la ruta que se le pase.
THRESHOLD = 4
NUM_WORKERS = 0
BATCH_SIZE = 10


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

def run_audio_predict(audio_path: str, output_dir: str = OUTPUT_DIR, num_workers: int = NUM_WORKERS, batch_size: str = BATCH_SIZE, ms_channel: str = None, verbose = False):
    ''' Evalúa y retorna si la calidad de un audio supera el umbral, usando el modelo NISQA.
    
        Recibe los parámetros que recibiría NISQA por consola y los mete en un dict, tomado del archivo 'run_predict.py' de NISQA.
        Inicializa el modelo con los parámetros y ejecuta la predicción. Esta devuelve un DataFrame, del que se obtiene la predicción de MOS.
        Con este valor se obtiene el booleano que indica si el archivo superó el umbral. '''

    set_verbose(verbose)
    if verbose:
        print('Running run_audio_predict')
    args = {'mode': 'predict_file', 'output_dir': output_dir, 'pretrained_model': 'weights/nisqa.tar', 'deg': audio_path, 'num_workers': num_workers, 'bs': batch_size, 'ms_channel': ms_channel }

    nisqa = nisqaModel(args)
    df = nisqa.predict()
    mos = df.loc[0, 'mos_pred']

    return mos >= THRESHOLD