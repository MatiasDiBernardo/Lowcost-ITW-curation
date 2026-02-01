from QualityPredition.dnsmos.dnsmos_local import main, set_verbose
import numpy as np
import pandas as pd
import librosa
import os
import yaml
import glob
from tqdm import tqdm

# Carga configuración
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

nisqa_config = config["quality_prediction"]
VERBOSE = config["verbose"]

OUTPUT_DIR = ''  # Si se le asigna valor, graba un csv con los resultados de NISQA en la ruta que se le pase.
THRESHOLD = nisqa_config["threshold"]
MAX_SECONDS = nisqa_config["max_seconds"]
MIN_SECONDS = nisqa_config["min_seconds"]
NUM_WORKERS = nisqa_config["num_workers"]
BATCH_SIZE = nisqa_config["batch_size"]
SAVE_SCORES = nisqa_config["save_scores"]

def run_audio_predict(audio_path: str, output_dir: str, personalized_MOS: bool = False, result_type: str = 'bool', threshold: float = THRESHOLD):
    set_verbose(VERBOSE)
    if VERBOSE:
        print('Running run_audio_predict DNSMOS...')
    args = {'testset_dir': audio_path, 'personalized_MOS': personalized_MOS }

    if SAVE_SCORES:
        csv_path = f'{output_dir}/DNSMOS_Results.csv'
        args['csv_path'] = csv_path

    
    # El audio no es válido si supera el máximo de tiempo establecido
    y, sr = librosa.load(audio_path, mono=True)
    audio_duration = len(y)/sr
    if audio_duration > MAX_SECONDS or audio_duration < MIN_SECONDS:
        return False

    df = main(args)
    # df = pd.read_csv(csv_path)
    mos = df.loc[0, 'OVRL']

    if result_type == 'bool':
        return mos >= THRESHOLD
    elif result_type == 'df':
        return df
    elif result_type == 'dnsmos':
        return mos

def run_folder_predict(input_dir: str, output_dir: str, personalized_MOS: bool = False, result_type: str = 'bool', threshold: float = THRESHOLD):
    ''' If personalized_MOS is True, it penalizes interfering speakers.
    '''
    set_verbose(VERBOSE)
    if VERBOSE:
        print('Running run_folder_predict DNSMOS...')
    csv_path = f'{output_dir}/DNSMOS_Results.csv'
    args = {'testset_dir': input_dir, 'csv_path': csv_path, 'personalized_MOS': personalized_MOS }
    
    main(args)
    df = pd.read_csv(csv_path)

    if result_type == 'bool':
        out = []
        wavs = glob(os.path.join(input_dir, '*.wav'))
        mp3s = glob(os.path.join(input_dir, '*.mp3'))
        files = wavs + mp3s
        for file in files:
            mos = df.loc[df['filename'] == file, 'OVRL'].values[0]
            out.append(mos >= threshold)
        return out
    elif result_type == 'df':
        return df
    elif result_type == 'dnsmos':
        return np.array(df['OVRL'])
    
def run_multifolder_predict(root_path: str, result_type: str = 'mean', starting_point: int = 0, finishing_point: int = 1000000):
    folder_paths = []
    paths = glob.glob(os.path.join(root_path, '**'))
    is_root_added = False
    for p in paths:
        if os.path.isdir(p):
            folder_paths.append(p)
        elif not is_root_added:
            if os.path.isfile(p):
                file, extension = os.path.splitext(p)
                if extension == '.wav' or extension == '.mp3':
                    folder_paths.insert(0, root_path)

    print(f'Number of folders to evaluate: {len(folder_paths)}')
    total_mos = np.array([])
    num_audios = []
    means = []
    stds = []
    for i in tqdm(range(len(folder_paths))):
        f = folder_paths[i]
        print(f)
        if i < starting_point or i > finishing_point:
            print('Reading csv...')
            res = pd.read_csv(os.path.join(f, f'DNSMOS_Results.csv'))
            mos = np.array(res['mos_pred'])
        else:
            try:
                mos = run_folder_predict(f, f, result_type='dnsmos')
            except ValueError:
                num_audios.append(-1)
                means.append(-1)
                stds.append(-1)
                continue

        print(mos)

        num_audios.append(len(mos))
        mean = np.round(np.nanmean(mos), 2)
        std = np.round(np.nanstd(mos), 2)
        print(f'Mean +/- std: {mean:.2f} +/- {std:.2f}')
        means.append(mean)
        stds.append(std)
        total_mos = np.concatenate((total_mos, mos))

    total_mean = np.round(np.nanmean(total_mos), 2)
    total_std = np.round(np.nanstd(total_mos), 2)

    data = {'folder': folder_paths, 'num_audios': num_audios, 'mean': means, 'std': stds}
    df = pd.DataFrame(data)
    filename_info = f'{os.path.basename(root_path)}-{total_mean}-{total_std}.csv'
    df.to_csv(root_path + f'/{filename_info}')
    print(f'Total Mean +/- std: {total_mean} +/- {total_std}')
    return total_mean, total_std