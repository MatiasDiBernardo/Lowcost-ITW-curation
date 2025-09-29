
# Low Cost pre-processing pipeline for ITW datasets generation

This repository contains an algorithm to select the best sections of noisy speech datasets, enabling the creation of high-quality datasets for ASR tasks.

## Folder Structure

All data folders are located inside the main `Data` directory. You can automatically create all required folders using the script `Utils/create_folder_estructure.py`.

- **Audio_to_Process**: Place new audios here to be processed. After processing, files are moved to the raw audios folder and this folder is emptied.
- **Audios_Raw**: Contains raw audios to be processed (e.g., podcasts, videos). Files are renamed with an ID and name separated by `_`.
- **Audios_Denoise**: Contains denoised audios.
- **Audios_VAD**: Contains subfolders for each long audio, with segments (chunks) named by audio ID and chunk number.
- **Audios_Clean**: Contains audio segments that passed quality filters. Also includes a `removed` subfolder for tracking discarded audios.
- **Audios_Transcript**: Contains segments with the best transcriptions. Also includes a `transcripts` subfolder for transcript CSVs.
- **Dataset**: Contains all the best fragments, merged into a single dataset.

## Main Configuration (`config.yaml`)

The `config.yaml` file allows you to adjust key parameters for each stage of the processing chain. Example options include:

- **Global**
    - `test`: Enable or disable test mode.
    - `verbose`: Enable or disable verbose output.

- **VAD (Voice Activity Detection)**
    - `mean_duration`: Mean duration for audio segments (seconds).
    - `std_desv`: Standard deviation for segment duration.

- **Quality Prediction**
    - `type`: Mos predictor model ("NISQA" or "DNS MOS").
    - `threshold`: Quality threshold for audio selection.
    - `max_seconds`: Maximum duration for audio segments.
    - `min_seconds`: Minimum duration for audio segments.
    - `num_workers`: Number of workers for batch processing.
    - `batch_size`: Batch size for processing.

- **Denoising**
    - `type`: Denoising model ("None", "Demucs" or "DeepFilterNet").

- **Transcription (STT)**
    - `type`: Only Whisper model (working on accurate alternatives).
    - `model_size`: Whisper model size ("Small", "Large", "Turbo").

You can modify these parameters in `config.yaml` to change how the pipeline processes your audio data.

## Installation — dependencies

### 1) Pipeline only (lightweight) — recommended for production / dataset generation
Create and activate a virtual environment, then install the core package (no heavy metric libraries):

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip setuptools wheel
pip install .
```

### 2) Pipeline + Metrics (heavy) — for evaluation and reproducing paper figures

Inside the same activated virtualenv, install the optional metrics extras (PESQ/STOI/NISQA/room acoustics, etc.):

```bash
pip install .[metrics]
```
What this provides: all metric calculation libraries required to run evaluation scripts such as utils/evaluate_dataset_metrics.py. These are heavier and may need system build tools or ML runtimes (e.g., torch).

### 3) Using Poetry (project-managed venv)

Install core dependencies with Poetry:

```bash
poetry install
```
Add metrics extras into the Poetry venv (reliable across Poetry versions):

```bash
# Run pip inside Poetry-managed venv to install extras from the local project
poetry run pip install .[metrics]
```
(Some Poetry versions support installing extras directly — consult your Poetry version docs. Using poetry run pip is broadly compatible.)

## Running the Program

The processing chain is executed via `main.py`. You can configure the pipeline to skip stages such as **Denoising** or **Cleaning** by adjusting the configuration in `config.yaml`.
