
# ASR Dataset Generator

This repository contains an algorithm to identify the best sections of long audio files, enabling the creation of high-quality datasets for ASR tasks.

## Folder Structure

All data folders are located inside the main `Datos` directory. You can automatically create all required folders using the script `Utils/create_folder_estructure.py`.

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

- **NISQA (Quality Prediction)**
    - `threshold`: Quality threshold for audio selection.
    - `max_seconds`: Maximum duration for audio segments.
    - `min_seconds`: Minimum duration for audio segments.
    - `num_workers`: Number of workers for batch processing.
    - `batch_size`: Batch size for processing.

- **VAD (Voice Activity Detection)**
    - `mean_duration`: Mean duration for audio segments (seconds).
    - `std_desv`: Standard deviation for segment duration.

You can modify these parameters in `config.yaml` to change how the pipeline processes your audio data.

## Installation

All dependencies required for this project are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.9.1 or later installed. Some modules (e.g., Whisper) may require additional system dependencies such as ffmpeg.

## Running the Program

The processing chain is executed via `main.py`. You can configure the pipeline to skip stages such as **Denoising** or **Cleaning** by adjusting the configuration in `config.yaml`.
