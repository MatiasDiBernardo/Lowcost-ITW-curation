from pathlib import Path
from pyannote.audio import Pipeline
import torch

def load_osd_pipeline(hf_token: str = None):
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            token=hf_token
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            use_auth_token=hf_token
        )
    
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        
    return pipeline

def detect_overlapped_speech(
    audio_path: Path, 
    waveform, 
    sample_rate: int, 
    pipeline: Pipeline, 
    min_overlap_sec: float = 0.05
) -> bool:
    """
    Process an audio (waveform) to detect overlapped speech.
    
    Args:
        audio_path (Path): Path (only for logging/debug).
        waveform (Tensor): Audio tensor [Channels, Samples].
        sample_rate (int): Sample rate.
        pipeline (Pipeline): OSD pipeline loaded.
        min_overlap_sec (float): Seconds threshold to consider the audio 'dirty'.

    Returns:
        bool: True if there is overlapped speech (NOT APPROPRIATE) or error.
              False if the audio is clean (APPROPRIATE).
    """
    try:
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        
        overlap_duration = 0.0
        for segment in result.get_timeline().support():
            overlap_duration += segment.duration
            
        return overlap_duration > min_overlap_sec

    except Exception as e:
        return True