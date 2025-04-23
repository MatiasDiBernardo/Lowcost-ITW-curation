import whisper

# Por ahora solo usamos Whisper para transcripciones (plantear WER con whisper Turbo)
MODEL_SIZE = "turbo"
    
def stt_whisper(audio_path):
    """Aplica Whisper para Speech-To-Text a un audio y devuelve un string con la transcripción.

    Args:
        audio_path (str): Path del audio a transcribir.
    Returns:
        str: Transcripción del audio
    """
    model = whisper.load_model(MODEL_SIZE)

    # Transcribe audio
    result = model.transcribe(audio_path)

    return result