from faster_whisper import WhisperModel

def stt_transcribe_faster(audio_path):
    model_size = "large-v3"

    # # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(audio_path, beam_size=5)
    text = ""

    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text += segment.text
    
    return text
    
