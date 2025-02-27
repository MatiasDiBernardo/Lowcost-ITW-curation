from df.enhance import enhance, init_df, load_audio, save_audio

def denoise_deep_net(input_audio, output_audio):
    """Aplica el algorítmo de denoising a un audio y los guarda 
    en la dirección específicada. 

    Args:
        input_audio (str): Path del audio de entrada a denoisear
        output_audio (str): Path donde se guarda el audio denoiseado
    """
    # Load default model
    model, df_state, _ = init_df(log_file=None, log_level="NONE")
    
    audio, _ = load_audio(input_audio, sr=df_state.sr())

    # Denoise the audio
    enhanced = enhance(model, df_state, audio)
    # Save for listening
    save_audio(output_audio, enhanced, df_state.sr())
