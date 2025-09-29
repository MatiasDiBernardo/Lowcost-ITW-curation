from df.enhance import enhance, init_df, load_audio, save_audio

def denoise_deep_net(input_audio, output_audio):
    """ Applied DeepFilterNet denoising algorithm. 

    Args:
        input_dir (str): Path to folder with audios in wav format and any sample rate.
        output_dir (str): Path to the output folder of denoised audios.
    """
    # Load default model
    model, df_state, _ = init_df(log_file=None, log_level="NONE")
    
    audio, _ = load_audio(input_audio, sr=df_state.sr())

    # Denoise the audio
    enhanced = enhance(model, df_state, audio)
    # Save for listening
    save_audio(output_audio, enhanced, df_state.sr())
