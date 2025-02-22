import os
from df.enhance import enhance, init_df, load_audio, save_audio

def denoise_deep_net(input_dir, output_dir):
    """Aplica el algorítmo de denoising a una carpeta de audios y los guarda 
    en el directorio especificado. 

    Args:
        input_dir (str): Path de los audios de entrada a denoisear
        output_dir (str): Path donde se guardan los audios denoiseados
    """
    # Load default model
    model, df_state, _ = init_df()
    
    for filename in os.listdir(input_dir):
        audio, _ = load_audio(os.path.join(input_dir, filename), sr=df_state.sr())

        # Denoise the audio
        enhanced = enhance(model, df_state, audio)
        # Save for listening
        save_audio(os.path.join(output_dir, filename), enhanced, df_state.sr())
