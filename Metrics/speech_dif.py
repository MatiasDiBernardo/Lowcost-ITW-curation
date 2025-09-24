from pymcd.mcd import Calculate_MCD

def calculate_MCD(audio_based, audio_denoised):
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd_val = mcd_toolbox.calculate_mcd(audio_based, audio_denoised)
    return mcd_val

