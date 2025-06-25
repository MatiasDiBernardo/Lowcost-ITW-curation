import os
from groq import Groq

# También podría probar con "whisper-large-v3-turbo" (es más rápido pero creo que pude ser menos preciso)
def STT_Groq(filename):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="whisper-large-v3",
        response_format="verbose_json",
        )
    
    return transcription.text
      