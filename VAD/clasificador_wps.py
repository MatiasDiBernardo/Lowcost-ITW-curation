def classify_segments_by_speed(result, unit="segundo"):
    """Clasifica segmentos según la velocidad de habla (WPS) con atributos id, start, end y speed."""
    time_factor = 1 if unit == "segundo" else 60
    speed_thresholds = {"slow": 2, "normal": 4}  # Umbrales para WPS

    segments = result["segments"]
    
    classified_segments = []
    for idx, segment in enumerate(segments):
        duration = segment["end"] - segment["start"]
        word_count = len(segment["text"].split())
        wps = word_count / (duration / time_factor)

        if wps < speed_thresholds["slow"]:
            speed = "Slow"
        elif wps <= speed_thresholds["normal"]:
            speed = "Normal"
        else:
            speed = "Fast"

        classified_segments.append({
            "id": idx,
            "start": segment["start"],
            "end": segment["end"],
            "speed": speed
        })

    return {"segments": classified_segments}

def get_lowest_speed_category(result):
    """Devuelve la categoría de velocidad más baja presente en los segmentos."""
    # Mapeo de categorías a valores numéricos
    speed_map = {"Slow": 1, "Normal": 2, "Fast": 3}

    # Extraer todas las velocidades de los segmentos
    speeds = [segment["speed"] for segment in result["segments"]]

    # Encontrar el valor más bajo según el mapeo
    lowest_speed = min(speeds, key=lambda speed: speed_map[speed])

    return lowest_speed.capitalize()


def categorize_and_filter_segments(silero_segments, whisper_segments, excluded_speed):
    """
    Categoriza los segmentos de Silero basándose en los segmentos de Whisper, 
    y elimina los solapamientos dentro de cada velocidad, considerando la exclusión
    de una categoría de velocidad en el análisis final.

    Args:
        silero_segments (list): Lista de segmentos de Silero con 'start' y 'end'.
        whisper_segments (list): Lista de segmentos de Whisper con 'start', 'end' y 'speed'.
        excluded_speed (str): Velocidad a excluir ('slow', 'normal', 'fast').

    Returns:
        dict: Diccionario con las velocidades restantes como claves y listas de índices como valores.
    """
    # Convertir la velocidad excluida a minúsculas para consistencia
    excluded_speed = excluded_speed.lower()

    # Inicializamos el diccionario para almacenar los segmentos por velocidad
    categorized_segments = {
        "slow": set(),
        "normal": set(),
        "fast": set()
    }

    # Paso 1: Asignación de segmentos a todas las velocidades posibles
    for idx, silero in enumerate(silero_segments):
        for whisper in whisper_segments:
            if silero["start"] < whisper["end"] and silero["end"] > whisper["start"]:
                speed = whisper["speed"].lower()
                if speed in categorized_segments:
                    categorized_segments[speed].add(idx)

    # Paso 2: Eliminar solapamientos en el orden de velocidad
    # Primero eliminamos de Fast, luego de Normal
    # Comenzamos con la categoría más alta, Fast
    for idx in list(categorized_segments["fast"]):
        if idx in categorized_segments["normal"]:
            categorized_segments["normal"].remove(idx)
        if idx in categorized_segments["slow"]:
            categorized_segments["slow"].remove(idx)

    # Ahora eliminamos de Normal los que también están en Slow
    for idx in list(categorized_segments["normal"]):
        if idx in categorized_segments["slow"]:
            categorized_segments["normal"].remove(idx)

    # Paso 3: Excluir la categoría indicada
    if excluded_speed in categorized_segments:
        del categorized_segments[excluded_speed]

    # Convertimos las listas a un formato adecuado
    categorized_segments = {speed.capitalize(): list(indices) for speed, indices in categorized_segments.items() if indices}

    return categorized_segments