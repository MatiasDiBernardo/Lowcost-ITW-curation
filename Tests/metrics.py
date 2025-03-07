import os
import json

# Accuracy, preccision, recall and F1 de toda la cadena (se base en resultados del VAD)

# Resultados esperados (completar de forma manual en el spected_results.json)
with open(os.path.join("Tests", "expected_results.json"), "r") as json_file:
    expected_results = json.load(json_file)

# Se computa los resultados obtenidos por la cadena de procesamientos
predicted_results = {}
folders_vad = os.listdir(os.path.join("Datos", "Audios_VAD"))

for folder in folders_vad:
    audios = os.listdir(os.path.join("Datos", "Audios_VAD", folder))
    for audio in audios:
        audios_end_of_the_chain = os.listdir(os.path.join("Datos", "Audios_Transcript", folder))
        chain_result = audio in audios_end_of_the_chain
        name = audio.split(".")[0]
        predicted_results[name] = chain_result

# Calculation of confusion matrix
TP = 0  # TruePositive (Correctamente clasificado como positivo) 
TN = 0  # TrueNegative (Correctamente clasificado como negativo)
FP = 0  # FalsoPositvo (Erroneamente clasificado como positivo)
FN = 0  # FalsoNegativo (Erroneamente clasificado como negativo)

for name_audio in expected_results.keys():
    if predicted_results[name_audio] and expected_results[name_audio]:
        TP += 1

    if not predicted_results[name_audio] and not expected_results[name_audio]:
        TN += 1

    if predicted_results[name_audio] and not expected_results[name_audio]:
        FP += 1
    
    if not predicted_results[name_audio] and expected_results[name_audio]:
        FN += 1
    
# Calcular resultados
accuarcy = (TP + TN)/(TP + TN + FP + FN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)
F1 = 2 * (precision*recall)/(precision+recall)

# Mostrar resultados
print("Resultados")
print(f"Accuracy: {str(round(accuarcy, 3))}")
print(f"Precision: {str(round(precision, 3))}")
print(f"Recall: {str(round(recall, 3))}")
print(f"F1: {str(round(F1, 3))}")
