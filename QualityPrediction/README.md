# NISQA

Esta es un inferencia de NISQA, un modelo entrenado para evaluar la calidad de audios de voz.
Esta permite evaluar un audio solo o toda una carpeta. Va a devolver True o False dependiendo de que el MOS del audio evaluado haya superado el umbral.

## Parámetros globales
  - THRESHOLD. El umbral según el que define si un audio es aceptable o no para ser utilizado.
  - NUM_WORKERS. Opcional. Permite la configuración del PyTorch Dataloader utilizado.
  - BATCH_SIZE. Opcional. Permite la configuración del PyTorch Dataloader utilizado.
  - OUTPUT_DIR. Opcional. Si tiene un valor distinto de '', NISQA graba en un csv los resultados que retorna.	

### Requerimientos

Estas son las librerías incluidas en el env.yaml de nisqa.

  - cudatoolkit=10.2.89
  - libsndfile=1.0.31
  - pillow=8.4.0
  - pip=21.3.1
  - python=3.9.9
  - pytorch=1.10.1
  - scipy=1.7.3
  - torchvision=0.11.2
  - librosa==0.8.1
  - matplotlib==3.5.1
  - numba==0.54.1
  - numpy==1.20.3
  - pandas==1.3.5
  - pyyaml==6.0
  - scikit-learn==1.0.2
  - seaborn==0.11.2
  - tqdm==4.62.3

### NISQA: Speech Quality and Naturalness Assessment

Dejo acá una breve intro del README de NISQA. Para más info verlo entero dentro de su carpeta.

**Speech Quality Prediction:**   
NISQA is a deep learning model/framework for speech quality prediction. The NISQA model weights can be used to predict the quality of a speech sample that has been sent through a communication system (e.g telephone or video call). Besides overall speech quality, NISQA also provides predictions for the quality dimensions *Noisiness*, *Coloration*, *Discontinuity*, and *Loudness* to give more insight into the cause of the quality degradation. 

**Speech Quality Datasets:**  
We provide a large corpus of more than 14,000 speech samples with subjective speech quality and speech quality dimension labels. 
