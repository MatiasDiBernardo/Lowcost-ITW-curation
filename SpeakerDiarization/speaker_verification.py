import numpy as np
import torch
import torchaudio
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import warnings
from pyannote.audio import Model, Inference

def load_embedding_model(hf_token: str):
    """
    Loads the Pyannote embedding model.
    Tries to use 'token' (new standard) or 'use_auth_token' (legacy) for compatibility.
    """
    try:
        try:
            model = Model.from_pretrained("pyannote/embedding", token=hf_token)
        except TypeError:
            model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
            
        inference = Inference(model)
        
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
            
        return inference
    except Exception as e:
        print(f"Error loading Embedding model: {e}")
        return None

def extract_fixed_chunks(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration_s: float = 5.0, 
    step_duration_s: float = 2.5,
    min_chunk_duration_s: float = 3.0
):
    """
    Extracts overlapping fixed-size chunks from an audio file.
    Chunks shorter than min_chunk_duration_s are discarded.
    """
    num_frames = waveform.shape[1]
    chunk_size_frames = int(chunk_duration_s * sample_rate)
    step_size_frames = int(step_duration_s * sample_rate)
    min_chunk_frames = int(min_chunk_duration_s * sample_rate)

    start = 0
    while start + min_chunk_frames <= num_frames:
        end = start + chunk_size_frames
        chunk = waveform[:, start:end]
        yield chunk
        
        start += step_size_frames

def verify_multi_speaker_by_clustering(
    audio_path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
    embedding_pipeline,
    cluster_method: str = 'agglomerative', 
    agglomerative_threshold: float = 0.6,
    spectral_min_clusters: int = 2,        
    spectral_max_clusters: int = 4
) -> bool:
    """
    Verifies if there are multiple speakers using a Pyannote embedding pipeline
    and fixed-size audio chunks.
    
    Returns:
        bool: True if detects multiple speakers or there is an error (NOT APPROPRIATE).
              False if detects a single speaker (APPROPRIATE).
    """
    try:
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        speech_chunks = list(extract_fixed_chunks(waveform, sample_rate))
        
        if len(speech_chunks) < 2:
            return False

        all_embeddings = []
        for chunk in speech_chunks:
            embedding = embedding_pipeline({"waveform": chunk, "sample_rate": sample_rate})
            all_embeddings.append(embedding)

        if not all_embeddings:
             return False
        
        embeddings_np = np.vstack(all_embeddings)
        
        if embeddings_np.shape[0] < 2:
            return False
            
        X_normalized = normalize(embeddings_np, norm='l2', axis=1)

        num_clusters = 1
        if cluster_method == 'agglomerative':
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=agglomerative_threshold, 
                metric='cosine', 
                linkage='complete'
            ).fit(X_normalized)
            num_clusters = clustering.n_clusters_
            print(f"    [Agglomerative] Found {num_clusters} cluster(s) with threshold {agglomerative_threshold}.")
        
        elif cluster_method == 'spectral':
            best_n_clusters = 1
            if X_normalized.shape[0] >= spectral_min_clusters:
                best_score = -1
                
                max_clusters_to_test = min(spectral_max_clusters + 1, X_normalized.shape[0])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for k in range(spectral_min_clusters, max_clusters_to_test):
                        try:
                            clustering = SpectralClustering(
                                n_clusters=k, 
                                affinity='cosine', 
                                assign_labels='kmeans', 
                                random_state=0
                            ).fit(X_normalized)
                            
                            if len(set(clustering.labels_)) > 1:
                                score = silhouette_score(X_normalized, clustering.labels_, metric='cosine')
                                if score > best_score:
                                    best_score = score
                                    best_n_clusters = k
                        except Exception:
                            continue
            
            num_clusters = best_n_clusters
            print(f"    [Spectral] Best fit found for {num_clusters} cluster(s).")

        return num_clusters > 1

    except Exception as e:
        import traceback
        print(f"    CLUSTERING ERROR processing {audio_path.name}: {e}")
        traceback.print_exc()
        return True
