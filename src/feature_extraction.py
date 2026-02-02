import librosa
import numpy as np

def extract_features(file_path, n_mfcc=13):
    """
    Extract MFCC and spectral features from audio file.
    
    Args:
        file_path (str): Path to audio file
        n_mfcc (int): Number of MFCC coefficients
    
    Returns:
        np.array: Feature vector
    """
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features = np.hstack([mfccs_mean, spectral_centroid])
    return features
