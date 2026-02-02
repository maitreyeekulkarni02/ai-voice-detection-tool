import numpy as np
from src.feature_extraction import extract_features

def predict(model, file_path):
    """
    Predict if audio is AI-generated or Human.
    
    Args:
        model: Trained classifier
        file_path: Path to audio file
    
    Returns:
        str: "AI" or "Human"
    """
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    pred = model.predict(features)[0]
    return "AI" if pred == 1 else "Human"
