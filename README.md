# AI Voice Detection Tool

## Overview
This project focuses on detecting AI-generated speech versus human speech using
machine learning and audio signal processing techniques. The system is designed
for applications in cybersecurity, fraud prevention, and audio authentication.

This project received the DIPEX 2025 – WATI (Women and Technological Innovation)
State-Level Award and is associated with a research paper accepted in an STM Journal.

## Problem Statement
With the rise of AI-generated voice cloning, there is an increasing risk of
impersonation, fraud, and misinformation. This project aims to build a reliable
system to distinguish between AI-generated and genuine human speech.

## Dataset
- Audio samples consisting of human speech and AI-generated speech
- Preprocessed into uniform sampling rates and durations
- Feature extraction performed on audio signals

## Methodology
1. Audio preprocessing and noise handling
2. Feature extraction using:
   - MFCCs
   - Spectral features
   - Pitch and temporal characteristics
3. Model training using supervised machine learning
4. Performance evaluation using precision, recall, and F1-score

## Results
The trained model demonstrated effective differentiation between AI-generated
and human voices with strong classification performance across evaluation metrics.

## Technologies Used
- Python
- Librosa
- NumPy, Pandas
- Scikit-learn
- Jupyter Notebook

## Research & Recognition
- Research paper accepted in Journal of Instrumentation Technology & Innovations (2026)
- Presented at SIESM 2026, Sanjivani University
- Awarded prize for innovation and technical contribution
- DIPEX 2025 – WATI State-Level Winner

## Future Work
- Extend model to multilingual datasets
- Explore deep learning architectures
- Improve real-time inference capabilities
