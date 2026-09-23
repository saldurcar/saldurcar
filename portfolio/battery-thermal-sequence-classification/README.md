# Battery Thermal Sequence Classification — GRU & LSTM

Public portfolio version of recurrent deep-learning workflows developed for temporal classification of lithium-ion battery thermal stages.

## Problem

Battery experiments generate synchronized time-series signals from thermal images and electrical/temperature sensors. A single frame may not contain enough context to identify the current thermal stage, so recurrent models are used to learn from a recent temporal window.

## Workflow

1. Read synchronized experimental features.
2. Remove invalid/calibration samples.
3. Build thermal and sensor-derived features.
4. Create **30-step temporal windows**.
5. Fit `StandardScaler` using training data only.
6. Train GRU and LSTM classifiers.
7. Evaluate with accuracy, precision, recall, F1-score and confusion matrices.

## Representative architectures

### GRU
```
GRU(32)
Dropout(0.30)
Dense(16, ReLU)
Dropout(0.20)
Dense(n_classes, Softmax)
```

### LSTM
```
LSTM(32)
Dropout(0.30)
Dense(16, ReLU)
Dropout(0.20)
Dense(n_classes, Softmax)
```

## Notes

The full doctoral dataset and experiment-specific files are not distributed. The Python files in this folder are a cleaned public portfolio implementation reflecting the modeling workflow used in the research.

## Skills demonstrated

Python · Pandas · NumPy · scikit-learn · TensorFlow/Keras · GRU · LSTM · time-series analysis · feature scaling · model evaluation
