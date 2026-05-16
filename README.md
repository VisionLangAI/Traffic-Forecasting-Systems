
# MSA-GCRNN for Smart Mobility Traffic Forecasting

This repository contains the implementation of the proposed **Multi-Step Attention-based Graph Convolutional Recurrent Neural Network (MSA-GCRNN)** framework for intelligent urban traffic forecasting using cyber-physical and social network features.

## Overview

The proposed framework integrates:

- Graph Convolutional Learning
- Recurrent Sequential Modeling
- Multi-Head Attention Mechanism
- Multimodal Cyber-Physical and Social Feature Fusion
- SHAP and LIME Explainable AI Analysis
- Statistical Validation and Cross-Validation

The model is designed for robust multi-step traffic congestion prediction in smart mobility environments.

---

## Dataset

Dataset used in this study:

**Smart Mobility and Traffic Optimization Dataset**  
Source: Kaggle

The dataset contains:

- Vehicle count
- Traffic speed
- Road occupancy
- Weather conditions
- Accident reports
- Ride-sharing demand
- Sentiment scores
- Parking availability
- Emission levels
- Energy consumption
- Geographic coordinates

Target classes:

- Low Congestion
- Medium Congestion
- High Congestion

---

## Preprocessing Pipeline

The preprocessing workflow includes:

- Missing value handling using KNN Imputation
- Categorical encoding
- Min-Max normalization
- Temporal sequence generation
- Graph adjacency matrix construction
- Temporal train-validation-test split
- Cross-validation analysis

---

## Proposed Framework

The MSA-GCRNN architecture contains:

- Graph Convolutional Recurrent Layers
- Multi-Head Attention Blocks
- Residual Gating Mechanism
- Dropout Regularization
- Softmax Prediction Layer

---

## Baseline Models

The following models are implemented for fair comparative evaluation:

- SVM
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LSTM
- Bi-LSTM
- GCN-LSTM

---

## Evaluation Metrics

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-Validation Accuracy
- Confidence Interval Analysis
- Statistical Significance Testing

---

## Explainability Analysis

Interpretability validation includes:

- SHAP Feature Attribution
- LIME Local Explanation Analysis
- Explanation Stability
- Fidelity Analysis
- Perturbation Robustness

---

## Statistical Validation

The framework includes:

- Paired t-test
- Wilcoxon Signed-Rank Test
- Friedman Test
- Confidence Interval Analysis
- Bootstrap Validation

---

## Repository Contents

- `MSA_GCRNN_Full_Framework.py`
- Preprocessing pipeline
- Graph construction
- Baseline model evaluation
- Proposed model implementation
- SHAP and LIME analysis
- Statistical testing
- Visualization generation

---

## Reproducibility

The implementation includes:

- Fixed random seeds
- Explicit hyperparameter settings
- Temporal validation strategy
- Cross-validation setup
- Confidence interval computation

---

## Requirements

Main dependencies:

```bash
numpy
pandas
scikit-learn
xgboost
torch
matplotlib
seaborn
networkx
shap
lime
scipy
```

---

## Citation

If you use this work, please cite the corresponding manuscript related to the MSA-GCRNN framework for intelligent traffic forecasting.


Multi-Step Attention-based Graph Convolutional Recurrent Neural Network with Explainable AI for Urban Traffic Forecasting Using Cyber-Physical and Social Network Features 


Multi-Step Attention-based Graph Convolutional Recurrent Neural Network with Explainable AI for Urban Traffic Forecasting Using Cyber-Physical and Social Network Features 
