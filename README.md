# AutoAnalyst AI

A full-stack Django AutoML platform that automatically analyzes datasets, trains machine learning models, and provides explainable AI insights.

## Overview

Users can:
1. Upload a CSV or XLSX dataset
2. View a data profile (missing values, statistics, duplicates)
3. Select a target column to predict
4. Automatically detect problem type (classification or regression)
5. Train multiple ML models (Logistic Regression, Random Forest, KNN, SVM for classification; Linear/Ridge/RF Regressor for regression)
6. Compare model performance and view the best model
7. Make predictions using the best model
8. View feature importance and download the trained model

## Tech Stack

- **Backend**: Django (Python 3.11)
- **ML**: scikit-learn, pandas, numpy, joblib
- **Frontend**: Pure HTML/CSS/JS (Inter/Poppins fonts, black & white SaaS design)
- **Database**: SQLite
- **Static files**: WhiteNoise

## Structure

```
AutoAnalyst/
├── manage.py
├── AutoAnalyst/        # Django config (settings, urls, wsgi)
├── apps/
│   ├── core/           # Home page, auth (signup)
│   ├── dataset/        # Upload, preview, profiling, target selection
│   ├── ml/             # Training, evaluation, results
│   └── predict/        # Predictions, model download
├── templates/          # HTML templates
├── static/             # CSS + JS
├── media/              # Uploaded datasets
└── saved_models/       # Trained model + metadata
```

## Running

```bash
cd AutoAnalyst && python manage.py runserver 0.0.0.0:5000
```

## User preferences

- Black & white minimal SaaS UI design
- Pure HTML/CSS/JS only (no frontend frameworks)
- No authentication required for core features (auth is optional)
