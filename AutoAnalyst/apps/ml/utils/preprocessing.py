import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def detect_problem_type(series: pd.Series) -> str:
    if not is_numeric(series):
        return 'classification'
    if series.nunique() <= 20:
        return 'classification'
    return 'regression'


def clean_df(df: pd.DataFrame, target_col: str):
    rows_before = len(df)
    df = df.dropna(subset=[target_col])

    ID_LIKE = {'id', 'index', 'row', 'no', 'num', '#', 'uuid',
               'email', 'phone', 'name', 'rowid'}

    cols_to_drop = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isna().mean() > 0.80:
            cols_to_drop.append(col)
            continue
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
            continue
        if col.lower() in ID_LIKE:
            cols_to_drop.append(col)
            continue
        if not is_numeric(df[col]) and df[col].nunique() / max(len(df), 1) > 0.90:
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    rows_after = len(df)
    return df.copy(), rows_before, rows_after, cols_to_drop


def preprocess(df: pd.DataFrame, target_col: str):
    encoders = {}
    label_mappings = {}
    imputation_log = []

    for col in df.columns:
        if df[col].isna().any():
            n_missing = int(df[col].isna().sum())
            if is_numeric(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                imputation_log.append({
                    'column': col, 'missing': n_missing,
                    'strategy': 'Median imputation',
                    'fill_value': round(float(fill_val), 4),
                })
            else:
                mode = df[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill)
                imputation_log.append({
                    'column': col, 'missing': n_missing,
                    'strategy': 'Mode imputation',
                    'fill_value': str(fill),
                })

    for col in df.columns:
        if not is_numeric(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            label_mappings[col] = list(le.classes_)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df.drop(columns=[target_col]).astype(float)
    y = df[target_col]
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    return (X_train, X_test, y_train, y_test,
            scaler, encoders, label_mappings, feature_names, imputation_log)
