import os
import pandas as pd


ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_df(path: str) -> pd.DataFrame:
    ext = path.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        return pd.read_csv(path)
    return pd.read_excel(path)


def get_file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
