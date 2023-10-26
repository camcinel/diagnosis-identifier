import pandas as pd
import text_utils


def fix_slashes(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df['file_idx'] = df.file_idx.str.split('\\').str.join('/')
    return new_df


def clean_ent_df(ent_df: pd.DataFrame) -> pd.DataFrame:
    new_df = ent_df.copy()
    new_df['text'] = new_df.text.str.strip()
    new_df['end_idx'] = new_df.end_idx.str.split(';').str[-1]
    for pos in ['start', 'end']:
        new_df[f'{pos}_idx'] = new_df[f'{pos}_idx'].astype(int)
    return new_df


def clean_rel_df(rel_df: pd.DataFrame) -> pd.DataFrame:
    new_df = rel_df.copy()
    new_df['entity2'] = rel_df.entity2.str.strip()
    for i in ['1', '2']:
        new_df[f'entity{i}'] = new_df[f'entity{i}'].str.split(':').str[-1]
    return new_df


def get_diagnoses(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['diagnosis'] = txt_df.text.map(lambda x: text_utils.get_category(x, ('Discharge Diagnosis',
                                                                                'Discharge Diagnoses',
                                                                                'Final Discharge Diagnosis',
                                                                                'Final Discharge Diagnoses')))
    return new_df.dropna()


def get_history(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['history'] = txt_df.text.map(lambda x: text_utils.get_category(x, ('History of Present Illness',)))
    return new_df


def get_complaint(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['complaint'] = txt_df.text.map(lambda x: text_utils.contains_category(x, ('Chief Complaint',)))
    return new_df


def split_txt_df(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    for category in ['diagnoses', 'history', 'complaint']:
        new_df = globals()[f'get_{category}'](new_df)
    return new_df
