import pandas as pd
import text_utils
from nlp import DiseaseSearcher
from typing import List


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


def get_diagnosis(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['diagnosis'] = txt_df.text.map(lambda x: text_utils.get_category(x, ('Discharge Diagnosis',
                                                                                'Discharge Diagnoses',
                                                                                'Final Discharge Diagnosis',
                                                                                'Final Discharge Diagnoses')))
    return new_df.dropna()


def get_primary_diagnosis(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['primary_diagnosis'] = new_df.diagnosis.map(lambda x: text_utils.find_primary_diagnoses(x))
    return new_df


def get_history(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['history'] = new_df.text.map(lambda x: text_utils.get_category(x, ('History of Present Illness',)))
    return new_df


def get_complaint(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['complaint'] = new_df.text.map(lambda x: text_utils.contains_category(x, ('Chief Complaint',)))
    return new_df


def split_txt_df(txt_df: pd.DataFrame) -> pd.DataFrame:
    new_df = txt_df.copy()
    categories = ['diagnosis', 'primary_diagnosis', 'history', 'complaint']
    for category in categories:
        new_df = globals()[f'get_{category}'](new_df)
    new_df[categories] = new_df[categories].replace('\n', ' ', regex=True)
    return new_df


def get_primary_diseases(txt_df: pd.DataFrame, searcher: DiseaseSearcher) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['primary_diseases'] = new_df.primary_diagnosis.map(lambda x: searcher.get_diseases(x))
    return new_df


def get_underlying_factors(txt_df: pd.DataFrame, searcher: DiseaseSearcher) -> pd.DataFrame:
    new_df = txt_df.copy()
    new_df['underlying_factors'] = new_df \
        .apply(lambda x: text_utils.find_factors(
            x['primary_diagnosis'],
            x['history'],
            x['complaint'],
            x['primary_diseases'],
            searcher
        ),
        axis=1
    )
    return new_df


def create_disease_df(txt_df: pd.DataFrame) -> pd.DataFrame:
    disease_dict = {}
    for index, row in txt_df[['primary_diseases', 'underlying_factors']].iterrows():
        diseases, factors = row.primary_diseases, row.underlying_factors
        for disease in diseases:
            if disease not in disease_dict:
                disease_dict[disease] = set(factors)
            else:
                disease_dict[disease] = disease_dict[disease].union(set(factors))
    disease_dict = {key: list(value) for key, value in disease_dict.items()}
    return pd.DataFrame({
        'disease': disease_dict.keys(),
        'factors': disease_dict.values()
    })
