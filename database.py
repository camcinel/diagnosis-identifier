import os
import pandas as pd
from utils import df_utils
import load_data
from nlp import DiseaseSearcher
from typing import List


class DiagnosisDatabase:
    def __init__(self, data_dir: str, model_name: str):
        self.searcher = DiseaseSearcher(model_name)
        self.database = self.create_or_load_database(data_dir)
        print('Database Ready')

    def create_database(self, data_dir: str) -> pd.DataFrame:
        data_path = f'{data_dir}/training_20180910/'
        txt_df = load_data.load_txt(data_path)
        txt_df = df_utils.split_txt_df(txt_df)
        txt_df = df_utils.get_primary_diseases(txt_df, self.searcher)
        txt_df = df_utils.get_underlying_factors(txt_df, self.searcher)

        disease_db = df_utils.create_disease_df(txt_df)
        disease_db.to_pickle(f'{data_dir}/disease_db.pkl')
        return disease_db

    def create_or_load_database(self, data_dir: str) -> pd.DataFrame:
        if os.path.isfile(f'{data_dir}/disease_db.pkl'):
            print('Loading saved database')
            return pd.read_pickle(f'{data_dir}/disease_db.pkl')
        print('Creating and saving new database')
        return self.create_database(data_dir)

    def find_factors(self, disease: str) -> List[str]:
        doc = self.searcher.nlp(disease)
        if len(doc.ents) == 0:
            return ['']

        entity = doc.ents[0]
        if len(entity._.kb_ents) == 0:
            return ['']

        disease = self.searcher.linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name

        try:
            return self.database.loc[self.database.disease == disease].factors.values[0]
        except IndexError:
            return ['']
