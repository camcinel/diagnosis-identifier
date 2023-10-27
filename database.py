import os
import sys
import warnings
import pandas as pd
from utils import df_utils, load_data
from nlp import DiseaseSearcher
from typing import List


class DiagnosisDatabase:
    """
    A database and associated NER model for medical terms

    Attributes:
        searcher: DiseaseSearcher
            A scispacy NER model and UMLS linker
        database: pandas DataFrame
            A DataFrame of diseases and corresponding underlying factors
    """
    def __init__(self, data_dir: str, model_name: str):
        """
        Initializes DiagnosisDatabase with given directory for clinical notes .txt files
        and model name for scispacy NER model

        Parameters
        ----------
        data_dir: str
            The directory where the clinical notes file are located in the subdirectory training_20180910
        model_name: str
            The model name of the scispacy NER model
            For avaible models, see https://allenai.github.io/scispacy/
        """
        self.searcher = DiseaseSearcher(model_name)
        self.database = self.create_or_load_database(data_dir)
        print('Database Ready')

    def create_database(self, data_dir: str) -> pd.DataFrame:
        """
        Creates pandas DataFrame database of diseases and underlying factors
        Also saves the database for future retrieval

        Parameters
        ----------
        data_dir: str
            Directory where clinical notes data is located in the subdirectory training_20180910

        Returns
        -------
        pandas DataFrame
            A DataFrame consisting of columns:
                disease: diseases found in the primary diagnosis of clinical notes
                factors: factors for each disease as found in the clinical notes

        """
        # create data path
        data_path = os.path.join(data_dir, 'training_20180910')

        # load the clinical notes data
        txt_df = load_data.load_txt(data_path)

        # clearn and expand the clinical notes data
        txt_df = df_utils.split_txt_df(txt_df)
        txt_df = df_utils.get_primary_diseases(txt_df, self.searcher)
        txt_df = df_utils.get_underlying_factors(txt_df, self.searcher)

        # create the disease database
        disease_db = df_utils.create_disease_df(txt_df)

        # save the disease database for faster retrieval in the future
        disease_db.to_pickle(os.path.join(data_dir, 'disease_db.pkl'))
        return disease_db

    def create_or_load_database(self, data_dir: str) -> pd.DataFrame:
        """
        Load the saved database if it exists, otherwise create and save one

        Parameters
        ----------
        data_dir: str
            Directory where clinical notes data is located in the subdirectory training_20180910

        Returns
        -------
        pandas DataFrame
            The saved data frame disease_db.pkl if it exists
            otherwise the output of self.create_database

        """

        # check if we can load the file
        if os.path.isfile(os.path.join(data_dir, 'disease_db.pkl')):
            print('Loading saved database')
            return pd.read_pickle(os.path.join(data_dir, 'disease_db.pkl'))

        # file does not exist so create one instead
        print('Creating and saving new database')
        return self.create_database(data_dir)

    def find_factors(self, disease: str) -> List[str]:
        """
        Find the corresponding factors for the given disease in the database

        Parameters
        ----------
        disease: str
            The name of a disease
            Note it does not have to be a canonical name

        Returns
        -------
        List[str]
            The canonical names of the diseases which are factors of the given disease,
            as extracted from the database

        """
        # extract entities
        doc = self.searcher.nlp(disease)

        # if no entities, return nothing
        if len(doc.ents) == 0:
            return []

        # get first entity
        entity = doc.ents[0]

        # if no canonical names of this disease, return nothing
        if len(entity._.kb_ents) == 0:
            return []

        # get the canonical name of the disease
        disease = self.searcher.linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name

        # look for canonical name in database and return factors if it exists
        # otherwise return nothing
        try:
            return self.database.loc[self.database.disease == disease].factors.values[0]
        except IndexError:
            return []


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    if len(sys.argv) <= 1:
        raise KeyError('No disease given')

    db = DiagnosisDatabase('data', 'en_ner_bc5cdr_md')

    diseases = [' '.join(x.split('_')) for x in sys.argv[1:]]
    for disease in diseases:
        print(f'Underlying Factors for {disease}:')
        factors = db.find_factors(disease)

        if not factors:
            print('Disease not found in database')

        for index, factor in enumerate(factors):
            print(f'{index}. {factor}')
