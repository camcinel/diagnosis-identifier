from database import DiagnosisDatabase
from utils import text_utils
import sys
import os
import warnings
from typing import Tuple, List


class DiagnosisDetector:
    def __init__(self, database: DiagnosisDatabase):
        self.db = database

    def give_diagnosis(self, text: str, print_out: bool = False) -> dict[str: str]:
        primary_disease, factors = self.extract_diseases_and_factors(text)
        return self.get_diagnosis_and_factors(primary_disease, factors, print_out)

    def extract_diseases_and_factors(self, text: str) -> Tuple[List[str], List[str]]:
        diagnosis, history, complaint = self.find_contexts(text)
        primary_diagnosis = text_utils.find_primary_diagnoses(diagnosis)
        primary_diseases = self.db.searcher.get_diseases(primary_diagnosis)
        factors = text_utils.find_factors(diagnosis, history, complaint, primary_diseases, self.db.searcher)
        return primary_diseases, factors

    def get_diagnosis_and_factors(self, primary_diseases: List[str], factors: List[str], print_out: bool = False) -> dict[str: str]:
        disease_factor_dict = {}
        for disease in primary_diseases:
            underlying_factors = self.db.find_factors(disease)
            relevant_factors = [factor for factor in underlying_factors if factor in factors]
            disease_factor_dict[disease] = relevant_factors

        if print_out:
            self.pretty_print(disease_factor_dict)
        return disease_factor_dict

    @staticmethod
    def find_contexts(text: str) -> Tuple[str, str, str]:
        diagnosis = text_utils.get_category(text, ('Discharge Diagnosis',
                                                   'Discharge Diagnoses',
                                                   'Final Discharge Diagnosis',
                                                   'Final Discharge Diagnoses'))
        complaint = text_utils.contains_category(text, ('Chief Complaint',))
        history = text_utils.get_category(text, ('History of Present Illness',))

        return diagnosis, history, complaint

    @staticmethod
    def strip_contexts(contexts: Tuple[str, str, str, str]) -> Tuple[str, str, str, str]:
        contexts = [context.replace('\n', ' ').strip() if context else '' for context in contexts]
        return tuple(contexts)

    @staticmethod
    def pretty_print(diagnosis_factor_dict: dict[str: str]) -> None:
        if not diagnosis_factor_dict:
            print('No diagnoses found in database')
            return

        print('Diagnoses:')
        for index, (primary_diagnosis, factors) in enumerate(diagnosis_factor_dict.items()):
            print(f'{index+1}. {primary_diagnosis}')
            for sub_index, factor in enumerate(factors):
                print(f'\t{index+1}.{sub_index+1}. {factor}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    if len(sys.argv) <= 1:
        raise FileNotFoundError('No File Path Given')
    file_paths = sys.argv[1:]

    db = DiagnosisDatabase('data', 'en_ner_bc5cdr_md')
    detector = DiagnosisDetector(db)
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f'No such file {file_path}')
            continue

        print(f'Diagnosis for {file_path}')
        text = text_utils.load_text(file_path)
        try:
            detector.give_diagnosis(text, print_out=True)
        except AttributeError:
            print('No diagnoses found')
        print()

