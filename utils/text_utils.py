import pandas as pd
from typing import Tuple, Union, List
from nlp import DiseaseSearcher


def load_text(file_path: str) -> str:
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def contains_category(text: str, category: Tuple[str, ...]) -> Union[str, None]:
    category += tuple([x.upper() for x in category])

    lines = text.split('\n')
    for index, line in enumerate(lines):
        for cat in category:
            if cat in line:
                return lines[index + 1].strip()
    return None


def get_category(text: str, category: Tuple[str, ...]) -> Union[str, None]:
    category += tuple([x.upper() for x in category])

    categories = text.split('\n\n')
    category_chunks = [chunk.strip() for chunk in categories if chunk.strip().startswith(category)]
    if category_chunks:
        category_chunk = category_chunks[0]
        for cat in category:
            category_chunk = category_chunk.replace(cat, '')
        category_chunk = category_chunk.strip(':').strip()
    else:
        category_chunk = None
    return category_chunk


def find_primary_diagnoses(diagnosis: str) -> str:
    if 'secondary' not in diagnosis.lower():
        return diagnosis.lower().strip()
    primary_diagnosis = diagnosis.lower().split('secondary')[0]
    primary_diagnosis = primary_diagnosis.strip()
    return primary_diagnosis


def find_factors(
        diagnosis: str,
        history: str,
        complaint: str,
        primary_diseases: List[str],
        searcher: DiseaseSearcher
) -> List[str]:
    if not history:
        history = ''
    if not complaint:
        complaint = ''

    full_context = diagnosis.lower() + ' ' + history.lower() + ' ' + complaint.lower()
    return searcher.get_factors(full_context, primary_diseases)
