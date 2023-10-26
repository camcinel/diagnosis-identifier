import pandas as pd
from typing import Tuple


def contains_category(text: str, category: Tuple[str, ...]) -> str | None:
    category += tuple([x.upper() for x in category])

    lines = text.split('\n')
    for index, line in enumerate(lines):
        for cat in category:
            if cat in line:
                return lines[index + 1].strip()
    return None


def get_category(text: str, category: Tuple[str, ...]) -> str | None:
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
    if diagnosis.lower().startswith('primary'):
        primary_diagnosis = diagnosis.split('\n')[1].strip('0123456789.- )')
    else:
        primary_diagnosis = diagnosis.split('\n')[0].strip('0123456789.- )')

    return primary_diagnosis.split(',')[0]


def find_factors(
        file_idx: str,
        ent_df: pd.DataFrame,
        diagnosis: str,
        primary_diagnosis: str,
        history: str,
        complaint: str
) -> str | None:
    relevant_entities = set()

    entities = ent_df[(ent_df.category == 'Reason') & (ent_df.file_idx == file_idx)]

    if not history:
        history = ''
    if not complaint:
        complaint = ''

    if len(entities) == 0:
        return None

    for entity in entities.text:
        if entity.lower() in primary_diagnosis.lower():
            continue
        elif entity in diagnosis.lower() + ' ' + history.lower() + ' ' + complaint.lower():
            relevant_entities.add(entity)

    if len(relevant_entities) == 0:
        return None

    return ', '.join(relevant_entities)
