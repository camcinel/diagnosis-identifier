import pandas as pd
from typing import Tuple, Union, List
from nlp import DiseaseSearcher


def load_text(file_path: str) -> str:
    """
    Extract text from .txt files
    Parameters
    ----------
    file_path: str
        Path to .txt file

    Returns
    -------
    str
        Text within the .txt file

    """
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def contains_category(text: str, category: Tuple[str, ...]) -> Union[str, None]:
    """
    Searches a text for line containing given categories
    Parameters
    ----------
    text: str
        Text to be searched through
    category: Tuple[str, ...]
        Tuple of strings that contains the relevant categories to be found

    Returns
    -------
    str or None
        Returns line after the first line containing one of the categories if one is found
        Otherwise returns None
    """
    # add uppercase version of the categories since some notes have sections in all caps
    category += tuple([x.upper() for x in category])

    # split text into lines
    lines = text.split('\n')

    # search for category in lines, and return the following line
    for index, line in enumerate(lines):
        for cat in category:
            if cat in line:
                return lines[index + 1].strip()
    return None


def get_category(text: str, category: Tuple[str, ...]) -> Union[str, None]:
    """
    Searches a text for text chunks separated by empty lines which start with a given category
    Parameters
    ----------
    text: str
        Text to be searched through
    category: Tuple[str, ...]
        Tuple of strings that contains the categories to be found

    Returns
    -------
    str or None
        Returns the first chunk of text to start with one of the given categories if any are found
        Otherwise returns None
    """
    # add uppercase version of the categories since some notes have sections in all caps
    category += tuple([x.upper() for x in category])

    # split the text into chunks separated by blank lines
    categories = text.split('\n\n')

    # find chunks that start with our category
    category_chunks = [chunk.strip() for chunk in categories if chunk.strip().startswith(category)]

    # if we find chunks, iterate through them and remove their titles
    if category_chunks:
        category_chunk = category_chunks[0]
        for cat in category:
            category_chunk = category_chunk.replace(cat, '')
        category_chunk = category_chunk.strip(':').strip()

    # if we don't find any chunks, return nothing
    else:
        category_chunk = None
    return category_chunk


def find_primary_diagnoses(diagnosis: str) -> str:
    """
    Finds primary diagnoses in block of text
    Parameters
    ----------
    diagnosis: str
        Text pertaining to the diagnoses of a patient

    Returns
    -------
    str
        Subset of text pertaining to primary diagnoses if diagnosis is split into primary and secondary
        Otherwise returns the original diagnosis

    """
    if 'secondary' not in diagnosis.lower():
        return diagnosis.lower().strip()
    primary_diagnosis = diagnosis.lower().split('secondary')[0]
    primary_diagnosis = primary_diagnosis.strip()
    return primary_diagnosis


def find_factors(
        diagnosis: str,
        history: Union[str, None],
        complaint: Union[str, None],
        primary_diseases: List[str],
        searcher: DiseaseSearcher
) -> List[str]:
    """

    Parameters
    ----------
    diagnosis: str
        Text pertaining to the diagnoses of a patient
    history: str
        Text pertaining to the history of the present illness
    complaint: str
        Text pertaining to the chief complaint of the patient
    primary_diseases: List[str]
        List of diseases found in the primary diagnosis
    searcher: DiseaseSearcher
        A NER model linked to a medical database

    Returns
    -------
    List[str]
        List of canonical names of illnesses found in the diagnosis, history, and complaint that are not
        found in the primary diagnosis

    """
    # turn empty sections into empty strings
    if not history:
        history = ''
    if not complaint:
        complaint = ''

    full_context = diagnosis.lower() + ' ' + history.lower() + ' ' + complaint.lower()
    return searcher.get_factors(full_context, primary_diseases)
