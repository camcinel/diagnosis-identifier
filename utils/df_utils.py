import pandas as pd
from utils import text_utils
from nlp import DiseaseSearcher


def fix_slashes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces backward slashes in file names in a pandas DataFrame with forward slashes

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame containing a column of file names

    Returns
    -------
    pandas DataFrame
        A copy of the original DataFrame where the file names have had all \ replaced by /

    """
    new_df = df.copy()
    new_df['file_idx'] = df.file_idx.str.split('\\').str.join('/')
    return new_df


def clean_ent_df(ent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning on ent_df
        Strips entity name text
        Removes semicolons from end indices
        Converts start and end indices to integers

    Parameters
    ----------
    ent_df: pandas DataFrame
        The entity dataframe output by utils.load_data.load_ann

    Returns
    -------
    pandas DataFrame
        A copy of ent_df with the above cleaning procedures done

    """
    new_df = ent_df.copy()
    new_df['text'] = new_df.text.str.strip()
    new_df['end_idx'] = new_df.end_idx.str.split(';').str[-1]
    for pos in ['start', 'end']:
        new_df[f'{pos}_idx'] = new_df[f'{pos}_idx'].astype(int)
    return new_df


def clean_rel_df(rel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning procedures for rel_df
        Strips newline characters from entity2
        Removes Arg1 and Arg2 from entity1 and entity2, respectively, just leaving entity identifiers

    Parameters
    ----------
    rel_df: pandas DataFrame
        The relation dataframe output by utils.load_data.load_ann

    Returns
    -------
    pandas DataFrame
        A copy of rel_df with the above cleaning procedures done

    """
    new_df = rel_df.copy()
    new_df['entity2'] = rel_df.entity2.str.strip()
    for i in ['1', '2']:
        new_df[f'entity{i}'] = new_df[f'entity{i}'].str.split(':').str[-1]
    return new_df


def get_diagnosis(txt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Procedure to create a new column in a pandas DataFrame of clinical notes
    of the section of the note that corresponds to the discharge diagnoses

    Parameters
    ----------
    txt_df: pandas DataFrame
        A DataFrame which includes a column "text" of clinical data notes as strings

    Returns
    -------
    pandas DataFrame
        A copy of txt_df with an additional column of "diagnosis" which corresponds
        to the section of the clinical note that has to do with the discharge diagnoses

    """
    new_df = txt_df.copy()
    new_df['diagnosis'] = txt_df.text.map(lambda x: text_utils.get_category(x, ('Discharge Diagnosis',
                                                                                'Discharge Diagnoses',
                                                                                'Final Discharge Diagnosis',
                                                                                'Final Discharge Diagnoses')))
    return new_df.dropna()


def get_primary_diagnosis(txt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Procedure to create a new column in a pandas DataFrame of primary diagnoses

    Parameters
    ----------
    txt_df: pandas DataFrame
        A DataFrame which includes a column "diagnosis" which corresponds to the
        diagnosis section of a clinical note

    Returns
    -------
    pandas DataFrame
        A copy of txt_df with an additional column "primary_diagnosis" which is
        the section of the diagnosis which is marked as primary, if it exists,
        otherwise it is the entire original diagnosis

    """
    new_df = txt_df.copy()
    new_df['primary_diagnosis'] = new_df.diagnosis.map(lambda x: text_utils.find_primary_diagnoses(x))
    return new_df


def get_history(txt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Procedure to create a new column in a pandas DataFrame of clinical notes
    of the section of the note that corresponds to the history of present illness

    Parameters
    ----------
    txt_df: pandas DataFrame
        A DataFrame which includes a column "text" of clinical data notes as strings

    Returns
    -------
    pandas DataFrame
        A copy of txt_df with an additional column of "history" which corresponds
        to the section of the clinical note that has to do with the history of present illness

    """
    new_df = txt_df.copy()
    new_df['history'] = new_df.text.map(lambda x: text_utils.get_category(x, ('History of Present Illness',)))
    return new_df


def get_complaint(txt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Procedure to create a new column in a pandas DataFrame of clinical notes
    of the section of the note that corresponds to the chief complaint

    Parameters
    ----------
    txt_df: pandas DataFrame
        A DataFrame which includes a column "text" of clinical data notes as strings

    Returns
    -------
    pandas DataFrame
        A copy of txt_df with an additional column of "complaint" which corresponds
        to the section of the clinical note that has to do with the chief complaint

    """
    new_df = txt_df.copy()
    new_df['complaint'] = new_df.text.map(lambda x: text_utils.contains_category(x, ('Chief Complaint',)))
    return new_df


def split_txt_df(txt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs the methods get_diagnosis, get_primary_diagnosis, get_history, and get_complaint
    in order on the same DataFrame

    Parameters
    ----------
    txt_df: pandas DataFrame
        The clinical notes DataFrame as output by utils.load_data.load_txt

    Returns
    -------
    pandas DataFrame
        A copy of txt_df with the columns "diagnosis", "primary_diagnosis", "history",
        and "complaint" added by the above procedures

    """
    new_df = txt_df.copy()
    categories = ['diagnosis', 'primary_diagnosis', 'history', 'complaint']
    for category in categories:
        new_df = globals()[f'get_{category}'](new_df)
    return new_df


def get_primary_diseases(txt_df: pd.DataFrame, searcher: DiseaseSearcher) -> pd.DataFrame:
    """
    Procedure to create a new column in a pandas DataFrame which consists of List[str]
    that are the canonical names of diseases found in the "primary_diagnosis" column

    Parameters
    ----------
    txt_df: pandas DataFrame
        A DataFrame of clinical notes as output by split_txt_df
    searcher: DiseaseSearcher
        A NER model linked to a medical knowledge database

    Returns
    -------
    pandas DataFrame
        A copy of txt_df with the additional column "primary_diseases" as described above

    """
    new_df = txt_df.copy()
    new_df['primary_diseases'] = new_df.primary_diagnosis.map(lambda x: searcher.get_diseases(x))
    return new_df


def get_underlying_factors(txt_df: pd.DataFrame, searcher: DiseaseSearcher) -> pd.DataFrame:
    """
    Procedure to create a new column in a pandas DataFrame which consists of List[str]
    that are the canonical names of diseases found in the diagnosis, chief complaint, and
    history of primary illness that are not one of the primary diagnoses

    Parameters
    ----------
    txt_df: pandas DataFrame
        A DataFrame of clinical notes as output by get_primary_disease
    searcher: DiseaseSearcher
        A NER model linked to a medical knowledge database

    Returns
    -------
    pandas DataFrame
        A copy of txt_df with the additional column of "underlying_factors" as described above

    """
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
    """
    Procedure to create a new DataFrame that consists of diseases
    and their corresponding underlying factors as found in a
    DataFrame of clinical notes

    Parameters
    ----------
    txt_df: pandas DataFrame
        A DataFrame of clinical notes as output by get_underlying_factors

    Returns
    -------
    pandas DataFrame
        A DataFrame with two columns:
            "disease": the name of a disease found in the "primary_diseases" column of txt_df
            "factors": the underlying factors as found in the "underlying_factors" column of
                       txt_df for the given disease

    """
    # create empty dictionary to store diseases and their factors
    disease_dict = {}

    # iterate through row of the DataFrame
    for index, row in txt_df[['primary_diseases', 'underlying_factors']].iterrows():
        diseases, factors = row.primary_diseases, row.underlying_factors

        # iterate through diseases in the row
        for disease in diseases:
            if disease not in disease_dict:
                # add set to remove duplicates
                disease_dict[disease] = set(factors)
            else:
                # union sets to prevent adding duplicates
                disease_dict[disease] = disease_dict[disease].union(set(factors))

    # convert sets back into lists
    disease_dict = {key: list(value) for key, value in disease_dict.items()}

    # create pandas DataFrame and return it
    return pd.DataFrame({
        'disease': disease_dict.keys(),
        'factors': disease_dict.values()
    })
