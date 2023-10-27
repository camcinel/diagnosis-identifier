import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking import EntityLinker
from typing import List


class DiseaseSearcher:
    """
    A NER model linked to a medical knowledge database in order to identify and given canonical names to diseases

    Attributes:
        nlp: A scispacy NER model
        linker: scispacy ULMS linker
    """
    def __init__(self, model_name: str):
        """
        Initializes DiseaseSearcher with the given NER model

        Parameters
        ----------
        model_name: str
            The name of one of scispacy's NER models
            For available models, see https://allenai.github.io/scispacy/
        """
        print('Loading NLP model')

        # create initial model
        self.nlp = spacy.load(model_name)

        # add abbreviation detector
        self.nlp.add_pipe("abbreviation_detector")

        # add UMLS linker
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

        # split off the linker
        self.linker = self.nlp.get_pipe('scispacy_linker')
        print('NLP model loaded')

    def get_diseases(self, text: str) -> List[str]:
        """
        Extract diseases and find canonical names from a string

        Parameters
        ----------
        text: str
            A subsection of a clinical note

        Returns
        -------
        List[str]
            A list of the canonical names of diseases found in the text

        """
        # extract entities
        doc = self.nlp(text)

        # filter for those that are diseases
        entities = [x for x in doc.ents if x.label_ == 'DISEASE']

        # convert each disease to its canonical name
        diseases = [
            self.linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name for entity in entities if len(entity._.kb_ents) > 0
        ]

        # remove duplicates and return
        return list(set(diseases))

    def get_factors(self, text: str, primary_diseases: List[str]) -> List[str]:
        """
        Extract diseases and find canonical names from a string,
        discarding those already in primary_diseases

        Parameters
        ----------
        text: str
            A subsection of a clinical note
        primary_diseases:
            A list of canonical names of diseases as output in self.get_diseases

        Returns
        -------
        List[str]
            A list of the canonical names of diseases found in the text,
            with those found in primary_diseases removed

        """
        # extract entities
        doc = self.nlp(text)

        # filter for those that are diseases
        entities = [x for x in doc.ents if x.label_ == 'DISEASE']

        # convert each disease to its canonical name
        diseases = [
            self.linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name for entity in entities if len(entity._.kb_ents) > 0
        ]

        # filter for those already in primary_diseases
        factors = [disease for disease in diseases if disease not in primary_diseases]

        # remove duplicates and return
        return list(set(factors))
