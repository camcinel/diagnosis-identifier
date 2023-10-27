import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking import EntityLinker
from typing import List


class DiseaseSearcher:
    def __init__(self, model_name: str):
        print('Loading NLP model')
        self.nlp = spacy.load(model_name)
        self.nlp.add_pipe("abbreviation_detector")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = self.nlp.get_pipe('scispacy_linker')
        print('NLP model loaded')

    def get_diseases(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = [x for x in doc.ents if x.label_ == 'DISEASE']
        diseases = [
            self.linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name for entity in entities if len(entity._.kb_ents) > 0
        ]
        return list(set(diseases))

    def get_factors(self, text: str, primary_diseases: List[str]) -> List[str]:
        doc = self.nlp(text)
        entities = [x for x in doc.ents if x.label_ == 'DISEASE' and x.text not in primary_diseases]
        diseases = [
            self.linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name for entity in entities if len(entity._.kb_ents) > 0
        ]
        factors = [disease for disease in diseases if disease not in primary_diseases]
        return list(set(factors))
