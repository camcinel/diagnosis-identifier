# Diagnosis Database and Detector

This repository contains methods to create databases of primary diagnoses and underlying factors,
determine possible relevant factors from a given disease, and extract primary diagnoses with relevant
factors from a clinical note

## Installation

In order to install this module, `cd` into the root directory and run
```commandline
pip install -r requirements.txt
```

Additionally, the `en_ner_bc5cdr_md` must be installed by running
```commandline
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
```

## Usage

### Underlying Factor Extraction

In order to get underlying factors for a disease found in the database, the following should be run
```commandline
python database.py disease_name
```

Multiple diseases can be looked up at the same time, via the syntax
```commandline
python database.py disease_name_1 disease_name_2 ... disease_name_n
```

#### Example
Running the command
```commandline
python database.py pulmonary_embolism
```
gives the output
```commandline
Underlying Factors for pulmonary embolism:
0. Ankle edema (finding)
1. Pain
2. Dyspnea
3. Edema
4. Exanthema
5. Hemoptysis
6. Syncope
7. Hyperlipidemia
8. Tachypnea
9. Palpitations
10. Ischemia
...
```

### Diagnosis and Evidence Extraction

In order to extract the primary diagnoses and relevant factors from a clinical note,
the clinical note should first be saved as .txt file.
Then the following command can be run
```commandline
python main.py /path/to/clinical/note.txt
```

Multiple notes can be extracted from at the same time via the syntax
```commandline
python main.py /path/to/clinical/note_1.txt /path/to/clinical/note_2.txt ... /path/to/clinical/note_n.txt
```

#### Example
Running the command
```commandline
python main.py data/training_20180910/100187.txt
```

gives the output
```commandline
Diagnosis for data/training_20180910/100187.txt
Diagnoses:
1. Hematoma
        1.1. Chronic Obstructive Airway Disease
        1.2. Coughing
        1.3. Abdominal Pain
2. Pulmonary Embolism
        2.1. Abdominal Pain
        2.2. Chronic Obstructive Airway Disease
        2.3. Coughing
3. Ear Inflammation
4. Bacteremia
        4.1. Abdominal Pain
        4.2. Chronic Obstructive Airway Disease
        4.3. Coughing
...
```