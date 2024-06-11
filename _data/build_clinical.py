import compress_json, pickle, random, json, tiktoken
from datasets import load_dataset
import pandas as pd

random.seed(42)
SIZE = 200
encoding = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens

def build_clinical_notes(target = "behavior_alcohol"):
    data = [ ]
    phenotype = pd.read_csv("_data/MIMIC-SBDH.csv")
    phenotype=phenotype.rename(columns={'row_id': 'ROW_ID'})
    patient = pd.read_csv( "_data/PATIENTS.csv.gz", compression="gzip", low_memory=False )
    note = pd.read_csv( "_data/NOTEEVENTS.csv.gz", compression="gzip", low_memory=False )
    table = phenotype.merge(note.merge(patient,on="SUBJECT_ID").rename(columns={"ROW_ID_x": "ROW_ID"}), on="ROW_ID")
    table['tokens'] = [ num_tokens_from_string(t) for t in table["TEXT"].tolist() ]
    table = table[table["tokens"] <= 2048 ]
    positives = table[ (table[target] == 1) | (table[target] == 2)  ]
    negatives = table[ (table[target] == 0) | (table[target] == 3)  ]

    posF = positives[ positives["sdoh_economics"] == 1 ].to_dict("records")
    negF = negatives[ negatives["sdoh_economics"] == 1 ].to_dict("records")
    posM = positives[ positives["sdoh_economics"] != 1 ].to_dict("records")
    negM = negatives[ negatives["sdoh_economics"] != 1 ].to_dict("records")

    size = min( [ len(posF), len(negF), len(posM), len(negM), SIZE//4 ] )
    random.shuffle(posF)
    random.shuffle(negF)
    random.shuffle(posM)
    random.shuffle(negM)
    print(len(posF), len(negF), len(posM), len(negM), size)
    for ex in posF[:size]+negF[:size]+posM[:size]+negM[:size]:
        group = 1 if ex["sdoh_economics"] == 1 else 0
        target_value = "yes" if ex[target] in [1,2] else "no"
        adj_value = [0,1] if target_value == "yes" else [1,0]
        data += [ { "input": ex["TEXT"], "target": target_value, "group": group, "adjustment": adj_value } ]
    return { "employed": data }

def build_pop_clinical_notes(target = "behavior_alcohol"):

    data = [ ]
    phenotype = pd.read_csv("_data/MIMIC-SBDH.csv")
    phenotype=phenotype.rename(columns={'row_id': 'ROW_ID'})
    patient = pd.read_csv( "_data/PATIENTS.csv.gz", compression="gzip", low_memory=False )
    note = pd.read_csv( "_data/NOTEEVENTS.csv.gz", compression="gzip", low_memory=False )
    table = phenotype.merge(note.merge(patient,on="SUBJECT_ID").rename(columns={"ROW_ID_x": "ROW_ID"}), on="ROW_ID")
    table['tokens'] = [ num_tokens_from_string(t) for t in table["TEXT"].tolist() ]
    table = table[table["tokens"] <= 2048 ]
    positives = table[ (table[target] == 1) | (table[target] == 2)  ]
    negatives = table[ (table[target] == 0) | (table[target] == 3)  ]

    posF = positives[ positives["sdoh_economics"] == 1 ].to_dict("records")
    negF = negatives[ negatives["sdoh_economics"] == 1 ].to_dict("records")
    posM = positives[ positives["sdoh_economics"] == 0 ].to_dict("records")
    negM = negatives[ negatives["sdoh_economics"] == 0 ].to_dict("records")

    _data = posF + negF + posM + negM
    random.shuffle(_data)

    for ex in _data[:SIZE]:
        group = 1 if ex["sdoh_education"] == 1 else 0
        target_value = "yes" if ex[target] in [1,2] else "no"
        adj_value = [0,1] if target_value == "yes" else [1,0]
        data += [ { "input": ex["TEXT"], "target": target_value, "group": group, "adjustment": adj_value } ]
    return data

if __name__ == "__main__":
    
    data = compress_json.load("_data/data.json")

    data["clinical_notes"] = build_clinical_notes()
    data["clinical_notes"]["population"] = build_pop_clinical_notes()

    compress_json.dump(data, "_data/data.json")