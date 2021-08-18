
# reticulate::use_virtualenv(here::here("nlp_env"),TRUE)

import pandas as pd
import spacy
import json

articles_df = pd.read_csv("data/raw/swissdox/210809_request/Angst1.tsv", sep='\t', encoding = 'utf-8')
articles_df

# doccano expects in json:
# {"text": "Terrible customer service.", "labels": ["negative"], "meta": {"wikiPageID": 1}}
# {"text": "Really great transaction.", "labels": ["positive"], "meta": {"wikiPageID": 2}}
# {"text": "Great price.", "labels": ["positive"], "meta": {"wikiPageID": 3}}

# header classification prep for binary sdg related classification

# turn into dict
articles_dict_list = articles_df.to_dict("records")

# create data format in dict concatenation

def make_doccano_dict_from_dict_list(input_dict_list: list, text_key: str, labels_key: list = None, meta_keys: list = None) -> dict:
  # output = []
  # for dic in input_dict_list:
  #   doc = {}
  #   doc["text"] = dic.get(text_key)
  #   if labels_key is not None:
  #     doc["labels"] = [dic[key] for key in labels_key]
  #   if labels_key is None:
  #     doc["labels"] = [""]
  #   if meta_keys is not None:
  #     doc["meta"] = {key:dic[key] for key in meta_keys}
  #   output.append(doc)
  output = [{
    "text":dic.get(text_key),
    # https://stackoverflow.com/questions/44807107/how-to-check-if-object-is-not-none-within-a-list-comprehension
    "labels":[dic[key] for key in (labels_key or [])],
    "meta":{key:dic[key] for key in (meta_keys or {})}
    }
    for dic in input_dict_list
  ]
  return(output)
  
# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/

# header classification

header_classification = make_doccano_dict_from_dict_list(articles_dict_list, "head", meta_keys = ["pubtime"])

with open('data/processed/annotations/annotation_input/header_classification.txt', 'w') as outfile:
    json.dump(header_classification, outfile)
