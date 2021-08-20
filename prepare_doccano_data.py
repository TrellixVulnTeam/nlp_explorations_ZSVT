
# reticulate::use_virtualenv(here::here("nlp_env"),TRUE)

import pandas as pd
import spacy
import json

# define function for dict list to doccano json -----------------

# doccano expects in json:
# {"text": "Terrible customer service.", "labels": ["negative"], "meta": {"wikiPageID": 1}}
# {"text": "Really great transaction.", "labels": ["positive"], "meta": {"wikiPageID": 2}}
# {"text": "Great price.", "labels": ["positive"], "meta": {"wikiPageID": 3}}

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

# header classification prep for binary sdg related classification -----------------------------

articles_df = pd.read_csv("data/raw/swissdox/210809_request/Angst1.tsv", sep='\t', encoding = 'utf-8')
articles_df

# turn into dict
articles_dict_list = articles_df.to_dict("records")
  
# header classification

header_classification = make_doccano_dict_from_dict_list(articles_dict_list, "head", meta_keys = ["pubtime"])

# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/

with open('data/processed/annotations/annotation_input/header_classification.json', 'w') as outfile:
    json.dump(header_classification, outfile)
    
    
# paragraph classification prep for binary sdg related classification --------------------------------

# beautifulsoup for html splits
from bs4 import BeautifulSoup 

articles_df = pd.read_csv("data/raw/swissdox/210809_request/Angst1.tsv", sep='\t', encoding = 'utf-8')
articles_df

# turn into dict
articles_dict_list = articles_df.to_dict("records")

par_docs = []

# # for loop
# 
# for dic in articles_dict_list:
#   soup = BeautifulSoup(dic["content"])
#   paragraphs = [par for par in soup.findAll('p')]
#   for par in paragraphs:
#     par_dic = {key:value for key,value in dic.items() if key not in ["content"]}
#     par_dic["par"] = par
#     par_docs.append(par_dic)

# dict comprehension - much much faster

par_docs = [{key:value for d in (dic, {'par':par.string}) for key,value in d.items() if key not in ["content"]}
  for dic in articles_dict_list 
  for par in BeautifulSoup(dic["content"]).findAll('p')
  ]

len(par_docs)  
len(articles_dict_list)

paragraph_classification = make_doccano_dict_from_dict_list(par_docs, "par", 
  meta_keys = [key for key in par_docs[0].keys() if key not in "par"])

# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/

with open('data/processed/annotations/annotation_input/paragraph_classification.json', 'w') as outfile:
    json.dump(paragraph_classification, outfile)

