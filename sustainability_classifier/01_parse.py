import pandas as pd
import spacy

nlp = spacy.load("de_core_news_md")

articles_df = pd.read_csv("../data/raw/swissdox/210809_request/Angst1.tsv", sep='\t', encoding = 'utf-8')
articles_df
