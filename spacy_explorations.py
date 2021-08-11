
# create python 3.9 virtual env called nlp_env with py (python launcher) 
# in terminal

# py -3.9 -m venv nlp_env
# source nlp_env/Scripts/activate

# from power shell just target activate: .\nlp_env\Scripts\activate

# to use in R Studio via reticulate use reticulate::use_virtualenv(here::here("nlp_env"),TRUE)

# the basic tokenization stuff ---------

from spacy.lang.de import German

nlp = German()

doc = nlp("So viele lustige Gnomen? 33 davon!")

doc.text

for token in doc:
  print(token.text, token.is_alpha, token.is_punct, token.like_num)

# now with a pretrained pipeline ----------

# check https://spacy.io/models/de - models in different sizes

# how to install: https://spacy.io/usage/models

import spacy

nlp = spacy.load("de_core_news_sm") # small German one

text = "So viele lustige Gnomen? 33 davon! Inklusive der Zürcher Kantonalbank"

doc = nlp(text)

doc.ents # eine named entity

for token in doc:
  print(
    token.text,
    token.pos_, # part of speech
    token.dep_, # dependency relation
    token.ent_iob_, # is the token an/ part of an entity? 
    #“B” means the token begins an entity, “I” means it is inside an entity, 
    # “O” means it is outside an entity, and "" means no entity tag is set.
    token.ent_type_ #named entity type
    )

# rule based matching ---------

# this is really cool - you can use the tokens from the pretrained model to formulate
# quite complex queries. eg. find a lemmatized verb followed by one or two nouns...

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

pattern = [{"POS":"NOUN"},{"IS_PUNCT":True}] #patterns are dictionaries
# this means find a noun followed by a punctuation

matcher.add("PATTERN_TEST", [pattern]) #add pattern to matcher

matches = matcher(doc)

for match_id, start, end in matches:
  print("Match:", doc[start:end].text)
  
# how to compute similarities --------

# if using spaCy's built-in similarity, need to use medium or large size model
# the nlp model encodes by default with word to vec and uses cosine similarity to compare

nlp = spacy.load("de_core_news_md")

doc1 = nlp("Nachhaltigkeit ist politisch")
doc2 = nlp("Fliegen ist nicht nachhaltig")

doc1.vector #one big vector

similarity = doc1.similarity(doc2) #compare one doc to the other
print(similarity)

# combine rule  based matching and model -------

#PhraseMatcher can use tokens in a Doc object to match

from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

# let's say we have a list of mobility types

MOBILITY = ["fliegen","autofahren","laufen"]

doc = nlp("In der Stadt können wir autofahren und laufen")

matcher = PhraseMatcher(nlp.vocab)

patterns = list(nlp.pipe(MOBILITY)) #faster way to import a list
matcher.add("MOBILITY",None,*patterns) #add the list to the matcher

matches = matcher(doc)
matches #a list of pattern hashes and start and end points

nlp.vocab.strings[matches[0][0]]


for match_id, start, end in matches:
  print(doc[start:end])

# pipelines (calling nlp()) --------

nlp = spacy.load("de_core_news_md")

# see pipeline component names
print(nlp.pipe_names) 
# component name/ component function tuples list
print(nlp.pipeline)

# adding a custom component

# custom components are functions that take a doc as input and return it modified
# example: simple component that returns length of doc

from spacy.language import Language

@Language.component('compute_doc_length')
def compute_doc_length(doc):
  print("Doc length:", len(doc))
  return(doc)

# add to pipeline

nlp.add_pipe("compute_doc_length", first = True) #first means it's first in the pipeline

nlp.pipe_names # see it added first in the pipeline
nlp("Schau mal das an")

# a more complicated component - rule-based entity matching
# we use the mobility phrase matcher from above

MOBILITY = ["fliegen","autofahren","laufen"]
matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(MOBILITY)) #faster way to import a list
matcher.add("MOBILITY",None,*patterns) #add the list to the matcher

@Language.component("recognize_mobility_form")
def recognize_mobility_form(doc):
    # Apply the matcher to the doc
    matches = matcher(doc)
    # Create a Span for each match and assign the label "Mobility"
    spans = [Span(doc, start, end, label="Mobility") for match_id, start, end in matches]
    # Overwrite the doc.ents with the matched spans
    doc.ents = spans
    return doc
    
nlp.add_pipe("recognize_mobility_form", after = "ner") #add after named entity recognition
nlp.pipe_names

doc = nlp("In der Stadt können wir autofahren und laufen")

# print the ent text and label
print([(ent.text, ent.label_) for ent in doc.ents])

# extending Doc, Span or Token with custom method extensions ---------

