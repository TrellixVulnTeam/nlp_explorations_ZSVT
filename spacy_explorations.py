
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

# nlp.vocab.strings can translate hash to string and vice versa
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

# extending Doc, Span, Token or entities with custom method extensions ---------

nlp = spacy.load("de_core_news_md")

# for example, find a certain type of entity label and and create a method that turns it into a wikipedia
# search URL
# to do so we write an extension for the Span class

# first we define a getter function - it takes a span as input and returns what we want
def get_wikipedia_url(span):
    # Get a Wikipedia URL if the span has one of the labels
    if span.label_ in ("PER", "ORG", "GPE", "LOCATION"):
        entity_text = span.text.replace(" ", "_")
        return "https://en.wikipedia.org/w/index.php?search=" + entity_text
        
Span.set_extension("wikipedia_url", getter = get_wikipedia_url, force=True)

doc = nlp(
  "So viele Gnomen arbeiten in der Zürcher Kantonalbank unter ihnen Henri Dunant"
)

for ent in doc.ents:
  print(ent.text, ent.label_, ent._.wikipedia_url) # the empty underscore indcates custom extension

# combine custom pipeline component and custom extension -----

# this is quite cool, as it enables to add structured data to a spaCy pipeline

nlp = spacy.load("de_core_news_md")

# let's have our mobility matcher

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

nlp.add_pipe("recognize_mobility_form", after="ner")
nlp.pipe_names

# now a getter that looks up a categorization of mobility in this dictionary
mobility_categorization = {"fliegen":"dirty","autofahren":"dirty","laufen":"clean"}

# getter:
def get_mobility_categorization(span):
  category = mobility_categorization.get(span.text)
  return(category)

# register as span extension attribute

Span.set_extension("mobility_category", getter=get_mobility_categorization, force=True)

doc = nlp(
  "manche fliegen, andere laufen"
)

print([(ent.text, ent.label_,ent._.mobility_category) for ent in doc.ents])


# Getting real - efficient processing of more text ------

import pandas as pd

nlp = spacy.load("de_core_news_md")

articles_df = pd.read_csv("data/raw/swissdox/210809_request/Angst1.tsv", sep='\t', encoding = 'utf-8')
articles_df

articles_df["rubric"].unique()
# only zurich articles
articles_zh = articles_df[articles_df["rubric"].isin(["Zürich","Zürich und Region"])]

#turn text column in Pandas (which is a pyhton series) to a list to pass to spacy
# to be fast, we'll work first with the heads (titles)
zh_heads = articles_zh["head"].to_list()

docs = list(nlp.pipe(zh_heads))

# get all entities in titles

entities = [doc.ents for doc in docs]
entities[0:5]

#show all person entities

# find persons named
persons_in_head = [ent.text for doc in docs for ent in doc.ents if ent.label_ == "PER"]
persons_in_head

# with matcher is probably faster but I have not gotten full solution yet

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

any_person = [{"ENT_TYPE":"PER"}] #patterns are dictionaries
# this means find a noun followed by a punctuation

matcher.add("FIND_PERSON", [any_person]) #add pattern to matcher

persons_in_head = [Span(doc,start,end).text for doc in docs for match_id, start, end in matcher(doc)]
len(persons_in_head)
persons_in_head

# # for ref (as I am a bit shaky with List comprehension still - outermost loop comes always first)
# for doc in docs[0:10]:
#   for match_id, start, end in matcher(doc):
#     Span(doc, start, end).text

# visualize things - this is quite cool

from spacy import displacy

html = displacy.render(docs[0:30], style="ent", page=True)

# to display the generated html

import tempfile
import webbrowser

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
    url = 'file://' + f.name
    f.write(html)
webbrowser.open(url)

# now read in context information about text as well

# one way is to iterate over all pandas rows. this is likely very slow

# another way would be to convert the pandas df to a list of named tuples
#https://stackoverflow.com/questions/9758450/pandas-convert-dataframe-to-array-of-tuples/34551914#34551914

from spacy.tokens import Doc

dict_list = articles_zh.to_dict("records")
articles_zh_tuples = [[item.get('content'),item] for item in dict_list][0]

# Register some custom extensions
Doc.set_extension("pubtime", default=None)
Doc.set_extension("medium_code", default=None)
Doc.set_extension("head", default=None)

#https://spacy.io/usage/processing-pipelines
doc_tuples = nlp.pipe(articles_zh_tuples[0], as_tuples = True)

docs = []
for doc, context in nlp.pipe(articles_zh_tuples, as_tuples = True):
    # Set the custom doc attributes from the context
    doc._.pubtime = context["pubtime"]
    doc._.medium_code = context["medium_code"]
    doc._.head = context["head"]
    docs.append(doc)

