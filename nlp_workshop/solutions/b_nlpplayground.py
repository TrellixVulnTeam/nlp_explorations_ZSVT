"""
This file may serve as a sandbox to get an understanding what spacy, a state-of-the-art
Python library for common NLP tasks, does. Most of the examples are taken from
spacy's website, e.g., https://spacy.io/usage/spacy-101#features

Feel free to change (or break) things, we won't need this file later.
"""

import spacy

# TODO scroll to pipeline()


def tokenize(doc):
    # https://spacy.io/usage/spacy-101#annotations-token
    for token in doc:
        print(token.text)


def pos_tagging(doc):
    # https://spacy.io/usage/spacy-101#annotations-pos-deps
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.dep_, token.is_stop)


def named_entity_recognition(doc):
    # https://spacy.io/usage/spacy-101#annotations-ner
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


def text_similarity():
    # https://spacy.io/usage/spacy-101#vectors-similarity
    # TODO change the next two lines if you want
    your_text_1 = "I like salty fries and hamburgers."
    your_text_2 = "Fast food tastes very good."

    # load spacy
    nlp = spacy.load("en_core_web_lg")

    # preprocess both texts
    doc1 = nlp(your_text_1)
    doc2 = nlp(your_text_2)

    # Similarity of two documents
    print(doc1, "<->", doc2, doc1.similarity(doc2))

    # Similarity of tokens and spans
    french_fries = doc1[2:4]
    burgers = doc1[5]
    print(french_fries, "<->", burgers, french_fries.similarity(burgers))


def pipeline():
    # TODO change my_text, if you want
    my_text = "Apple is looking at buying U.K. startup for $1 billion"

    # Load the library using the large (lg) model of the English language (en) trained
    # on various datasets representing online documents (web)
    nlp = spacy.load("en_core_web_lg")

    # Process your text with spacy
    doc = nlp(my_text)

    # TODO Call the function(s) you want. Each shows different information that spacy
    #  extracted. Hint: Use CMD + Click to jump to each function (works only if the
    #  function is not mentioned in a comment but in a line of code).
    tokenize(doc)
    # pos_tagging(doc)
    # named_entity_recognition(doc)
    # text_similarity()


if __name__ == "__main__":
    pipeline()
