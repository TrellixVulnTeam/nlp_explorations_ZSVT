"""
In this exercise, we'll learn
* which models are available for spacy. A (language) model enables spacy to perform
  NLP tasks, such as those we saw in b_nlpplayground
* how to apply spacy on a given text (it's easy :-))
"""

import spacy
from loguru import logger
from tqdm import tqdm

# Here we import the parse_dataset function we wrote earlier to reuse it in the current
# exercise (if you look carefully, you will notice that we use the a_parsing file from
# the solutions folder ("solutions.a_parsing") and not the one you just wrote).
# We will do this in all future exercises, too, i.e., whenever making use of code we
# created in an earlier exercise, we use the
# script provided in the solutions folder, just so that everyone is on the same page.
# In case you successfully completed an earlier exercise, feel free to remove the
# "solutions." from the import statement (e.g., "from a_parsing import parse_dataset")
# so that you use your own script.
from solutions.a_parsing import parse_dataset

# TODO Visit https://spacy.io/usage/models and see what models are available. Note that
#  there are language-specific models, so always watch out to use the right model. What
#  other factors may influence your decision which model to use
#  (performance/efficiency)?

# Load spacy with the large model for the English language (en_core_web_lg)
nlp = spacy.load("en_core_web_lg")

def get_spacy():
    """
    A function we'll use in later exercise to retrieve the previously initialized
    spacy object
    """
    return nlp


def preprocess_dataset(df, column_name="fulltext"):
    """
    Iterates the rows (=articles) in the dataset and uses spacy to preprocess them.
    Specifically, it accesses the text in the attribute column_name and passes it to spacy.
    Returns a list of preprocessed spacy documents.
    """
    logger.info("preprocessing...")

    # create an empty list to store the preprocessed documents
    docs = []

    # iterate the dataset (and use tqdm to show a progress bar because the preprocessing
    # may take some time)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # get the text attribute (column_name defaults to "fulltext", which is the
        # column we created earlier and which consists of the article's headline and
        # maintext)
        text = row[column_name]

        # TODO invoke spacy to NLP preprocess the text
        # Hint: Use the nlp(...) function offered by spacy (see b_playground for how to
        # apply spacy on a given text in order to retrieve a preprocessed document)
        # Hint 2: Apply spacy on the variable text
        doc = nlp(text)

        # add the document to the docs list
        docs.append(doc)

    logger.info("done")

    # TODO Set a breakpoint, for example on line 70 (logger.info(...)) and debug the
    #  file. Using the debugger view, investigate the content of docs and individual
    #  preprocessed documents in that list. You'll notice that the debugger shows a
    #  human-friendly text representation of each document and that a document also
    #  consists of many spacy-internal variables and functions (which are not of
    #  interest for us). FYI: The same is true for many Python objects and when using
    #  the debugger in your own projects or throughout this course, you'll find that
    #  it's sometimes difficult to "find" the information in an object that you might
    #  be interested in. Learning by doing is my suggestion, and this may take some time
    #  to get used to, so don't worry in case you're overwhelmed with the many internals
    #  shown for some objects.

    # return the preprocessed documents
    return docs


if __name__ == "__main__":
    # Parse the Excel file
    df = parse_dataset()

    # Preprocess the dataset
    # TODO CMD + Click the function
    preprocess_dataset(df)
