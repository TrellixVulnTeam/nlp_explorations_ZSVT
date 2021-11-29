import spacy
from loguru import logger
from tqdm import tqdm

from util import prepare_system_to_run_from_solutions_folder

# Here we import the parse_dataset function we wrote earlier to reuse it in the current
# exercise
from solutions.a_parsing import parse_dataset

# TODO Load spacy with the large model for the English language
# nlp = ...
nlp = spacy.load("en_core_web_lg")
# then scroll to the bottom of this file


def get_spacy():
    return nlp


# TODO scroll to bottom of file


def preprocess_dataset(df, column_name="fulltext"):
    """
    Iterates the rows (=articles) in the dataset and uses spacy to preprocess them.
    Specifically it access the text in the attribute column_name and passes it to spacy.
    Returns a list of preprocessed spacy documents.
    :param column_name:
    :param df:
    :return:
    """
    logger.info("preprocessing...")

    # create an empty list to store the preprocessed documents
    docs = []

    # iterate the dataset (and use tqdm to show a progress bar because the preprocessing
    # may take some time)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # get the text attribute (column_name defaults to "fulltext", which you created
        # earlier and which consists of the article's headline and maintext)
        text = row[column_name]

        # TODO invoke spacy to NLP preprocess the text
        # doc = ...
        doc = nlp(text)

        # add the document to the docs list
        docs.append(doc)

    logger.info("done")

    # return the preprocessed documents
    return docs


if __name__ == "__main__":
    prepare_system_to_run_from_solutions_folder()

    # Parse the Excel file
    df = parse_dataset()

    # Preprocess the dataset
    # TODO CMD + Click the function
    preprocess_dataset(df)
