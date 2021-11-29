"""
In the final exercise, we'll train our own classifier. Note that typically you would use
a larger dataset, both so that the classifier has "enough" training data to learn
patterns (=knowledge) from but also for more reliable evaluation. Training a larger
dataset, however, also takes (much) more time. Yet, the concept is the same (no matter
the size of the dataset), so for the sake of simplicity and due to our
limited time, we'll use our small dataset. When training and evaluating a classifier
later in your own project, you will be able to adapt and apply the steps below - just
use more data :-) We'll create a classifier for three sentiment classes: positive,
neutral, and negative.

In this exercise specifically, we'll complete the first of in total four tasks for the
training and evaluation of our very own sentiment classifier. Specifically, we'll
convert the content of our Excel file into examples that can be used for training the
classifier and evaluating it.
"""
from collections import Counter

from loguru import logger

from solutions.a_parsing import parse_dataset
from solutions.c_preprocessing import preprocess_dataset
from util import prepare_system_to_run_from_solutions_folder


# TODO Scroll to pipeline()


def first_k_sentence_of_doc_to_vector(doc, k):
    # Build the vector representation consisting of sentence embeddings of the first
    # k sentences
    # First, let's get the first five sentences
    sentences = list(doc.sents)
    sentences = sentences[:k]

    # Create an empty list that will hold the vector (one vector for one document)
    vector = []

    # Iterate the sentences
    for sentence in sentences:
        # Get the vectors of the first five sentences
        cur_vector = sentence.vector

        # TODO *Extend* the vector list with the cur_vector
        # Hint: Don't use append, because this will result in vector being a list of
        # lists, whereas we want a list of scalars
        # vector.TODO
        vector.extend(cur_vector)

    return vector


def create_examples(df, docs):
    """
    Given our dataframe and the preprocessed docs, creates and returns two lists, where
    one contains the input sequences for our model, and the other the true values
    :param df:
    :param docs:
    :return:
    """

    # For training (and evaluation), independently of the classification task, we need
    # basically only two things:
    # 1) the input sequence passed to the model, e.g., a sentence or other (rather
    # short) text from an article of our dataset
    # 2) the value that the model should predict, e.g., the sentiment score of the input
    # sequence (typically in ML, this is called the true value or y_true)
    # A pair consisting of these two pieces of information is called an example. The
    # more examples, the better, i.e., the model can learn better from many examples
    # (during training), and the results of the evaluation are more reliable.

    # Let's create examples! Given an article, we'll create the following:
    # 1) input sequence: a vector containing the word embeddings of the first five
    # sentences
    # 2) y_true: the article's attribute in the column "TONE" in the dataset
    input_vector_all = []
    y_true_all = []

    # Run an index from 0 to the number of docs
    for index in range(len(docs)):
        # Get the current preprocessed spacy doc and the corresponding Excel row
        doc = docs[index]
        df_row = df.iloc[index]

        # Create a vector for the first five sentences of the doc
        vector = first_k_sentence_of_doc_to_vector(doc, 5)

        # We built the vector of the current document.
        # TODO Append the vector to input_vector_all
        # input_vector_all.TODO
        input_vector_all.append(vector)

        # TODO Get the true value for the sentiment of the current article
        # Hint: From the df_row object
        # y_true = ...
        y_true = df_row["TONE"]

        # Append the true value to the list of true values
        y_true_all.append(y_true)

    # print some info incl. distribution
    logger.info(
        "created {} examples (distribution: {})",
        len(input_vector_all),
        Counter(y_true_all).most_common(),
    )

    return input_vector_all, y_true_all


def pipeline():
    """
    Invokes each step in our sequence of analysis steps
    :return:
    """
    # Parse the Excel file
    df = parse_dataset(5)

    # Preprocess
    docs = preprocess_dataset(df)

    # Create the examples
    # TODO CMD+Click
    create_examples(df, docs)


if __name__ == "__main__":
    prepare_system_to_run_from_solutions_folder()
    pipeline()
