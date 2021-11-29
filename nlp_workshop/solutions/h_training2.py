"""
Second of four exercises on the way to training and testing our own sentiment
classifier.

In this exercise, we'll learn
* what a train and a test set is (and very briefly why we need both)
* how to create these sets
* how to normalize and convert the raw y_true value so that we can use it with our
classifier

Tasks 1 and 2 are independent of the classification task and should be done always in
any ML project. Task 3 is also necessary in most projects but what specifically needs to
be done depends on the data and on the classifier. We'll do task 3, of course, for the
specific Excel file and classification task at hand :-)
"""

from collections import Counter

from solutions.a_parsing import parse_dataset
from solutions.c_preprocessing import preprocess_dataset
from solutions.g_training1 import create_examples
from util import prepare_system_to_run_from_solutions_folder

from loguru import logger
from random import shuffle
from math import floor

# TODO Scroll to pipeline


def normalize_sentiment(y_true_all):
    """
    Converts 5 sentiment labels to 3 numerical classes (both "POS" and "POS -"
    should become 1, "NEUT" -> 0, and both "NEG" classes -1, respectively. Then, creates
    a vector from these three classes, with the order of scalars being [NEGATIVE,
    NEUTRAL, POSITIVE]
    :param y_true_all:
    :return:
    """

    # We'll create and use a dict for the conversion, where each key is the label as
    # used in the Excel, and its corresponding value is the normalized value we'll use
    LABEL2CLASS = {
        "POS": 1,
        "POS -": 1,
        "NEUT": 0,
        "NEG -": -1,
        "NEG - ": -1,
        "NEG": -1,
        "NEG ": -1,
        "POS  -": 1,
        "POS ": 1,
        # TODO add the remaining classes (there might be mistakes in the Excel, which
        #  you'll notice later...)
        # TODO FELIX REMOVE THE CLASSES
    }

    # create an empty list to hold the normalized values
    tmp = []
    for y_true in y_true_all:
        normalized = LABEL2CLASS[y_true]
        tmp.append(normalized)

    # print some info incl. distribution
    logger.info(
        "normalized y_true (distribution: {})", Counter(tmp).most_common(),
    )

    # convert from single numerical scalar (with 3 values) to a 3-sized vector
    y_true_all = []
    for y_true in tmp:
        vector = None
        if y_true == -1:
            vector = [1, 0, 0]
        elif y_true == 0:
            vector = [0, 1, 0]
        elif y_true == 1:
            vector = [0, 0, 1]
        else:
            raise ValueError(f"unknown value: {y_true}")
        y_true_all.append(vector)

    # TODO go back to pipeline()

    return y_true_all


def create_splits(input_vector_all, y_true_all):

    # Compared to shuffling and splitting two lists, it's more convenient to do these
    # steps with one list only. Thus, let's join the items of both lists.
    examples = list(zip(input_vector_all, y_true_all))

    # TODO Shuffle
    # Hint: use the shuffle function (it doesn't return anything but changes the list
    # passed to it)
    # TODO
    shuffle(examples)

    # Split
    # In ML, a commonly used split is 80%/20% for the training and test set. Although we
    # have exactly 100 examples and could just split at position 80, let's implement
    # this the general way so that the splitting will also work with any number of
    # examples.
    # TODO get the number of examples
    # num_examples = ...
    num_examples = len(examples)
    # TODO determine the position in the list where we should split (everything before
    #  that position will be one set, everything after another set)
    # split_position =
    split_position = num_examples * 0.8
    # TODO Round it down
    # Hint https://www.google.de/search?q=python+round+down
    # split_position = ...
    split_position = floor(split_position)

    # do the actual split
    examples_train = examples[:split_position]
    examples_test = examples[split_position:]

    # TODO go back to pipeline()

    return examples_train, examples_test


def pipeline():
    # Parse the Excel file
    df = parse_dataset()

    # Preprocess
    docs = preprocess_dataset(df)

    # Create the examples
    input_vector_all, y_true_all = create_examples(df, docs)

    # TODO Normalize y_true_all (CMD+Click)
    y_true_all = normalize_sentiment(y_true_all)

    # Training (and evaluating) a classifier is a traditional machine learning (ML)
    # task. In ML, is not only a best practice but also compulsory to use separate,
    # non-overlapping sets for training and evaluation (tl;dr Background info: we are
    # typically interested in how well a classifier generalizes from the examples
    # it was trained with when used on new, unseen examples. Thus, examples used in
    # the evaluation must be different from those used in training. We will achieve this
    # by shuffling our lists of examples and then splitting them into two sets.
    # TODO CMD+Click
    examples_train, examples_test = create_splits(input_vector_all, y_true_all)


if __name__ == "__main__":
    prepare_system_to_run_from_solutions_folder()
    pipeline()
