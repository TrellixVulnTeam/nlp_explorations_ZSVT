"""
Third of four exercises and the last that is concerned with the actual training (the
fourth will be about the evaluation). What are we going to do?
* learn how to train a classifier from scikit-learn on our data
* fix a common real-world issue
* almost nothing to implement this time but follow the code actively and perhaps add a few
  debug points and debug to see what's going on
"""

from solutions.a_parsing import parse_dataset
from solutions.c_preprocessing import preprocess_dataset
from solutions.g_training1 import create_examples
from solutions.h_training2 import normalize_sentiment, create_splits

from sklearn.neural_network import MLPClassifier

from loguru import logger

# TODO go to training_pipeline()


def padding_to_specific_length(vectors, desired_length):
    """
    Pads each vector in the parameter vectors, i.e., adds 0s where necessary, to be of
    length desired_length. Padding is a common technique in ML to make sure that the
    input to a neural network is always of same length (which is required because - and
    now it gets nerdy again - a neural network (or an "AI") is not much more than
    multiple matrices multiplied with other matrices, and (perhaps you remember this
    from high school, you can't just multiply a vector/matrix of any size with another
    matrix but certain dimension restriction apply. And so years ago it became common
    practice to just add 0s until everything is in order (i.e., all vectors/matrices are
    of same length).
    """
    logger.info(
        "padding {} vectors to length of {} scalars", len(vectors), desired_length
    )

    # Iterate vectors
    for vector in vectors:
        # Determine the difference between the desired length and the current vector's
        # length
        length_difference = desired_length - len(vector)
        # Continue with the next vector if the current vector has the desired length
        if length_difference == 0:
            continue

        # Create a 0-vector to be added to the original vector so that the latter will
        # be of the desired length
        zeros = [0] * length_difference
        vector.extend(zeros)

    return vectors


def padding_of_input_vectors(vectors):
    """
    Appends 0s to all vectors that are not of the same length as the longest vector in
    vectors.
    """
    # Get the length of the longest vector
    max_length = 0
    for vector in vectors:
        max_length = max(max_length, len(vector))

    # TODO CMD+Click (we'll create a specific function just for the padding, because
    #  we will use that function later during the evaluation)
    vectors = padding_to_specific_length(vectors, max_length)

    return vectors


def training_pipeline():
    # Parse the Excel file
    df = parse_dataset()

    # Preprocess
    docs = preprocess_dataset(df)

    # Create the examples
    input_vector_all, y_true_all = create_examples(df, docs)

    # Padding of input vectors
    input_vector_all = padding_of_input_vectors(input_vector_all)

    # Normalize y_true_all
    y_true_all = normalize_sentiment(y_true_all)

    # Create splits
    examples_train, examples_test = create_splits(input_vector_all, y_true_all)

    # Alright, so at this point we got all the data ready in the form we need it (we
    # have 2 splits (1 for training, 1 for testing), we converted each document
    # into a numerical vector representation, and likewise did we convert the sentiment
    # into a numerical vector representation -- text as data at its best :-) Time for
    # some actual training. As always, scikit-learn, the Python machine learning library
    # has got our back, and we can use any of the models they provide (after the
    # workshop, why not have a look at this nice and informative graphical overview
    # of the results of many models that scikit-learn offers:
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    # We'll use MLPClassifier because it's a neural network and because if we were about
    # to run a startup we could call our approach AI or deep learning (since we are
    # using a neural network ;-) After the workshop, you could try any other of the
    # models available (they can simply be exchanged, without one having to rewrite any
    # code).

    # We set random_state to a fixed value so that everyone gets the same values. Note
    # that one can pass various other (hyper)parameters to the MLPClassifier, but the
    # defaults are in many cases just fine. (In contrast, if you wanted to submit your
    # work to a CS/CL conference, you typically would want to try to find the best
    # combination of these parameters to hopefully improve the classification, even if
    # only by one percent points.)
    classifier = MLPClassifier(random_state=1)#random state is like set.seed() - replicability

    # For each training example, get the input_vector and the sentiment_vector (and
    # do this for all examples and create a list of this)
    input_vector = [example[0] for example in examples_train]
    sentiment_vector = [example[1] for example in examples_train]

    logger.info("training...")
    # TODO Train the classifier on the training data
    # Hint 1: Use input_vector and sentiment_vector as parameters
    # Hint 2: Training, in scikit-learn's terminology (the ML library we're using) means
    # fit (we used this function already in the context of clustering).
    # Hint 3: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.fit
    # TODO add one line of code: classifier.TODO(TODO, TODO)
    classifier.fit(input_vector, sentiment_vector)

    logger.info("done")

    # TODO Execute this file.
    # Hint: it will crash. Try to find the cause of the issue using Google and
    # Stackoverflow and the debugger. Spend only a few minutes on this and see how far
    # you can get, after 5 minutes, continue with Hint 2. Hint 3 describes how we can
    # fix the issue.
    # Hint 2: https://stackoverflow.com/a/25495091 The issue is that at least one vector
    # in input_vector is of different size compared to all other vectors, i.e., one
    # document contains less than 5 sentences so that word embeddings of less than 5
    # sentences were appended in our function create_examples(...), resulting in a
    # shorter vector.
    # Hint 3: Retry by removing the comment form line 71 (so that
    # padding_of_input_vectors is executed). Also, go into the padding_of_input_vectors
    # function and understand what it does, i.e., how it solves the above issue.

    return classifier, examples_test


if __name__ == "__main__":
    training_pipeline()
