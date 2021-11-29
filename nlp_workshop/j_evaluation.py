"""
In this exercise, we'll
* use the model we trained in the previous exercise and
* quantitatively evaluate it using the examples from our test set
* learn a bit (really just a first glance) about evaluation metrics ("how to evaluate")
* qualitatively "evaluate" (=play around) with your own sentences and see how well the
  classifier works
"""

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from loguru import logger
import numpy as np

from solutions.c_preprocessing import get_spacy
from solutions.g_training1 import first_k_sentence_of_doc_to_vector
from solutions.i_training3 import training_pipeline, padding_to_specific_length

nlp = get_spacy()


def evaluate():
    # The training_pipeline() function returns the trained classifier and the text set
    classifier, examples_test = training_pipeline()

    # Quantitative evaluation
    # Similar to the previous exercise, let's get the input and target (=sentiment)
    # vectors - this time of the test set
    input_vectors = [example[0] for example in examples_test]
    true_sentiment_vectors = [example[1] for example in examples_test]

    # TODO Use the classifier to predict the sentiment of each test example
    # Hint: You don't need a for loop but can directly pass all input vectors to the
    # classifier. In the previous exercise, we used the MLPClassifier (this is the
    # class of the classifier object returned by training_pipeline(); google for
    # "scikit learn MLPClassifier predict" to find out how you can predict with this
    # classifier
    # Hint 2: use the predict function and the input_vectors
    predicted_sentiment_vectors = classifier  # TODO: classifier.TODO(TODO)

    # Et voila! Let's evaluate the classifier. predicted_sentiment_vectors now contains
    # the
    # predicted sentiment values for the test examples. Since we also know the correct
    # value (from the Excel file, now stored in true_sentiment_vectors) - let's do a
    # quantitative evaluation by comparing them.
    # Since this is such a common task, the machine learning
    # library we use, scikit-learn, offers a set of functions that do the job for us.
    # Below, we calculate a few of the most common evaluation metrics. The most
    # intuitive one is perhaps accuracy, which is just the fraction of correctly
    # predicted examples compared to the total number of all examples. So, an accuracy
    # of 80% means that 80% of all examples were correctly predicted by the classifier.
    accuracy = accuracy_score(true_sentiment_vectors, predicted_sentiment_vectors)
    f1_macro = f1_score(
        true_sentiment_vectors, predicted_sentiment_vectors, average="macro"
    )
    f1_all = f1_score(true_sentiment_vectors, predicted_sentiment_vectors, average=None)

    # The confusion_matrix() function requires a specific format (FYI/nerd info:
    # we need the np.asarray(...).argmax(...) stuff because otherwise we would run into
    # an error as shown on https://stackoverflow.com/questions/46953967 - from there I
    # also got this solution)
    conf_matrix = confusion_matrix(
        np.asarray(true_sentiment_vectors).argmax(axis=1),
        np.asarray(predicted_sentiment_vectors).argmax(axis=1),
    )

    # Print the evaluation metrics
    logger.info("f1_m:     {}", f1_macro)
    logger.info("f1_all:   {} (order: negative, neutral, positive)", f1_all)
    logger.info("accuracy: {}", accuracy)
    logger.info("confusion matrix:\n{}", conf_matrix)

    # TODO Debug the code (set a breakpoint on one of the lines) and have a look at the
    #  evaluation scores (google what a confusion matrix is). Do you think the
    #  performance is good? Would you use the
    #  classifier or not? Why? How could you improve the performance?

    # Qualitative evaluation (playground)
    # TODO Add a few sentences of your own
    my_documents = [
        "Facebook is awesome.",
        "Facebook sucks.",
        "Twitter's stock fell by 3%.",
        "Privacy concerns concerning YouTube and co.",
        "Add your own sentences or documents. You can also add documents with multiple "
        "sentences",
    ]
    # Iterate the sentences and convert them to
    my_input_vectors = []
    for sentence in my_documents:
        doc = nlp(sentence)
        input_vector = first_k_sentence_of_doc_to_vector(doc, 5)
        my_input_vectors.append(input_vector)

    # TODO Perform padding on the input vectors.
    #  Similar to the previous exercise, we'll need all vectors to be of the same length
    # (in comparison to the training vectors, which were 5 sentences * 300 dimensions
    # = 1500).
    # Hint: The function takes two parameters (click on its name and then F1 for its
    # documentation).
    my_input_vectors = padding_to_specific_length(..., ...)

    # Use the classifier for prediction.
    my_predicted_sentiment_vectors = classifier.predict(my_input_vectors)

    # Print the documents and their predicted sentiment
    for sentence, sentiment in zip(my_documents, my_predicted_sentiment_vectors):
        logger.info("Document:  {}", sentence)
        logger.info("Sentiment: {}", sentiment)

    # TODO Run the file and see how well (or bad) your sentences are classified.
    #  Can you identify any pattern, e.g., a type of sentence or sentiment expression
    #  that works particularly well or bad? How could you improve the bad cases?


if __name__ == "__main__":
    evaluate()
