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
from solutions.i_training3 import (
    training_pipeline,
    padding_of_input_vectors,
    padding_to_specific_length,
)
from util import prepare_system_to_run_from_solutions_folder

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
    # predicted_sentiment_vectors = classifier....
    predicted_sentiment_vectors = classifier.predict(input_vectors)

    # Et voila! Let's evaluate it. Evaluation is basically comparing the predicted
    # values (stored in predicted_sentiment_vectors) with the true values (stored in
    # true_sentiment_vectors). Since this is such a common task, the machine learning
    # library we use, scikit learn, offers a set of functions that do the job for us.
    # Below, we calculate a few of the most common evaluation metrics. The most
    # intuitive one is perhaps accuracy, which is just the fraction of correctly
    # predicted examples compared to the total number of all examples. So, an accuracy
    # of 80% means that 80% of all examples were correctly predicted by the classifier.
    accuracy = accuracy_score(true_sentiment_vectors, predicted_sentiment_vectors)
    f1_macro = f1_score(
        true_sentiment_vectors, predicted_sentiment_vectors, average="macro"
    )
    f1_all = f1_score(true_sentiment_vectors, predicted_sentiment_vectors, average=None)

    # The confusion_matrix() function requires a specific format (FYI and to demonstrate
    # how easy it will be to fix most of the issues you might run into: I wasn't aware
    # of this and just googled the error message I got, clicked the first link (which of
    # course led to Stack Overflow ;o), and implemented the snippet shown there, i.e.
    # https://stackoverflow.com/questions/46953967)
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
    #  numbers. Do you think the performance is good? Would you use the classifier
    #  and why? How could you improve the performance?

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
    my_input_vectors = []
    for sentence in my_documents:
        doc = nlp(sentence)
        input_vector = first_k_sentence_of_doc_to_vector(doc, 5)
        my_input_vectors.append(input_vector)

    # Remember how we implemented and used the function padding_of_input_vectors() in
    # the previous exercise? Except if each of the "documents" contains only a single
    # sentence, we will need this function, too.
    # TODO Pad my_input_vectors to be of the same length as the input vectors the
    #  classifier was trained on
    # Hint: The function takes to parameters (click F1 for its documentation), and the
    # value of the second parameter was logged earlier (will be visible if you run this
    # script or the previous exercise's script)
    # my_input_vectors = padding_to_specific_length(...)
    my_input_vectors = padding_to_specific_length(my_input_vectors, 1500)

    # Use the classifier for prediction.
    my_predicted_sentiment_vectors = classifier.predict(my_input_vectors)

    # Print the documents and their predicted sentiment
    for sentence, sentiment in zip(my_documents, my_predicted_sentiment_vectors):
        logger.info("Document:  {}", sentence)
        logger.info("Sentiment: {}", sentiment)


if __name__ == "__main__":
    prepare_system_to_run_from_solutions_folder()
    evaluate()
