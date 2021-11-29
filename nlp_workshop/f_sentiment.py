"""
In this exercise, we'll learn how to
* use a model from Huggingface Hub, incl. how to initialize its tokenizer and the actual
  model (we'll do this with a sentiment classification model, but the steps are
  generally independent of the classification task, so you will be apply them easily in
  your own projects where other classification tasks are necessary)

FYI: Models from Huggingface Hub under the hood employ pytorch or tensorflow, which are
both deep learning Python libraries, offering rather low-level deep-learning-related
functionality, e.g., enabling developers to design and implement neural networks. We
won't really use pytorch or tensorflow but will use the more convenient functions
offered by Huggingface models and the Huggingface framework.
"""

from solutions.a_parsing import parse_dataset
from solutions.c_preprocessing import preprocess_dataset


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from loguru import logger


# Language models don't work with texts and words directly, but require token
# indexes, i.e., the index of a word in the model's vocabulary. For example,
# in a model's vocabulary, the index 100 might represent the word "the", index 101
# might represent "she", etc. We'll use a tokenizer to convert a text sequence into
# its index representation that we can then pass to the model.
# TODO Load our model's tokenizer (siebert/sentiment-roberta-large-english) so that we
#  can use it later.
# Hint: Go to
# https://huggingface.co/siebert/sentiment-roberta-large-english, click
# on "Use in Transformers" (top right), and copy paste the corresponding line of code.
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")

# We'll also need to load the model itself.
# TODO Load our model (siebert/sentiment-roberta-large-english)
# Hint: See previous hint
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")


# TODO scroll to pipeline()


def predict_sentiment_of_sentence(sentence):
    """
    Predicts the sentiment of a given sentence.
    """
    # TODO: Convert sentence into its index representation by calling the
    #  tokenizer.
    # Hint: google "huggingface process text"
    # Hint 2: https://huggingface.co/transformers/preprocessing.html#base-use
    # FYI: Remember the top text in this file about pytorch and tensorflow?
    # return_tensors="pt" just means that we want the tokenizer to return a pytorch
    # object
    inputs = tokenizer(sentence, return_tensors="pt")

    # Call the model. This will pass the tokenized and converted inputs to the model,
    # i.e., the pre-trained and fine-tuned model we downloaded from Huggingface Hub, and
    # will return the results of its last neural layer (Nerd fact: Neural
    # networks consist of multiple "layers" of neurons, where (typically) the input is
    # passed into the first layer, processed by the first layer, and then the processed
    # values are passed to the next layer, and so on; what we will have in the variable
    # outputs is the result of the last layer).
    outputs = model(**inputs)

    # In machine learning and especially deep learning, for efficiency, one typically
    # processes data in so called batches, e.g., 8, 16, or 32 sentences at the same
    # time. For the sake of simplicity and demonstration, we here only process a single
    # sentence. Yet, model(...) returns a tuple (=list) of results (because it is
    # typically invoked with a batch of texts). Before we proceed, let's thus access
    # only the first (and only) result item in this tuple.
    outputs = outputs[0]

    # outputs is a tensor (=vector or matrix) of size 1x2 because the model classifies
    # the given input according two classes, NEGATIVE and POSITIVE (Nerd fact: In
    # technical terms, the last layer of the neural network consists of two neurons, one
    # for each class). The tensor does not
    # contain the probability but rather an internal score (called the activation value)
    # of the neural network. The softmax function let's us convert this value into
    # probabilities
    sentiment_probabilities = softmax(outputs)

    # TODO Set a breakpoint and debug to see the contents of inputs, outputs, and
    #  sentiment_probabilities. Note how the dimensionality of the variables changes
    #  as "inputs" (high-dimensional vector representation of a sentence) is processed
    #  by the model, which gives us a two-dimensional outputs variable.

    # TODO Go back to the function sentiment_classification
    return sentiment_probabilities


def print_sentiment(sentence, sentiment_probabilities):
    """
    Given the predicted probabilities of a single sentence, print the more likely
    sentiment class, "neutral" if neither class is likely, and "ambivalent" if both
    classes are likely
    """
    # If the probability of a sentiment class is higher than our threshold, we'll print
    # that.
    SENTIMENT_THRESHOLD = 0.5

    # Convert sentiment_probabilities tensor (of size 1x2) to list
    sentiment_probabilities = sentiment_probabilities.tolist()[0]

    # The predicted classes are NEGATIVE and POSITIVE (in this order, see the example at
    # https://huggingface.co/siebert/sentiment-roberta-large-english)

    # get the predicted probability of each class
    probability_negative = sentiment_probabilities[0]
    probability_positive = sentiment_probabilities[1]

    # is the sentiment higher than our threshold
    is_negative = probability_negative >= SENTIMENT_THRESHOLD
    is_positive = probability_positive >= SENTIMENT_THRESHOLD

    # decide which sentiment was predicted
    predicted_class = "neutral"
    if is_positive and is_negative:
        predicted_class = "ambivalent"
    elif is_positive:
        predicted_class = "positive"
    elif is_negative:
        predicted_class = "negative"

    # print this information
    logger.info(
        "\nprediction of sentence: {}\n{} (probabilities: {})",
        sentence,
        predicted_class,
        sentiment_probabilities,
    )


def sentiment_classification(docs):
    """
    Performs sentiment classification using a model from Huggingface Hub on
    a few of our own texts (feel free to add your own) and some news articles' sentences
    from our Excel file.
    """

    # Let's build a small set of a few exemplary sentences for each of which we will
    # predict their sentiment (on sentence-level). We'll discuss the results briefly
    # later. Feel free to add more sentences to this list.
    my_sentences = [
        "Awesome.",
        "I am happy about the results.",
        "I hate it.",
        "I would have expected more.",
        "The camera is awesome but it costs too much.",
        "The camera costs too much but still it's awesome.",
        "Bob said that on the one hand he was happy about the results but on the other hand he had expected more.",
        "This is neutral.",
    ]
    # Iterate all sentences and predict and print their sentiment.
    for sentence in my_sentences:
        # TODO CMD+Click on the function
        sentiment = predict_sentiment_of_sentence(sentence)
        print_sentiment(sentence, sentiment)

    # Iterate the docs and predict and print their sentiment
    for doc_index, doc in enumerate(docs):
        # Iterate the current doc's sentences
        for index, sentence in enumerate(doc.sents):
            # Get the sentence as a string
            sentence_text = sentence.text

            # Predict the sentiment
            sentiment = predict_sentiment_of_sentence(sentence_text)
            print_sentiment(sentence_text, sentiment)

            # Let's stop after 2 sentences
            if index >= 1:
                break


def pipeline():
    """
    Invokes each step in our sequence of analysis steps
    """
    # Parse the Excel file (only get the first two rows/articles in this exercise)
    df = parse_dataset(2)

    # Preprocess using spacy. Note that we need spacy only for sentence segmentation
    # but except for that won't use any other NLP functionality provided by spacy.
    # Retrieving the individual sentences is necessary because most (not all!) language
    # models, including the one that we decided to use, work only on individual
    # sentences, tweets, or generally short texts.
    docs = preprocess_dataset(df)

    # TODO CMD+Click
    sentiment_classification(docs)


if __name__ == "__main__":
    pipeline()
