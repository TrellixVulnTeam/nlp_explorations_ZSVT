"""
In this exercise, we'll
* practically learn what clustering is and does
* apply two common clustering algorithms (optionally a third one)
* compare their results
* learn how to predict the cluster of a new document using a trained clustering method

Clustering is useful if you have a large collection of documents and want to group them
according to their content (or theoretically you can cluster according to any property
that you can represent numerically, such as date, time, locations, etc., but since this
is an NLP workshop, we'll of course cluster using text :-)

Moreover, once you trained a clustering method on a dataset, you can also use the
trained method to predict the cluster of new documents.
"""

from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation
from loguru import logger

from solutions.c_preprocessing import get_spacy
from util import plot_dendrogram

# TODO scroll to pipeline()


nlp = get_spacy()


def convert_documents_to_vectors(documents):
    """
    Converts each document (being a text consisting of one or more sentences) into
    a vector representation using spacy.
    """
    # Preprocess using spacy
    spacy_docs = []
    for text in documents:
        # TODO Use spacy to preprocess the text into a document
        # Hint: use the nlp(...) function as in the earlier exercises
        doc = nlp(text)

        # Append the current doc to the list spacy_docs
        spacy_docs.append(doc)

    # Convert each document into a vector representation:

    # create an empty list to store the documents' vectors
    vectors = []
    # iterate our documents
    for doc in spacy_docs:
        # TODO Get the vector representation of the current doc.
        # Hint: https://spacy.io/api/doc (you'd find it after some reading, but for
        # time reasons, you may also just search for "meaning representation")
        # The vector represents a given text as a numerical vector (or matrix). Under
        # the hood, spacy uses state-of-the-art language models, such as BERT or
        # RoBERTa, and thus the vector representation we are retrieving here is a
        # numerical representation that conveys syntactic, semantic, and other
        # information (we'll see a few practical examples of what this means later in
        # the workshop).
        vector_representation = doc.vector  # TODO: article_embedding = doc.TODO

        # add the current doc's vector to our vector list
        vectors.append(vector_representation)

    return vectors


def agglomerative_clustering(vectors):
    """
    Performs agglomerative clustering on the list of vectors. Goodie: Also prints a
    "beautiful" dendrogram :-)
    """

    # Agglomerative clustering is algorithmically a quite simple technique (basically
    # pair-wise comparison of documents). Still, we don't need to implement it, because
    # the scikit-learn library (a Python library for machine-learning-related tasks)
    # got it for us.

    # Create the clustering model (after the workshop, you can find more info on
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    # TODO "Fit" the model on our vectors
    # Fitting a model is a technical term that basically means that the model should
    # learn from the data, i.e., in case of clustering models that they should learn
    # which clusters are in the texts (or rather their vector representations).
    # Hint: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    # (look for the function fit; as you can see it takes two parameters, whereby the
    # second is ignored so that you only need to pass the first one, which is our data,
    # i.e., the variable named vectors, containing the list of embeddings
    model = model.fit(vectors)

    # Plot the results of agglomerative clustering in a dendrogram (this is a prepared
    # function)
    plot_dendrogram(model, truncate_mode="level", p=30)

    # TODO A dendogram should be visible now. Investigate it and compare the clustering
    #  to the contents of the Excel file. How well does the clustering work?
    # The numbers at the bottom (x-axis) reflect the index of the element in our list
    # of documents.
    # Hint: For example, investigate the clusters using a bottom-up approach, i.e.,
    # which two documents were clustered first? These have the highest similarity
    # according to their vector representation. (You can identify what was clustered
    # first by looking at the height of the horizontal "merging" connections, the lower
    # these are, the more similar the documents or clusters are to another). Some
    # "interesting" pairs are: 0 and 5 (good), 7 and 8 (weird, why?), 1 and 4 and 9.
    # TODO Let's discuss your findings in the group. So, write down your findings here
    #  (as a comment)

    return model


def print_clustered_docs(model, documents):
    logger.info("printing cluster information for {} documents", len(documents))
    for index in range(len(documents)):
        cluster_label = model.labels_[index]
        doc = documents[index]
        logger.info("{}\t{}", cluster_label, doc)
    logger.info("")


def k_means_clustering(vectors, documents):
    """
    Performs k-means clustering. k-means is a relatively old (1967) technique that is
    due to its intuitiveness and generally "good" results still used quite commonly
    nowadays. As the name k-means suggests, one (you) needs to define a number k so that
    k-means knows how many clusters to form.
    """
    # Let's first define how many cluster we are looking for
    # TODO Set a number. Since you already had a look at the documents we'll cluster
    #  and may have an idea how many topical clusters there are, start with that.
    num_clusters = 4

    # One good thing about scikit-learn is that once you know how to invoke one
    # clustering method, you already know how to invoke any clustering method (though
    # some parameters might be different across different clustering methods, cf.
    # the documentation). Compare the lines of codes here with those in
    # agglomerative_clustering, they are basically identical.
    model = KMeans(n_clusters=num_clusters)

    # Fit the model on our vectors
    model = model.fit(vectors)

    # Unfortunately, there's no visualization readily available. So instead, we'll
    # print the cluster labels, i.e., for each document, which cluster it belongs to.
    # As a free goodie (yippy), I wrote a function that prints for each documents first
    # the cluster label and then the document's text.
    print_clustered_docs(model, documents)

    # TODO execute the file multiple times and change num_clusters (above) each time.
    # With which number do you think the results are best? Note your findings here (as
    # a comment) and let's discuss them later

    return model


def affinity_propagation_clustering(vectors, documents):
    """
    Affinity propagation! This is an optional exercise, and as you can see, there's
    nothing in here. Please add the corresponding code (should be 2 lines for the
    model creation and clustering, and 1 for printing the results)
    """
    # TODO Write three lines of code for clustering and printing the results
    # Hint: Google "scikit learn affinity propagation" and see the other clustering
    # functions we implemented, e.g., k_means_clustering

    # TODO Execute the file and write down: How good is the clustering quality? How
    # does it compare to k-means and agglomerative clustering? Which one would you
    # choose? Why?


def pipeline():
    """
    Invokes each step in our sequence of analysis steps
    """

    # Let's create a list of exemplary documents (each consisting single or multiple
    # sentence. Feel free to add your own.
    my_documents = [
        "The battery of my mobile phone is empty too fast.",
        "This recipe is yummy.",
        "Why the Retail Industry Is Fighting Vaccine Mandates",
        "Computers enable you to do things really fast by automating stuff",
        "Yesterday, I ate chicken wings so today I think it's gonna be something vegetarian.",
        "The iPhone really changed the world but the Internet perhaps even more",
        "Children’s Vaccines Bring Relief to Families in Time for the Holiday Season",
        "CS 101 - the computer science introductory lecture - will start on Monday, November 27. Please make sure to be there on time.",
        "When Can the Covid Masks Finally Come Off?",
        "I like peanuts but I hate fries.",
    ]

    # Convert the documents into a numerical vector representation
    # TODO CMD + Click
    vectors = convert_documents_to_vectors(my_documents)

    # Let's do some clustering. Why not start with agglomerative clustering? :-)
    # TODO CMD + Click
    agglomerative_clustering(vectors)

    # Agglomerative clustering, with its hierarchical structure is quite special. Let's
    # try a "normal" clustering approach. Perhaps you know k-means
    # (https://en.wikipedia.org/wiki/K-means_clustering) which is a very well-known
    # and traditional approach. Let's try that first.
    # TODO CMD + Click
    model = k_means_clustering(vectors, my_documents)

    # Optional:
    # Alright, so k-means requires us to set a value k, i.e., how many clusters we want
    # to retrieve. This can be an advantage (if we know roughly how many clusters there
    # might be in the data, because we can help k-means find good clusters) and also a
    # disadvantage (if we don't know that, because a "wrong" value will lead k-means to
    # find clusters that are not there). Some modern clustering approaches can find
    # clusters without such prior information. Let's try affinity propagation, which
    # I found pretty useful in various projects.
    # TODO CMD + Click
    affinity_propagation_clustering(vectors, my_documents)

    # Last part of this exercise: prediction! Once we trained a clustering method
    # (technically also called a model) we can use that model to predict for new
    # documents, which cluster they belong to. Let's use the k-means clustering model
    # and predict a few sentences
    # TODO: Feel free to add your own sentences here
    my_new_sentences = [
        "I'm a foodie.",
    ]

    # Convert them to vectors
    new_vectors = convert_documents_to_vectors(my_new_sentences)

    # TODO Call the predict function on the new_vectors
    # Hint: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.predict
    # Hint 2: You only need to pass one argument, the variable new_vectors
    cluster_labels = model  # TODO: model.TODO(TODO)

    # Iterate the new sentences and predict their cluster
    for index in range(len(cluster_labels)):
        # Get sentence and cluster label
        new_sentence = my_new_sentences[index]
        cluster_label = cluster_labels[index]

        # Print some info
        logger.info(
            "The cluster according to the k-means model is as follows:\nSentence: {}\nCluster: {}",
            new_sentence,
            cluster_label,
        )

    # TODO Execute the file and check whether the sentences are clustered as you'd
    #  expect (note that you need to scroll up in the console log because the last lines
    #  show the results of affinity_propagation (if you implemented the optional
    #  exercise) in order to compare the cluster label of the new sentences with those
    #  from the k-means execution on our initial set of documents). How well does it
    #  work? Write down your findings.


if __name__ == "__main__":
    pipeline()
