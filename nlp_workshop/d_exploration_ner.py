"""
In this exercise, we'll learn
* how to use named entity recognition (NER) to retrieve GPEs (countries, etc.)
* find the most frequently mentioned GPEs in a dataset
* find out what they "do"
* filter & convert data to get what you want from spacy (persons and their actions)

Having a question, e.g., "Which GPEs are most frequently mentioned in my dataset, and
what actions do they perform?", and knowing which NLP-technique(s) to use is often not
sufficient to solve the task. Especially in exploratory text analysis (and perhaps in
any other use case as well), a large part of the things we'll need to do to get the job
done is filtering for relevant data and converting it from one structure into another.
This exercise broadly covers all of these challenges :-) Specifically, we'll try to find
out what the most frequent GPEs are and what they do - without ever reading any of
the actual articles, so this could easily be done on 1000s of articles! Sounds nice?
Let's do it!
"""
from collections import Counter, defaultdict

from solutions.a_parsing import parse_dataset
from solutions.c_preprocessing import preprocess_dataset
from util import prepare_system_to_run_from_solutions_folder

from loguru import logger


# TODO scroll to pipeline()


def get_gpes_and_mentions(docs):
    """
    Given a set of docs, retrieves all GPEs and their mentions. Returns a list sorted
    by the frequency of the GPE (most frequent GPEs first), e.g.,
    [
        (Switzerland, [Switzerland, Switzerland, Switzerland]),
        (USA, [USA, USA]),
        ...
    ]
    """
    # This variable will hold all GPE names and for each GPE name all of its mentions.
    gpe_name_and_mentions = defaultdict(list)
    # *Nerd info* regarding why we use a defaultdict (not required to read but for those
    # interested - or feel free to skip):
    # defaultdict is a special python object that behaves like a dict BUT additionally
    # allows us to access elements that we haven't added yet. In such case, a normal
    # dict would crash the program (giving us a KeyError because we tried to access
    # something that is not there) but the defaultdict instead quietly creates a list
    # for us as the value of the accessed value. Sounds complicated? Using a defaultdict
    # allows us to do this:
    # gpe_name_and_mentions = defaultdict(list)
    # gpe_name_and_mentions["notaddedyet"].append("hi)
    # Whereas in a regular dict, we would have to do this
    # gpe_name_and_mentions = dict()
    # if not "notaddedyet" in gpe_name_and_mentions:
    #     gpe_name_and_mentions["notaddedyet"] = []
    # gpe_name_and_mentions["notaddedyet"].append("hi")
    # More information for after the workshop can be found in the Python documentation.

    # Iterate all docs (because we want to find entities in all docs)
    for doc in docs:
        # As you might remember from the b_nlpplayground.py exercise, doc.ents lets us
        # access all named entities found in a document. Iterate all such entities
        for ent in doc.ents:
            # TODO Check whether the type of the entity is a geopolitical entity (2
            #  lines of code)
            # Hint 1: As https://spacy.io/usage/spacy-101#annotations-ner shows (search
            # for "named entity", the type we are looking for is "GPE"
            # Hint 2: As the same website shows, the type can be accessed by using
            # "ent.label_"
            entity_type = ...  # TODO
            if entity_type == "TODO":  # TODO
                # Add the current entity mention to the list of all mentions of this
                # entity
                gpe_name_and_mentions[ent.text].append(ent)

    # Sort the GPEs by how often they occur. FYI for the more advanced Python devs, feel
    # free to examine how this function works, for the others you may just skip this
    # function.
    gpe_name_and_mentions_sorted_by_frequency = sorted(
        gpe_name_and_mentions.items(), key=lambda item: len(item[1]), reverse=True
    )

    # Print some info
    logger.info("found {} GPEs in {} docs", len(gpe_name_and_mentions), len(docs))

    # TODO Set a breakpoint on the logger.info(...) line and see what the content of
    #  gpe_name_and_mentions is. What is its structure (it's a dict of ...)? Also note
    #  how gpe_name_and_mentions_sorted_by_frequency is sorted by the length of the
    #  lists (containing the GPEs' mentions). What is the structure of
    #  gpe_name_and_mentions_sorted_by_frequency (it's a list of ...)?

    return gpe_name_and_mentions_sorted_by_frequency


def find_actions_of_gpe(gpe_mentions):
    """
    Finds actions performed by GPE. Actually, the function doesn't really find the
    actions performed by the GPE but just looks which verbs do co-occur with the GPE's
    mentions. Let's discuss later on how this could be improved :-)
    """
    # This will hold all verbs that occur in the any sentence where our current GPE
    # occurs
    verbs_cooccurring_with_gpe_mentions = []

    # Iterate all mentions of the GPE
    for gpe_mention in gpe_mentions:
        # Now, we need to investigate for the current mention, which verbs are in the
        # same sentence.
        # TODO Get the sentence of the current GPE mention
        # Hint: You can get the sentence of a GPE mention (which is called a Span in
        # spacy's terminology, as it spans multiple individual tokens) using the
        # sent attribute of the gpe_mention
        sentence = ...  # TODO: sentence_of_mention = gpe_mention...

        # Now that we have the current GPE's sentence, iterate all the sentence's tokens
        for token in sentence:
            # ... and for each token, check what its part of speech is
            # TODO Get the part of speech of the current token
            # Hint: We already did this in b_nlpplayground, line 23
            # Hint 2: Use the pos_ attribute
            pos_of_token = ...

            # TODO Check if pos_of_token is a verb
            # Hint: As stated on the spacy website, spacy uses a specific set of POS
            # labels and provides a link to this website where you can find
            # what the label of a verb is: https://universaldependencies.org/u/pos/
            if pos_of_token == "TODO":
                # Get the lemmatized version of the verb
                lemmatized_verb = token.lemma_

                # Save the lemmatized verb
                verbs_cooccurring_with_gpe_mentions.append(lemmatized_verb)

                # TODO Set a breakpoint on all_verbs_cooccurring_with_gpe.append(...)
                #  and look at the values of token and lemmatized_verb. What is the
                #  difference between a verb and its lemmatized version? (You may use
                #  the debugger's "Resume" button (on the left side) to resume execution
                #  until the breakpoint is reached again.

    # Finally, let's sort the verbs by how often they occur and get only the 10 most
    # frequent ones.
    verbs_sorted_by_frequency = Counter(
        verbs_cooccurring_with_gpe_mentions
    ).most_common(10)

    return verbs_sorted_by_frequency


def find_most_frequent_gpes_and_what_they_do(docs):
    """
    As the function's name suggests, this function gets the most frequent GPEs and
    lists for each of them the actions performed by it.
    """
    # Get a list of all GPEs and their mentions. Note the list is sorted by how frequent
    # the GPEs are
    gpes_and_mentions = get_gpes_and_mentions(docs)

    # We are interested in only the most frequent GPEs, so let's keep only the top 10
    gpes_and_mentions = gpes_and_mentions[:10]

    # Alright, so at this point in time, we know what the most frequent GPEs are. So,
    # let's find out what each of them is doing (according to the articles). We'll do
    # this for one GPE after another.

    # Iterate the GPEs and their mentions (to handle one GPE after another)
    for one_gpe_and_its_mentions in gpes_and_mentions:
        gpe_text = one_gpe_and_its_mentions[0]
        gpe_mentions = one_gpe_and_its_mentions[1]

        # For the current GPE, let's find all its actions
        # TODO CMD+Click
        verbs = find_actions_of_gpe(gpe_mentions)

        logger.info("{}: {}", gpe_text, verbs)

    # TODO Execute (no need to debug) the program, look at the output in the console.
    #  What are the most frequent GPEs? And what do they do? Are there duplicate GPEs?
    #  Are there verbs that are not really meaningful for your text exploration?
    #  Let's discuss.


def pipeline():
    """
    Invokes each step in our sequence of analysis steps
    """
    # Parse the Excel file.
    df = parse_dataset()

    # Preprocess the dataset using the function you wrote earlier to get spacy documents
    docs = preprocess_dataset(df)

    # Let's find out what the most frequently mentioned persons in our dataset are, and
    # what they do. Here's roughly how we're going to achieve that.

    # * We'll use spacy to retrieve all geopolitical entities (named entities of
    #   type GPE, i.e., countries, cities, and states, more info for after the workshop
    #   can be found at https://spacy.io/usage/spacy-101#annotations-ner)
    # * Count the entities
    # * Get only the most frequent ones
    # * Only for these GPEs, get their actions (we'll do that by looking up the
    #   sentences where the GPEs occur and retrieving the verbs in such sentences)
    # TODO CMD+Click
    find_most_frequent_gpes_and_what_they_do(docs)


if __name__ == "__main__":
    pipeline()
