import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


def prepare_system_to_run_from_solutions_folder():
    os.chdir("..")


def plot_dendrogram(model, **kwargs):
    """
    Plots a "beautiful" (=better than using the debugger's explore functions :-) )
    dendrogram.
    :return:
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    plt.show()


def get_platforms(df):
    platforms = []
    for index, row in df.iterrows():
        platform = row["Platform"]
        platforms.append(platform)

    return platforms


def get_tone_vectors(df):
    tones = []
    for index, row in df.iterrows():
        platform = row["TONE"]
        tones.append(platform)

    unique_tones = set(tones)
    # Convert this back to a list (reason: the list still contains only the unique
    # values but in contrast to the set it has an order - we will exploit this property
    # when we create the vectors)
    unique_tones = list(unique_tones)
    # get the number of unique tones (We will create vectors of the same size for
    # all docs)
    num_tones = len(unique_tones)

    # Create an empty list to store each doc's platform_vector
    tone_vectors = []

    # Iterate tones (the list of the docs' tones)
    for tone in tones:
        # TODO create the current doc's tone vector and initialize it with as many
        #  0s as we have tones
        # Hint: We need all vectors to be of size num_platforms.
        # Hint 2: https://stackoverflow.com/a/8528626
        # platform_vector = ...
        tone_vector = [0] * num_tones

        # TODO Find out which scalar represents the current doc's tone
        # Hint: Use the index(...) function of the unique_platforms list
        # Google for "python list index"
        # platform_index = ...
        tone_index = unique_tones.index(platform)

        # TODO Set the corresponding scalar to 1
        # Hint: in the platform_vector using the platform_index
        # platform_vector[...
        tone_vector[tone_index] = 1

        # Add the platform_vector to the list
        tone_vectors.append(tone_vector)

    return tone_vectors
