import pandas as pd
from loguru import logger

from util import prepare_system_to_run_from_solutions_folder

# TODO scroll to the bottom of this file


def get_full_text_from_row(headline, text):
    # TODO Both, headline and text, may contain leading and trailing spaces. Remove
    #  them. https://stackoverflow.com/a/959218
    # headline = headline.TODO
    # text = ...
    headline = headline.strip()
    text = text.strip()

    # Note that spacy, the NLP library we'll use
    # later, requires that each sentence ends with a period in order to detect it as an
    # individual sentence (this is called sentence segmentation or sentence splitting,
    # a common technique in NLP). See if this is always the case for the headline by
    # setting a breakpoint and debugging the program. If not, add a period to the
    # headline.
    # headline = headline + TODO
    headline = headline + "."

    # Now, concatenate both attributes. Be sure to add a space between them (again, for
    # sentence segmentation).
    # fulltext = headline + ...
    fulltext = headline + " " + text

    # TODO CMD + Click on the function name (scroll up a bit) to get back to the
    #  place where this function is invoked.
    return fulltext


def parse_dataset(k=100):
    """
    Reads the Excel file "211018_GroundTruth_SENTIMENT.xlsx" and returns it as a pandas
    dataframe, whereby it truncates the df to k rows. Also, adds a new column containing
    the row's article's fulltext (consisting of headline and main text).
    :param k:
    :return:
    """

    # Before programming anything, have a look at the file we will use. Open
    # "211018_GroundTruth_SENTIMENT.xlsx" in Excel and get a first understanding
    # what this file is about. Briefly think about the following questions:
    # What data does the file contain? How is the data's quality? Are all rows
    # structured identically? If not, which rows differ, how?
    # TODO open the file in Excel, then continue here

    filename = "211018_GroundTruth_SENTIMENT.xlsx"

    # TODO Load the file as a dataframe using pandas
    # Hint: Use the pd.read_excel function to load the Excel file (pd is an abbreviation
    # for the pandas library). Make sure to use the parameter sheet_name to open the
    # right Excel tab.
    # https://stackoverflow.com/questions/47975866
    # df = pd.TODO
    df = pd.read_excel(filename, sheet_name="100 Sample")

    # Always be nice to your "future" you and your fellows :-) Thus, print some
    # helpful info, for example the number of rows the file contains. Writing proper
    # comments, using human-friendly names for variables and functions, and logging
    # are key requirements to high maintainability, readability, and reusability.
    logger.info("read {} rows from {}", len(df), filename)

    # TODO set a breakpoint on the previous logger.info row
    #  https://stackoverflow.com/a/53075641
    #  and then debug the current file (right-click anywhere and click on 'Debug
    #  workshop').
    # Once the debugger reaches your breakpoint, it will stop execution and you can,
    # for example, use the Debug view (it will popup automatically) to inspect the
    # values of your variable.
    # TODO investigate the content of df by clicking on "View as DataFrame"
    #  on the right (just have a quick look, no need for a detailed examination)

    # When looking at the file in Excel earlier, you might have noticed that only the
    # first 100 rows contain a value in the column "TONE". Since we will need this
    # attribute later in the workshop, truncate the dataframe to the value k (which
    # is a parameter of the function parse_dataset that defaults to 100)
    # Hint: Look for "items from the beginning through stop-1" in
    # https://stackoverflow.com/a/509295
    # TODO df = df <-- add something at the end here
    df = df[:k]

    # Log the new number of rows
    logger.info("truncated to {} rows", len(df))

    # Each row in the Excel file represents a single article's attributes, including
    # its headline, its maintext, and some other information. We are interested in the
    # headline (column name "headline 1") and maintext (column name "text"). Lets
    # concatenate them so that we can later analyze them as one attribute. Since this
    # workshop is not about how pandas works, here's some prepared code. Have a look
    # at the function get_full_text_from_row

    # iterate all rows in df
    for index, row in df.iterrows():
        # build the full text
        # TODO Hold CMD and click on the function name (CTRL on Windows/Linux) to
        #  go to the function
        fulltext = get_full_text_from_row(row["headline 1"], row["text"])
        # set the fulltext as a new value in the row
        df.at[index, "fulltext"] = fulltext

    logger.info("parsing completed")

    # now that we're finished parsing the Excel file, return the df
    return df


# entry point / Python's start routine
if __name__ == "__main__":
    prepare_system_to_run_from_solutions_folder()

    """
    In this exercise, we'll read an Excel file and turn it into a dataset structure. 
    We'll use the pandas dataframe for the latter.
    """
    # TODO CMD + Click on the function name (CTRL on Windows/Linux)
    parse_dataset()
