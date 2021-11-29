"""
In this exercise, we'll learn how to
* use PyCharm and its debugger
* read an Excel file with pandas (a Python library for tabular data, often abbreviated
  as "pd")
* create a new attribute that combines two attributes (we'll add two text attributes)
* perform very basic text cleanup (removal of leading and trailing spaces)

The latter three steps can be subsumed as *parsing*, i.e., we parse a given dataset and
turn it into a data structure that we can further process (in later exercises). Parsing,
thus, is an essential (and usually the first) step in any NLP-based project.
"""

import pandas as pd
from loguru import logger

# TODO scroll to the bottom of this file


def get_full_text_from_row(headline, text):
    """
    This function creates and returns a new text given two texts (headline and text) by
    concatenating them (="combining")
    """
    # TODO Both, headline and text, may contain leading and trailing spaces. Remove
    #  them. Doing so is a best practice in cleaning the data. (FYI: There are of course
    #  more advanced text cleaning techniques, such as removal of "errors" in a given
    #  text. What kind of errors does our Excel file contain and how could we remove
    #  them?
    # Hint: Google "python remove leading and trailing spaces"
    # Hint 2: https://stackoverflow.com/a/959218
    headline = headline  # TODO: headline = headline.TODO
    text  # TODO: text = text.TODO

    # Note that spacy, the NLP library we'll use
    # later, requires that each sentence ends with a period in order to detect it as an
    # individual sentence (this is called sentence segmentation or sentence splitting,
    # a common technique in NLP). See if this is always the case for the headline by
    # setting a breakpoint and debugging the program or by looking at the Excel file.
    # If not, add a period to the headline. (Hint: it is not always the case)
    headline  # TODO: headline = headline + ...

    # TODO: Now, concatenate (=add) both attributes. Be sure to add a space between them
    #  (again, this is required by spacy for sentence segmentation).
    # Hint: In Python, you can concatenate texts by simply using the "+" operator.
    fulltext = headline  # TODO: fulltext = headline + ...

    # TODO CMD + Click on the function name (scroll up a bit) to get back to the
    #  place where this function is invoked.
    return fulltext


def parse_dataset(k=100):
    """
    Reads the Excel file "211018_GroundTruth_SENTIMENT.xlsx" and returns it as a pandas
    dataframe, whereby it truncates (=shortens) the df to k rows. Also, adds a new
    column containing the row's article's fulltext (consisting of headline and main
    text).
    """

    # Before programming anything, have a look at the file we will use. Open
    # "211018_GroundTruth_SENTIMENT.xlsx" in Excel and get a first understanding
    # what this file is about - open the tab "100 Sample", that's the one we will be
    # working with. Briefly think about the following questions:
    # What data does the file contain? How is the data's quality? Are all rows
    # structured identically? If not, which rows differ, how?
    # TODO open the file in Excel, then continue here

    filename = "211018_GroundTruth_SENTIMENT.xlsx"
    sheet = "100 Sample"

    # TODO Load the file as a dataframe using pandas
    # Hint: Use the pd.read_excel function to load the Excel file (pd is an abbreviation
    # for the pandas library). Make sure to use filename and sheet to open the correct
    # file and the correct sheet within.
    # Hint: google "pandas read excel sheet"
    # Hint 2: https://stackoverflow.com/questions/47975866
    df = pd  # TODO: df = pd.read_excel(..., ...=...)

    # Always be nice to your "future" you and your fellows :-) Thus, print some
    # helpful info, for example the number of rows the file contains. Writing proper
    # comments, using human-friendly names for variables and functions, and logging
    # are key requirements to high maintainability, readability, and reusability.
    logger.info("read {} rows from {} ({})", len(df), filename, sheet)

    # TODO set a breakpoint on the previous logger.info row
    #  https://stackoverflow.com/a/53075641
    #  and then debug the current file (right-click anywhere and click on 'Debug
    #  a_parsing'). In future exercises, set breakpoints yourself and debug the file (I
    #  recommend to set a breakpoint on those lines of codes where you are asked to
    #  change or add some code (don't set a breakpoint on an empty line or a comment
    #  because the debugger will not stop at these lines).
    # Once the debugger reaches your breakpoint, it will stop execution and you can,
    # for example, use the Debug view (it will popup automatically) to inspect the
    # values of your variable.
    # TODO investigate the content of df by clicking on "View as DataFrame"
    #  on the right (just have a quick look, no need for a detailed examination)
    # You will notice that the df contains the Excel file's content.

    # When looking at the file in Excel earlier, you might have noticed that only the
    # first 100 rows contain a value in the column "TONE". Since we will need this
    # attribute later in the workshop, truncate (=shorten) the dataframe to the value k
    # (which is a parameter of the function parse_dataset that defaults to 100)
    df = df[:k]

    # Log the new number of rows
    logger.info("truncated to {} rows", len(df))

    # Each row in the Excel file represents a single article's attributes, including
    # its headline, its maintext, and some other information. We are interested in the
    # headline (column name "headline 1") and maintext (column name "text"). Lets
    # concatenate (=add) them so that we can later analyze them as one attribute. Since
    # this workshop is not about how pandas works, here's some prepared code. Have a
    # look at the function get_full_text_from_row and understand what it does.

    # Iterate all rows in df
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
    # TODO CMD + Click on the function name (CTRL on Windows/Linux)
    parse_dataset()
