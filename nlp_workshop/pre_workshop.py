"""
Hi! Glad that you're here :-)

This file contains some information about how to prepare your computer for the workshop
as well as the files contained in the zip folder.

---- What's in the zip folder I just extracted?
Basically four parts:
- pre_workshop.py: This is the only file that is important prior to the workshop. It
    contains setup instructions. For now, you can ignore all other files as we will need
    them only during the workshop.
- 211018_GroundTruth_SENTIMENT.xlsx: A real-world data file that we'll use throughout
    the workshop. Thanks to Nahema Marchal and the team of the prodigi project for
    providing this file!
- Multiple *.py files: The exercises and demos that we'll work on in the workshop.
- solution folder: Containing one solved python file for each exercise or demo


---- What do I need to do prior to the workshop?
Prior to the workshop, please follow the instructions and commands below (most of them
can just be copy-pasted into your CLI / shell and they all should work out of the
box).

Because we'll learn and apply a variety of techniques in the workshop, the day
of the workshop will be quite packed. Thus, please make sure that your Python
environment is up and running *before* the workshop.

In case you run into any issues, I'm happy to help. Please just send me an email
(felix.hamborg@uni-konstanz.de, I speak German and English). If possible, I would
recommend that you first try to reproduce and/or fix the issue with a colleague who uses
the same operating system, which is typically easier and faster compared to remotely
investigating and fixing issues.

What are we going to install? The tools installed in steps 1 and 2 is commonly used
among computer science researchers, software developers, etc., as they provide
convenient solutions and functionality to tasks and issues that are common in everyday
programming. I hope that you may find them beneficial for your own work, too, and maybe
continue to use them after the workshop. In case not, don't worry! You can simply
uninstall them after the workshop, and any software installed will be gone from your
computer :-)

Please follow these instructions prior to the workshop! For steps 1 - 4, there's also
a video tutorial that I recorded and that you can watch additionally:
Video Part 1: https://www.youtube.com/watch?v=Sh7DhzWoCy0

1) Install PyCharm: You can download it here https://www.jetbrains.com/pycharm/download
   Select the Community version (if you already have a license you may also select the
   Professional version, but Community offers everything we'll need for the workshop).

   What is PyCharm? PyCharm is the IDE (Integrated Development Environment) we'll use
   (in case you're familiar with only Jupyter notebooks: think of an IDE as a strongly
   advanced version of such Jupyter).

2) Install Anaconda: You can download it here https://www.anaconda.com/products/individual

   What is it and why do I need it? Anaconda manages your Python environments. You may
   have installed Python already on your computer, so why do you need multiple environments
   or even another software to manage them?! There're are many benefits of using a
   manager such as Anaconda, one of the most important ones is that some Python packages
   (you will install a few later) require a specific version of other Python packages.
   While this may be fine if you have only one project that you work on, as soon as you
   work on multiple projects there's an increased chance of version conflicts (one package
   requires version 1 of package XY, whereas another package requires version 2 of package
   XY). So, one good practice is to create a individual, isolated Python environment for
   each of your projects.

3) Open a terminal / CLI / shell (if you don't know how to do that, PyCharm also has a
   Terminal, see https://www.jetbrains.com/help/pycharm/terminal-emulator.html - in that
   case first follow steps 5-7) or just Google your operating system's name and terminal).
   On Windows (or in general, if you get an error that the command/program "conda" is not known or
   recognized when executing the first command from below), you should use the Anaconda Prompt
   to execute all these commands. This program will be installed during step 2.

4) Create and prepare the Python environment we'll use in the workshop. You can simply copy
   and paste each of the following lines (one after the after, and please double check
   that you don't forget a line - the order of the following commands is important!). In case
   you run into a problem, e.g., forgot a line, you can delete the environment using the
   following two commands (only execute them if needed, otherwise just ignore them):
### IGNORE ME
conda activate base
conda remove --name nlpworkshop --all --yes
### END IGNORE ME

### The following commands create and prepare the Python environment. Some might take
### a few moments but none of them should require any manual interaction from you.
conda create --yes -n nlpworkshop python=3.7
conda activate nlpworkshop
conda install -y -c conda-forge tqdm loguru
conda install -y -c anaconda pandas xlrd openpyxl
pip install spacy
python -m spacy download en_core_web_lg
conda install --yes pandas scikit-learn
pip install "transformers>=3.1.0,<4"
conda install --yes "pytorch=1.7.1" torchvision -c pytorch
pip install matplotlib
### END OF COMMANDS FOR THE TERMINAL

### UPDATE ###
conda activate nlpworkshop
conda install --yes "pytorch=1.7.1" torchvision -c pytorch
pip install matplotlib
### END UPDATE ###


5) Follow the video instructions of video part 2:
https://www.youtube.com/watch?v=1_49B16IuPw

Note that since the video was recorded a few things were updated:

* In the video, I mention that the command "python -m spacy download en_core_web_lg" may
  fail and I also mention an alternative (i.e., to install the en_core_web_lg model
  instead). This issue has been solved so you can simply ignore that part (and thus I
  also removed the alternative from the list of commands above)

* For Windows users: the "where python" command that I mention in the video may list
  multiple paths to python.exe files. The one you're looking for most likely looks
  something like this:
  "C:\ProgramData\Anaconda3\envs\nlpworkshop\python.exe" (Note that in contrast to what
  I said in the video - it seems that on Windows no "bin" folder is part of this path,
  which is also fine. You can identify the correct path by checking if it contains
  "Anaconda3", "envs", and "nlpworkshop")

"""


if __name__ == '__main__':
    from solutions.j_evaluation import evaluate
    evaluate()

    from loguru import logger
    logger.info("Congratulations, you are now on the way to becoming an NLP expert")

