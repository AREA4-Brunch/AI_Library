"""
    In case you choose to rename generated models' files
    this might come in handy.
"""

from fileinput import filename
import os
import re

PATH_CWD = os.path.dirname(os.path.abspath(__file__))
PATH_DATASETS = os.path.join(PATH_CWD, "Mini_Library/_datasets/")
PATH_MODELS = os.path.join(PATH_CWD, "Mini_Library/_models")


for file_name in os.listdir(PATH_MODELS):
    print(file_name)

    x = re.search(r"(\d)_e=(\d.\d*)_lr=(\d*)(_?[a-z]*.pkl)", file_name)
    model_name = x.group(1)
    e = x.group(2)
    lr = x.group(3)
    rest = x.group(4)
    new_name = "{}_e={}_lr={}{}".format(model_name, e, lr, rest)

    print(new_name)
    os.rename(os.path.join(PATH_MODELS, file_name),
              os.path.join(PATH_MODELS, new_name))
