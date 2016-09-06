# -*- conding:utf-8 -*-

import csv
from operator import itemgetter
import os
import json
import pickle
import pandas as pd


def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths


def read_train(nrows=None):
    train_path = get_paths()["train_path"]
    # get column names
    col_names = pd.read_csv(train_path, nrows=1).columns.tolist()
    # extract useful column
    col_names.remove("click_bool")
    col_names.remove("gross_bookings_usd")
    # col_names.remove("data_time")
    col_names.remove("position")
    if nrows == None:
        train_samples = pd.read_csv(train_path, usecols=col_names)
    else:
        train_samples = pd.read_csv(train_path, useclos=col_names, nrows=nrows)
    return train_samples


def read_test(nrows=None):
    test_path = get_paths()["test_path"]
    # get column names
    col_names = pd.read_csv(test_path, nrows=1).columns.tolist()
    # extract useful column
    col_names.remove("gross_bookings_usd")
    col_names.remove("data_time")
    col_names.remove("position")
    if nrows == None:
        test_samples = pd.read_csv(test_path, usecols=col_names)
    else:
        test_samples = pd.read_csv(test_path, usecols=col_names, nrows=nrows)
    return test_samples


def save_model(model_name=None):
    if model_name is None:
        in_path = get_paths()["model_path"]
    else:
        path, _ = os.path.split(get_paths()["model_path"])
        in_path = os.path.join(path, model_name)
    return pickle.dump(open(in_path))


def load_model(model_name=None):
    if model_name is None:
        in_path = get_paths()["model_path"]
    else:
        path, _ = os.path.split(get_paths()["model_path"])
        in_path = os.path.join(path, model_name)
    return pickle.load(open(in_path))


def write_submission(recommendations, submission_file=None):
    if submission_file is None:
        submission_path = get_paths()["submission_path"]
    else:
        path, file_name = os.path.split(get_paths()["submission_path"])
        submission_path = os.path.join(path, submission_file)
    path, _ = os.path.split(submission_path)
    if not os.path.exists(path):
        os.makedirs(path)
    rows = [(int(srch_id), int(prod_id))
            for srch_id, prod_id, rank_float
            in sorted(recommendations, key=itemgetter(0, 2))]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerow(["TARGET"])
    writer.writerows(rows)
