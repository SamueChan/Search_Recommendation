# -*- coding:utf-8 -*-

import data_io
import pandas as pd
import numpy as np
import os
from data_io import get_paths
from feature_extraction import extract_features


def process_train_samples(samples, max_srch_size=10, each_saved_size=1000000):
    sorted_samples = samples.sorted_values(by=["srch_id"])
    sorted_samples = sorted_samples.reset_index(drop=True)
    samples_in_one_srch = pd.DataFrame()
    for r_idx, sample in sorted_samples.iterrows():
        if (r_idx + 1) % 1000 == 0:
            print "Processed %i sample of %i" % (r_idx + 1, sorted_samples.shape[0])
        is_next_in_same_search = True
        samples_in_one_srch = pd.concat((sample.to_frame().transpose(), samples_in_one_srch), axis=0)
        current_srch_id = sample["srch_id"]
        if (r_idx + 1) == sorted_samples.shape[0]:
            is_next_in_same_search == False
        else:
            next_srch_id = sorted_samples["srch_id"][r_idx + 1]
            if current_srch_id != next_srch_id:
                is_next_in_same_search = False
        if not is_next_in_same_search:
            ext_samples_in_one_srch = extract_features(samples_in_one_srch)
            n_samples = ext_samples_in_one_srch.shape[0]
            if n_samples > max_srch_size:
                if np.any(ext_samples_in_one_srch["bookings_bool"]):
                    pos_samples = ext_samples_in_one_srch[ext_samples_in_one_srch["booking_bool"] == 1]
                    neg_samples = ext_samples_in_one_srch[ext_samples_in_one_srch["booking_bool"] == 0]
                    selected_neg_samples = neg_samples.samples(n=max_srch_size - pos_samples.shape[0])
                    selected_samples = pd.concat((pos_samples, selected_neg_samples), axis=0)
                else:
                    selected_samples = ext_samples_in_one_srch.sample(n=max_srch_size)
            else:
                selected_samples = ext_samples_in_one_srch.copy()
            processed_samples = pd.concat((processed_samples, selected_samples), axis=0)
            samples_in_one_srch = pd.DataFrame()
        if (r_idx + 1) % each_saved_size == 0:
            save_file_name = "proc_train_samples_%i.csv" % (r_idx + 1)
            save_path = get_paths()["proc_train_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if np.any(np.isnan(processd_samples.values)):
                processd_samples = processd_samples.fillna(value=0)
            processd_samples.to_csv(os.path.join(save_path, save_file_name), index=None)
            save_file_name = "proc_train_samples%i.csv" % (r_idx + 1)
            save_path = get_paths()["proc_train_path"]
            if np.any(np.isnan(processd_samples.values)):
                processd_samples = processd_samples.fillna(value=0)
            processd_samples.to_csv(os.path.join(save_path, save_file_name), index=None)


def do_train_samples_processing():
    print "Reading training data..."
    train_samples = data_io.read_train()
    print "Processing training data..."
    train_samples = train_samples.fillna(value=0)
    process_train_samples(train_samples)


if __name__ == "__main__":
    do_train_samples_processing()
