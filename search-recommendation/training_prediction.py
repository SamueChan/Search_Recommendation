# -*- conding:utf-8 -*-

import data_io
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import numpy as np
from scipy import stats
import pandas as pd
import random
from data_io import get_paths
import os
from feature_extraction import extract_features


def process_test_samples(test_samples):
    processed_samples = pd.DataFrame()
    samples_in_one_srch = pd.DataFrame()
    for r_idx, sample in test_samples.iterrows():
        if (r_idx + 1) % 1000 == 0:
            print "Processd %i sample of %i" % (r_idx + 1, test_samples.shape[0])
        is_next_in_same_search = True
        samples_in_one_srch = pd.concat((sample.yo_frame().transpose(), samples_in_one_srch), axis=0)


def training(processed_train_csv_file):
    processed_train_samples = pd.read_csv(processed_train_csv_file)
    processed_train_samples = processed_train_samples.replace([np.inf, -np.inf], np.nan)
    processed_train_samples = processed_train_samples.fillna(value=0)
    processed_train_samples_index_lst = processed_train_samples.index.tolist()
    random.shuffle(processed_train_samples_index_lst)
    shuffled_train_samples = processed_train_samples.ix[processed_train_samples_index_lst]
    col_names = shuffled_train_samples.columns.tolist()
    col_names.remove("booking_bool")
    features = shuffled_train_samples[col_names].values
    labels = shuffled_train_samples["booking_bool"].values

    print "Training Random Forest Classifier"
    rf_classifier = RandomForestClassifier(n_estimators=150, verbose=2, learning_rate=0.1, min_samples_split=10)
    rf_classifier.fit(features, labels)
    print "Saving the Random Forest Classifier"
    data_io.save_model(rf_classifier, model_name="rf_classifier.pkl")

    print "Training Gradient Boosting Classifier"
    gb_classifier = GradientBoostingClassifier(n_estimators=150, verbose=2, learning_rate=0.1, min_samples_split=10)
    gb_classifier.fit(features, labels)
    print "Saving the Gradient Boosting Classifier"
    data_io.save_model(gb_classifier, model_name="gb_classifier.pkl")

    print "Training SGD Classifier"
    sgd_classifier = SGDClassifier(loss="modifier_huber", verbose=2, n_jobs=-1)
    sgd_classifier.fit(features, labels)
    print "Saving the SGD Classifier"
    data_io.save_model(sgd_classifier, model_name="sgd_classifier.pkl")


def prediction(n_train_samples):
    proc_test_samples_file = get_paths()["proc_test_samples_path"]
    if os.path.exists(proc_test_samples_file):
        print "Loading processed test data..."
        new_test_samples = pd.read_csv(proc_test_samples_file)
    else:
        print "Reading test data..."
        test_samples = data_io.read_test()
        test_samples = test_samples.fillna(value=0)
        print "Porcessing test samples"
        new_test_samples = process_test_samples(test_samples)
        new_test_samples.to_csv(proc_test_samples_file, index=None)
    test_feature = new_test_samples.values

    print "Loading the Random Forest Classifier"
    rf_classifier = data_io.load_model(model_name="rf_classifier.pkl")
    print "Random Forest Predicting"
    rf_predictions = rf_classifier.predict_proba(test_feature)[:, 1]

    print "Loading the Gradient Boosting Classifier"
    gb_classifier = data_io.load_model(model_name="gb_classifier.pkl")
    print "Gradient Boosting Predicting"
    gb_predictions = gb_classifier.predict_proba(test_feature)[:, 1]

    print "Loading the SGD Classifier"
    sgd_classifier = data_io.load_model(model_name="sgd_classifier.pkl")
    print "SGD Predicting"
    sgd_predictions = sgd_classifier.predict_proba(test_feature)[:, 1]

    prob_arr = np.vstack((rf_predictions, gb_predictions, sgd_predictions))
    mean_score = np.mean(prob_arr, axis=0)
    mean_score = -1.0 * mean_score

    mean_recommendations = zip(new_test_samples["srch_id"], new_test_samples["prod_id"], mean_score)

    print "Writing predictions to file"
    data_io.write_submission(mean_recommendations, submission_file="mean_result_%i.csv" % n_train_samples)


if __name__ == "__main__":
    n_train_samples = 8930723
    save_train_sample_file = "proc_train_sample_%i.csv" % n_train_samples
    processed_train_csv_file = os.path.join(get_paths()["proc_train_path"], save_train_sample_file)
    training(processed_train_csv_file)
    prediction(n_train_samples)