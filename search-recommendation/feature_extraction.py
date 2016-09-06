# -*- coding:utf-8 -*-

def normalize_series(pd_series):
    norm_pd_series = (pd_series - pd_series.mean()) / (pd_series.max() - pd_series.min())
    norm_pd_series = norm_pd_series.fillna(value=0)
    return norm_pd_series

def extract_features(samples_in_one_srch):
    price_usd_series = samples_in_one_srch["price_usd"]
    norm_price_usd_series = normalize_series(price_usd_series)
    mean_price_usd_series = price_usd_series.mean()
    std_price_usd_series = price_usd_series.std()
    median_price_usd_series = price_usd_series.median()

    orig_destination_distance_series = samples_in_one_srch["orig_destination_distance"]
    norm_orig_destination_distance_series = normalize_series(orig_destination_distance_series)
    mean_orig_destination_distance_series = orig_destination_distance_series.mean()
    std_orig_destination_distance_series = orig_destination_distance_series.std()
    median_orig_destination_distance_series = orig_destination_distance_series.median()

    ext_samples_in_one_srch = samples_in_one_srch.copy()
    ext_samples_in_one_srch["price_usd"] = norm_price_usd_series
    ext_samples_in_one_srch["orig_destination_distance"] = norm_price_usd_series

    ext_samples_in_one_srch["price_usd_mean"] = mean_price_usd_series
    ext_samples_in_one_srch["price_usd_std"] = std_price_usd_series
    ext_samples_in_one_srch["price_usd_median"] = median_price_usd_series

    ext_samples_in_one_srch["orig_destination_distance_mean"] = mean_orig_destination_distance_series
    ext_samples_in_one_srch["orig_destination_distance_std"] = std_orig_destination_distance_series
    ext_samples_in_one_srch["orig_destination_distance_median"] = median_orig_destination_distance_series

    return ext_samples_in_one_srch
