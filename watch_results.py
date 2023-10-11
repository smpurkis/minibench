import json

import polars as pl

if __name__ == "__main__":
    results_df = pl.read_parquet("results/results.parquet")
    # dataset_scores = results_df.get_column("scores").to_list()
    dataset_scores = results_df.get_column("normalized_scores").to_list()
    dataset_scores_dict = {}
    for key in dataset_scores[0].keys():
        key = key.split("-")[0]
        dataset_scores_dict[key] = []
    for dataset_score in dataset_scores:
        for key, value in dataset_score.items():
            key = key.split("-")[0]
            dataset_scores_dict[key].append(value)
    dataset_scores_df = pl.DataFrame(dataset_scores_dict)
    dataset_scores_df.glimpse()
