import streamlit as st
import polars as pl
from pathlib import Path
from run_bench import save_overall_results, get_datasets, calculate_category_scores
import json


def load_results():
    return pl.read_parquet(Path("results/results.parquet")).sort("score").reverse()


def process_results(results_df, columns):
    row_dicts = results_df.select(columns).to_dicts()
    for row in row_dicts:
        for d in sorted(list(row[columns[-1]])):
            row[d] = row[columns[-1]].get(d)
        del row[columns[-1]]
    return pl.DataFrame(row_dicts)


def load_and_process_results():
    results_folder = Path("results")
    results_df = pl.DataFrame()
    results_folders = [f for f in Path(results_folder).iterdir() if not f.is_file()]
    for result_folder in results_folders:
        metadata = json.load(open(result_folder / "metadata.json", "r"))
        datasets_df = pl.read_parquet(result_folder / "datasets_results.parquet")
        number_of_samples = metadata["number_of_samples"]
        datasets = get_datasets(number_of_samples, metadata["dataset_seed"])
        scores = {
            r["dataset"]: r["score"]
            for r in datasets_df.group_by("dataset")
            .agg(pl.sum("score").alias("score"))
            .to_dicts()
        }
        normalized_scores = {k: v / number_of_samples for k, v in scores.items()}
        category_scores = calculate_category_scores(datasets_df)
        normalized_category_scores = {
            k: v / (len([d for d in datasets if d.category == k]) * number_of_samples)
            for k, v in category_scores.items()
        }
        score = sum(scores.values())
        normalized_score = score / (len(metadata["datasets"]) * number_of_samples)
        result = {
            "model": metadata["model"],
            "model_seed": metadata["model_seed"],
            "dataset_seed": metadata["dataset_seed"],
            "score": score,
            "normalized_score": normalized_score,
            "scores": scores,
            "normalized_scores": normalized_scores,
            "category_scores": category_scores,
            "normalized_category_scores": normalized_category_scores,
            "number_of_samples": number_of_samples,
            "run_folder_name": result_folder.name,
        }
        results_df = pl.DataFrame([*results_df.to_dicts(), result])
    return results_df.sort("normalized_score").reverse()


st.title("Results Analysis")

fig = save_overall_results(results_folder=Path("results"), method="plotly")
st.plotly_chart(fig)

results_df = load_results()
st.table(results_df.select(["model", "normalized_score", "score"]))

scores_df = process_results(
    results_df, ["model", "normalized_score", "normalized_category_scores"]
)
st.table(scores_df)

scores_df = process_results(
    results_df, ["model", "normalized_score", "score", "normalized_scores"]
)
st.table(scores_df)

results_df = load_and_process_results()
st.table(results_df.select(["model", "normalized_score", "score"]))
