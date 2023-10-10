from pathlib import Path

import polars as pl
import json

if __name__ == "__main__":
    results_dir = Path("results")

    results = []
    for result in results_dir.glob("*"):
        datasets_df = pl.read_parquet(result / "datasets_results.parquet")
        metadata = json.load((result / "metadata.json").open())
        metadata["timestamp"] = result.name
        json.dump(
            metadata, (result / "metadata.json").open("w"), indent=4, sort_keys=True
        )
        result = {
            "model": metadata["model"],
            "model_seed": metadata["model_seed"],
            "dataset_seed": metadata["dataset_seed"],
            "score": datasets_df["score"].sum(),
            "scores": datasets_df.group_by("dataset").agg(pl.sum("score")),
            "number_of_samples": metadata["number_of_samples"],
            "run_folder_name": result.name,
        }
        p = 0
