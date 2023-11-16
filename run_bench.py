import argparse
import json
import os
from datetime import datetime
from functools import cache
from multiprocessing.dummy import Pool
from pathlib import Path
from time import sleep, time
from typing import Any, Optional

import openai
import polars as pl
from datasets import load_dataset
from llama_cpp import Llama
from tqdm.auto import tqdm
from os import PathLike

from Datasets.Dataset import CACHE_DATASET_PATH, Dataset, DatasetMeta
from Datasets.plot import make_radar_chart
from Datasets.ProcessRows import (
    process_row_agieval,
    process_row_anli,
    process_row_arc,
    process_row_cosmoqa,
    process_row_hellaswag,
    process_row_mathqa,
    process_row_mmlu,
    process_row_race,
    process_row_truthfulqa,
    process_row_winogrande,
)

# from threading import Thread
# from save_thread_result import ThreadWithResult as Thread
# from multiprocessing.pool import Pool


"""
dataset categories:
Knowledge
    Question Answering
        Natural Questions N
        WebQuestions N
        TriviaQA N
    Multi-subject Test
        MMLU Y
        AGIEval-EN Y 
        ARC-e Y
        ARC-c Y
Reasoning
    Commonsense Reasoning
        Anli Y
        Cosmos-QA Y
        LAMBADA N
        HellaSwag Y
        WinoGrande Y
Comprehension
    Reading Comprehension
        RACE-m Y
        RACE-h Y
        DROP N
Math
    Mathmatical Reasoning
        MATHQA Y
Safety
    Truthfulness
        TruthfulQA Y
    Toxicity
        RealToxicityPrompts N
"""

dataset_metadata = [
    DatasetMeta(
        path="cais/mmlu",
        name="all",
        split="validation",
        process_row_fn=process_row_mmlu,
        category="Knowledge/Multi-subject Test",
    ),  # 1.53k
    # DatasetMeta(
    #     path="baber/agieval",
    #     # name="*",
    #     name="",
    #     split="test",
    #     process_row_fn=process_row_agieval,
    #     category="Knowledge/Multi-subject Test",
    # ),
    DatasetMeta(
        path="ai2_arc",
        name="ARC-Challenge",
        split="test",
        process_row_fn=process_row_arc,
        category="Knowledge/Multi-subject Test",
    ),  # 1.17k
    # DatasetMeta(
    #     path="ai2_arc",
    #     name="ARC-Easy",
    #     split="validation",
    #     process_row_fn=process_row_arc,
    #     category="Knowledge/Multi-subject Test",
    # ),
    DatasetMeta(
        path="anli",
        split="test_r1",
        process_row_fn=process_row_anli,
        category="Reasoning/Commonsense Reasoning",
    ),  # 1k
    DatasetMeta(
        path="cosmos_qa",
        split="validation",
        process_row_fn=process_row_cosmoqa,
        category="Reasoning/Commonsense Reasoning",
    ),
    DatasetMeta(
        path="AlekseyKorshuk/hellaswag",
        name="ARC-Easy",
        split="validation",
        process_row_fn=process_row_hellaswag,
        category="Reasoning/Commonsense Reasoning",
    ),  # 2.99k
    DatasetMeta(
        path="winogrande",
        name="winogrande_s",
        split="validation",
        process_row_fn=process_row_winogrande,
        category="Reasoning/Commonsense Reasoning",
    ),  # 1.27k
    # DatasetMeta(
    #     path="race",
    #     name="middle",
    #     split="validation",
    #     process_row_fn=process_row_race,
    #     category="Comprehension/Reading Comprehension",
    # ),
    DatasetMeta(
        path="race",
        name="high",
        split="validation",
        process_row_fn=process_row_race,
        category="Comprehension/Reading Comprehension",
    ),  # 4.89k
    DatasetMeta(
        path="math_qa",
        split="validation",
        process_row_fn=process_row_mathqa,
        category="Math/Mathmatical Reasoning",
    ),  # 4.48k
    DatasetMeta(
        path="truthful_qa",
        name="multiple_choice",
        split="validation",
        process_row_fn=process_row_truthfulqa,
        category="Safety/Truthfulness",
    ),  # 817
]


os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:5001/v1"
os.environ["OPENAI_API_HOST"] = "http://127.0.0.1:5001"
os.environ["OPENAI_API_KEY"] = "dummy"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]


def save_example_prompt(dataset: Dataset):
    print(f"Loading dataset: {dataset.path}/{dataset.name}")
    dataset.load_dataset()
    prompt = dataset.form_prompt(dataset.df.iloc[0])

    # save to prompts folder
    save_name = f"{dataset.path.split('/')[-1]}_{dataset.name}.txt"
    save_path = Path(f"prompts/{save_name}")
    print(f"Saving prompt to {save_path}")
    save_path.write_text(prompt)


def check_correct_answer(row: pl.Series, completion: str) -> int:
    if len(completion) == 0:
        return 0
    score = completion[0] == str(row["answer"])
    if len(completion) > 1 and score:
        confirm = completion[1] == ")"
        score = score and confirm
    return score


def run_query(llm: Llama, row: pl.Series, prompt: str) -> tuple[str, float]:
    s = time()
    # resp = openai.Completion.create(
    #     engine="",
    #     prompt=prompt,
    #     max_tokens=2,
    #     n=1,
    #     stop=None,
    #     temperature=0.0,
    # )
    # completion = resp.choices[0].text.strip()

    resp = llm(prompt, max_tokens=2, echo=True, temperature=0.0)
    completion = resp["choices"][0]["text"].replace(prompt, "").strip()
    time_taken = time() - s
    score = check_correct_answer(row, completion)
    return completion, score, time_taken


def load_dataset(
    dataset: Dataset,
    cache_dir: str = CACHE_DATASET_PATH,
):
    dataset.load_dataset(cache_dir=cache_dir)


def load_all_datasets(
    datasets: list[Dataset],
    dataset_seed: Optional[int] = None,
    cache_dir: str = CACHE_DATASET_PATH,
):
    s = time()

    # parallel
    # pool = Pool(5)
    # try:
    #     pool.starmap(
    #         load_dataset,
    #         [(dataset, cache_dir) for dataset in datasets],
    #     )
    # except Exception as e:
    #     print(e)
    # pool.close()

    # serial
    for dataset in datasets:
        dataset.load_dataset(cache_dir=cache_dir)
    # print(time() - s)


def start_model_process(model_path: str, model: str, seed: int) -> Llama:
    full_model_path = Path(model_path, model)
    assert full_model_path.exists(), f"Model path {full_model_path} does not exist"
    print(f"Loading model from: {full_model_path}")
    llm = Llama(
        model_path=full_model_path.as_posix(),
        n_gpu_layers=1,
        seed=seed,
        n_ctx=2048,
        verbose=False,
    )
    return llm


def run_bench(
    llm: Llama, model: str, datasets: list[Dataset], number_of_samples: int
) -> int:
    start = time()
    query_time_taken = 0
    count = 0
    pbar = tqdm(total=len(datasets) * number_of_samples)
    for dataset in datasets:
        dtime = time()
        rows = dataset.df.to_dicts()
        rows = [row for row in rows if "completion" not in row]
        for _, row in enumerate(rows):
            prompt = dataset.form_prompt(row)
            completion, score, time_taken = run_query(llm, row, prompt)
            row["prompt"] = prompt
            row["completion"] = completion
            row["model"] = model
            row["time_taken"] = time_taken
            row["score"] = score
            dataset.results_df = pl.DataFrame(rows)
            count += 1
            pbar.update(1)
            query_time_taken += time_taken
        print(f"Dataset: {dataset.path}/{dataset.name}, time taken: {time() - dtime}")
    total_time_taken = time() - start
    print(f"Total time taken: {total_time_taken}, query time taken: {query_time_taken}")

    score = sum([d.results_df.get_column("score").sum() for d in datasets])
    print(score)
    return score


def calculate_category_scores(df: pl.DataFrame) -> dict[str, float]:
    return {
        r["category"]: r["score"]
        for r in df.group_by("category").agg(pl.sum("score").alias("score")).to_dicts()
    }


def calculate_random_category_scores(datasets: list[Dataset]) -> dict[str, float]:
    # sum over each category
    random_category_scores = {}
    for dataset in datasets:
        if dataset.category not in random_category_scores:
            random_category_scores[dataset.category] = []
        random_category_scores[dataset.category].append(dataset.random_guess_score)
    random_category_scores = {
        k: sum(v) / len(v) for k, v in random_category_scores.items()
    }
    return random_category_scores


def get_datasets(number_of_samples: int, dataset_seed: int = 1) -> list[Dataset]:
    datasets = [
        Dataset(
            path=meta.path,
            name=meta.name,
            split=meta.split,
            number_of_samples=number_of_samples,
            process_row_fn=meta.process_row_fn,
            category=meta.category,
            seed=dataset_seed,
        )
        for meta in dataset_metadata
    ]
    # print(f"Loading datasets with seed: {dataset_seed}")
    load_all_datasets(datasets)
    return datasets


def save_overall_results(
    results_folder: Optional[PathLike] = None,
    method: str = "matplotlib",
):
    if results_folder is not None:
        assert Path(
            results_folder
        ).exists(), f"Results folder {results_folder} does not exist"
    # form results_df from all the results so far
    results_df = pl.DataFrame()
    results_folders = [f for f in Path(results_folder).iterdir() if not f.is_file()]
    for result_folder in results_folders:
        metadata = json.load(
            open(result_folder / "metadata.json", "r"),
        )
        datasets_df = pl.read_parquet(
            result_folder / "datasets_results.parquet",
        )
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
    data_list = [
        calculate_random_category_scores(datasets),
        *results_df["normalized_category_scores"].to_list(),
    ]
    labels = ["random", *results_df["model"].to_list()]
    if result_folder is not None:
        results_df.write_parquet(results_folder / "results.parquet")
    return make_radar_chart(
        data_list=data_list,
        labels=labels,
        save_path=results_folder / "radar_chart" if method == "matplotlib" else None,
        method=method,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        help="comma separated, e.g. EleutherAI/gpt-neo-2.7B,EleutherAI/gpt-neo-2.7B",
        required=True,
    )
    parser.add_argument(
        "--model_path", type=str, help="path to model, e.g. /home/models", required=True
    )
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--dataset_seeds", type=str, default="1")
    parser.add_argument("--number_of_samples", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DATASET_PATH)
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--results_location", type=str, default="results")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    number_of_samples = args.number_of_samples
    results_folder = Path(args.results_location)
    results_folder.mkdir(exist_ok=True)
    if len(list(results_folder.iterdir())) > 0:
        save_overall_results(results_folder)
    results_df = pl.DataFrame()
    if (results_folder / "results.csv").exists():
        results_df = pl.read_parquet(results_folder / "results.parquet")

    for model in args.models.split(","):
        model_seed = args.model_seed
        # print(
        #     f"Running model: {model}, with seed: {model_seed}, on datasets: {args.datasets}, with number of samples: {number_of_samples}"
        # )
        llm = start_model_process(
            model_path=args.model_path, model=model, seed=model_seed
        )

        dataset_seeds = [int(seed) for seed in args.dataset_seeds.split(",")]
        for dataset_seed in dataset_seeds:
            timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
            run_folder_name = f"{model}-ms-{model_seed}-ds-{dataset_seed}-{timestamp}"
            datasets = get_datasets(number_of_samples, dataset_seed=dataset_seed)
            score = run_bench(llm, model, datasets, number_of_samples)

            # save metadata as json to run name in results folder
            metadata = {
                "model": model,
                "model_path": args.model_path,
                "model_seed": model_seed,
                "dataset_seed": dataset_seed,
                "number_of_samples": number_of_samples,
                "timestamp": timestamp,
                "datasets": [
                    {
                        "path": dataset.path,
                        "name": dataset.name,
                        "split": dataset.split,
                        "category": dataset.category,
                    }
                    for dataset in datasets
                ],
            }

            Path(results_folder / run_folder_name).mkdir(exist_ok=True)

            json.dump(
                metadata,
                open(results_folder / run_folder_name / "metadata.json", "w"),
                indent=4,
                sort_keys=True,
            )

            # combine all the datasets results into one and save to run name in results folder
            for dataset in datasets:
                dataset.results_df = dataset.results_df.with_columns(
                    pl.lit(dataset.identifier).alias("dataset")
                )
                dataset.results_df = dataset.results_df.with_columns(
                    pl.lit(dataset.category).alias("category")
                )
                print(dataset.identifier, dataset.results_df.get_column("id").dtype)

            all_dataset_results_df: pl.DataFrame = pl.concat(
                [dataset.results_df for dataset in datasets]
            )
            all_dataset_results_df.write_parquet(
                results_folder / run_folder_name / "datasets_results.parquet"
            )

            scores = {
                r["dataset"]: r["score"]
                for r in all_dataset_results_df.group_by("dataset")
                .agg(pl.sum("score").alias("score"))
                .to_dicts()
            }
            normalized_scores = {k: v / number_of_samples for k, v in scores.items()}
            category_scores = calculate_category_scores(all_dataset_results_df)
            normalized_category_scores = {
                k: v
                / (len([d for d in datasets if d.category == k]) * number_of_samples)
                for k, v in category_scores.items()
            }

            make_radar_chart(
                data_list=[
                    normalized_category_scores,
                    calculate_random_category_scores(datasets),
                ],
                labels=[model, "random"],
                save_path=results_folder / run_folder_name / "radar_chart",
            )

            # save results_df after each run
            result = {
                "model": model,
                "model_seed": model_seed,
                "dataset_seed": dataset_seed,
                "score": score,
                "normalized_score": score / (len(datasets) * number_of_samples),
                "scores": scores,
                "normalized_scores": normalized_scores,
                "category_scores": category_scores,
                "normalized_category_scores": normalized_category_scores,
                "number_of_samples": number_of_samples,
                "run_folder_name": run_folder_name,
            }

            results_df = pl.DataFrame([*results_df.to_dicts(), result])

            results_df.write_parquet(results_folder / "results.parquet")
            save_overall_results(results_folder)
        results_df.write_parquet(results_folder / "results.parquet")
    results_df.write_parquet(results_folder / "results.parquet")

    # show results_df after all runs
    print(results_df)


if __name__ == "__main__":
    main()
