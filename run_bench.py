from datetime import datetime
from pathlib import Path
from time import sleep, time
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm
import os
import openai
import pandas as pd
import json

# from threading import Thread
# from save_thread_result import ThreadWithResult as Thread
# from multiprocessing.pool import Pool



from multiprocessing.dummy import Pool
from Datasets.Dataset import Dataset, DatasetMeta, CACHE_DATASET_PATH
from Datasets.ProcessRows import (
    process_row_mmlu,
    process_row_agieval,
    process_row_arc,
    process_row_anli,
    process_row_cosmoqa,
    process_row_hellaswag,
    process_row_winogrande,
    process_row_race,
    # process_row_mathqa,
    process_row_truthfulqa,
)

import argparse

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
    # RowMetadata(path="cais/mmlu", name="*", split="validation"),
    DatasetMeta(
        path="cais/mmlu",
        name="all",
        split="validation",
        process_row_fn=process_row_mmlu,
        category="Knowledge/Multi-subject Test",
    ),
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
        split="validation",
        process_row_fn=process_row_arc,
        category="Knowledge/Multi-subject Test",
    ),
    DatasetMeta(
        path="ai2_arc",
        name="ARC-Easy",
        split="validation",
        process_row_fn=process_row_arc,
        category="Knowledge/Multi-subject Test",
    ),
    DatasetMeta(
        path="anli",
        split="test_r1",
        process_row_fn=process_row_anli,
        category="Reasoning/Commonsense Reasoning",
    ),
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
    ),
    DatasetMeta(
        path="winogrande",
        name="winogrande_s",
        split="validation",
        process_row_fn=process_row_winogrande,
        category="Reasoning/Commonsense Reasoning",
    ),
    DatasetMeta(
        path="race",
        name="middle",
        split="validation",
        process_row_fn=process_row_race,
        category="Comprehension/Reading Comprehension",
    ),
    DatasetMeta(
        path="race",
        name="high",
        split="validation",
        process_row_fn=process_row_race,
        category="Comprehension/Reading Comprehension",
    ),
    # DatasetMeta(path="math_qa", split="validation", process_row_fn=process_fn=process_row_mathqa),
    DatasetMeta(
        path="truthful_qa",
        name="multiple_choice",
        split="validation",
        process_row_fn=process_row_truthfulqa,
        category="Safety/Truthfulness",
    ),
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


def check_correct_answer(row: pd.Series, completion: str) -> int:
    score = completion[0] == str(row["answer"])
    if len(completion) > 1 and score:
        confirm = completion[1] == ")"
        score = score and confirm
    return score


def run_query(row: pd.Series, prompt: str) -> tuple[str, float]:
    s = time()
    resp = openai.Completion.create(
        engine="",
        prompt=prompt,
        max_tokens=2,
        n=1,
        stop=None,
        temperature=0.0,
    )
    completion = resp.choices[0].text.strip()
    time_taken = time() - s
    score = check_correct_answer(row, completion)
    return completion, score, time_taken


def load_dataset(
    dataset: Dataset,
    dataset_seed: Optional[int] = None,
    cache_dir: str = CACHE_DATASET_PATH,
):
    dataset.load_dataset(dataset_seed=dataset_seed, cache_dir=cache_dir)


def load_all_datasets(
    datasets: list[Dataset],
    dataset_seed: Optional[int] = None,
    cache_dir: str = CACHE_DATASET_PATH,
):
    s = time()
    pool = Pool(10)
    try:
        pool.starmap(
            load_dataset,
            [
                (dataset, dataset_seed, cache_dir)
                for dataset in tqdm(datasets, desc="Loading datasets")
            ],
        )
    except Exception as e:
        print(e)
    pool.close()
    print(time() - s)


def start_model_server(seed: int, model: str)


def run_bench(datasets: list[Dataset], model: str, number_of_samples: int) -> int:
    start = time()
    query_time_taken = 0
    count = 0
    pbar = tqdm(total=len(datasets) * number_of_samples)
    for dataset in datasets:
        dtime = time()
        rows = dataset.df.to_dict(orient="records")
        rows = [row for row in rows if "completion" not in row]
        for _, row in enumerate(rows):
            prompt = dataset.form_prompt(row)
            completion, score, time_taken = run_query(row, prompt)
            row["prompt"] = prompt
            row["completion"] = completion
            row["model"] = model
            row["time_taken"] = time_taken
            row["score"] = score
            dataset.results_df = pd.DataFrame(rows)
            dataset.save()
            count += 1
            pbar.update(1)
            query_time_taken += time_taken
        print(f"Dataset: {dataset.path}/{dataset.name}, time taken: {time() - dtime}")
    total_time_taken = time() - start
    print(f"Total time taken: {total_time_taken}, query time taken: {query_time_taken}")

    score = sum([d.results_df.score.sum() for d in datasets])
    print(score)
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="comma separated, e.g. EleutherAI/gpt-neo-2.7B",
        required=True,
    )
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--dataset_seed", type=int, default=1)
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
    results_df = pd.DataFrame() or pd.read_csv(results_folder / "results.csv")
    for model in args.model.split(","):
        process = start_model_server(seed=args.model_seed, model=model)

        run_folder_name = datetime.now().isoformat().replace(":", "-")
        model_seeds = [int(seed) for seed in args.model_seed.split(",")]
        dataset_seeds = [int(seed) for seed in args.dataset_seed.split(",")]
        for dataset_seed in dataset_seeds:
            datasets = [
                Dataset(
                    path=meta.path,
                    name=meta.name,
                    split=meta.split,
                    number_of_samples=number_of_samples,
                    process_row_fn=meta.process_row_fn,
                )
                for meta in tqdm(dataset_metadata)
            ]
            print(f"Running dataset seed: {dataset_seed}")
            load_all_datasets(datasets, dataset_seed=dataset_seed)
            score = run_bench(datasets, model, number_of_samples)

            # save metadata as json to run name in results folder
            metadata = {
                "model": model,
                "model_seed": model_seeds,
                "dataset_seed": dataset_seed,
                "number_of_samples": number_of_samples,
            }
            json.dump(
                metadata, Path(results_folder / f"{run_folder_name}.json").open("w")
            )

            # combine all the datasets results into one and save to run name in results folder
            all_dataset_results_df = pd.DataFrame()
            for dataset in datasets:
                dataset.results_df["dataset"] = dataset.identifier
                all_dataset_results_df = all_dataset_results_df.append(
                    dataset.results_df
                )
            all_dataset_results_df.to_csv(
                results_folder / f"{run_folder_name}_{dataset_seed}.csv"
            )

            # save results_df after each run
            results_df = results_df.append(
                {
                    "model": model,
                    "model_seed": model_seeds,
                    "dataset_seed": dataset_seed,
                    "score": score,
                    "number_of_samples": number_of_samples,
                    "run_folder_name": run_folder_name,
                },
                ignore_index=True,
            )
            results_df.to_csv(results_folder / "results.csv")
    results_df.to_csv(results_folder / "results.csv")
    # show results_df after all runs
    print(results_df)


if __name__ == "__main__":
    main()
