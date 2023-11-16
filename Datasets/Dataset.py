import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import polars as pl
from datasets import load_dataset

CACHE_DATASET_PATH = "cache"
SYSTEM_INSTRUCTION = (
    "You are a helpful AI assistant. Your job is to answer the user's question."
)


@dataclass(frozen=True)
class DatasetMeta:
    path: str
    split: str
    category: str
    name: Optional[str] = ""
    process_row_fn: Optional[callable] = None


@dataclass(frozen=True)
class Row:
    options: list[str]
    answer: int  # index of options that is the answer, starting from 0
    question: Optional[str] = ""
    id: Optional[str] = ""
    context: Optional[str] = ""
    subject: Optional[str] = ""


class Dataset:
    def __init__(
        self,
        path: str,
        split: str,
        process_row_fn: callable,
        name: Optional[str] = None,
        cache_dir: str = CACHE_DATASET_PATH,
        number_of_samples: int = 10,
        category: Optional[str] = None,
        seed: int = 1,
    ):
        self.path = path
        self.name = name
        self.split = split
        self.category = category
        self.process_row_fn = process_row_fn
        self.cache_dir = cache_dir
        self.number_of_samples = number_of_samples
        self.df: Optional[pl.DataFrame] = None
        self.results_df: Optional[pl.DataFrame] = None
        self.identifier = f"{self.path.replace('/', '_')}{f'-{self.name}' if self.name is not None else ''}-{self.split}".lower()
        self.save_path = Path(f"cache/{self.identifier}.parquet")
        self.random_guess_score = None
        self.seed = seed

    def calculate_dataset_metrics(self) -> None:
        print(f"{self.identifier}, Number of samples: {len(self.df)}")
        self.random_guess_score = 1 / self.df["options"].apply(len).mean()

    def load_dataset(self, cache_dir: str) -> pl.DataFrame:
        cache_seed_path = (
            Path(cache_dir)
            / f"{self.identifier}_seed_{self.seed}_n_{self.number_of_samples}.parquet"
        )
        if cache_seed_path.exists():
            self.df = pl.read_parquet(cache_seed_path)
            self.calculate_dataset_metrics()
            return self.df
        else:
            if self.df is None:
                dataset_gen = load_dataset(
                    path=self.path,
                    name=self.name,
                    split=self.split,
                    cache_dir=cache_dir,
                    streaming=True,
                )
                if self.seed is not None:
                    dataset_gen = dataset_gen.shuffle(seed=self.seed)
                rows = []
                for i, row in enumerate(dataset_gen):
                    row = self.process_row(row)
                    rows.append(row)
                    if i >= self.number_of_samples - 1:
                        break
                self.df = pl.DataFrame(rows)
                self.calculate_dataset_metrics()
                self.df.write_parquet(cache_seed_path)
        return self.df

    def process_row(self, row: dict[str, Any]) -> Row:
        return self.process_row_fn(row)

    def form_prompt(self, row: pl.Series) -> str:
        options = [f"({i}) {option}" for i, option in enumerate(row["options"])]
        options = "\n".join(options)

        context = (
            f"CONTEXT \n" f"{row['context'].strip()}\n\n"
            if len(row["context"]) > 0
            else ""
        )
        return (
            "SYSTEM INSTRUCTION\n"
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"{context}"
            f"QUESTION: {row['question']}\n\n"
            "Select the correct answer from the following options:\n"
            f"{options}\n"
            "ANSWER: ("
        )
