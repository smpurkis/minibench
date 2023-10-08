from pathlib import Path
from typing import Any, Optional

# import pandas as pd
import polars as pd
from datasets import load_dataset
from dataclasses import dataclass

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
    ):
        self.path = path
        self.name = name
        self.split = split
        self.category = category
        self.process_row_fn = process_row_fn
        self.cache_dir = cache_dir
        self.number_of_samples = number_of_samples
        self.df: Optional[pd.DataFrame] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.identifier = f"{self.path.replace('/', '_')}{f'-{self.name}' if self.name is not None else ''}-{self.split}".lower()
        self.save_path = Path(f"cache/{self.identifier}.parquet")

    def save(self) -> None:
        # self.df.write_parquet(self.save_path)
        pass

    def load_dataset(self, dataset_seed: Optional[int], cache_dir: str) -> pd.DataFrame:
        if self.df is None:
            dataset_gen = load_dataset(
                path=self.path,
                name=self.name,
                split=self.split,
                cache_dir=cache_dir,
                streaming=True,
            )
            if dataset_seed is not None:
                dataset_gen = dataset_gen.shuffle(seed=dataset_seed)
            rows = []
            for i, row in enumerate(dataset_gen):
                row = self.process_row(row)
                rows.append(row)
                if i >= self.number_of_samples - 1:
                    break
            self.df = pd.DataFrame(rows)
        return self.df

    def process_row(self, row: dict[str, Any]) -> Row:
        return self.process_row_fn(row)

    def form_prompt(self, row: pd.Series) -> str:
        options = [f"({i}) {option}" for i, option in enumerate(row["options"])]
        options = "\n".join(options)

        # replace \n at the end of context with a space

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
