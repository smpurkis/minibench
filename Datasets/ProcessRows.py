from typing import Any
from .Dataset import Row


def process_row_mmlu(row: dict[str, Any]) -> Row:
    return Row(
        question=row["question"],
        options=row["choices"],
        answer=row["answer"],
    )


def process_row_agieval(row: dict[str, Any]) -> Row:
    return Row(
        context=row["context"],
        question=row["question"],
        options=row["options"],
        answer=row["answer"],
    )


def process_row_arc(row: dict[str, Any]) -> Row:
    return Row(
        id=row["id"],
        question=row["question"],
        options=row["choices"]["text"],
        answer=row["choices"]["label"].index(row["answerKey"]),
    )


def process_row_anli(row: dict[str, Any]) -> Row:
    return Row(
        id=row["uid"],
        context=f"Premise: {row['premise']}\n\nHypothesis: {row['hypothesis']}\n",
        question="Does the the hypothesis follow from the context?",
        options=["Entailment", "Neutral", "Contradiction"],
        answer=row["label"],
    )


def process_row_cosmoqa(row: dict[str, Any]) -> Row:
    return Row(
        id=row["id"],
        context=row["context"],
        question=row["question"],
        options=[row[k] for k in (k for k in row.keys() if "answer" in k)],
        answer=row["label"],
    )


def process_row_hellaswag(row: dict[str, Any]) -> Row:
    return Row(
        id=row["ind"],
        context=row["ctx"],
        question="What is the most likely ending to the sentence?",
        options=row["endings"],
        answer=row["label"],
    )


def process_row_winogrande(row: dict[str, Any]) -> Row:
    return Row(
        context=row["sentence"],
        question='Fill in the blank "_"',
        options=[
            row[option_key] for option_key in (k for k in row.keys() if "option" in k)
        ],
        answer=int(row["answer"]),
    )


def process_row_race(row: dict[str, Any]) -> Row:
    return Row(
        id=row["example_id"],
        context=row["article"],
        question=row["question"],
        options=row["options"],
        answer=ord(row["answer"]) - ord("A"),
    )


# def process_row_mathqa(row: dict[str, Any]) -> Row:
#     options = [r.split(") ") for r in row["options"].split(",")]
#     return Row(
#         context=row["Problem"],
#         question=row["problem"],
#         options=row["answer"],
#         answer=row["answer"],
#         subject=row["category"],
#     )


def process_row_truthfulqa(row: dict[str, Any]) -> Row:
    return Row(
        question=row["question"],
        options=row["mc1_targets"]["choices"],
        answer=row["mc1_targets"]["labels"].index(1),
    )
