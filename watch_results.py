# %%
from run_bench import save_overall_results, dataset_metadata
from pathlib import Path
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import dash

# %%
fig = save_overall_results(results_folder=Path("results"), method="plotly")
fig

# %%
results_df = pl.read_parquet(Path("results/results.parquet"))
# sort by score
results_df = results_df.sort("score").reverse()
# extract score and model columns
# max model col width
with pl.Config(fmt_str_lengths=50, tbl_rows=1000):
    print(results_df.select(["model", "normalized_score", "score"]))

# %%
row_dicts = results_df.select(
    ["model", "normalized_score", "normalized_category_scores"]
).to_dicts()
for row in row_dicts:
    for d in sorted(list(row["normalized_category_scores"])):
        row[d] = row["normalized_category_scores"].get(d)
    del row["normalized_category_scores"]
scores_df = pl.DataFrame(row_dicts)
scores_df


# %%
row_dicts = results_df.select(
    ["model", "normalized_score", "score", "normalized_scores"]
).to_dicts()
for row in row_dicts:
    for d in sorted(list(row["normalized_scores"])):
        row[d] = row["normalized_scores"].get(d)
    del row["normalized_scores"]
scores_df = pl.DataFrame(row_dicts)
scores_df

# %%
# from transformers import AutoTokenizer
# from petals import AutoDistributedModelForCausalLM

# # Choose any model available at https://health.petals.dev
# model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)

# # Connect to a distributed network hosting model layers
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
# prompt = """
# There are three sisters in a room alone. Anna is reading a book. Alice is playing a game called Zonda. Zonda requires two people to play it.
# What is the third sister, Amanda, most likely doing? Explain why.
# Select from the following options:
# (A) Also reading a book, like Anna
# (B) Having music lessons from Alice
# (C) Playing Zonda with Alice
# (D) Observing the other sisters, while they do their activities only
# (E) Trying to think of something to so
# Answer: (
# """
# from time import time
# # Run the model as if it were on your computer
# inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
# s = time()
# outputs = model.generate(inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))
# print(f"Time taken: {time() - s:.2f}s")

# %%
from pathlib import Path
import json
from run_bench import get_datasets, calculate_category_scores

results_folder = Path("results")

results_df = pl.DataFrame()
results_folders = [f for f in Path(results_folder).iterdir() if not f.is_file()]
for result_folder in results_folders:
    print(result_folder)
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
    # normalized_score = sum(normalized_category_scores.values())/len(normalized_category_scores)

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
results_df.sort("normalized_score").reverse().select(
    ["model", "normalized_score", "score"]
)

# %%
