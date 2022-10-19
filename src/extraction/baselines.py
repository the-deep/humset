import pandas as pd
import numpy as np
import rust_utils
import json
import os
import multiprocessing
import glob
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
import datasets
from rouge import Rouge
from sklearn.metrics import balanced_accuracy_score
from transformers import HfArgumentParser
from dataclasses import dataclass
import wandb

tqdm.pandas()

# when run script provide model name as command line argument
# example: python baselines.py --model_name_or_path=xlm-roberta-base 
@dataclass
class Args:
    model_name_or_path: str


(args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

# provide here a train/val/test splitting of HumSet dataset, as show in the notebook
# in classification directory  
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

df = pd.concat([train_df, valid_df, test_df])

texts = {}

# load raw text documents provided in dataset archive tar.gz
for path_str in tqdm(glob.glob("./json/*.json")):
    path = Path(path_str)

    lead_id = int(path.stem)
    texts[lead_id] = "\n\n".join(
        "\n".join(section for section in page) for page in json.load(open(path))
    )


def has_match(row):
    lead_id = row["lead_id"]

    if lead_id not in texts:
        return False
    elif row["excerpt"] in texts[lead_id]:
        return True
    else:
        return np.nan  # can not be determined just yet, needs fuzzy matching


df["has_match"] = df.apply(has_match, axis=1)


def find_fuzzy_match(row):
    if row["has_match"] is True or row["has_match"] is False:
        return np.nan

    lead_id = int(row["lead_id"])

    text = texts.get(lead_id)

    matches = sorted(
        rust_utils.levenshtein_search(row["excerpt"], text), key=lambda m: m[2]
    )

    if len(matches) == 0:
        return np.nan

    m = matches[0]
    excerpt = np.nan

    for i in (0, -1, 1):
        for j in (0, 1, -1):
            try:
                excerpt = text.encode("utf-8")[m[0] + i : m[1] + j].decode("utf-8")
                break
            except UnicodeDecodeError:
                pass
        if excerpt is not None:
            break

    return excerpt


if not os.path.exists("proc_df.csv"):
    df["fuzzy_match"] = df.progress_apply(find_fuzzy_match, axis=1)
    df["has_match"] = df["has_match"].fillna(False) | (~df["fuzzy_match"].isna())

    def compute_matched_excerpt(row):
        if row["has_match"]:
            matched_excerpt = (
                row["excerpt"] if row["fuzzy_match"] is np.nan else row["fuzzy_match"]
            )
        else:
            matched_excerpt = np.nan

        if matched_excerpt is np.nan:
            start = np.nan
            end = np.nan
        else:
            text = texts[row["lead_id"]]
            start = text.index(matched_excerpt)
            end = start + len(matched_excerpt)

        row["start"] = start
        row["end"] = end
        row["matched_excerpt"] = matched_excerpt

        del row["has_match"]
        del row["fuzzy_match"]

        return row

    proc_df = df.progress_apply(compute_matched_excerpt, axis=1)
    proc_df.to_csv("proc_df.csv", index=False)
else:
    proc_df = pd.read_csv("proc_df.csv")

## training

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

dsets = {}
seen_ids = set()

for (name, lead_ids) in [
    ("train", train_df["lead_id"].astype(int).unique()),
    ("valid", valid_df["lead_id"].astype(int).unique()),
    ("test", test_df["lead_id"].astype(int).unique()),
]:
    lead_ids = [lead_id for lead_id in lead_ids if lead_id in texts]

    assert len(seen_ids & set(lead_ids)) == 0
    seen_ids.update(lead_ids)

    dsets[name] = datasets.Dataset.from_dict(
        {"text": [texts[lead_id] for lead_id in lead_ids], "lead_id": lead_ids}
    )

dsets = datasets.DatasetDict(dsets)

lead_id_groups = proc_df.groupby("lead_id")

max_document_length = 4096
max_length = 512

assert max_document_length % max_length == 0


def encode(examples):
    text = examples["text"][0]
    lead_id = examples["lead_id"][0]

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_document_length,
        truncation=True,
        padding="max_length",
    )
    labels = np.zeros(max_document_length)
    labels[np.array(encoding["attention_mask"]) == 0] = -100

    for i, row in lead_id_groups.get_group(lead_id).iterrows():
        if row["matched_excerpt"] is np.nan:
            continue

        start = row["start"]
        end = row["end"]

        def is_in_excerpt(offset):
            return offset[0] != offset[1] and offset[0] >= start and offset[1] <= end

        for i, offset in enumerate(encoding["offset_mapping"]):
            if is_in_excerpt(offset):
                labels[i] = 1

    actual_length = max_document_length - sum(
        np.array(encoding["input_ids"]) == tokenizer.pad_token_id
    )
    actual_length_tiled = next(
        i
        for i in range(max_length, max_document_length + max_length, max_length)
        if i >= actual_length
    )

    return {
        "input_ids": [
            encoding["input_ids"][i : i + max_length]
            for i in range(0, actual_length_tiled, max_length)
        ],
        "attention_mask": [
            encoding["attention_mask"][i : i + max_length]
            for i in range(0, actual_length_tiled, max_length)
        ],
        "offset_mapping": [
            encoding["offset_mapping"][i : i + max_length]
            for i in range(0, actual_length_tiled, max_length)
        ],
        "labels": [
            labels[i : i + max_length]
            for i in range(0, actual_length_tiled, max_length)
        ],
        "lead_id": [lead_id] * (actual_length_tiled // max_length),
    }


dsets = dsets.map(
    encode,
    remove_columns=["text", "lead_id"],
    batched=True,
    batch_size=1,
    num_proc=multiprocessing.cpu_count(),
)

model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path)

data_collator = DataCollatorForTokenClassification(tokenizer)


def compute_rouge(predicted_texts, true_texts):
    scorer = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])

    # bug in `rouge`: texts with only "." are treated as empty but not ignored
    evaluatable_pairs = [
        # lowercase because the model tokenizer might lowercase.
        # in that case it is not trivial to get uppercase predictions
        (hyp.lower(), ref.lower())
        for hyp, ref in zip(predicted_texts, true_texts)
        if len(hyp.replace(".", "")) > 0 and len(ref.replace(".", "")) > 0
    ]
    n_empty_hypotheses = sum(
        1
        for hyp, ref in zip(predicted_texts, true_texts)
        if (len(hyp.replace(".", "")) == 0) and len(ref.replace(".", "")) > 0
    )
    if len(evaluatable_pairs) == 0:
        scores = {}
    else:
        hyps, refs = list(zip(*evaluatable_pairs))
        all_scores = scorer.get_scores(hyps, refs, ignore_empty=True, avg=False)
        # flattens the dicts
        all_scores = [
            pd.json_normalize(score, sep="/").to_dict(orient="records")[0]
            for score in all_scores
        ]

        scores = {k: [0] * n_empty_hypotheses for k in all_scores[0].keys()}

        for score in all_scores:
            for key, value in score.items():
                scores[key].append(value)

        scores = {k: np.mean(v) for k, v in scores.items()}

    return scores


def compute_metrics(results):
    preds = {}
    input_ids = {}
    for i, row in enumerate(dsets["test"]):
        lead_id = row["lead_id"]

        if lead_id not in preds:
            preds[lead_id] = []
            input_ids[lead_id] = []

        preds[lead_id].append(results.predictions[i])
        input_ids[lead_id].append(row["input_ids"])

    preds = {k: np.concatenate(v, 0) for k, v in preds.items()}
    input_ids = {k: np.concatenate(v, 0) for k, v in input_ids.items()}

    predicted_texts = []
    true_texts = []

    for lead_id in preds.keys():
        mask = np.argmax(preds[lead_id], -1) == 1
        predicted_texts.append(
            tokenizer.decode(input_ids[lead_id][mask], skip_special_tokens=True)
        )
        true_texts.append(" ".join(lead_id_groups.get_group(lead_id)["excerpt"]))

    metrics = {}
    metrics.update(compute_rouge(predicted_texts, true_texts))
    mask = results.label_ids != -100
    metrics["balanced_accuracy_score"] = balanced_accuracy_score(
        results.label_ids[mask].flatten(),
        np.argmax(results.predictions, -1)[mask].flatten(),
    )
    return metrics


training_args = TrainingArguments(
    "output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    warmup_ratio=0.3,
    report_to="wandb",
    save_strategy="epoch",
    logging_steps=100,
    adafactor=True,
    fp16=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
)
wandb.init(name=args.model_name_or_path)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dsets["train"],
    eval_dataset=dsets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
