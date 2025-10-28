import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

finetuned_model_path = r"C:\Users\Pavan\Downloads\NLP Project\biobert-ner-jnlpba"   # change if saved elsewhere
model = AutoModelForTokenClassification.from_pretrained(finetuned_model_path)
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)


data_files = {
    "train": r"C:\Users\Pavan\Downloads\NLP Project\NERDatasets\JNLPBA\train.tsv",
    "validation": r"C:\Users\Pavan\Downloads\NLP Project\NERDatasets\JNLPBA\devel.tsv",
    "test": r"C:\Users\Pavan\Downloads\NLP Project\NERDatasets\JNLPBA\test.tsv"
}
raw_datasets = load_dataset("text", data_files=data_files)

def parse_conll(example):
    tokens, ner_tags = [], []
    for line in example["text"].split("\n"):
        if line.strip() == "":
            if tokens:
                yield {"tokens": tokens, "ner_tags": ner_tags}
                tokens, ner_tags = [], []
        else:
            splits = line.split()
            if len(splits) == 2:
                token, tag = splits
                tokens.append(token)
                ner_tags.append(tag)
    if tokens:
        yield {"tokens": tokens, "ner_tags": ner_tags}

dataset_splits = {}
for split, dataset in raw_datasets.items():
    parsed = []
    for example in dataset:
        parsed.extend(list(parse_conll(example)))
    dataset_splits[split] = parsed

datasets = DatasetDict({
    split: Dataset.from_list(dataset_splits[split])
    for split in dataset_splits
})


labels = sorted(list({tag for d in datasets["train"] for tag in d["ner_tags"]}))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels_out = []
    for i, labels_example in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word:
                label_ids.append(label2id[labels_example[word_id]])
            else:
                label_ids.append(label2id[labels_example[word_id]])
            prev_word = word_id
        labels_out.append(label_ids)
    tokenized_inputs["labels"] = labels_out
    return tokenized_inputs

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[pred] for (pred, lab) in zip(pred_row, label_row) if lab != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lab] for (pred, lab) in zip(pred_row, label_row) if lab != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


args = TrainingArguments(
    output_dir="./ner_eval",
    per_device_eval_batch_size=16,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

print("\n📊 Validation metrics:")
val_results = trainer.evaluate(tokenized_datasets["validation"])
print(val_results)

print("\n📊 Train metrics:")
train_results = trainer.evaluate(tokenized_datasets["train"])
print(train_results)

print("\n📊 Test metrics:")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)

# Detailed classification report on test set
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions, axis=2)

true_preds = [
    [id2label[p] for (p, l) in zip(pred_row, label_row) if l != -100]
    for pred_row, label_row in zip(preds, labels)
]
true_labels = [
    [id2label[l] for (p, l) in zip(pred_row, label_row) if l != -100]
    for pred_row, label_row in zip(preds, labels)
]

print("\n📑 Test classification report:\n")
print(classification_report(true_labels, true_preds))
