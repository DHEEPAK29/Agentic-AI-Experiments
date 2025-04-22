 from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
import torch, datasets, json

# load the YAML
import yaml, argparse
with open("tiny_gpt_config.yaml") as f:
    cfg = yaml.safe_load(f)

config = GPT2Config(**{k: cfg[k] for k in
                       ["vocab_size","n_positions","n_embd",
                        "n_layer","n_head","n_inner"]})
model = GPT2LMHeadModel(config)

# pin <pad> to zero loss
model.config.pad_token_id = cfg["pad_token_id"]

train_ds = datasets.load_from_disk("train.arrow")
val_ds   = datasets.load_from_disk("val.arrow")

args = TrainingArguments(
        output_dir="tiny-gpt",
        per_device_train_batch_size=cfg["trainer"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["trainer"]["per_device_eval_batch_size"],
        num_train_epochs=cfg["trainer"]["num_train_epochs"],
        learning_rate=cfg["trainer"]["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg["trainer"]["logging_steps"],
        weight_decay=cfg["trainer"]["weight_decay"],
        lr_scheduler_type=cfg["trainer"]["lr_scheduler_type"],
        warmup_steps=cfg["trainer"]["warmup_steps"])

Trainer(model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds).train()
