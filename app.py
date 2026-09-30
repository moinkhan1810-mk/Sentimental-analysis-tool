"""
Polite Response Generator (T5-small fine-tuned on tweet_eval sentiment)

Pipeline: Data loading -> Cleaning -> Formatting -> Tokenization -> Training -> Inference (Gradio UI)

Run:  python app.py
On first run it fine-tunes T5-small on ~100 tweets and saves the model to ./t5_polite_model.
Later runs load the saved model directly.
"""

import os
import re
import html
import random

import torch
import gradio as gr
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_NAME = "t5-small"
SAVE_DIR = "./t5_polite_model"
SEED = 42
N_TRAIN, N_VAL, N_TEST_POOL = 100, 20, 200
MAX_INPUT_LEN, MAX_TARGET_LEN = 64, 48

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
RESPONSES = {
    0: "I'm really sorry to hear that. I apologize for the inconvenience, and I'll do my best to make it right.",
    1: "Thank you for your message. Could you please clarify a bit more so I can help you better?",
    2: "Thank you so much for your kind words! We truly appreciate your support.",
}

set_seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
SPECIAL_RE = re.compile(r"[^a-z0-9\s.,!?']")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.lower()
    text = SPECIAL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def make_prompt(cleaned: str) -> str:
    return f"Generate a polite response: {cleaned}"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data():
    try:
        raw = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    except Exception:
        raw = load_dataset("tweet_eval", "sentiment")

    splits = {
        "train": raw["train"].shuffle(seed=SEED).select(range(N_TRAIN)),
        "val": raw["validation"].shuffle(seed=SEED).select(range(N_VAL)),
        "test": raw["test"].shuffle(seed=SEED).select(range(N_TEST_POOL)),
    }

    def prep(ex):
        ex["clean_text"] = clean_text(ex["text"])
        ex["input_text"] = make_prompt(ex["clean_text"])
        ex["target_text"] = RESPONSES[ex["label"]]
        return ex

    for k in splits:
        splits[k] = splits[k].map(prep).filter(lambda x: len(x["clean_text"]) > 0)
    return splits


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(splits, tokenizer):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    def tokenize_fn(batch):
        enc = tokenizer(batch["input_text"], max_length=MAX_INPUT_LEN, truncation=True)
        lab = tokenizer(text_target=batch["target_text"], max_length=MAX_TARGET_LEN, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc

    train_tok = splits["train"].map(tokenize_fn, batched=True, remove_columns=splits["train"].column_names)
    val_tok = splits["val"].map(tokenize_fn, batched=True, remove_columns=splits["val"].column_names)

    args = TrainingArguments(
        output_dir="./t5_polite_out",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    trainer.train()
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Saved fine-tuned model to {SAVE_DIR}")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    model.eval()
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Startup: data, models
# --------------------------------------------------------------------------- #
splits = load_data()
test_ds = splits["test"]

if not os.path.isdir(SAVE_DIR):
    train_model(splits, T5TokenizerFast.from_pretrained(MODEL_NAME))

tokenizer = T5TokenizerFast.from_pretrained(SAVE_DIR)
base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
ft_model = T5ForConditionalGeneration.from_pretrained(SAVE_DIR).to(device)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def sample_tweet():
    ex = test_ds[random.randrange(len(test_ds))]
    return ex["text"], LABEL_NAMES[ex["label"]]


def respond(tweet: str):
    cleaned = clean_text(tweet or "")
    if not cleaned:
        return "", "(empty after cleaning)", "(empty after cleaning)"
    prompt = make_prompt(cleaned)
    return (
        cleaned,
        generate(base_model, tokenizer, prompt),
        generate(ft_model, tokenizer, prompt),
    )


with gr.Blocks(title="Polite Response Generator") as demo:
    gr.Markdown("# Polite Response Generator\nT5-small fine-tuned on tweet_eval sentiment. "
                "Negative → apology, Neutral → clarification, Positive → appreciation.")
    with gr.Row():
        tweet_box = gr.Textbox(label="Tweet", lines=3, scale=4)
        true_label = gr.Textbox(label="True sentiment (from dataset)", interactive=False, scale=1)
    with gr.Row():
        sample_btn = gr.Button("Load random test tweet")
        run_btn = gr.Button("Generate", variant="primary")
    cleaned_box = gr.Textbox(label="Cleaned text", interactive=False)
    with gr.Row():
        before_box = gr.Textbox(label="Before fine-tuning (base t5-small)", lines=4, interactive=False)
        after_box = gr.Textbox(label="After fine-tuning", lines=4, interactive=False)

    sample_btn.click(sample_tweet, outputs=[tweet_box, true_label])
    run_btn.click(respond, inputs=tweet_box, outputs=[cleaned_box, before_box, after_box])

if __name__ == "__main__":
    demo.launch()
