"""
Polite Response Generator (T5-small fine-tuned on tweet_eval sentiment) - Streamlit

Run:  streamlit run app.py
Pehli baar "Train model" button dabayein (~100 samples, jaldi ho jata hai).
Model ./t5_polite_model mein save hota hai, phir agli baar direct load hota hai.
"""

import os
import re
import html
import random

import torch
import streamlit as st
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
@st.cache_resource(show_spinner="Dataset load ho raha hai...")
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
# Training (with live logs in the UI)
# --------------------------------------------------------------------------- #
class StreamlitLogCallback(TrainerCallback):
    def __init__(self, progress_bar, log_box):
        self.progress_bar = progress_bar
        self.log_box = log_box
        self.lines = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.lines.append(f"step {state.global_step}/{state.max_steps}: {logs}")
            self.log_box.code("\n".join(self.lines[-15:]))
        if state.max_steps:
            self.progress_bar.progress(min(state.global_step / state.max_steps, 1.0))


def train_model(splits, progress_bar, log_box):
    set_seed(SEED)
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
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
        disable_tqdm=True,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[StreamlitLogCallback(progress_bar, log_box)],
    )
    trainer.train()
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Models load ho rahe hain...")
def load_models():
    tokenizer = T5TokenizerFast.from_pretrained(SAVE_DIR)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    ft_model = T5ForConditionalGeneration.from_pretrained(SAVE_DIR).to(device)
    return tokenizer, base_model, ft_model


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    model.eval()
    dev = next(model.parameters()).device  # hamesha model ke device par
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Polite Response Generator", page_icon="💬")
st.title("💬 Polite Response Generator")
st.caption(
    "T5-small fine-tuned on tweet_eval sentiment. "
    "Negative → apology, Neutral → clarification, Positive → appreciation."
)
st.sidebar.write(f"Device: **{device}**")

splits = load_data()
test_ds = splits["test"]

# ---- Training section ----
if not os.path.isdir(SAVE_DIR):
    st.warning("Fine-tuned model abhi nahi mila. Pehle train karein.")
    if st.button("🚀 Train model", type="primary"):
        progress = st.progress(0.0)
        log_box = st.empty()
        with st.spinner("Training chal rahi hai..."):
            train_model(splits, progress, log_box)
        st.success("Training complete! Model save ho gaya.")
        st.rerun()
    st.stop()

tokenizer, base_model, ft_model = load_models()

# ---- Inference section ----
if "tweet" not in st.session_state:
    st.session_state["tweet"] = ""
    st.session_state["true_label"] = ""


def load_random_tweet():
    ex = test_ds[random.randrange(len(test_ds))]
    st.session_state["tweet"] = ex["text"]
    st.session_state["true_label"] = LABEL_NAMES[ex["label"]]


st.button("🎲 Load random test tweet (dataset se)", on_click=load_random_tweet)

tweet = st.text_area("Tweet", key="tweet", height=100)
if st.session_state["true_label"]:
    st.write(f"True sentiment (dataset): **{st.session_state['true_label']}**")

if st.button("Generate", type="primary"):
    cleaned = clean_text(tweet or "")
    if not cleaned:
        st.error("Cleaning ke baad text khali ho gaya. Koi doosra tweet try karein.")
    else:
        prompt = make_prompt(cleaned)
        with st.spinner("Generating..."):
            before = generate(base_model, tokenizer, prompt)
            after = generate(ft_model, tokenizer, prompt)

        st.subheader("Cleaned text")
        st.code(cleaned, language=None)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before fine-tuning")
            st.info(before or "(empty output)")
        with col2:
            st.subheader("After fine-tuning")
            st.success(after or "(empty output)")
