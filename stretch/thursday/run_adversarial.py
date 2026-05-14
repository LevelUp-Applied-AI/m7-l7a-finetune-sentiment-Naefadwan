"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    df = pd.read_csv(adv_csv_path)
    id2label = model.config.id2label
    
    predicted_labels = []
    predicted_probs = []
    correct_list = []

    model.eval()
    for _, row in df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            
            pred_label = id2label[pred_idx]
            pred_prob = probs[0, pred_idx].item()
            
            predicted_labels.append(pred_label)
            predicted_probs.append(pred_prob)
            correct_list.append(pred_label == row["expected_label"])

    df["predicted_label"] = predicted_labels
    df["predicted_probability"] = predicted_probs
    df["correct"] = correct_list
    
    return df


def main() -> None:
    """Orchestrate; write results.csv."""
    # Find paths relative to this script's location as a fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    
    model_path = os.environ.get("MODEL_PATH", "model")
    # If default "model" isn't here, check the project root
    if model_path == "model" and not os.path.exists(model_path):
        potential_root_model = os.path.join(root_dir, "model")
        if os.path.exists(potential_root_model):
            model_path = potential_root_model

    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    # If default "adversarial_set.csv" isn't here, check the script directory
    if adv_csv == "adversarial_set.csv" and not os.path.exists(adv_csv):
        potential_adv_csv = os.path.join(script_dir, "adversarial_set.csv")
        if os.path.exists(potential_adv_csv):
            adv_csv = potential_adv_csv

    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()
