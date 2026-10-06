import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from rintag.features import sent2features, sent2labels

def evaluate():
    print("Loading test data and model...")
    project_root = Path(__file__).resolve().parents[1]
    test_df = pd.read_csv(
        project_root / "data" / "processed" / "test.csv",
        keep_default_na=False,
    )
    train_df = pd.read_csv(
        project_root / "data" / "processed" / "train.csv",
        keep_default_na=False,
    )
    
    with open(project_root / "src" / "rintag" / "models" / "crf_model.pkl", "rb") as f:
        crf = pickle.load(f)
        
    sentences = []
    for _, group in test_df.groupby("sentence_id"):
        sentences.append(list(zip(group["word"].astype(str), group["pos_tag"].astype(str))))
        
    X_test = [sent2features([word for word, _ in sent]) for sent in sentences]
    y_test = [sent2labels(sent) for sent in sentences]
    
    # Flatten lists for scikit-learn metrics
    y_test_flat = [tag for tags in y_test for tag in tags]
    y_pred = crf.predict(X_test)
    y_pred_flat = [tag for tags in y_pred for tag in tags]
    
    print("\nClassification Report:")
    report = classification_report(y_test_flat, y_pred_flat)
    print(report)
    
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)
        
    print("\nCalculating OOV Accuracy...")
    train_vocab = set(train_df["word"].astype(str).str.lower())
    
    test_tokens = []
    for i, sent in enumerate(sentences):
        for j, (word, true_tag) in enumerate(sent):
            test_tokens.append((word, true_tag, y_pred[i][j]))
            
    oov_tokens = [t for t in test_tokens if t[0].lower() not in train_vocab]
    if oov_tokens:
        oov_accuracy = sum(1 for t in oov_tokens if t[1] == t[2]) / len(oov_tokens)
        print(f"OOV Accuracy: {oov_accuracy:.4f} ({sum(1 for t in oov_tokens if t[1] == t[2])}/{len(oov_tokens)})")
    else:
        print("No OOV tokens found in test set.")
        
    print("\nGenerating Confusion Matrix...")
    labels = sorted(list(set(y_test_flat) | set(y_pred_flat)))
    cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
    print(f"Saved confusion matrix to {results_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    evaluate()
