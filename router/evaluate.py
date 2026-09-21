import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data.dataset_loader import DatasetLoader
from data.ml_dataset_prep import MLDatasetPrep
from embeddings.embedding_model import EmbeddingModel, load_embedding_config
from router.keras_mlp import KerasMLPRouter


def evaluate_router(
    model_path: str = "models/router/best_model.keras",
    metadata_path: str = "models/router/metadata.json"
) -> dict[str, Any]:
    """
    Comprehensive evaluation of the trained router.
    
    Computes:
    - Accuracy, Precision, Recall, F1-score
    - Macro-F1, Per-class F1
    - Confusion Matrix
    - Confidence calibration (ECE)
    """
    # Load model
    router = KerasMLPRouter.load(model_path)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load data and prepare test set
    loader = DatasetLoader()
    data = loader.load_dataset()

    prep = MLDatasetPrep()
    train_idx, val_idx, test_idx = prep.split_data(data)

    # Get test texts and labels
    test_texts = [data[i]['prompt'] for i in test_idx]
    test_labels = np.array([prep.label_mapping.map_to_difficulty(
        data[i]['model_label']) for i in test_idx])

    # Convert to one-hot for evaluation
    class_to_idx = {"easy": 0, "medium": 1, "hard": 2}
    y_true_idx = np.array([class_to_idx[label] for label in test_labels])

    # Load embedding model
    emb_config = load_embedding_config()
    emb_model = EmbeddingModel(emb_config)

    # Generate test embeddings
    print("Generating test embeddings...")
    X_test = emb_model.encode_batch(test_texts)

    # Predict
    print("Predicting...")
    y_pred_probs = router.predict(X_test)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_pred_labels = [list(class_to_idx.keys())[list(class_to_idx.values()).index(i)] for i in y_pred_idx]

    # Compute metrics
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    precision_macro = precision_score(y_true_idx, y_pred_idx, average='macro')
    recall_macro = recall_score(y_true_idx, y_pred_idx, average='macro')
    f1_macro = f1_score(y_true_idx, y_pred_idx, average='macro')

    # Per-class metrics
    precision_per_class = precision_score(y_true_idx, y_pred_idx, average=None)
    recall_per_class = recall_score(y_true_idx, y_pred_idx, average=None)
    f1_per_class = f1_score(y_true_idx, y_pred_idx, average=None)

    class_names = ["easy", "medium", "hard"]

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)

    # Confidence analysis
    confidence = np.max(y_pred_probs, axis=1)
    correct = (y_pred_idx == y_true_idx)

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = (y_true_idx == i)
        if mask.sum() > 0:
            per_class_acc[name] = float(correct[mask].mean())

    # Classification report
    report = classification_report(y_true_idx, y_pred_idx, target_names=class_names, output_dict=True)

    results = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'per_class_metrics': per_class_metrics,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': class_names,
        'ece': float(ece),
        'classification_report': report
    }

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print evaluation results in a readable format"""
    print("=" * 60)
    print("ROUTER EVALUATION RESULTS")
    print("=" * 60)
    print("\nOverall Metrics:")
    print(f"  Accuracy:       {results['accuracy']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro):    {results['recall_macro']:.4f}")
    print(f"  F1-score (macro):  {results['f1_macro']:.4f}")
    print(f"  ECE:               {results['ece']:.4f}")

    print("\nPer-Class Metrics:")
    for class_name, metrics in results['per_class_metrics'].items():
        acc = results['per_class_accuracy'].get(class_name, 0)
        print(f"  {class_name.capitalize():>6}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Acc={acc:.4f}")

    print("\nConfusion Matrix:")
    print(f"  Labels: {results['confusion_matrix_labels']}")
    for i, row in enumerate(results['confusion_matrix']):
        print(f"  {results['confusion_matrix_labels'][i]:>6}: {row}")

    print(f"\nECE (Expected Calibration Error): {results['ece']:.4f}")


def save_results(results: dict[str, Any], output_path: str = "evaluation_results.json") -> None:
    """Save evaluation results to JSON"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    """Run full evaluation"""
    print("Running comprehensive router evaluation...")
    results = evaluate_router()
    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
