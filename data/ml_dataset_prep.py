import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from data.dataset_loader import DatasetLoader


@dataclass
class LabelMapping:
    """
    Maps the dataset's single model_label (which router-tier model actually
    handled the sample) to a difficulty class (Easy/Medium/Hard), matching
    the model registry in config.yaml:
    - model_label 0 (qwen3_1.7b)  -> easy
    - model_label 2 (qwen3_4b)    -> medium
    - model_label 1 (llama3.2_3b) -> hard

    Unlike the earlier dataset, each sample already carries exactly one
    label, so there is no pattern-collapsing step - map_to_difficulty just
    looks up the model_label directly.
    """

    # Default mapping rules: model_label (int) -> difficulty
    DEFAULT_RULES = {
        0: "easy",
        2: "medium",
        1: "hard",
    }

    DIFFICULTY_TO_ONE_HOT = {
        "easy": (1, 0, 0),
        "medium": (0, 1, 0),
        "hard": (0, 0, 1),
    }

    ONE_HOT_TO_DIFFICULTY = {v: k for k, v in DIFFICULTY_TO_ONE_HOT.items()}

    def __init__(self, custom_rules: dict[int, str] | None = None):
        self.rules = custom_rules or self.DEFAULT_RULES.copy()

    def map_to_difficulty(self, model_label: int) -> str:
        """Map the dataset's model_label to a single difficulty class"""
        return self.rules.get(model_label, "medium")  # Default fallback

    def to_one_hot(self, difficulty: str) -> tuple[int, int, int]:
        """Convert difficulty class to one-hot encoding"""
        return self.DIFFICULTY_TO_ONE_HOT.get(difficulty, (0, 1, 0))

    def from_one_hot(self, one_hot: tuple[int, int, int]) -> str:
        """Convert one-hot to difficulty class"""
        return self.ONE_HOT_TO_DIFFICULTY.get(one_hot, "medium")

    def get_all_difficulties(self) -> list[str]:
        """Get list of all difficulty classes"""
        return ["easy", "medium", "hard"]

    def save(self, path: str) -> None:
        """Save mapping to JSON file"""
        data = {
            'rules': {str(k): v for k, v in self.rules.items()},
            'difficulty_to_one_hot': {k: list(v) for k, v in self.DIFFICULTY_TO_ONE_HOT.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LabelMapping':
        """Load mapping from JSON file"""
        with open(path) as f:
            data = json.load(f)

        rules = {int(k): v for k, v in data['rules'].items()}
        return cls(custom_rules=rules)


class MLDatasetPrep:
    """
    Prepare ML dataset: verify label mapping, create stratified splits.

    Handles the multi-label format in the dataset and maps it to
    3 difficulty classes for the Keras MLP classifier.
    """

    def __init__(self, loader: DatasetLoader | None = None):
        self.loader = loader or DatasetLoader()
        self.label_mapping = LabelMapping()

    def verify_label_mapping(self, data: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Verify and document the label mapping from data"""
        if data is None:
            data = self.loader.load_dataset()

        analysis = self.loader.get_label_mapping_analysis(data)

        # Add difficulty class distribution
        difficulty_counts = Counter()
        for sample in data:
            difficulty = self.label_mapping.map_to_difficulty(sample['model_label'])
            difficulty_counts[difficulty] += 1

        analysis['difficulty_distribution'] = dict(difficulty_counts)
        analysis['mapping_rules'] = {str(k): v for k, v in self.label_mapping.rules.items()}

        return analysis

    def split_data(
        self,
        data: list[dict[str, Any]] | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Create 80/10/10 stratified split.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if data is None:
            data = self.loader.load_dataset()

        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        # Get difficulty labels for stratification
        labels = []
        for sample in data:
            difficulty = self.label_mapping.map_to_difficulty(sample['model_label'])
            labels.append(difficulty)

        indices = list(range(len(data)))

        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels,
            random_state=seed
        )

        # Get labels for temp set
        temp_labels = [labels[i] for i in temp_idx]

        # Second split: val vs test from temp
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=seed
        )

        return train_idx, val_idx, test_idx

    def prepare_training_data(
        self,
        data: list[dict[str, Any]] | None = None,
        embeddings: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Prepare final training data with embeddings and labels.

        Args:
            data: Dataset samples
            embeddings: Pre-computed embeddings (n_samples, embedding_dim)

        Returns:
            Dict with X_train, y_train, X_val, y_val, X_test, y_test
        """
        if data is None:
            data = self.loader.load_dataset()

        train_idx, val_idx, test_idx = self.split_data(data)

        # Convert labels to one-hot for categorical crossentropy
        def get_one_hot(idx: int) -> np.ndarray:
            difficulty = self.label_mapping.map_to_difficulty(data[idx]['model_label'])
            return np.array(self.label_mapping.to_one_hot(difficulty), dtype=np.float32)

        result = {
            'train_indices': train_idx,
            'val_indices': val_idx,
            'test_indices': test_idx,
        }

        if embeddings is not None:
            result['X_train'] = embeddings[train_idx]
            result['X_val'] = embeddings[val_idx]
            result['X_test'] = embeddings[test_idx]

        result['y_train'] = np.array([get_one_hot(i) for i in train_idx])
        result['y_val'] = np.array([get_one_hot(i) for i in val_idx])
        result['y_test'] = np.array([get_one_hot(i) for i in test_idx])

        return result

    def save_splits(
        self,
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
        path: str
    ) -> None:
        """Save split indices to JSON file"""
        data = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_splits(self, path: str) -> dict[str, list[int]]:
        """Load split indices from JSON file"""
        with open(path) as f:
            return json.load(f)

    def get_class_weights(self, data: list[dict[str, Any]] | None = None) -> dict[int, float]:
        """Compute class weights for imbalanced dataset"""
        if data is None:
            data = self.loader.load_dataset()

        difficulty_counts = Counter()
        for sample in data:
            difficulty = self.label_mapping.map_to_difficulty(sample['model_label'])
            difficulty_counts[difficulty] += 1

        # Map difficulty to class index
        class_to_idx = {"easy": 0, "medium": 1, "hard": 2}

        total = len(data)
        weights = {}
        for difficulty, count in difficulty_counts.items():
            idx = class_to_idx[difficulty]
            weights[idx] = total / (len(difficulty_counts) * count)

        return weights


def main():
    """Quick test when run directly"""
    prep = MLDatasetPrep()
    data = prep.loader.load_dataset()

    print("Verifying label mapping...")
    analysis = prep.verify_label_mapping(data)

    print(f"Unique labels: {analysis['unique_labels']}")
    print(f"Label counts: {analysis['label_counts']}")
    print(f"Label to model name: {analysis['label_to_model_name']}")
    print(f"Difficulty distribution: {analysis['difficulty_distribution']}")
    print(f"Mapping rules: {analysis['mapping_rules']}")

    print("\nCreating stratified splits...")
    train_idx, val_idx, test_idx = prep.split_data(data)

    print(f"Train: {len(train_idx)} ({len(train_idx)/len(data)*100:.1f}%)")
    print(f"Val: {len(val_idx)} ({len(val_idx)/len(data)*100:.1f}%)")
    print(f"Test: {len(test_idx)} ({len(test_idx)/len(data)*100:.1f}%)")

    # Verify stratification
    train_labels = [prep.label_mapping.map_to_difficulty(data[i]['model_label']) for i in train_idx]
    val_labels = [prep.label_mapping.map_to_difficulty(data[i]['model_label']) for i in val_idx]
    test_labels = [prep.label_mapping.map_to_difficulty(data[i]['model_label']) for i in test_idx]

    print(f"\nTrain distribution: {Counter(train_labels)}")
    print(f"Val distribution: {Counter(val_labels)}")
    print(f"Test distribution: {Counter(test_labels)}")

    print(f"\nClass weights: {prep.get_class_weights(data)}")

    # Save splits
    prep.save_splits(train_idx, val_idx, test_idx, "data/splits.json")
    print("\nSplits saved to data/splits.json")

    # Save label mapping
    prep.label_mapping.save("data/label_mapping.json")
    print("Label mapping saved to data/label_mapping.json")


if __name__ == "__main__":
    main()
