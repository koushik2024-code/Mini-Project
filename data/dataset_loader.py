import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetSchema:
    """Schema definition for the dataset"""
    required_fields: list[str] | None = None
    label_field: str = "model_label"
    valid_labels: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if self.required_fields is None:
            self.required_fields = [
                'id', 'task', 'prompt', 'answer', 'dataset', 'model_label', 'model_name'
            ]

    def validate(self, data: list[dict[str, Any]]) -> list[str]:
        """Validate dataset against schema. Returns list of error messages."""
        errors = []

        for i, sample in enumerate(data):
            # Check required fields
            required_fields = self.required_fields or []
            for field in required_fields:
                if field not in sample:
                    errors.append(f"Sample {i}: missing required field '{field}'")

            # Check label field
            if self.label_field in sample:
                label = sample[self.label_field]
                if not isinstance(label, int):
                    errors.append(f"Sample {i}: '{self.label_field}' should be an int")
                elif label not in self.valid_labels:
                    errors.append(f"Sample {i}: '{self.label_field}' should be one of {self.valid_labels}")

        return errors


class DatasetLoader:
    """Load and validate the single-label router dataset (model_label per sample)"""

    def __init__(self, dataset_path: str | None = None):
        if dataset_path is None:
            # Default to config.yaml path
            import yaml
            config_path = Path(__file__).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                dataset_path = config.get('data', {}).get('dataset_path', 'data/router_dataset_single_label.jsonl')
            else:
                dataset_path = 'data/router_dataset_single_label.jsonl'

        self.dataset_path = Path(dataset_path)
        self.schema = DatasetSchema()

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load dataset from JSONL file"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        data = []
        with open(self.dataset_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    data.append(sample)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

        return data

    def validate_dataset(self, data: list[dict[str, Any]]) -> list[str]:
        """Validate dataset against schema"""
        return self.schema.validate(data)

    def get_label_distribution(self, data: list[dict[str, Any]]) -> dict[int, int]:
        """Get distribution of model_label values"""
        counter: Counter[int] = Counter()
        for sample in data:
            if 'model_label' in sample:
                counter[sample['model_label']] += 1
        return dict(counter)

    def get_dataset_distribution(self, data: list[dict[str, Any]]) -> dict[str, int]:
        """Get distribution of samples per dataset"""
        counter: Counter[str] = Counter()
        for sample in data:
            ds = sample.get('dataset', 'unknown')
            counter[ds] += 1
        return dict(counter)

    def get_task_distribution(self, data: list[dict[str, Any]]) -> dict[str, int]:
        """Get distribution of samples per task"""
        counter: Counter[str] = Counter()
        for sample in data:
            task = sample.get('task', 'unknown')
            counter[task] += 1
        return dict(counter)

    def get_label_mapping_analysis(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze the model_label distribution, including the model_name each label maps to"""
        label_dist = self.get_label_distribution(data)
        label_to_name: dict[int, str] = {}
        for sample in data:
            label_to_name.setdefault(sample['model_label'], sample['model_name'])

        analysis: dict[str, Any] = {
            'unique_labels': len(label_dist),
            'label_counts': dict(sorted(label_dist.items())),
            'label_to_model_name': dict(sorted(label_to_name.items())),
        }
        return analysis


def main() -> None:
    """Quick test when run directly"""
    loader = DatasetLoader()
    data = loader.load_dataset()
    print(f"Loaded {len(data)} samples")

    # Validate
    errors = loader.validate_dataset(data)
    if errors:
        print(f"Validation errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print("Validation passed!")

    # Distributions
    label_dist = loader.get_label_distribution(data)
    print("\nLabel distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"  {label}: {count}")

    dataset_dist = loader.get_dataset_distribution(data)
    print("\nDataset distribution:")
    for ds, count in sorted(dataset_dist.items()):
        print(f"  {ds}: {count}")

    task_dist = loader.get_task_distribution(data)
    print("\nTask distribution:")
    for task, count in sorted(task_dist.items()):
        print(f"  {task}: {count}")

    # Label mapping analysis
    analysis = loader.get_label_mapping_analysis(data)
    print("\nLabel mapping analysis:")
    print(f"  Unique labels: {analysis['unique_labels']}")
    for label, name in analysis['label_to_model_name'].items():
        print(f"  {label} -> {name}")


if __name__ == "__main__":
    main()
