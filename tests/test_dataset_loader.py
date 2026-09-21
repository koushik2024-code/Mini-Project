import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import DatasetLoader, DatasetSchema


class TestDatasetLoader:
    """Unit tests for PRD-0001-FR-1: Load and validate dataset"""

    def test_dataset_file_exists(self):
        """Test that dataset file exists at expected path"""
        loader = DatasetLoader()
        assert loader.dataset_path.exists(), f"Dataset not found at {loader.dataset_path}"

    def test_load_dataset_returns_list(self):
        """Test that load_dataset returns a list of samples"""
        loader = DatasetLoader()
        data = loader.load_dataset()
        assert isinstance(data, list), "Dataset should be a list"
        assert len(data) > 0, "Dataset should not be empty"

    def test_sample_has_required_fields(self):
        """Test that each sample has all required fields"""
        loader = DatasetLoader()
        data = loader.load_dataset()

        required_fields = ['id', 'task', 'prompt', 'answer', 'dataset', 'model_label', 'model_name']
        for sample in data[:100]:  # Check first 100 samples
            for field in required_fields:
                assert field in sample, f"Missing required field: {field}"

    def test_model_label_is_valid_int(self):
        """Test that model_label is a single int in {0, 1, 2}"""
        loader = DatasetLoader()
        data = loader.load_dataset()

        for sample in data[:100]:
            label = sample['model_label']
            assert isinstance(label, int), "model_label should be an int"
            assert label in [0, 1, 2], "model_label should be 0, 1, or 2"

    def test_no_missing_values_in_required_fields(self):
        """Test that required fields have no missing/empty values"""
        loader = DatasetLoader()
        data = loader.load_dataset()

        for sample in data:
            assert sample['id'] is not None and sample['id'] != "", "id should not be empty"
            assert sample['prompt'] is not None and sample['prompt'] != "", "prompt should not be empty"
            assert sample['answer'] is not None and sample['answer'] != "", "answer should not be empty"
            assert sample['task'] is not None and sample['task'] != "", "task should not be empty"
            assert sample['dataset'] is not None and sample['dataset'] != "", "dataset should not be empty"

    def test_schema_validation(self):
        """Test that schema validation works"""
        loader = DatasetLoader()
        data = loader.load_dataset()
        schema = DatasetSchema()

        errors = schema.validate(data)
        assert len(errors) == 0, f"Schema validation errors: {errors}"

    def test_get_label_distribution(self):
        """Test label distribution calculation"""
        loader = DatasetLoader()
        data = loader.load_dataset()
        dist = loader.get_label_distribution(data)

        assert isinstance(dist, dict), "Distribution should be a dict"
        assert len(dist) > 0, "Should have at least one label pattern"
        total = sum(dist.values())
        assert total == len(data), f"Total samples mismatch: {total} vs {len(data)}"

    def test_get_dataset_distribution(self):
        """Test dataset distribution calculation"""
        loader = DatasetLoader()
        data = loader.load_dataset()
        dist = loader.get_dataset_distribution(data)

        assert isinstance(dist, dict), "Distribution should be a dict"
        assert len(dist) > 0, "Should have at least one dataset"
        total = sum(dist.values())
        assert total == len(data), f"Total samples mismatch: {total} vs {len(data)}"
