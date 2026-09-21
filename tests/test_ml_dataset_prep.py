import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ml_dataset_prep import LabelMapping, MLDatasetPrep


class TestMLDatasetPrep:
    """Unit tests for PRD-0001-FR-2, FR-5: Label mapping verification and data splitting"""

    def setup_method(self):
        """Set up ML dataset preparation"""
        self.prep = MLDatasetPrep()
        self.loader = self.prep.loader
        self.data = self.loader.load_dataset()

    def test_label_mapping_analysis(self):
        """Test that label mapping analysis works correctly"""
        analysis = self.loader.get_label_mapping_analysis(self.data)

        assert 'unique_labels' in analysis
        assert analysis['unique_labels'] == 3
        assert 'label_counts' in analysis
        assert 'label_to_model_name' in analysis

        # Check all 3 labels are identified
        assert set(analysis['label_counts'].keys()) == {0, 1, 2}
        assert analysis['label_to_model_name'][0] == "qwen3_1.7b"
        assert analysis['label_to_model_name'][1] == "llama3.2_3b"
        assert analysis['label_to_model_name'][2] == "qwen3_4b"

    def test_label_mapping_to_difficulty(self):
        """Test mapping from model_label to difficulty class"""
        mapping = LabelMapping()

        # Test known labels (matches config.yaml's model registry)
        assert mapping.map_to_difficulty(0) == "easy"    # qwen3_1.7b
        assert mapping.map_to_difficulty(2) == "medium"  # qwen3_4b
        assert mapping.map_to_difficulty(1) == "hard"    # llama3.2_3b

        # Test one-hot encoding
        assert mapping.to_one_hot("easy") == (1, 0, 0)
        assert mapping.to_one_hot("medium") == (0, 1, 0)
        assert mapping.to_one_hot("hard") == (0, 0, 1)

    def test_stratified_split(self):
        """Test 80/10/10 stratified split"""
        train_idx, val_idx, test_idx = self.prep.split_data(self.data)

        total = len(self.data)
        assert len(train_idx) + len(val_idx) + len(test_idx) == total

        # Check proportions (approximately 80/10/10)
        assert abs(len(train_idx) / total - 0.8) < 0.02
        assert abs(len(val_idx) / total - 0.1) < 0.02
        assert abs(len(test_idx) / total - 0.1) < 0.02

    def test_stratified_split_preserves_distribution(self):
        """Test that stratified split preserves label distribution"""
        train_idx, val_idx, test_idx = self.prep.split_data(self.data)

        # Get label distributions for each split
        train_labels = [self.data[i]['model_label'] for i in train_idx]
        val_labels = [self.data[i]['model_label'] for i in val_idx]
        test_labels = [self.data[i]['model_label'] for i in test_idx]

        from collections import Counter
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        test_dist = Counter(test_labels)
        full_dist = Counter(d['model_label'] for d in self.data)

        # Check each label is represented in all splits
        for label in full_dist:
            assert label in train_dist, f"Label {label} missing in train"
            assert label in val_dist, f"Label {label} missing in val"
            assert label in test_dist, f"Label {label} missing in test"

    def test_no_data_leakage(self):
        """Test that there's no overlap between splits"""
        train_idx, val_idx, test_idx = self.prep.split_data(self.data)

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        assert len(train_set & val_set) == 0, "Train/Val overlap"
        assert len(train_set & test_set) == 0, "Train/Test overlap"
        assert len(val_set & test_set) == 0, "Val/Test overlap"

    def test_reproducible_split(self):
        """Test that split is reproducible with same seed"""
        train_idx1, val_idx1, test_idx1 = self.prep.split_data(self.data, seed=42)
        train_idx2, val_idx2, test_idx2 = self.prep.split_data(self.data, seed=42)

        assert train_idx1 == train_idx2
        assert val_idx1 == val_idx2
        assert test_idx1 == test_idx2

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits"""
        train_idx1, _, _ = self.prep.split_data(self.data, seed=42)
        train_idx2, _, _ = self.prep.split_data(self.data, seed=123)

        assert train_idx1 != train_idx2

    def test_save_and_load_splits(self, tmp_path):
        """Test saving and loading split indices"""
        train_idx, val_idx, test_idx = self.prep.split_data(self.data)

        split_path = tmp_path / "splits.json"
        self.prep.save_splits(train_idx, val_idx, test_idx, str(split_path))

        # Load back
        loaded = self.prep.load_splits(str(split_path))

        assert loaded['train'] == train_idx
        assert loaded['val'] == val_idx
        assert loaded['test'] == test_idx

    def test_save_and_load_label_mapping(self, tmp_path):
        """Test saving and loading label mapping"""
        mapping = LabelMapping()
        mapping_path = tmp_path / "label_mapping.json"
        mapping.save(str(mapping_path))

        loaded = LabelMapping.load(str(mapping_path))

        # Test that mapping is preserved
        for label in [0, 1, 2]:
            assert loaded.map_to_difficulty(label) == mapping.map_to_difficulty(label)

    def test_prepare_training_data(self):
        """Test preparing final training data with embeddings"""
        # This will be tested more thoroughly after embedding integration
        train_idx, val_idx, test_idx = self.prep.split_data(self.data)

        # Get difficulty labels for each split
        train_difficulties = [self.prep.label_mapping.map_to_difficulty(
            self.data[i]['model_label']) for i in train_idx]

        # Check all three classes present in train
        assert "easy" in train_difficulties
        assert "medium" in train_difficulties
        assert "hard" in train_difficulties


class TestLabelMapping:
    """Tests for LabelMapping class"""

    def test_default_mapping(self):
        """Test default model_label to difficulty mapping"""
        mapping = LabelMapping()

        # model_label -> single difficulty (matches config.yaml model registry)
        assert mapping.map_to_difficulty(0) == "easy"    # qwen3_1.7b
        assert mapping.map_to_difficulty(2) == "medium"  # qwen3_4b
        assert mapping.map_to_difficulty(1) == "hard"    # llama3.2_3b

    def test_one_hot_conversion(self):
        """Test difficulty to one-hot conversion"""
        mapping = LabelMapping()

        assert mapping.to_one_hot("easy") == (1, 0, 0)
        assert mapping.to_one_hot("medium") == (0, 1, 0)
        assert mapping.to_one_hot("hard") == (0, 0, 1)

    def test_invalid_label(self):
        """Test handling of invalid/unknown label"""
        mapping = LabelMapping()

        # Unknown label should default to medium
        result = mapping.map_to_difficulty(99)
        assert result == "medium"  # Default fallback

    def test_custom_mapping(self):
        """Test custom mapping configuration"""
        custom_rules = {
            0: "hard",    # Override default
            1: "medium",
            2: "easy",
        }
        mapping = LabelMapping(custom_rules=custom_rules)

        assert mapping.map_to_difficulty(0) == "hard"
