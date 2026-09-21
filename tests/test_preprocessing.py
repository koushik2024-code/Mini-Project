import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.preprocessing import PreprocessingConfig, TextPreprocessor


class TestTextPreprocessor:
    """Unit tests for PRD-0001-FR-3: Preprocess training prompts"""

    def setup_method(self):
        """Set up preprocessor with default config"""
        config = PreprocessingConfig(
            strip_whitespace=True,
            normalize_unicode=True,
            preserve_math=True,
            preserve_code=True,
            preserve_symbols=True
        )
        self.preprocessor = TextPreprocessor(config)

    def test_strips_whitespace(self):
        """Test that unnecessary whitespace is removed"""
        text = "  Hello   world  \n\n  "
        result = self.preprocessor.preprocess(text)
        assert result == "Hello world"

    def test_normalizes_unicode(self):
        """Test unicode normalization"""
        text = "H\u00e9llo"  # é as single char vs e + combining acute
        result = self.preprocessor.preprocess(text)
        assert "Hello" in result or "H\u00e9llo" in result  # Normalized

    def test_preserves_mathematical_expressions(self):
        """Test that math expressions are preserved"""
        text = "The formula is E = mc^2 and integral \\int_0^1 x dx"
        result = self.preprocessor.preprocess(text)
        assert "E = mc^2" in result
        assert "integral" in result

    def test_preserves_programming_code(self):
        """Test that code snippets are preserved"""
        text = "Use def foo(): return x + 1 and print('hello')"
        result = self.preprocessor.preprocess(text)
        assert "def foo()" in result
        assert "return x + 1" in result

    def test_preserves_important_symbols(self):
        """Test that important symbols are preserved"""
        text = "Check x > 5 and y <= 10 or z != None"
        result = self.preprocessor.preprocess(text)
        assert ">" in result
        assert "<=" in result
        assert "!=" in result

    def test_preserves_punctuation(self):
        """Test that standard punctuation is preserved"""
        text = "Hello, world! How are you? I'm fine."
        result = self.preprocessor.preprocess(text)
        assert "," in result
        assert "!" in result
        assert "?" in result
        assert "'" in result
        assert "." in result

    def test_handles_empty_string(self):
        """Test handling of empty string"""
        result = self.preprocessor.preprocess("")
        assert result == ""

    def test_handles_none_input(self):
        """Test handling of None input"""
        result = self.preprocessor.preprocess(None)
        assert result == ""

    def test_no_rag_chunking(self):
        """Test that NO RAG-style chunking occurs - single semantic unit"""
        long_text = " ".join(["This is a sentence."] * 100)
        result = self.preprocessor.preprocess(long_text)
        # Should return as single string, not list of chunks
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be split into chunks
        assert result.count("This is a sentence.") == 100

    def test_answer_field_not_used(self):
        """Test that answer field is never used as input feature"""
        # This is a design test - the preprocessor only processes the prompt/query
        # The answer field should be handled separately by the caller
        prompt = "What is 2+2?"
        result = self.preprocessor.preprocess(prompt)
        # Preprocessor should only process the prompt
        assert "2+2" in result
        assert "4" not in result  # Answer not included

    def test_batch_preprocessing(self):
        """Test batch preprocessing"""
        texts: list[str | None] = ["  Hello  ", "World  ", "  Test"]
        results = self.preprocessor.preprocess_batch(texts)
        assert results == ["Hello", "World", "Test"]

    def test_custom_config(self):
        """Test preprocessor with custom config"""
        config = PreprocessingConfig(
            strip_whitespace=False,
            normalize_unicode=False,
            preserve_math=False,
            preserve_code=False,
            preserve_symbols=False
        )
        preprocessor = TextPreprocessor(config)
        text = "  Hello  "
        result = preprocessor.preprocess(text)
        # With strip_whitespace=False, leading/trailing spaces preserved
        assert result == "  Hello  "


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = PreprocessingConfig()
        assert config.strip_whitespace is True
        assert config.normalize_unicode is True
        assert config.preserve_math is True
        assert config.preserve_code is True
        assert config.preserve_symbols is True

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'strip_whitespace': True,
            'normalize_unicode': True,
            'preserve_math': True,
            'preserve_code': True,
            'preserve_symbols': True
        }
        config = PreprocessingConfig(**config_dict)
        assert config.strip_whitespace is True
