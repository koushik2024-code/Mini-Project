import re
import unicodedata
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing"""
    strip_whitespace: bool = True
    normalize_unicode: bool = True
    preserve_math: bool = True
    preserve_code: bool = True
    preserve_symbols: bool = True


class TextPreprocessor:
    """
    Preprocess text for embedding generation.

    Key principles:
    - NO RAG-style chunking: each query is one complete semantic unit
    - Preserve mathematical expressions, code, and important symbols
    - Answer field is NEVER used as input feature
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

        # Patterns for preservation
        self.math_pattern = re.compile(r'\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]')
        self.code_pattern = re.compile(r'`[^`]+`|```[\s\S]*?```')
        self.symbol_pattern = re.compile(r'[<>=!]=|[+\-*/^%&|]|[{}[\]]')

    def preprocess(self, text: str | None) -> str:
        """
        Preprocess a single text string.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        # Step 1: Unicode normalization
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)

        # Step 2: Strip unnecessary whitespace
        if self.config.strip_whitespace:
            text = self._strip_whitespace(text)

        # Note: We do NOT chunk the text. Each query is treated as
        # one complete semantic unit (no RAG-style fixed-size chunking).

        return text

    def preprocess_batch(self, texts: list[str | None]) -> list[str]:
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to NFC form"""
        return unicodedata.normalize('NFC', text)

    def _strip_whitespace(self, text: str) -> str:
        """Strip unnecessary whitespace while preserving internal spacing"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing
        text = text.strip()
        return text


def load_preprocessing_config(config_path: str = "config.yaml") -> PreprocessingConfig:
    """Load preprocessing config from YAML file"""
    from pathlib import Path

    import yaml

    path = Path(config_path)
    if not path.exists():
        return PreprocessingConfig()

    with open(path) as f:
        config = yaml.safe_load(f)

    prep_config = config.get('data', {}).get('preprocessing', {})
    return PreprocessingConfig(
        strip_whitespace=prep_config.get('strip_whitespace', True),
        normalize_unicode=prep_config.get('normalize_unicode', True),
        preserve_math=prep_config.get('preserve_math', True),
        preserve_code=prep_config.get('preserve_code', True),
        preserve_symbols=prep_config.get('preserve_symbols', True)
    )


if __name__ == "__main__":
    # Quick test
    config = PreprocessingConfig()
    preprocessor = TextPreprocessor(config)

    test_cases = [
        "  Hello   world  ",
        "E = mc^2",
        "def foo(): return x + 1",
        "x > 5 and y <= 10",
        None,
        "",
        "What is 2+2?",
    ]

    for tc in test_cases:
        result = preprocessor.preprocess(tc)
        print(f"Input: {repr(tc)}")
        print(f"Output: {repr(result)}")
        print()
