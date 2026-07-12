import pytest

from src.ai.text_splitter import RecursiveCharacterTextSplitter


class TestRecursiveCharacterTextSplitter:

    def setup_method(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
        )

    def test_split_text_by_separators(self):
        text = "Hello world, this is a test of the recursive text splitter."
        result = self.splitter.split_text(text)
        assert result == [
            "Hello world",
            "this is a test of the recursive text splitter",
        ]

    def test_split_text_updated_chunk_size(self):
        self.splitter.chunk_size = 100
        text = (
            "Hello world, this is a test of the recursive text splitter. "
            "If I have a period, it should split along the period."
        )
        result = self.splitter.split_text(text)
        assert result == [
            "Hello world, this is a test of the recursive text splitter",
            "If I have a period, it should split along the period.",
        ]

    def test_split_text_with_newlines(self):
        self.splitter.chunk_size = 110
        text = (
            "Hello world, this is a test of the recursive text splitter. "
            "If I have a period, it should split along the period.\n"
            "Or, if there is a new line, it should prioritize splitting on new lines instead."
        )
        result = self.splitter.split_text(text)
        assert result == [
            "Hello world, this is a test of the recursive text splitter",
            "If I have a period, it should split along the period.",
            "Or, if there is a new line, it should prioritize splitting on new lines instead.",
        ]

    def test_empty_string(self):
        assert self.splitter.split_text("") == []

    def test_special_characters_and_large_texts(self):
        large_text = "A" * 1000
        self.splitter.chunk_size = 200
        result = self.splitter.split_text(large_text)
        assert result == ["A" * 200] * 5

        special_text = "Hello!@# world$%^ &*( this) is+ a-test"
        result = self.splitter.split_text(special_text)
        assert result == ["Hello!@#", "world$%^", "&*( this)", "is+", "a-test"]

    def test_chunk_size_equal_to_chunk_overlap(self):
        self.splitter.chunk_size = 50
        self.splitter.chunk_overlap = 50
        with pytest.raises(ValueError, match="Cannot have chunkOverlap >= chunkSize"):
            self.splitter.split_text("Invalid configuration")
