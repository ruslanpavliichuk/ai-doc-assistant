from transformers import AutoTokenizer
import re


def split_paragraphs(text: str) -> list[str]:
    """Splits text by two or more newline characters."""
    return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]


class TokenChunker:
    """
    A class to split text into chunks based on token count using a Hugging Face tokenizer.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the chunker and loads the tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model whose tokenizer will be used.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chunk(self, text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> list[str]:
        """
        Splits a text into chunks of a specified token size with overlap.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The desired number of tokens in each chunk.
            chunk_overlap (int): The number of tokens to overlap between consecutive chunks.

        Returns:
            list[str]: A list of text chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        # Encode the text into token IDs, without adding special tokens like [CLS] or [SEP]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        # Iterate through tokens with a step equal to chunk_size minus overlap
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if not chunk_tokens:
                continue

            # Decode the token chunk back into a string
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks
