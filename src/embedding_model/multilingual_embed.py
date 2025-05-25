from typing import List, Union
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path

# Add src/ to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from embedding_model.embed_base import BaseEmbeddingModel



class MultilingualEmbeddingModel(BaseEmbeddingModel):
    """
    A concrete implementation of BaseEmbeddingModel using a multilingual model
    from the Sentence Transformers library.

    Specifically, it loads 'distiluse-base-multilingual-cased-v1', 
    which supports English, German, and several other languages.

    Attributes:
        model (SentenceTransformer): The underlying sentence-transformers model.
    """

    def __init__(self, model_path: Union[str, Path] = "sentence-transformers/distiluse-base-multilingual-cased-v1") -> None:
        """
        Initialize the multilingual embedding model.

        Args:
            model_path (Union[str, Path], optional): Path or model name.
                Defaults to the pretrained multilingual model.
        """
        super().__init__(model_path)
        self.model: SentenceTransformer = None  # Will be loaded in load_model()
        self.model_name = model_path

    def name(self) -> str:
        return self.model_name

    def load_model(self) -> None:
        """
        Load the sentence-transformers multilingual model from HuggingFace
        or from a local path.
        """
        self.model = SentenceTransformer(str(self.model_name))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): The input texts to embed.

        Returns:
            List[List[float]]: A list of vector embeddings.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() before embedding.")
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def save_model(self, save_path: Union[str, Path] = None) -> None:
        """
        Save the current model to disk.

        Args:
            save_path (Union[str, Path], optional): The destination path to save the model.
                If not provided, defaults to the original `model_path`.
        """
        save_path = Path(save_path) if save_path else self.model_path
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))


if __name__ == "__main__":
    # Example usage and testing
    model = MultilingualEmbeddingModel()
    model.load_model()

    sample_texts = [
        "This is a test sentence in English.",
        "Dies ist ein Testsatz auf Deutsch."
    ]




    for idx, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
        print(f"Text {idx + 1}: {text}")
        print(f"Embedding (first 5 dimensions): {embedding[:5]}\n")