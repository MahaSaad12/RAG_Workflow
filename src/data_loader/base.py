from abc import ABC, abstractmethod
from typing import List, Union, Any

class BaseEmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.
    Defines the required interface for any embedding model implementation.
    """

    def _init_(self, model_name: str = None):
        self.model_name = model_name or self.default_model_name()
    
    @abstractmethod
    def default_model_name(self) -> str:
        """
        Return the default model name for the embedding model.
        """
        pass

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the input text(s).
        
        Args:
            texts (str or List[str]): A single string or a list of strings to embed.
        
        Returns:
            List[float] or List[List[float]]: Embedding(s) for the input text(s).
        """
        pass

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Default batching logic for models that support single embedding at a time.
        Override this in subclasses if the model supports batch embedding natively.
        
        Args:
            texts (List[str]): List of texts to embed.
            batch_size (int): Number of texts per batch.

        Returns:
            List[List[float]]: List of embeddings.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed(batch)
            if isinstance(batch_embeddings[0], float):  # In case only one embedding was returned
                batch_embeddings = [batch_embeddings]
            embeddings.extend(batch_embeddings)
        return embeddings