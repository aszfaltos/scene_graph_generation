import torch
from sentence_transformers import SentenceTransformer


def obj_edge_vectors_minilm(names, wv_dir, wv_dim=384):
    """
    Generate class name embeddings using all-MiniLM-L6-v2 sentence transformer.

    :param names: List of object class names
    :param wv_dir: Path to pretrained MiniLM model directory
    :param wv_dim: Embedding dimension (default 384 for MiniLM-L6-v2)
    :return: Tensor of shape (len(names), wv_dim)
    """
    # Load sentence transformer model
    model = SentenceTransformer(wv_dir)

    # Initialize with random vectors (fallback)
    vectors = torch.randn(len(names), wv_dim)

    # Encode all names at once (more efficient than one-by-one)
    with torch.no_grad():
        embeddings = model.encode(names, convert_to_tensor=True)
        vectors = embeddings.float()

    return vectors  # (len(names), 384)
