import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "policy_chunks.json"
)

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDINGS_FILE = (
    VECTORSTORE_DIR
    / "policy_embeddings.npy"
)

FAISS_INDEX_FILE = (
    VECTORSTORE_DIR
    / "policy.index"
)

METADATA_FILE = (
    VECTORSTORE_DIR
    / "policy_metadata.json"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


# ============================================================
# Load Policy Chunks
# ============================================================

def load_chunks():
    """
    Load policy chunks generated during Day 1.

    Returns:
        list: Policy chunk records.
    """

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Policy chunks file not found: {CHUNKS_FILE}"
        )

    try:
        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            chunks = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in policy chunks file: {error}"
        )

    if not isinstance(chunks, list):
        raise ValueError(
            "Policy chunks must be stored as a list."
        )

    if not chunks:
        raise ValueError(
            "Policy chunks file is empty."
        )

    return chunks


# ============================================================
# Validate Policy Chunks
# ============================================================

def validate_chunks(chunks):
    """
    Validate that every chunk contains the required fields.
    """

    required_fields = {
        "chunk_id",
        "text",
        "metadata"
    }

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} is not a valid dictionary."
            )

        missing_fields = (
            required_fields - chunk.keys()
        )

        if missing_fields:
            raise ValueError(
                f"Chunk {index} is missing fields: "
                f"{missing_fields}"
            )

        if not chunk["text"] or not chunk["text"].strip():
            raise ValueError(
                f"Chunk {index} contains empty text."
            )


# ============================================================
# Load Embedding Model
# ============================================================

def load_embedding_model():
    """
    Load the Sentence Transformer embedding model.

    The model is loaded from the local Hugging Face cache
    whenever available. This prevents network requests when
    running the RAG pipeline or MCP server.
    """

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # 1. Try to locate the model in the Hugging Face cache
    # --------------------------------------------------------

    hf_home = Path(
        os.getenv(
            "HF_HOME",
            Path.home() / ".cache" / "huggingface"
        )
    )

    model_cache_dir = (
        hf_home
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )

    local_model_path = None

    if model_cache_dir.exists():

        snapshots = [
            path
            for path in model_cache_dir.iterdir()
            if path.is_dir()
        ]

        if snapshots:
            # Use the most recently modified snapshot
            local_model_path = max(
                snapshots,
                key=lambda path: path.stat().st_mtime
            )

    # --------------------------------------------------------
    # 2. Load local model if available
    # --------------------------------------------------------

    if local_model_path is not None:

        print(
            f"Using local embedding model:\n"
            f"{local_model_path}"
        )

        try:

            model = SentenceTransformer(
                str(local_model_path),
                local_files_only=True
            )

            print(
                "Embedding model loaded successfully "
                "from local cache."
            )

            return model

        except Exception as error:

            raise RuntimeError(
                "The local Sentence Transformer model "
                "was found but could not be loaded.\n"
                f"Model path: {local_model_path}\n"
                f"Original error: {error}"
            ) from error

    # --------------------------------------------------------
    # 3. Model not available locally
    # --------------------------------------------------------

    raise FileNotFoundError(
        "\nLocal embedding model not found.\n\n"
        "Expected Hugging Face cache location:\n"
        f"{model_cache_dir}\n\n"
        "Download the model once using an environment "
        "with Hugging Face access, then run the application "
        "in offline mode."
    )

# ============================================================
# Generate Embeddings
# ============================================================

def generate_embeddings(chunks, model):
    """
    Generate normalized embeddings for all
    policy chunks.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    if not texts:
        raise ValueError(
            "No text available for embedding."
        )

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Generated embeddings must be a "
            "2-dimensional array."
        )

    return embeddings.astype(
        "float32"
    )


# ============================================================
# Save Embeddings
# ============================================================

def save_embeddings(embeddings):
    """
    Save generated embeddings as a NumPy file.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    print(
        f"Embeddings saved to: "
        f"{EMBEDDINGS_FILE}"
    )


# ============================================================
# Create FAISS Vector Store
# ============================================================

def create_faiss_index(embeddings):
    """
    Create a FAISS index using inner product similarity.

    Because the embeddings are normalized, inner product
    is equivalent to cosine similarity.
    """

    if embeddings is None:
        raise ValueError(
            "Embeddings cannot be None."
        )

    if len(embeddings) == 0:
        raise ValueError(
            "Cannot create FAISS index from empty embeddings."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional array."
        )

    dimension = embeddings.shape[1]

    print(
        f"Creating FAISS index "
        f"with dimension: {dimension}"
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        np.asarray(
            embeddings,
            dtype="float32"
        )
    )

    print(
        "FAISS index created successfully."
    )

    print(
        f"Vectors stored in index: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# Save FAISS Index
# ============================================================

def save_faiss_index(index):
    """
    Save the FAISS index to disk.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE)
    )

    print(
        f"FAISS index saved to: "
        f"{FAISS_INDEX_FILE}"
    )


# ============================================================
# Create Metadata
# ============================================================

def create_metadata(chunks):
    """
    Create metadata corresponding to each vector.

    The order of metadata must remain identical to
    the order of embeddings added to FAISS.
    """

    metadata = []

    for chunk in chunks:

        chunk_metadata = chunk.get(
            "metadata",
            {}
        )

        metadata.append(
            {
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "source": chunk_metadata.get(
                    "source"
                ),
                "policy_type": chunk_metadata.get(
                    "policy_type"
                ),
                "country": chunk_metadata.get(
                    "country"
                ),
                "chunk_index": chunk_metadata.get(
                    "chunk_index"
                ),
                "total_chunks": chunk_metadata.get(
                    "total_chunks"
                ),
                "text": chunk.get(
                    "text"
                )
            }
        )

    return metadata


# ============================================================
# Save Metadata
# ============================================================

def save_metadata(metadata):
    """
    Save policy metadata used by the RAG retrieval layer.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Metadata saved to: "
        f"{METADATA_FILE}"
    )


# ============================================================
# Main Embedding + Vector Store Pipeline
# ============================================================

def main():

    print("=" * 70)
    print("POLICY EMBEDDING + VECTOR STORE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"\nLoaded chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # 2. Validate chunks
    # --------------------------------------------------------

    validate_chunks(chunks)

    print(
        "Chunk validation completed successfully."
    )

    # --------------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # 4. Generate embeddings
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        chunks,
        model
    )

    print("\nEmbedding information:")

    print(
        f"Number of embeddings : "
        f"{embeddings.shape[0]}"
    )

    print(
        f"Embedding dimensions : "
        f"{embeddings.shape[1]}"
    )

    # --------------------------------------------------------
    # 5. Save embeddings
    # --------------------------------------------------------

    save_embeddings(
        embeddings
    )

    # --------------------------------------------------------
    # 6. Create FAISS index
    # --------------------------------------------------------

    index = create_faiss_index(
        embeddings
    )

    # --------------------------------------------------------
    # 7. Save FAISS index
    # --------------------------------------------------------

    save_faiss_index(
        index
    )

    # --------------------------------------------------------
    # 8. Create metadata
    # --------------------------------------------------------

    metadata = create_metadata(
        chunks
    )

    if len(metadata) != index.ntotal:
        raise ValueError(
            "Metadata count does not match "
            "the number of vectors in FAISS."
        )

    # --------------------------------------------------------
    # 9. Save metadata
    # --------------------------------------------------------

    save_metadata(
        metadata
    )

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Policy chunks       : {len(chunks)}"
    )

    print(
        f"Embeddings          : {embeddings.shape}"
    )

    print(
        f"FAISS vectors       : {index.ntotal}"
    )

    print(
        f"Embedding model     : {EMBEDDING_MODEL}"
    )

    print(
        "\nVector store files:"
    )

    print(
        f"  - {EMBEDDINGS_FILE}"
    )

    print(
        f"  - {FAISS_INDEX_FILE}"
    )

    print(
        f"  - {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()