import json
import os
from pathlib import Path

import faiss
import numpy as np

# Use the centralized embedding model loader.
# This ensures RAG uses the local Hugging Face cache
# instead of trying to connect to Hugging Face.
from src.embeddings import load_embedding_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_FILE = BASE_DIR / "vectorstore" / "policy.index"
METADATA_FILE = BASE_DIR / "vectorstore" / "policy_metadata.json"

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

TOP_K = int(
    os.getenv("TOP_K", "3")
)

MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.35")
)


# --------------------------------------------------
# Policy Retriever
# --------------------------------------------------

class PolicyRetriever:
    """
    Retrieves relevant policy chunks using
    semantic search with Sentence Transformers
    and FAISS.

    This class is responsible ONLY for retrieval.

    It does not generate the final answer.
    """

    def __init__(self):

        # ------------------------------------------
        # Validate vector store files
        # ------------------------------------------

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}"
            )

        # ------------------------------------------
        # Load FAISS index
        # ------------------------------------------

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        if self.index.ntotal == 0:
            raise ValueError(
                "FAISS index contains no vectors."
            )

        # ------------------------------------------
        # Load metadata
        # ------------------------------------------

        print("Loading metadata...")

        try:

            with open(
                METADATA_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                self.metadata = json.load(file)

        except json.JSONDecodeError as error:

            raise ValueError(
                f"Invalid metadata JSON: {error}"
            )

        if not isinstance(
            self.metadata,
            list
        ):

            raise ValueError(
                "Policy metadata must be a list."
            )

        # ------------------------------------------
        # Validate index and metadata
        # ------------------------------------------

        if self.index.ntotal != len(
            self.metadata
        ):

            raise ValueError(
                "FAISS index count does not "
                "match metadata count."
            )

        # ------------------------------------------
        # Load embedding model
        # ------------------------------------------
        #
        # IMPORTANT:
        # Do NOT directly use:
        #
        # SentenceTransformer(MODEL_NAME)
        #
        # because that can attempt to contact
        # Hugging Face.
        #
        # Instead, use the centralized loader
        # from src.embeddings.py.
        # ------------------------------------------

        print(
            f"Loading embedding model: {MODEL_NAME}"
        )

        self.model = load_embedding_model()

        print(
            "Policy retriever initialized successfully."
        )

    # --------------------------------------------------
    # Semantic Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K
    ):
        """
        Search for the most relevant policy chunks.

        Parameters
        ----------
        query : str
            Employee's policy-related question.

        top_k : int
            Number of results to retrieve.

        Returns
        -------
        list
            Retrieved policy chunks with scores.
        """

        # ------------------------------------------
        # Validate query
        # ------------------------------------------

        if not query or not query.strip():

            raise ValueError(
                "Search query cannot be empty."
            )

        query = query.strip()

        # ------------------------------------------
        # Validate top_k
        # ------------------------------------------

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )

        top_k = min(
            top_k,
            self.index.ntotal
        )

        # ------------------------------------------
        # Generate query embedding
        # ------------------------------------------

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # ------------------------------------------
        # Validate embedding dimension
        # ------------------------------------------

        if query_embedding.shape[1] != self.index.d:

            raise ValueError(
                "Query embedding dimension "
                f"{query_embedding.shape[1]} does not "
                f"match FAISS index dimension "
                f"{self.index.d}."
            )

        # ------------------------------------------
        # FAISS similarity search
        # ------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        # ------------------------------------------
        # Process search results
        # ------------------------------------------

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            score = float(score)

            # Ignore weak matches
            if score < MIN_RELEVANCE_SCORE:
                continue

            result = self.metadata[
                index
            ].copy()

            result["score"] = score

            results.append(result)

        return results

    # --------------------------------------------------
    # Build Context
    # --------------------------------------------------

    def build_context(
        self,
        results
    ):
        """
        Convert retrieved policy chunks into
        structured context for the agent/LLM.
        """

        if not results:
            return ""

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            source = result.get(
                "source",
                "Unknown"
            )

            policy_type = result.get(
                "policy_type",
                "Unknown"
            )

            country = result.get(
                "country",
                "Unknown"
            )

            score = result.get(
                "score",
                0
            )

            text = result.get(
                "text",
                ""
            ).strip()

            context_parts.append(
                f"""
--- Policy Result {i} ---

Source: {source}
Policy Type: {policy_type}
Country: {country}
Relevance Score: {score:.4f}

Policy Content:
{text}
"""
            )

        return "\n".join(
            context_parts
        )

    # --------------------------------------------------
    # Retrieve Context
    # --------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        top_k: int = TOP_K
    ):
        """
        Retrieve policy chunks and build context.

        Returns
        -------
        tuple
            (context, results)
        """

        results = self.search(
            query=query,
            top_k=top_k
        )

        if not results:

            return "", []

        context = self.build_context(
            results
        )

        return context, results


# --------------------------------------------------
# Global Retriever
# --------------------------------------------------

_retriever = None


def get_retriever():
    """
    Create the PolicyRetriever lazily.

    This prevents the embedding model and FAISS
    index from being loaded repeatedly.
    """

    global _retriever

    if _retriever is None:

        _retriever = PolicyRetriever()

    return _retriever


# --------------------------------------------------
# Agent Tool: Search Policy
# --------------------------------------------------

def search_policy(
    query: str,
    top_k: int = TOP_K
):
    """
    Search company policies using semantic search.

    This function is intended to be exposed as a
    LangChain tool.

    The function performs retrieval only.

    It does NOT generate an answer.
    """

    if not query or not query.strip():

        return {
            "success": False,
            "error": "Search query cannot be empty.",
            "results": []
        }

    try:

        retriever = get_retriever()

        results = retriever.search(
            query=query,
            top_k=top_k
        )

        if not results:

            return {
                "success": True,
                "message": (
                    "No sufficiently relevant "
                    "policy information was found."
                ),
                "results": []
            }

        formatted_results = []

        for result in results:

            formatted_results.append(
                {
                    "chunk_id": result.get(
                        "chunk_id"
                    ),
                    "source": result.get(
                        "source"
                    ),
                    "policy_type": result.get(
                        "policy_type"
                    ),
                    "country": result.get(
                        "country"
                    ),
                    "relevance_score": round(
                        result.get(
                            "score",
                            0
                        ),
                        4
                    ),
                    "text": result.get(
                        "text",
                        ""
                    )
                }
            )

        return {
            "success": True,
            "results": formatted_results
        }

    except Exception as error:

        print(
            f"Policy retrieval error: {error}"
        )

        return {
            "success": False,
            "error": (
                "Policy retrieval failed. "
                "Please try again."
            ),
            "results": []
        }


# --------------------------------------------------
# Utility Function
# --------------------------------------------------

def retrieve_policy_context(
    query: str,
    top_k: int = TOP_K
):
    """
    Retrieve policy context for internal application use.

    Returns context and raw retrieval results.
    """

    retriever = get_retriever()

    return retriever.retrieve_context(
        query=query,
        top_k=top_k
    )


# --------------------------------------------------
# Test
# --------------------------------------------------

def main():

    print("=" * 70)
    print("POLICY RETRIEVAL TEST")
    print("=" * 70)

    questions = [

        "What is the airport travel reimbursement limit in India?",

        "What is the standard travel limit in the United States?",

        "Can an employee travel after 10 PM?",

        "What is the hotel reimbursement limit?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUERY")
        print("=" * 70)

        print(question)

        try:

            result = search_policy(
                query=question,
                top_k=3
            )

            print("\nRESULT:")

            if not result["results"]:

                print(
                    "No relevant policy found."
                )

                continue

            for item in result["results"]:

                print(
                    f"\nSource: {item['source']}"
                )

                print(
                    f"Country: {item['country']}"
                )

                print(
                    f"Score: "
                    f"{item['relevance_score']}"
                )

                print(
                    f"Text: {item['text'][:300]}..."
                )

        except Exception as error:

            print(
                f"\nError: {error}"
            )


if __name__ == "__main__":
    main()
