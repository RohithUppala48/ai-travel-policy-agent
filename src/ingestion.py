"""
Policy document ingestion, cleaning, metadata extraction and chunking.

This module:
1. Loads .txt policy documents
2. Handles missing/empty files
3. Cleans policy text
4. Adds metadata
5. Splits documents into chunks
6. Saves processed documents and chunks
"""

from pathlib import Path
import json
import re
import logging
from typing import List, Dict, Optional


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POLICY_DIR = PROJECT_ROOT / "data" / "company_policy"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CLEANED_FILE = OUTPUT_DIR / "cleaned_policies.json"
CHUNKS_FILE = OUTPUT_DIR / "policy_chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# POLICY METADATA
# ---------------------------------------------------------

POLICY_METADATA = {
    "travel_policy_india.txt": {
        "policy_type": "travel",
        "country": "India"
    },
    "travel_policy_us.txt": {
        "policy_type": "travel",
        "country": "United States"
    },
    "airport_policy.txt": {
        "policy_type": "airport",
        "country": "Global"
    },
    "employee_eligibility.txt": {
        "policy_type": "eligibility",
        "country": "Global"
    },
    "expense_policy.txt": {
        "policy_type": "expense",
        "country": "Global"
    },
    "cancellation_policy.txt": {
        "policy_type": "cancellation",
        "country": "Global"
    },
    "approval_policy.txt": {
        "policy_type": "approval",
        "country": "Global"
    }
}


# ---------------------------------------------------------
# EXPECTED POLICY DOCUMENTS
# ---------------------------------------------------------

EXPECTED_FILES = [
    "travel_policy_india.txt",
    "travel_policy_us.txt",
    "airport_policy.txt",
    "employee_eligibility.txt",
    "expense_policy.txt",
    "cancellation_policy.txt",
    "approval_policy.txt"
]


# ---------------------------------------------------------
# DOCUMENT LOADING
# ---------------------------------------------------------

def load_document(file_path: Path) -> Optional[str]:
    """
    Load a single text document.

    Handles:
    - Missing files
    - Empty files
    - Whitespace-only files
    - Encoding issues
    """

    if not file_path.exists():
        logger.warning(
            "Policy document does not exist: %s",
            file_path
        )
        return None

    if not file_path.is_file():
        logger.warning(
            "Path is not a file: %s",
            file_path
        )
        return None

    try:
        content = file_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decoding failed for %s. Trying latin-1.",
            file_path
        )

        try:
            content = file_path.read_text(
                encoding="latin-1"
            )

        except Exception as exc:
            logger.error(
                "Could not read %s: %s",
                file_path,
                exc
            )
            return None

    except OSError as exc:
        logger.error(
            "Error reading %s: %s",
            file_path,
            exc
        )
        return None

    if not content.strip():
        logger.warning(
            "Document is empty: %s",
            file_path
        )
        return None

    return content


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean policy text while preserving meaningful
    headings, numbers and policy information.
    """

    if not text:
        return ""

    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n[ ]*\n[ ]*\n+", "\n\n", text)

    # Remove spaces at beginning/end of lines
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # Final cleanup
    text = text.strip()

    return text


# ---------------------------------------------------------
# METADATA
# ---------------------------------------------------------

def get_metadata(file_name: str) -> Dict:
    """
    Return metadata for a policy document.
    """

    metadata = POLICY_METADATA.get(
        file_name,
        {
            "policy_type": "unknown",
            "country": "Unknown"
        }
    )

    return {
        "source": file_name,
        "policy_type": metadata["policy_type"],
        "country": metadata["country"]
    }


# ---------------------------------------------------------
# LOAD ALL POLICY DOCUMENTS
# ---------------------------------------------------------

def ingest_documents() -> List[Dict]:
    """
    Load all expected policy documents.

    Returns a list containing:
    - source
    - raw_content
    - cleaned_content
    - metadata
    """

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory does not exist: {POLICY_DIR}"
        )

    documents = []

    for file_name in EXPECTED_FILES:

        file_path = POLICY_DIR / file_name

        raw_content = load_document(file_path)

        if raw_content is None:
            logger.warning(
                "Skipping document: %s",
                file_name
            )
            continue

        cleaned_content = clean_text(raw_content)

        if not cleaned_content:
            logger.warning(
                "Cleaned content is empty: %s",
                file_name
            )
            continue

        metadata = get_metadata(file_name)

        document = {
            "source": file_name,
            "raw_content": raw_content,
            "content": cleaned_content,
            "metadata": metadata,
            "character_count": len(cleaned_content)
        }

        documents.append(document)

        logger.info(
            "Loaded: %-35s | Characters: %d",
            file_name,
            len(cleaned_content)
        )

    return documents


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks.

    Uses paragraph boundaries where possible.

    Parameters:
        text: document text
        chunk_size: maximum approximate characters per chunk
        overlap: overlapping characters between chunks
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding the paragraph still fits
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            # Handle very large paragraphs
            if len(paragraph) > chunk_size:

                start = 0

                while start < len(paragraph):

                    end = start + chunk_size

                    piece = paragraph[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    start = end - overlap

                current_chunk = ""

            else:
                current_chunk = paragraph

    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------------------------
# CREATE CHUNK RECORDS
# ---------------------------------------------------------

def create_chunks(
    documents: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict]:

    all_chunks = []

    for document in documents:

        chunks = chunk_text(
            document["content"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(chunks):

            chunk_record = {
                "chunk_id": (
                    f"{Path(document['source']).stem}"
                    f"_chunk_{index}"
                ),
                "text": chunk,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": index,
                    "total_chunks": len(chunks)
                }
            }

            all_chunks.append(chunk_record)

    return all_chunks


# ---------------------------------------------------------
# SAVE JSON
# ---------------------------------------------------------

def save_json(
    data,
    output_path: Path
):
    """
    Save Python object as formatted JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    logger.info(
        "Saved: %s",
        output_path
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

def print_summary(
    documents: List[Dict],
    chunks: List[Dict]
):
    """
    Print ingestion summary.
    """

    print("\n" + "=" * 70)
    print("POLICY INGESTION SUMMARY")
    print("=" * 70)

    print(
        f"Expected documents : {len(EXPECTED_FILES)}"
    )

    print(
        f"Loaded documents   : {len(documents)}"
    )

    print(
        f"Total chunks       : {len(chunks)}"
    )

    print("\nDocuments:")
    print("-" * 70)

    for document in documents:

        source = document["source"]
        characters = document["character_count"]

        document_chunks = [
            chunk
            for chunk in chunks
            if chunk["metadata"]["source"] == source
        ]

        print(
            f"{source:<35}"
            f" {characters:>7} chars"
            f" | {len(document_chunks):>3} chunks"
        )

    print("=" * 70)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():

    logger.info(
        "Starting policy document ingestion..."
    )

    documents = ingest_documents()

    if not documents:
        raise RuntimeError(
            "No valid policy documents were loaded."
        )

    chunks = create_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    if not chunks:
        raise RuntimeError(
            "No chunks were generated."
        )

    # Save cleaned documents
    cleaned_documents = []

    for document in documents:

        cleaned_documents.append(
            {
                "source": document["source"],
                "content": document["content"],
                "metadata": document["metadata"],
                "character_count": document["character_count"]
            }
        )

    save_json(
        cleaned_documents,
        CLEANED_FILE
    )

    # Save chunks
    save_json(
        chunks,
        CHUNKS_FILE
    )

    print_summary(
        documents,
        chunks
    )

    logger.info(
        "Policy ingestion completed successfully."
    )


if __name__ == "__main__":
    main()