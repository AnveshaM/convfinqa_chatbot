import argparse
import os
import shutil
import json
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from get_embeddings import get_embeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data/train_subset.json"


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database and regenerate embeddings.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()
        print("✨ Database Cleared")

    # Create (or update) the data store.
    chunks = load_chunks()
    add_to_chroma(chunks)


def load_chunks():
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    chunks = []
    num_missing_entries = 0  # Counter for entries with missing keys

    for entry in data:
        chunk_id = entry["id"]

        # Check if any required keys are missing
        missing_keys = []
        required_keys = ["pre_text", "table", "post_text", "filename", "table_ori", "qa", "annotation"]
        for key in required_keys:
            if key not in entry:
                missing_keys.append(key)

        if missing_keys:
            num_missing_entries += 1  # Increment the counter
            # Print a message for the first missing entry
            if num_missing_entries == 1:
                print("\nEntries with missing keys:")
            print(f"  Entry {chunk_id}: {missing_keys}")
            # Skip this entry if any required key is missing
            continue

        # Concatenate all relevant fields into chunk text
        chunk_text = f"{entry['pre_text']} {entry['table']} {entry['post_text']} {entry['filename']}{entry['table_ori']}{entry['qa']}{entry['annotation']}"

        # Assuming 'page_content' refers to the chunk text
        page_content = chunk_text
        # Create a Document object for each entry with its ID, text, and page content
        chunk = Document(page_content=chunk_text, metadata={"id": chunk_id})
        chunks.append(chunk)

    # Print the total number of entries with missing keys
    print(f"\nTotal entries with missing keys: {num_missing_entries}")
    print(len(chunks))

    return chunks


def add_to_chroma(chunks):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embeddings()
    )

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
