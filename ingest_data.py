"""
Data Ingestion Script for RAG Application
Processes individual text files (pages) and stores them in a Chroma vector database
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from pathlib import Path
from typing import List


class DataIngestion:
    def __init__(
        self,
        data_directory: str,
        chroma_db_path: str = "./chroma_db",
        embedding_model: str = "mxbai-embed-large",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the data ingestion process

        Args:
            data_directory: Directory containing text files (e.g., 1.txt, 2.txt, etc.)
            chroma_db_path: Path where Chroma database will be stored
            embedding_model: Name of the embedding model (mxbai-embed-large)
            ollama_base_url: URL of the Ollama instance
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.data_directory = Path(data_directory)
        self.chroma_db_path = chroma_db_path
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_text_files(self) -> List[Document]:
        """
        Load all text files from the data directory

        Returns:
            List of LangChain Document objects
        """
        if not self.data_directory.exists():
            raise ValueError(f"Data directory not found: {self.data_directory}")

        documents = []
        text_files = sorted(
            self.data_directory.glob("*.txt"),
            key=lambda x: self._extract_page_number(x.name)
        )

        if not text_files:
            raise ValueError(f"No .txt files found in {self.data_directory}")

        print(f"Found {len(text_files)} text files")

        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract page number from filename
                page_number = self._extract_page_number(file_path.name)

                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path.name),
                        "page": page_number,
                        "file_path": str(file_path)
                    }
                )
                documents.append(doc)

                print(f"  Loaded: {file_path.name} (Page {page_number})")

            except Exception as e:
                print(f"  Error loading {file_path.name}: {str(e)}")

        return documents

    @staticmethod
    def _extract_page_number(filename: str) -> int:
        """
        Extract page number from filename (e.g., '4.txt' -> 4)

        Args:
            filename: Name of the file

        Returns:
            Page number as integer, or 0 if not found
        """
        match = re.search(r'(\d+)\.txt$', filename)
        if match:
            return int(match.group(1))
        return 0

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        print(f"\nSplitting {len(documents)} documents into chunks...")

        chunked_docs = self.text_splitter.split_documents(documents)

        print(f"Created {len(chunked_docs)} chunks")

        return chunked_docs

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create and persist a Chroma vector store from documents

        Args:
            documents: List of Document objects

        Returns:
            Chroma vector store
        """
        print(f"\nCreating Chroma vector store at {self.chroma_db_path}...")
        print(f"Using embedding model: {self.embedding_model}")
        print("This may take a few minutes depending on the number of documents...")

        # Create the vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.chroma_db_path
        )

        print(f"Vector store created successfully!")
        print(f"Total documents in store: {vectorstore._collection.count()}")

        return vectorstore

    def ingest(self, split_text: bool = True):
        """
        Run the full ingestion pipeline

        Args:
            split_text: Whether to split documents into chunks (recommended for large pages)
        """
        print("=" * 60)
        print("Starting Data Ingestion Process")
        print("=" * 60)
        print(f"Data Directory: {self.data_directory}")
        print(f"Chroma DB Path: {self.chroma_db_path}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Chunk Overlap: {self.chunk_overlap}")
        print("=" * 60)
        print()

        # Load documents
        documents = self.load_text_files()

        if not documents:
            print("No documents loaded. Exiting.")
            return

        # Split documents if requested
        if split_text:
            documents = self.split_documents(documents)
        else:
            print("\nSkipping text splitting (using whole pages)")

        # Create vector store
        self.create_vectorstore(documents)

        print("\n" + "=" * 60)
        print("Ingestion Complete!")
        print("=" * 60)
        print(f"\nYou can now query your data using rag_app.py")


def main():
    """Main function to run data ingestion"""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest text files into Chroma vector database")
    parser.add_argument("data_dir", help="Directory containing text files (e.g., 1.txt, 2.txt, etc.)")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to store Chroma database (default: ./chroma_db)")
    parser.add_argument("--embedding-model", default="mxbai-embed-large", help="Embedding model name (default: mxbai-embed-large)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    parser.add_argument("--no-split", action="store_true", help="Don't split text into chunks (use whole pages)")

    args = parser.parse_args()

    try:
        ingestion = DataIngestion(
            data_directory=args.data_dir,
            chroma_db_path=args.db_path,
            embedding_model=args.embedding_model,
            ollama_base_url=args.ollama_url,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        ingestion.ingest(split_text=not args.no_split)

    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
