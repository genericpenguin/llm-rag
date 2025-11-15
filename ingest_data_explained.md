# Data Ingestion Script Explained

This document explains how `ingest_data.py` works in simple terms. The script takes text files (like pages from a PDF) and converts them into a searchable database using vector embeddings.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Code Walkthrough](#code-walkthrough)
- [Data Flow](#data-flow)

## Overview

The data ingestion script performs three main tasks:
1. **Loads** text files from a directory
2. **Splits** large text into smaller chunks (optional)
3. **Creates** a vector database for fast semantic search

```mermaid
graph LR
    A[Text Files] --> B[Load Files]
    B --> C[Split into Chunks]
    C --> D[Generate Embeddings]
    D --> E[Store in Chroma DB]
    E --> F[Ready for Queries]

    style A fill:#e1f5ff
    style F fill:#c8e6c9
```

## How It Works

### What are Vector Embeddings?

Vector embeddings convert text into numbers (vectors) that capture meaning. Similar text has similar vectors, allowing computers to understand semantic relationships.

```mermaid
graph TD
    A["Text: 'The cat sat on the mat'"] --> B[Embedding Model]
    B --> C["Vector: [0.23, -0.45, 0.87, ...]"]
    D["Text: 'A feline rested on the rug'"] --> B
    B --> E["Vector: [0.21, -0.43, 0.89, ...]"]

    C -.Similar vectors.-> E

    style A fill:#e1f5ff
    style D fill:#e1f5ff
    style C fill:#fff9c4
    style E fill:#fff9c4
```

### Why Split Text into Chunks?

Large documents are split into smaller chunks for better retrieval accuracy. When you ask a question, the system finds the most relevant chunks rather than entire documents.

```mermaid
graph TD
    A[Long Document<br/>5000 words] --> B{Text Splitter}
    B --> C[Chunk 1<br/>1000 words]
    B --> D[Chunk 2<br/>1000 words]
    B --> E[Chunk 3<br/>1000 words]
    B --> F[Chunk 4<br/>1000 words]
    B --> G[Chunk 5<br/>1000 words]

    C --> H[More precise<br/>search results]

    style A fill:#ffccbc
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#c8e6c9
```

## Code Walkthrough

### 1. Imports and Dependencies

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

**What these do:**
- `OllamaEmbeddings`: Connects to Ollama to generate vector embeddings
- `Chroma`: Vector database for storing and searching embeddings
- `Document`: LangChain's document format that holds text + metadata
- `RecursiveCharacterTextSplitter`: Smart text splitter that preserves context

### 2. Class Initialization (`__init__` method)

**Location:** Lines 17-48

**What it does:** Sets up the ingestion pipeline with configuration

```mermaid
graph TD
    A[Initialize DataIngestion] --> B[Set Configuration]
    B --> C[Create OllamaEmbeddings]
    B --> D[Create Text Splitter]

    C --> E[Ready to generate<br/>vector embeddings]
    D --> F[Ready to split<br/>long texts]

    style A fill:#e1f5ff
    style E fill:#c8e6c9
    style F fill:#c8e6c9
```

**Key parameters:**
- `data_directory`: Where your .txt files are located
- `chroma_db_path`: Where to save the vector database
- `embedding_model`: Which model to use (default: mxbai-embed-large)
- `chunk_size`: How many characters per chunk (default: 1000)
- `chunk_overlap`: How much chunks overlap (default: 200)

**Why overlap?** Overlapping ensures important information at chunk boundaries isn't lost.

```mermaid
graph LR
    A["Chunk 1:<br/>...end of sentence"]
    B["Chunk 2:<br/>start of sentence..."]

    A -.200 char overlap.-> B

    style A fill:#c8e6c9
    style B fill:#c8e6c9
```

### 3. Loading Text Files (`load_text_files` method)

**Location:** Lines 58-103

**What it does:** Reads all .txt files and creates Document objects

```mermaid
sequenceDiagram
    participant Script
    participant FileSystem
    participant Document

    Script->>FileSystem: Find all .txt files
    FileSystem-->>Script: [1.txt, 2.txt, 3.txt, ...]

    loop For each file
        Script->>FileSystem: Read file content
        FileSystem-->>Script: Text content
        Script->>Script: Extract page number
        Script->>Document: Create Document with metadata
        Document-->>Script: Document object
    end

    Script->>Script: Return list of documents
```

**File naming:** Files are expected to be named by page number (e.g., `1.txt`, `2.txt`, `3.txt`)

**Metadata stored:**
- `source`: Original filename
- `page`: Page number extracted from filename
- `file_path`: Full path to the file

### 4. Extracting Page Numbers (`_extract_page_number` method)

**Location:** Lines 105-119

**What it does:** Uses regex to extract the number from filenames like "42.txt"

```mermaid
graph LR
    A["Filename: '42.txt'"] --> B[Regex Pattern:<br/>r'(\d+)\.txt$']
    B --> C[Extract: '42']
    C --> D[Convert to int: 42]

    style A fill:#e1f5ff
    style D fill:#c8e6c9
```

### 5. Splitting Documents (`split_documents` method)

**Location:** Lines 121-137

**What it does:** Breaks large documents into smaller chunks while preserving context

```mermaid
graph TD
    A[Original Document] --> B{RecursiveCharacterTextSplitter}

    B --> C[Try split by: '\n\n'<br/>paragraphs]
    C -->|Too large?| D[Try split by: '\n'<br/>new lines]
    D -->|Too large?| E[Try split by: '. '<br/>sentences]
    E -->|Too large?| F[Try split by: ' '<br/>words]
    F -->|Too large?| G[Split by character]

    C -->|Success| H[Chunks with context preserved]
    D -->|Success| H
    E -->|Success| H
    F -->|Success| H
    G --> H

    style A fill:#e1f5ff
    style H fill:#c8e6c9
```

**How RecursiveCharacterTextSplitter works:**
1. Tries to split at natural boundaries (paragraphs first)
2. Falls back to smaller boundaries if needed (sentences, then words)
3. Ensures each chunk is approximately `chunk_size` characters
4. Adds overlap between chunks to preserve context

### 6. Creating the Vector Store (`create_vectorstore` method)

**Location:** Lines 139-163

**What it does:** Generates embeddings and stores them in Chroma database

```mermaid
sequenceDiagram
    participant Script
    participant Ollama
    participant Chroma

    Script->>Script: Prepare documents

    loop For each document/chunk
        Script->>Ollama: Send text
        Ollama->>Ollama: Generate embedding vector
        Ollama-->>Script: Return vector [0.23, -0.45, ...]
        Script->>Chroma: Store text + vector + metadata
    end

    Chroma->>Chroma: Build searchable index
    Chroma-->>Script: Vector store ready
    Script->>Script: Persist to disk
```

**What happens:**
1. Each text chunk is sent to Ollama's embedding model
2. The model returns a vector (array of numbers) representing the meaning
3. The vector, text, and metadata are stored in Chroma
4. Chroma builds an index for fast similarity search
5. Everything is saved to disk at `chroma_db_path`

**Vector representation:**
```mermaid
graph TD
    A["Text Chunk:<br/>'The cat sat on the mat'"] --> B[Embedding Model:<br/>mxbai-embed-large]
    B --> C["Vector:<br/>[1024 dimensions]"]

    C --> D[Stored in Chroma DB]

    D --> E["Metadata:<br/>- source: '3.txt'<br/>- page: 3<br/>- file_path: '/path/to/3.txt'"]

    style A fill:#e1f5ff
    style C fill:#fff9c4
    style D fill:#c8e6c9
    style E fill:#f8bbd0
```

### 7. Full Ingestion Pipeline (`ingest` method)

**Location:** Lines 165-202

**What it does:** Orchestrates the entire process

```mermaid
flowchart TD
    A[Start Ingestion] --> B[Print Configuration]
    B --> C[Load Text Files]
    C --> D{Split Text?}

    D -->|Yes| E[Split into Chunks]
    D -->|No| F[Use Whole Pages]

    E --> G[Create Vector Store]
    F --> G

    G --> H[Generate Embeddings]
    H --> I[Store in Chroma DB]
    I --> J[Persist to Disk]
    J --> K[Complete!]

    style A fill:#e1f5ff
    style K fill:#c8e6c9
```

### 8. Command-Line Interface (`main` function)

**Location:** Lines 205-240

**What it does:** Parses command-line arguments and runs ingestion

**Available arguments:**
- `data_dir` (required): Directory with text files
- `--db-path`: Where to save the database
- `--embedding-model`: Which embedding model to use
- `--ollama-url`: URL of Ollama server
- `--chunk-size`: Size of text chunks
- `--chunk-overlap`: Overlap between chunks
- `--no-split`: Don't split text, use whole pages

## Data Flow

### Complete Process Flow

```mermaid
flowchart TD
    subgraph Input
        A1[1.txt]
        A2[2.txt]
        A3[3.txt]
    end

    subgraph "Load & Parse"
        B[Read Files]
        C[Extract Metadata]
    end

    subgraph "Process"
        D{Split?}
        E[Text Splitter]
        F[Keep Whole]
    end

    subgraph "Embed"
        G[Ollama Embedding Model]
        H[Generate Vectors]
    end

    subgraph "Store"
        I[Chroma Vector DB]
        J[Persist to Disk]
    end

    A1 --> B
    A2 --> B
    A3 --> B
    B --> C
    C --> D
    D -->|Yes| E
    D -->|No| F
    E --> G
    F --> G
    G --> H
    H --> I
    I --> J

    style Input fill:#e1f5ff
    style Store fill:#c8e6c9
```

### Vector Database Structure

```mermaid
erDiagram
    CHROMA_DB ||--o{ DOCUMENT : contains
    DOCUMENT ||--|| VECTOR : has
    DOCUMENT ||--|| METADATA : has

    DOCUMENT {
        string id
        string text_content
    }

    VECTOR {
        float[] embedding
        int dimensions
    }

    METADATA {
        string source
        int page
        string file_path
    }
```

## Key Concepts Explained

### 1. Why Use Embeddings?

Traditional search uses keyword matching. Vector embeddings understand meaning:

```mermaid
graph TD
    subgraph "Keyword Search"
        A1["Query: 'automobile'"] -.X.-> B1["Document: 'car'"]
        A1 -.No match!.-> C1[No results]
    end

    subgraph "Vector Search"
        A2["Query: 'automobile'<br/>Vector: [0.8, 0.2, ...]"] --> B2["Document: 'car'<br/>Vector: [0.81, 0.19, ...]"]
        B2 --> C2[Very similar!<br/>Found it!]
    end

    style C1 fill:#ffccbc
    style C2 fill:#c8e6c9
```

### 2. Chunk Overlap Illustration

```mermaid
graph TD
    A["Full Text:<br/>'...context before. Important sentence spans here. Context after...'"]

    A --> B["Chunk 1 (1000 chars):<br/>'...context before. Important'"]
    A --> C["Chunk 2 (1000 chars):<br/>'Important sentence spans here. Context...'"]

    B -.200 char overlap.-> C

    D["Without overlap:<br/>Sentence might be split<br/>and lose meaning"]

    E["With overlap:<br/>Full sentence captured<br/>in at least one chunk"]

    style A fill:#e1f5ff
    style B fill:#fff9c4
    style C fill:#fff9c4
    style D fill:#ffccbc
    style E fill:#c8e6c9
```

### 3. From Text to Searchable Database

```mermaid
sequenceDiagram
    participant User
    participant Script
    participant Ollama
    participant Chroma

    User->>Script: Run: python ingest_data.py data/
    Script->>Script: Load 5 text files
    Script->>Script: Split into 23 chunks

    loop For each chunk
        Script->>Ollama: "Generate embedding for this text"
        Ollama-->>Script: [0.23, -0.45, 0.87, ..., 1024 numbers]
        Script->>Chroma: Store text + vector + metadata
    end

    Chroma->>Chroma: Build similarity search index
    Script->>User: Success! 23 documents stored
```

## Example Usage

### Basic Usage
```bash
python ingest_data.py ./my_documents/
```

### With Custom Settings
```bash
python ingest_data.py ./my_documents/ \
  --db-path ./custom_db \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --embedding-model mxbai-embed-large
```

### Without Splitting (Whole Pages)
```bash
python ingest_data.py ./my_documents/ --no-split
```

## What Gets Created

After running the ingestion script:

```mermaid
graph TD
    A[Your Directory] --> B[chroma_db/]
    B --> C[Collection Data]
    B --> D[Vector Index]
    B --> E[Metadata Store]

    C --> F[Can be queried<br/>by rag_app.py]
    D --> F
    E --> F

    style A fill:#e1f5ff
    style F fill:#c8e6c9
```

The `chroma_db/` directory contains:
- Vector embeddings for all text chunks
- Original text content
- Metadata (source file, page number)
- Searchable index for fast similarity search

## Summary

The ingestion script transforms unstructured text files into a searchable knowledge base:

1. **Reads** text files with page numbers
2. **Splits** long text into manageable chunks (optional)
3. **Converts** text to vector embeddings using Ollama
4. **Stores** everything in a Chroma vector database
5. **Enables** semantic search by the RAG application

This creates a foundation for intelligent question-answering based on your documents!
