# LLM RAG Application

A Retrieval-Augmented Generation (RAG) application using LangChain, Ollama, and Chroma vector database. This application allows you to query information from text files (extracted PDF pages) using local LLMs.

## Features

- Local LLM inference using Ollama
- Vector embeddings using mxbai-embed-large model
- Chroma vector database for efficient retrieval
- Processes individual text files (e.g., PDF pages as separate .txt files)
- Interactive query mode
- Source document tracking

## Prerequisites

1. **Python 3.10**

2. **Ollama** installed and running locally
   - Install from: https://ollama.ai
   - Default URL: `http://localhost:11434`

3. **Required Ollama models**:
   ```bash
   # Pull the embedding model (required)
   ollama pull mxbai-embed-large

   # Pull an LLM model (e.g., llama3.2)
   ollama pull llama3.2
   ```

## Installation

1. Clone this repository or download the files

2. Install Python dependencies:
   ```bash
   brew install pyenv
   pyenv install 3.10
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Prepare Your Data

Place your text files in a directory. Files should be named by page number:
```
data/
├── 1.txt
├── 2.txt
├── 3.txt
├── 4.txt
└── 5.txt
```

Each file should contain the text content of a single page from your PDF.

### Step 2: Ingest Data into Chroma Database

Run the ingestion script to process your text files and create the vector database:

```bash
python ingest_data.py data/
```

**Options:**
- `--db-path`: Path to store Chroma database (default: `./chroma_db`)
- `--embedding-model`: Embedding model name (default: `mxbai-embed-large`)
- `--ollama-url`: Ollama base URL (default: `http://localhost:11434`)
- `--chunk-size`: Size of text chunks (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--no-split`: Don't split text into chunks, use whole pages

**Example with custom options:**
```bash
python ingest_data.py data/ --db-path ./my_db --chunk-size 500
```

### Step 3: Query Your Data

#### Interactive Mode (Recommended)

Start an interactive Q&A session:

```bash
python rag_app.py
```

You'll be able to ask multiple questions interactively. Type 'exit' or 'quit' to end the session.

#### Single Query Mode

Run a single query from the command line:

```bash
python rag_app.py --query "What is the main topic discussed in the document?"
```

**Options:**
- `--model`: Ollama model name (default: `llama3.2`)
- `--embedding-model`: Embedding model name (default: `mxbai-embed-large`)
- `--db-path`: Path to Chroma database (default: `./chroma_db`)
- `--ollama-url`: Ollama base URL (default: `http://localhost:11434`)
- `--query`: Single query to run (otherwise starts interactive mode)

**Example with custom model:**
```bash
python rag_app.py --model llama3.1 --query "Summarize the key points"
```

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running and has the required models
ollama pull mxbai-embed-large
ollama pull llama3.2

# 3. Ingest your text files
python ingest_data.py ./my_pdf_pages/

# 4. Start querying (interactive mode)
python rag_app.py

# Example questions you might ask:
# - What is this document about?
# - Summarize the main points on page 5
# - What does the author say about [specific topic]?
# - Are there any statistics mentioned?
```

## Project Structure

```
.
├── requirements.txt      # Python dependencies
├── ingest_data.py       # Data ingestion script
├── rag_app.py          # Main RAG application
├── README.md           # This file
├── chroma_db/          # Chroma vector database (created after ingestion)
└── data/               # Your text files (you create this)
```

## How It Works

1. **Ingestion Phase** (`ingest_data.py`):
   - Reads all .txt files from the specified directory
   - Optionally splits long pages into smaller chunks for better retrieval
   - Generates embeddings using mxbai-embed-large model via Ollama
   - Stores embeddings and text in Chroma vector database

2. **Query Phase** (`rag_app.py`):
   - Receives a question from the user
   - Converts question to an embedding using the same model
   - Retrieves the top 4 most relevant text chunks from Chroma
   - Sends the retrieved context + question to the LLM (llama3.2)
   - Returns the answer along with source information

## Troubleshooting

**Error: "Chroma database not found"**
- Make sure you've run `ingest_data.py` first to create the database

**Error: "Connection refused" or Ollama errors**
- Ensure Ollama is running: `ollama serve`
- Check that models are pulled: `ollama list`

**Poor quality answers**
- Try adjusting chunk size (smaller chunks for more precise retrieval)
- Try a different LLM model (e.g., `llama3.1`, `mistral`)
- Ensure your source text files are clean and well-formatted

**Slow performance**
- First query may be slower as models load
- Consider using a smaller model for faster responses
- Reduce the number of retrieved chunks (modify `k` in [rag_app.py:93](rag_app.py#L93))

## Customization

### Change the number of retrieved documents
Edit [rag_app.py:93](rag_app.py#L93):
```python
search_kwargs={"k": 4}  # Change 4 to your preferred number
```

### Modify the prompt template
Edit the template in [rag_app.py:74-83](rag_app.py#L74-L83) to customize how the LLM responds.

### Adjust text chunking strategy
Modify chunk_size and chunk_overlap in `ingest_data.py` or pass as arguments.
