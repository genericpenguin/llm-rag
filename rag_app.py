"""
RAG Application using LangChain, Ollama, and Chroma
Queries a Chroma vector database built from text files using mxbai-embed-large embeddings
"""

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os


class RAGApplication:
    def __init__(
        self,
        model_name: str = "llama3.2",
        embedding_model: str = "mxbai-embed-large",
        chroma_db_path: str = "./chroma_db",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the RAG application

        Args:
            model_name: Name of the Ollama model to use for generation
            embedding_model: Name of the embedding model (mxbai-embed-large)
            chroma_db_path: Path to the Chroma database
            ollama_base_url: URL of the Ollama instance
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chroma_db_path = chroma_db_path
        self.ollama_base_url = ollama_base_url

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )

        # Initialize LLM
        self.llm = OllamaLLM(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.7
        )

        # Load Chroma database
        self.vectorstore = None
        self.qa_chain = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """Load the Chroma vector store"""
        if not os.path.exists(self.chroma_db_path):
            raise ValueError(
                f"Chroma database not found at {self.chroma_db_path}. "
                "Please run the ingestion script first."
            )

        self.vectorstore = Chroma(
            persist_directory=self.chroma_db_path,
            embedding_function=self.embeddings
        )

        # Create retrieval QA chain
        self._create_qa_chain()

    def _create_qa_chain(self):
        """Create the retrieval chain using LCEL (LangChain Expression Language)"""

        # Custom prompt template
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context provided, just say that you don't know,
don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

        prompt = ChatPromptTemplate.from_template(template)

        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )

        # Build the RAG chain using LCEL pipe operator
        # This chain: retrieves docs -> formats them -> passes to prompt -> generates answer
        self.qa_chain = (
            RunnableParallel(
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Store retriever for source document access
        self.retriever = retriever

    def query(self, question: str, show_sources: bool = True) -> dict:
        """
        Query the RAG system

        Args:
            question: The question to ask
            show_sources: Whether to display source documents

        Returns:
            Dictionary containing the answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Check if vector store is loaded.")

        # Invoke the LCEL chain - it takes the question as direct input
        # The chain returns just the answer string (due to StrOutputParser)
        answer = self.qa_chain.invoke(question)

        response = {
            "question": question,
            "answer": answer,
            "sources": []
        }

        # If sources are requested, retrieve them separately using the retriever
        if show_sources and hasattr(self, 'retriever'):
            source_docs = self.retriever.invoke(question)
            for doc in source_docs:
                source_info = {
                    "page": doc.metadata.get("page", "unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                response["sources"].append(source_info)

        return response

    def interactive_mode(self):
        """Run an interactive Q&A session"""
        print("=" * 60)
        print("RAG Application - Interactive Mode")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Database: {self.chroma_db_path}")
        print("=" * 60)
        print("Type 'exit' or 'quit' to end the session")
        print("=" * 60)
        print()

        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("\nExiting interactive mode. Goodbye!")
                break

            if not question:
                continue

            print("\nThinking...\n")

            try:
                result = self.query(question)

                print(f"Answer: {result['answer']}\n")

                if result["sources"]:
                    print(f"Sources ({len(result['sources'])} documents):")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"\n  [{i}] Page {source['page']} - {source['source']}")
                        print(f"      Preview: {source['content_preview']}")

            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    """Main function to run the RAG application"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Application using LangChain, Ollama, and Chroma")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2)")
    parser.add_argument("--embedding-model", default="mxbai-embed-large", help="Embedding model name (default: mxbai-embed-large)")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to Chroma database (default: ./chroma_db)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--query", help="Single query to run (otherwise starts interactive mode)")

    args = parser.parse_args()

    # Initialize the RAG application
    try:
        rag = RAGApplication(
            model_name=args.model,
            embedding_model=args.embedding_model,
            chroma_db_path=args.db_path,
            ollama_base_url=args.ollama_url
        )

        if args.query:
            # Single query mode
            result = rag.query(args.query)
            print(f"\nQuestion: {result['question']}")
            print(f"\nAnswer: {result['answer']}\n")

            if result["sources"]:
                print(f"Sources ({len(result['sources'])} documents):")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n  [{i}] Page {source['page']} - {source['source']}")
                    print(f"      Preview: {source['content_preview']}")
        else:
            # Interactive mode
            rag.interactive_mode()

    except Exception as e:
        print(f"Error initializing RAG application: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
