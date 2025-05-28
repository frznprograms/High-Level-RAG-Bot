# AIAP® Hands-on Project: LLM RAG - Multimodal Educational Chatbot

**Version:** 1.0
**Date:** 2025-04-20

## 1. Objectives

To build an end-to-end Retrieval-Augmented Generation (RAG) system that functions as an educational chatbot. The system should be able to answer questions based on a diverse collection of multimodal documents (e.g., PDF textbook chapters, DOCX lecture notes, PPTX slides with images, related CSV data tables, images, short audio/video clips) covering a specific topic (e.g., AI/ML Interview preparation). The system should retrieve relevant information from the various sources and synthesize a coherent answer using a Large Language Model (LLM). The chatbot should be accessible via a simple interface (Streamlit/Gradio) served by a backend (FastAPI).

## 2. Tasks

### 2.1. Data Curation and Preprocessing (Corpus Building)

* **Deliverable:** A structured `corpus/` directory containing the multimodal documents, and potentially scripts (`preprocess_corpus.py`) used for initial extraction/conversion.
* **Content:**
  * Gather relevant open-source or create sample documents (PDF, DOCX, PPTX, CSV, JPG/PNG, MP3/MP4) related to the chosen educational topic. Ensure a mix of text, tables, images, and potentially audio/video.
  * Use tools (e.g., `unstructured`, `pypdf`, `python-pptx`, `opencv-python`, `moviepy`, `librosa`, `pandas`, potentially Docling) to extract content:
    * Text from PDF, DOCX, PPTX.
    * Tables from documents (convert to structured format like CSV or markdown).
    * Images from PDF, PPTX (save as separate files).
    * Text/captions associated with images if available.
    * Transcripts from audio/video (using speech-to-text models like Whisper).
    * Potentially break down long videos/audio into smaller segments.
  * Generate descriptions/summaries for images and video/audio segments using a vision/multimodal model (e.g., CLIP, BLIP, Llava).
  * Organize extracted content and metadata (source document, page/slide number, segment timestamp).

### 2.2. End-to-end RAG Pipeline and Application

* **Deliverables:**
  * Python scripts (`.py` files) in `src/` (e.g., `data_loader.py`, `embedder.py`, `vector_store.py`, `retriever.py`, `generator.py`, `app_backend.py` (FastAPI), `app_frontend.py` (Streamlit/Gradio)).
  * `run.sh`: Script to setup the vector store, start the backend, and potentially launch the frontend.
  * `requirements.txt`: Dependencies (e.g., langchain, llama-index, unstructured-io, transformers, sentence-transformers, faiss-cpu/chromadb/lancedb, fastapi, uvicorn, streamlit/gradio, openai/anthropic SDKs if using external LLMs, libraries for specific file types).
  * `config.yaml`: Parameters (corpus path, embedding model choice, LLM choice (local or API), vector DB path/settings, retriever settings, top-k results).
  * `README.md`: Documentation.
* **Pipeline Design:**
  * **Indexing/Embedding:**
    * Load processed text chunks, table data, image descriptions, audio/video transcripts/summaries.
    * Use an appropriate embedding model (e.g., Sentence-BERT, OpenAI Ada, a multimodal model if handling image embeddings directly) to generate vector embeddings for each piece of information.
    * Store embeddings and associated metadata in a vector database (e.g., FAISS, ChromaDB, LanceDB).
  * **Retrieval:**
    * User query comes via the frontend/backend.
    * Embed the user query using the same embedding model.
    * Perform a similarity search in the vector database to retrieve the top-k most relevant chunks of information (text, descriptions, etc.).
  * **Generation:**
    * Construct a prompt for the LLM, including the original user query and the retrieved context.
    * Use an LLM (e.g., a local model like Llama/Mistral via Ollama/HuggingFace, or an API like GPT-4/Claude) to generate an answer based on the provided context.
  * **Application Interface:**
    * FastAPI backend handles requests, orchestrates retrieval and generation.
    * Streamlit or Gradio frontend provides a simple chat interface for the user.
* **README.md Content:**
  * Name/Email.
  * Project Overview & Folder Structure.
  * Setup & Execution Instructions (`requirements.txt`, `run.sh`, environment variables for API keys, model download instructions).
  * Pipeline Logic/Flow (diagram showing multimodal ingestion, indexing, retrieval, generation, UI).
  * Data Corpus Description (types of files included, topic).
  * Key Components (Embedding model, Vector DB, Retriever logic, LLM, UI framework) & Rationale.
  * Evaluation Approach (qualitative assessment of answer relevance, factual consistency, handling of multimodal queries if applicable).
  * Limitations & Potential Improvements (e.g., handling complex queries, better multimodal fusion).

## 3. Data

* **Source:** User-curated collection of multimodal documents related to a specific educational topic (e.g., Photosynthesis). Should include:
  * Text: `.pdf`, `.docx`
  * Presentations: `.pptx` (containing text and images)
  * Tables: `.csv`
  * Images: `.jpg`, `.png`
  * Audio: `.mp3`, `.wav` (short clips or transcribed longer audio)
  * Video: `.mp4` (short clips or transcribed/summarized longer video)
* **Access:** Files should be placed in a local `corpus/` directory. The pipeline will process files from this directory.
* **Relevant URLs/Tools:**
  Please feel free to use any other publicly available resources/tools to do this hands-on project. Some useful tools and libraries include:
  * [Unstructured.io](https://unstructured-io.github.io/unstructured/) (File parsing)
  * [Docling (IBM)](https://www.analyticsvidhya.com/blog/2025/03/enhancing-multimodal-rag-capabilities-using-docling/) (Alternative parsing)
  * [LlamaIndex Multimodal](https://www.llamaindex.ai/blog/multimodal-rag-for-advanced-video-processing-with-llamaindex-lancedb-33be4804822e) (Framework)
  * [MachineLearningMastery Multimodal RAG](https://machinelearningmastery.com/implementing-multi-modal-rag-systems/) (Concepts)
  * [OpenAI CLIP](https://openai.com/index/clip/) (Image/Text Understanding)

## 4. List of Attributes / Data Format

* The "attributes" are the various pieces of information extracted and indexed:
  * Text chunks (from PDF, DOCX, PPTX, transcripts)
  * Table representations (e.g., markdown strings, CSV rows)
  * Image descriptions/summaries (text)
  * Audio/Video segment transcripts/summaries (text)
* Each indexed item should have associated metadata: source file name, page/slide/segment info, data type (text, table, image\_desc, etc.).

## (Optional for self-practice) Non-Chatbot based RAG Application

For a non-chatbot RAG application project (e.g., generating an educational report), the core RAG pipeline (ingestion, indexing, retrieval, generation) remains the same. Instead of a FastAPI/Streamlit interface, the final step would involve taking a topic or set of questions as input (e.g., via CLI parameters or a config file) and generating a structured markdown report. The generator.py script would need to orchestrate multiple retrieval/generation steps to build different sections of the report based on the input topic/questions. The output would be a .md file instead of interactive chat responses.
