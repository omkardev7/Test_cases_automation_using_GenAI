# Test_Cases Automation Tool 📝

## Overview
The Test_Cases Automation Tool is a Streamlit-based web application designed to automate the generation of test cases from uploaded SRS (Software Requirements Specification) documents in PDF format. This tool leverages advanced natural language processing techniques to extract relevant information and answer queries based on the uploaded documents.

## Features
1. PDF Document Upload: Users can upload SRS documents in PDF format.
2. Text Extraction and Vectorization: Extracts text from uploaded PDFs and converts it into vector embeddings.
3. Retrieval QA Chain: Utilizes a Retrieval Question-Answering (QA) chain to answer user queries based on the extracted content.
4. LLaMA-2 Integration: Employs the LLaMA-2 language model for generating responses.
5. Local Storage of Vector DB: Stores vectorized documents locally for faster retrieval in future sessions.

## Requirements
- Python 3.10 or higher
- Streamlit
- LangChain
- PyPDF2
- FAISS
- HuggingFace 
- Transformers

## How It Works
1. PDF Upload: Users can upload a PDF document which is stored locally.
2. Text Extraction: The tool extracts text from the PDF using PyPDFLoader.
3. Text Splitting: The extracted text is split into manageable chunks using RecursiveCharacterTextSplitter.
4. Embedding Generation: The text chunks are converted into vector embeddings using the HuggingFace model.
5. FAISS Vector Store: The embeddings are stored in a FAISS vector database for efficient retrieval.
6. LLaMA-2 Model: The LLaMA-2 model is loaded to process queries and generate responses.
7. Retrieval QA Chain: A QA chain is created using the vector database and the LLaMA-2 model to answer user queries.