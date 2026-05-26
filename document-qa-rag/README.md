# AI Document Q&A with Azure OpenAI, RAG, and Streamlit

## Project Overview

This project demonstrates an end-to-end AI-powered Document Question & Answer (Q&A) system using:

- Azure OpenAI
- TF-IDF Retrieval-Augmented Generation (RAG)
- Embedding-based Semantic Search
- Dynamic Pandas Routing
- Streamlit User Interface

The application allows users to ask natural language questions about Telco Customer Churn data and receive AI-generated insights backed by retrieved data context.

---

# Features

## 1. Azure OpenAI Integration

- Azure OpenAI GPT-4.1 Mini for response generation
- Azure OpenAI Embeddings for semantic search
- Environment-variable based secure configuration

---

## 2. Document Q&A

The notebook builds a reusable AI Q&A workflow using:
- Dataset summaries
- Business questions
- Prompt engineering
- Structured AI responses

---

## 3. TF-IDF RAG

Implemented basic Retrieval-Augmented Generation using:
- TfidfVectorizer
- Cosine similarity
- Top-k chunk retrieval

This enables keyword-based retrieval of relevant customer records.

---

## 4. Embedding-Based Semantic RAG

Improved retrieval using:
- text-embedding-3-small
- Semantic vector similarity
- Embedding-based chunk retrieval

This allows semantic understanding of questions such as:

```text
customers with expensive plans who left



## 5. Dynamic Pandas + GenAI Hybrid Routing

The application intelligently routes questions into:

Pandas Filter Queries

Examples:

Show customers with high monthly charges who churned
List customers with short tenure
Pandas Aggregation Queries

Examples:

Average monthly charges by churn
Count customers by contract type
RAG Questions

Examples:

Why do customers churn?
What business actions can reduce churn?

## 6. Streamlit User Interface

The Streamlit app provides:

Natural language Q&A
Dynamic Pandas code generation
AI explanations
Embedding search debugging
Chat history tracking
Tech Stack
Python
Pandas
NumPy
Scikit-learn
Azure OpenAI
Streamlit


Project Structure
document-qa-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── notebooks/
│   └── document_qa_rag.ipynb
│
├── data/
│   └── telco_churn.csv
│
├── artifacts/
│   ├── chunk_embeddings.npy
│   └── chunks.pkl



# Future Improvements

  - FAISS / ChromaDB vector database
  - LangChain integration
  - Multi-document RAG
  - Conversational memory
  - Azure AI Search integration
  - Production-safe query execution
  - Authentication and user management

# Learning Goals

This project demonstrates:

- Prompt engineering
- RAG implementation
- Embedding search
- AI + Pandas hybrid analytics
- Streamlit application development
- Azure OpenAI integration
