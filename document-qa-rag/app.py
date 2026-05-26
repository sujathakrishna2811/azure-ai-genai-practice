import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst (Churn Analysis)")


# -------------------------------
# Validate environment variables
# -------------------------------
if not os.getenv("AZURE_OPENAI_KEY"):
    st.error("Missing environment variable: AZURE_OPENAI_KEY")
    st.stop()

if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    st.error("Missing environment variable: AZURE_OPENAI_ENDPOINT")
    st.stop()


# -------------------------------
# Validate required files
# -------------------------------
required_files = [
    "telco_churn.csv",
    "chunk_embeddings.npy",
    "chunks.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Missing required file: {file}")
        st.stop()


# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("telco_churn.csv")
chunk_embeddings_np = np.load("chunk_embeddings.npy")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# -------------------------------
# Azure OpenAI client
# -------------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


# -------------------------------
# Question classifier
# -------------------------------
def classify_question(question):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a classifier.

Classify the user question into ONE category:

pandas_filter = exact filtering/listing rows
Examples:
- Show customers with high monthly charges who churned
- List customers with short tenure
- Show customers on month-to-month contracts

pandas_agg = aggregation, calculation, comparison
Examples:
- Do customers who churn have higher monthly charges?
- What is the average tenure by churn?
- Count customers by contract type

rag = general insights, explanations, business recommendations, unknown questions
Examples:
- Why do customers churn?
- What business actions can reduce churn?
- What patterns do you see?

Respond with ONLY one word:
pandas_filter OR pandas_agg OR rag
"""
            },
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()


# -------------------------------
# Dynamic Pandas code generator
# -------------------------------
def generate_pandas_code(question):
    columns = ", ".join(df.columns)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
You generate safe Pandas code for dataframe df.

Available columns:
{columns}

Rules:
- Return ONLY one Python expression.
- Do not use imports.
- Do not use print.
- Do not modify df.
- Do not use eval, exec, open, os, subprocess.
- For filters, return a dataframe expression.
- For aggregations, return a pandas result expression.

Examples:
df[(df["Churn"] == "Yes") & (df["MonthlyCharges"] >= 80)]
df.groupby("Churn")["MonthlyCharges"].mean()
df["Contract"].value_counts()
"""
            },
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# -------------------------------
# Safe Pandas executor
# -------------------------------
def run_pandas_code(code):
    blocked_terms = [
        "import", "open", "exec", "eval",
        "os.", "subprocess", "__"
    ]

    if any(term in code for term in blocked_terms):
        raise ValueError("Unsafe code detected.")

    allowed_globals = {
        "df": df,
        "pd": pd,
        "np": np
    }

    result = eval(code, {"__builtins__": {}}, allowed_globals)

    return result


# -------------------------------
# Explain Pandas result
# -------------------------------
def explain_pandas_result(question, result):
    if isinstance(result, pd.DataFrame):
        data_text = result.head(20).to_csv(index=False)
    elif isinstance(result, pd.Series):
        data_text = result.to_string()
    else:
        data_text = str(result)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a data analyst.
Explain the Pandas result clearly.

Rules:
- Use only the provided result.
- Be precise with numbers.
- Do not invent missing information.
- If the result is empty, say no matching records were found.
"""
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

Pandas result:
{data_text}
"""
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------------
# Embedding retrieval
# -------------------------------
def retrieve_chunks_embedding(question, top_k=5):
    question_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    question_embedding = np.array(question_embedding).reshape(1, -1)

    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings_np
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_indices]


# -------------------------------
# Embedding RAG
# -------------------------------
def ask_embedding_rag(question, top_k=5):
    relevant_chunks = retrieve_chunks_embedding(question, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a data analyst.
Answer only using the provided context.

Rules:
- Use only the retrieved context.
- Clearly distinguish between exact facts and general insights.
- Do not assume values unless explicitly present.
- If context is not enough, say what is missing.
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------------
# Smart routing
# -------------------------------
def smart_ask(question):
    category = classify_question(question)

    if category in ["pandas_filter", "pandas_agg"]:
        try:
            pandas_code = generate_pandas_code(question)
            result = run_pandas_code(pandas_code)
            answer = explain_pandas_result(question, result)

            return answer, result, pandas_code, f"Dynamic Pandas ({category})"

        except Exception as e:
            answer = ask_embedding_rag(question)
            return answer, None, None, f"Embedding RAG fallback - Pandas error: {e}"

    answer = ask_embedding_rag(question)
    return answer, None, None, "Embedding RAG"


# -------------------------------
# Streamlit UI
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about customer churn:")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            answer, pandas_result, pandas_code, method = smart_ask(question)

        st.caption(f"Used: {method}")

        if pandas_code:
            st.subheader("Generated Pandas Code")
            st.code(pandas_code, language="python")

        if isinstance(pandas_result, pd.DataFrame):
            st.subheader("Pandas Result")
            st.dataframe(pandas_result)

        elif isinstance(pandas_result, pd.Series):
            st.subheader("Pandas Result")
            st.dataframe(pandas_result)

        elif pandas_result is not None:
            st.subheader("Pandas Result")
            st.write(pandas_result)

        st.subheader("Answer")
        st.write(answer)

        st.session_state.history.append((question, answer))


# -------------------------------
# Debug retrieved chunks
# -------------------------------
if question and st.checkbox("Show embedding search chunks"):
    st.subheader("Retrieved Chunks (Debug)")
    st.caption("Used: Embedding Semantic Search")

    chunks_debug = retrieve_chunks_embedding(question, top_k=5)

    for i, c in enumerate(chunks_debug, start=1):
        st.markdown(f"### Chunk {i}")
        st.write(c)


# -------------------------------
# Chat history
# -------------------------------
st.subheader("Chat History")

for q, a in st.session_state.history:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")