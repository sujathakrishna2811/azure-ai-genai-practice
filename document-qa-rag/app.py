

import sys
!{sys.executable} -m pip show openai
!which python
!python -m pip show openai

import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd

#Correct ways to read a URI_FILE in Azure ML

# Step 1: Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id= os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name= os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name= os.getenv("AZURE_ML_WORKSPACE")
)

# Step 2 — Get the dataset
dataset = ml_client.data.get(
    name="IBM_Telco_Customer_Churn_CSV",
    version="1"
)
# Step 3 — Read using pandas
df = pd.read_csv(dataset.path)


Step 1: Improve Document Q&A
Goal: Instead of just sending raw rows, create- Dataset summary + relevant sample rows + user question. This gives the model better context.



# step 1: Create a useful dataset summary
summary = f"""
Dataset name: Telco Customer 

ChurnNumber of rows: {df.shape[0]}
Number of columns: {df.shape[1]}

Columns:{', '.join(df.columns)}

Churn distribution:
{df['Churn'].value_counts().to_string()}

Contract distribution:
{df['Contract'].value_counts().to_string()}

Internet service distribution:
{df['InternetService'].value_counts().to_string()}

Payment method distribution:
{df['PaymentMethod'].value_counts().to_string()}

Average monthly charges by churn:
{df.groupby('Churn')['MonthlyCharges'].mean().to_string()}

Average tenure by churn:
{df.groupby('Churn')['tenure'].mean().to_string()}

"""
print(summary)

# step 2: Ask a better question
question = "What patterns do you see related to customer churn?"

# step 3: Send summary to Azure OpenAI
response = client.chat.completions.create( 
    model=CHAT_MODEL,    
    messages=[  
        {
            "role": "system",
            "content": """
            You are a data analyst.
            Answer only using the provided dataset summary.
            Give clear insights in bullet points.
            If the data is not enough, say what additional data is needed.
            """
        },        
        {            
            "role": "user",            
            "content": f"""
            
            Dataset summary:{summary}
            Question:{question}            
            """        
        }   
        
        ],    
        
        temperature=0.2
    )
        
print(response.choices[0].message.content)


Step 2: Make reusable Document Q&A

def ask_question(question, summary, client):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a data analyst.

Use ONLY the provided dataset summary.
Be precise with numbers.
Clearly mention which group each value belongs to.
If the summary does not contain enough information, say what additional data is needed.

Format:
1. Answer
2. Supporting Data
3. Business Insight
4. Missing Information
"""
            },
            {
                "role": "user",
                "content": f"""
Dataset summary:
{summary}

Question:
{question}
"""
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

#Now ask questions easily:
answer = ask_question(
    "What patterns do you see related to customer churn?",
    summary,
    client
)

print(answer)

# more questions

questions = [
    "Do customers who churn have higher monthly charges?",
    "Do customers who churn have shorter tenure?",
    "What additional analysis should I perform next?",
    "What business actions can reduce churn?"
]

for q in questions:
    print("\nQUESTION:", q)
    print(ask_question(q, summary, client))
    print("-" * 80)


Step 3: RAG (Retrieval-Augmented Generation)
#Below is basic RAG using TF-IDF keyword search, just matching words


#Step 1: Convert rows into text chunks
chunks = []

for _, row in df.iterrows():
    chunk = f"""
Customer ID: {row['customerID']}
Gender: {row['gender']}
Senior Citizen: {row['SeniorCitizen']}
Partner: {row['Partner']}
Dependents: {row['Dependents']}
Tenure: {row['tenure']}
Contract: {row['Contract']}
Internet Service: {row['InternetService']}
Payment Method: {row['PaymentMethod']}
Monthly Charges: {row['MonthlyCharges']}
Total Charges: {row['TotalCharges']}
Churn: {row['Churn']}
"""
    chunks.append(chunk)


#Step 2: Create a simple search index
from sklearn.feature_extraction.text import TfidfVectorizer # this converts text → numbers
from sklearn.metrics.pairwise import cosine_similarity # Measures how similar two texts are 
#0 → not similar, 1 → exactly same

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

#Step 3: Retrieve relevant chunks
def retrieve_chunks(question, top_k=5):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    relevant_chunks = [chunks[i] for i in top_indices]
    return relevant_chunks


#Step 4: Ask Azure OpenAI using retrieved chunks
def ask_rag(question, top_k=5):
    relevant_chunks = retrieve_chunks(question, top_k)

    context = "\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a data analyst.
Answer only using the provided context.
If the answer is not available in the context, say:
'I don't know based on the retrieved data.'
Be clear and concise.
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


#Step 5: Test it
question = "Which customers are likely to churn?"
answer = ask_rag(question, top_k=5)

#print(ask_rag("Show customers with high monthly charges who churned", top_k=10))
#print(ask_rag("Do month-to-month customers churn?", top_k=10))

print(answer)

-----------------------------------------------------------------------------------------
STEP 4: Improve RAG

#Step 1: Convert rows into text chunks
chunks = []

for _, row in df.iterrows():
    chunk = f"""
Customer ID: {row['customerID']}
Gender: {row['gender']}
Senior Citizen: {row['SeniorCitizen']}
Partner: {row['Partner']}
Dependents: {row['Dependents']}
Tenure: {row['tenure']}
Contract: {row['Contract']}
Internet Service: {row['InternetService']}
Payment Method: {row['PaymentMethod']}
Monthly Charges: {row['MonthlyCharges']}
Total Charges: {row['TotalCharges']}
Churn: {row['Churn']}
"""
    chunks.append(chunk)


#Step 2: Create a simple search index
from sklearn.feature_extraction.text import TfidfVectorizer # this converts text → numbers
from sklearn.metrics.pairwise import cosine_similarity # Measures how similar two texts are 
#0 → not similar, 1 → exactly same

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

#Step 3: Retrieve relevant chunks
def retrieve_chunks(question, top_k=5):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    relevant_chunks = [chunks[i] for i in top_indices]
    return relevant_chunks

#Step 4: RAG-style approximate search
def ask_data_question(question):
    relevant_chunks = retrieve_chunks(question, top_k=10)
    context = "\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a data analyst.
Answer only using the provided filtered data.

Rules:
- Do not invent missing data.
- Summarize the customer patterns.
- Mention Customer ID, Churn, Monthly Charges, Tenure, and Contract when relevant.
- If the filtered data is empty, say no matching customers were found.
Format output clearly:
- Use bullet points
- Add spaces between values
- Keep numbers separated from text
- Use clean readable sentences
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


print(ask_data_question("List customers who churned"))

#Step 5: Build hybrid function- Pandas exact filter + GenAI explanation

def ask_filtered_customers(question, filtered_df, client):
    data_text = filtered_df.to_string(index=False)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a data analyst.
Answer only using the provided filtered data.

Rules:
- Do not invent missing data.
- Summarize the customer patterns.
- Mention Customer ID, Churn, Monthly Charges, Tenure, and Contract when relevant.
- If the filtered data is empty, say no matching customers were found.
- Use bullet points
- Add spaces between values
- Keep numbers separated from text
- Use clean readable sentences
"""
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

Filtered data:
{data_text}
"""
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

---------------------------------------------------------------------------------
# smart routing system:Exact filter / calculation question → use PandasSearch / explanation question → use RAG

# routing logic
def route_question(question):
    q = question.lower()

    pandas_keywords = [
        "how many", "count", "average", "mean",
        "greater than", "less than", "sum",
        "min", "max", "filter", "show customers",
        "list customers", "monthly charges", "tenure"
    ]

    for word in pandas_keywords:
        if word in q:
            return "pandas"

    return "rag"

# pandas handler
def handle_pandas(question, df):
    q = question.lower()

    # Example 1
    if "high monthly charges" in q and "churn" in q:
        filtered = df[
            (df["Churn"] == "Yes") &
            (df["MonthlyCharges"] >= 80)
        ][["customerID", "tenure", "Contract", "MonthlyCharges", "Churn"]]

        return filtered.head(10)

    # Example 2
    if "average monthly charges" in q:
        return df.groupby("Churn")["MonthlyCharges"].mean()

    return "Pandas logic not defined for this question."


# Combine everything
def smart_ask(question, df):
    route = route_question(question)

    if route == "pandas":
        result = handle_pandas(question, df)

        # If DataFrame → send to GenAI for explanation
        if hasattr(result, "to_string"):
            return ask_filtered_customers(question, result.head(10), client)

        return result

    else:
        return ask_data_question(question)


print(smart_ask("Show customers with high monthly charges who churned", df))
-----------------------------------------------------------------------------------------------
Embedding-based RAG

from openai import AzureOpenAI
import os

endpoint = "xxxxxxxxxxxxxxxxxxx"
model_name = EMBEDDING_MODEL
deployment = "xxxxxxxxxxxxxxx"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxx",
    api_version="2024-02-15-preview",
    azure_endpoint="xxxxxxxxxxxxxxxxxxxxxxxxxx"
)

# embedding deployed test
response = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input="customer churn high charges"
)

print(len(response.data[0].embedding))

#Step 3: Improve prompt slightly.Use your stronger version:

"content": """
You are a data analyst.
Answer only using the provided context.

Rules:
- Use only the retrieved context.
- If customers are requested, list all matching customers from the context.
- Include Customer ID, Churn, Monthly Charges, Tenure, and Contract when available.
- Do not invent missing data.
- If the context is not enough, say what is missing.
"""

# Step 1: Create embeddings for chunks 
batch_size = 100 
chunk_embeddings = [] 
for i in range(0, len(chunks), batch_size): 
    batch = chunks[i:i+batch_size] 
    response = client.embeddings.create( 
        model=EMBEDDING_MODEL, 
        input=batch ) 
    for item in response.data: 
        chunk_embeddings.append(item.embedding) 
import numpy as np 
chunk_embeddings_np = np.array(chunk_embeddings) 

# Save embeddings (VERY IMPORTANT) 
np.save("chunk_embeddings.npy", chunk_embeddings_np) 
    
# Step 2: Semantic retrieval function 
from sklearn.metrics.pairwise import cosine_similarity 

def retrieve_chunks_embedding(question, top_k=5): 
    question_embedding = client.embeddings.create( 
        model=EMBEDDING_MODEL, 
        input=question 
        ).data[0].embedding

    question_embedding = np.array(question_embedding).reshape(1, -1) 

    similarities = cosine_similarity( question_embedding, chunk_embeddings_np ).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1] 
    relevant_chunks = [chunks[i] for i in top_indices] 
    return relevant_chunks


# Step 3: Use in RAG 
def ask_embedding_rag(question, top_k=5): 
    relevant_chunks = retrieve_chunks_embedding(question, top_k) 
    context = "\n\n".join(relevant_chunks) 
    response = client.chat.completions.create( 
    model=CHAT_MODEL, 
    messages=[ 
        { "role": "system", 
        "content": """ 
        You are a data analyst. 
        Answer only using the provided context. 
        Rules: 
        - Use only the retrieved context 
        - List all matching customers if asked 
        - Do not invent data 
        """ },
        { "role": "user", 
        "content": f""" Context: {context} Question: {question} 
        """ 
        } ], temperature=0.2 ) 
    return response.choices[0].message.content 
        
#step 4: Test (THIS IS WHERE MAGIC HAPPENS) 
print(ask_embedding_rag("customers with expensive plans who left", top_k=10))

Build a Streamlit UI

pip install streamlit

import pickle

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

