import os
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Path to persist the vector database
DB_PATH = "medical_db"

# Load the CSV dataset
df = pd.read_csv("medicines.csv")

# Handle missing values safely
df = df.fillna("Not Available")

documents = []

# Convert each row into a LangChain Document
for _, row in df.iterrows():
    content = f"""
Medicine: {row['name']}
Active Ingredient: {row['active_ingredient']}
Excipients: {row['excipients']}
Uses: {row['uses']}
Safety Alerts: {row['safety_alerts']}
"""

    metadata = {
        "medicine_name": row["name"],
        "uses": row["uses"]
    }

    doc = Document(page_content=content.strip(), metadata=metadata)
    documents.append(doc)

# Initialize HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create and persist Chroma vector database
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_PATH
)

vector_db.persist()

print("Medical vector database created successfully.")