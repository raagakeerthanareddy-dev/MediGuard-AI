1. Problem Statement
Many people search for medicine information online but receive confusing or unstructured results. This project aims to build an AI-powered assistant that provides clear and structured information about medicines, including uses, ingredients, safety alerts, and alternatives.
2. System Architecture
The system follows a Retrieval-Augmented Generation (RAG) pipeline.
User Query → Streamlit Interface → Embedding Generation → Chroma Vector Database → Similarity Search → LLM (Groq Llama 3.1) → Structured Medicine Summary → Displayed to User
3. Embedding & Retrieval Approach
The project uses the embedding model:
sentence-transformers/all-MiniLM-L6-v2
Medicine data is converted into vector embeddings and stored in ChromaDB.
When a user asks a question, the system retrieves the most relevant medicines using semantic search.
4. Method used:
similarity_search_with_relevance_scores()
5. Prompt Design
The LLM receives retrieved medicine information and generates structured output.
Prompt format:
You are a medical AI assistant.
Using the provided medicine information, generate:
Medicine Name
Active Ingredient
Uses
Safety Alerts
Alternative Medicines
Short Summary
6. Failure Case
Query:
What is Ibuprofen used for?
Because Ibuprofen and Naproxen are both pain-relief medicines, semantic search may sometimes retrieve Naproxen first due to similarity in medical context.
7. Reflection
This project demonstrates how RAG can be used to build an AI healthcare assistant. Semantic search enables flexible queries, while LLMs generate understandable explanations. Future improvements could include drug interaction detection and larger medical datasets.
8. Example Queries
Tell me about Paracetamol
Medicine for fever
Drug used for diabetes
Safety alerts of Ibuprofen
9. requirements:
pandas
streamlit
langchain
langchain-community
chromadb
sentence-transformers
 langchain_huggingface
 langchain_core
 langchain_groq
10. how to run project?
1.pip install -r requirements.txt
2.streamlit run app.py
3.open browser

