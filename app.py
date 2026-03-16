import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# =================================
# PAGE CONFIG
# =================================

st.set_page_config(
    page_title="MediGuard AI",
    page_icon="💊",
    layout="wide"
)

# =================================
# SIDEBAR
# =================================

st.sidebar.title("💊 MediGuard AI")

st.sidebar.markdown("""
### AI Medicine Assistant

Powered by:

• Semantic Search  
• Chroma Vector Database  
• HuggingFace Embeddings  
• LLM Reasoning  

⚠ Educational use only.
""")

# =================================
# TITLE
# =================================

st.title("💊 MediGuard AI – Medicine Assistant")

st.write(
"Ask about **medicines, symptoms, ingredients, or safety alerts**."
)

# =================================
# LOAD VECTOR DATABASE
# =================================

DB_PATH = "medical_db"

@st.cache_resource
def load_vectordb():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return vectordb

vectordb = load_vectordb()

# =================================
# LOAD LLM
# =================================

def load_llm():
    os.getenv("GROQ_API_KEY")

    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key="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"
    )
    

llm = load_llm()

# =================================
# CHAT MEMORY
# =================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =================================
# USER INPUT
# =================================

user_input = st.chat_input("Ask about medicines or symptoms...")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Searching medical database..."):

        results = vectordb.similarity_search_with_relevance_scores(user_input, k=5)

        docs = []

        for doc, score in results:
            if score > 0.2:
                docs.append(doc)

        if len(docs) == 0:
            docs = [doc for doc, score in results[:3]]

        # Remove duplicate medicines
        unique_docs = {}
        for doc in docs:
            name = doc.metadata.get("medicine_name")
            if name not in unique_docs:
                unique_docs[name] = doc

        docs = list(unique_docs.values())

        if not docs:
            response_text = "No medicine information found."

        else:

            # =================================
            # MAIN MEDICINE + ALTERNATIVES
            # =================================

            main_doc = docs[0]

            main_name = main_doc.metadata.get("medicine_name", "Unknown")
            main_uses = main_doc.metadata.get("uses", "Not available")

            st.subheader("💊 Main Medicine")

            st.markdown(f"""
**{main_name}**

Uses: {main_uses}
""")

            if len(docs) > 1:

                st.subheader("🔄 Alternative Medicines")

                for doc in docs[1:]:

                    alt_name = doc.metadata.get("medicine_name", "Unknown")
                    alt_uses = doc.metadata.get("uses", "Not available")

                    st.markdown(f"""
• **{alt_name}** — {alt_uses}
""")

            # =================================
            # LLM SUMMARY
            # =================================

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are a medical AI assistant.

User question: {user_input}

Using the following medicine information:

{context}

Generate a structured explanation.

Sections:

1. Main Medicine
2. Active Ingredient
3. Uses
4. Safety Alerts
5. Alternative Medicines (if relevant)
6. Short Summary

Explain clearly in simple language.
"""

            response = llm.invoke(prompt)

            response_text = response.content

    with st.chat_message("assistant"):
        st.markdown("### 💊 Medicine Summary")
        st.write(response_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text
    })