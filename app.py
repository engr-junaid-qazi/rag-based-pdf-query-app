import os
import streamlit as st
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from groq import Groq
import numpy as np

# Load your environment variables
API = os.environ['GROQ_API_KEY'] = "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"

# Initialize Groq client
client = Groq(api_key=API)

# Initialize HuggingFace embedding model from langchain_community
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Determine the vector size dynamically by generating a sample embedding
sample_embedding = embedding_model.embed_query("test")
dimension = len(sample_embedding)

# Initialize FAISS
index = faiss.IndexFlatL2(dimension)

# Streamlit front-end
st.title("RAG-based PDF Query Application")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Extract text from PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.write("PDF Uploaded Successfully!")
    
    # Create chunks
    def create_chunks(text, chunk_size=500):
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    chunks = create_chunks(text)
    st.write(f"Created {len(chunks)} chunks.")

    # Generate embeddings
    embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
    embeddings = np.array(embeddings, dtype=np.float32)  # Convert to float32
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    st.write("Embeddings generated and stored in FAISS.")

    # Query input
    user_query = st.text_input("Enter your query:")
    if user_query:
        # Query embedding
        query_embedding = embedding_model.embed_query(user_query)
        query_embedding = np.array([query_embedding], dtype=np.float32)  # Convert to float32
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        k = 3  # Number of nearest neighbors
        distances, indices = index.search(query_embedding, k)
        relevant_chunks = [chunks[i] for i in indices[0]]

        # Pass to Groq API
        prompt = "\n\n".join(relevant_chunks) + f"\n\nUser Query: {user_query}"
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )
        response = chat_completion.choices[0].message.content
        
        # Display response
        st.write("### Response")
        st.write(response)
