import streamlit as st
import openai  # This line and subsequent lines should have no extra indent
import pinecone
from pinecone import Pinecone, ServerlessSpec 

   # Initialize Pinecone
   pc = Pinecone(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")  

   # Connect to or create an index
   if "openaiembeddings1" not in pc.list_indexes().names():  # Call names() 
       pc.create_index(
           name="openaiembeddings1",
           dimension=1536,                                       
        metric="cosine",  # Use "euclidean" if preferred
        spec=ServerlessSpec(
            cloud="aws",
            region=st.secrets["general"]["PINECONE_ENVIRONMENT"]
        )
    )

# Access the index
index = pc.Index("openaiembeddings1")

# OpenAI API Key
openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Streamlit app setup
st.title("AI-Assisted Chatbot")
st.write("Ask me anything based on the uploaded documents!")

# User input
query = st.text_input("Enter your question:")
if query:
    # Generate embedding for the query
    query_response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = query_response["data"][0]["embedding"]

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]

    # Generate a response with GPT
    context = "\n\n".join(retrieved_texts)
    chat_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Based on this information:\n\n{context}\n\nAnswer the query: {query}"}
        ]
    )

    # Display the response
    st.write("### Response:")
    st.write(chat_response["choices"][0]["message"]["content"])
