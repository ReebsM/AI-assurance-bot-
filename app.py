<import streamlit as st
import openai
import pinecone
from pinecone import Pinecone  # Import the Pinecone class

# Initialize Pinecone and OpenAI
# Instead of pinecone.init(), create a Pinecone instance
pc = Pinecone(api_key="pcsk_3cLjwg_357fBxn8Ya2UpgJnfP7QhGF3yE6EMpMcux9UFNTLEXwYrmppkHJC8otzSdrapXi", environment="us-east1-aws")  
index = pc.Index("openaiembeddings1")  # Access the index using the Pinecone instance


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
    retrieved_texts = [match['metadata']['text'] for match in results['matches']]

    # Generate a response with GPT
    # Generate a response with GPT
context = "\n\n".join(retrieved_texts)  # Combine retrieved texts with newlines
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

