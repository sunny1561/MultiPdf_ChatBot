import streamlit as st
import requests

# Define the FastAPI endpoint
FASTAPI_URL = "http://localhost:8000"

st.title("PDF QA Chatbot Interface")

# File upload
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

# If files are uploaded, send them to the FastAPI server
if uploaded_files:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        files = {"files": (uploaded_file.name, bytes_data)}
        response = requests.post(f"{FASTAPI_URL}/upload_pdf/", files=files)
        if response.status_code == 200:
            st.success(f"Uploaded {uploaded_file.name} successfully!")
        else:
            st.error(f"Failed to upload {uploaded_file.name}.")

# Question input
user_question = st.text_input("Enter your question:")

# Submit button for question
if st.button("Get Answer"):
    chat_history = []  # Assume you fetch or maintain chat history somehow
    if user_question:
        # Send the question to the FastAPI server
        response = requests.post(f"{FASTAPI_URL}/query/", json={"user_question": user_question, "chat_history": chat_history})
        if response.status_code == 200:
            result = response.json()
            st.write("Answer:", result.get("reply"))
        else:
            st.error("Failed to get an answer from the server.")
