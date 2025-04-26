import re
from flask import Flask, render_template, jsonify, request
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set API keys
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

hf_model = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Change this to the repo you want
    huggingfacehub_api_token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # Use your Hugging Face API token
    model_kwargs={"temperature": 0.4, "max_new_tokens": 500}  # Adjust these as necessary
)

# Download embeddings model
embeddings = download_hugging_face_embeddings()

# Define Pinecone index
index_name = "policybot"

# Load existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Define retriever and chains
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Context:\n{context}\n\nQuestion:\n{input}")
])

question_answer_chain = create_stuff_documents_chain(hf_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg")
    if not msg:
        return jsonify({"error": "No message received"}), 400

    print(f"User Input: {msg}")
    
    # Invoke the RAG chain to get the response
    response = rag_chain.invoke({"input": msg})
    
    # Extract only the part after "Assistant:"
    full_output = response.get("answer", "")
    match = re.search(r"Assistant:\s*(.+)", full_output, re.DOTALL)

    if match:
        final_answer = match.group(1).strip()
    else:
        final_answer = full_output.strip()
    
    print("Final Answer:", final_answer)
    
    # Return the extracted answer
    return jsonify({"answer": final_answer if final_answer else "Error generating response"})

if __name__ == '__main__':
    app.run(debug=True)
