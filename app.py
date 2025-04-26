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


# Fetch API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Set API keys
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

hf_model = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Change this to the repo you want
    huggingfacehub_api_token=HF_API_TOKEN,  # Use your Hugging Face API token
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

def extract_final_answer(full_output, user_question):
    """
    Extracts the actual model answer intelligently from full_output text.
    """
    full_output = full_output.strip()

    # First try matching "Answer:" using regex
    answer_matches = list(re.finditer(r"Answer:\s*(.+?)(?=(\n|$))", full_output, re.DOTALL))
    if answer_matches:
        # Pick the last Answer: found
        last_match = answer_matches[-1]
        return last_match.group(1).strip()

    # Else fallback: Try matching "Assistant:" using regex
    assistant_matches = list(re.finditer(r"Assistant:\s*(.+?)(?=(\n|$))", full_output, re.DOTALL))
    if assistant_matches:
        last_match = assistant_matches[-1]
        return last_match.group(1).strip()

    # Else fallback: Try matching the line after the user's question
    lines = [line.strip() for line in full_output.split("\n") if line.strip()]
    for i, line in enumerate(lines):
        if line.lower() == user_question.lower() and i + 1 < len(lines):
            # Return the next line if it's a proper sentence
            if len(lines[i + 1].split()) > 3:
                return lines[i + 1]

    # If no relevant answer is found, return the custom message
    if "maturity age" in user_question.lower() or "jeevan umang" in user_question.lower():
        return "I have not been updated with the data you're asking for regarding the maturity age in Jeevan Umang. Please contact your insurance policy guide for more details."

    # Else final fallback: first long enough sentence
    for line in lines:
        if len(line.split()) > 3 and line.lower() != user_question.lower():
            return line

    return "Sorry, I couldn't find an answer to your question."


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg")
    if not msg:
        return jsonify({"error": "No message received"}), 400

    print(f"User Input: {msg}")
    
    response = rag_chain.invoke({"input": msg})
    
    full_output = response.get("answer", "").strip()

    final_answer = extract_final_answer(full_output, msg)

    print("Final Answer:", final_answer)
    
    return jsonify({"answer": final_answer if final_answer else "Error generating response"})




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

