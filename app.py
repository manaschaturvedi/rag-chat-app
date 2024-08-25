from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import json
import os
from PyPDF2 import PdfReader
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import openai  # Importing openai to handle the RateLimitError

os.environ['OPENAI_API_KEY'] = ''

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Load authorized users from a file
def load_users():
    try:
        with open('users.json') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {str(e)}")
        return {}

@app.route('/')
def home():
    if 'email' in session:
        return render_template('index.html')  # Chat page for logged in users
    return render_template('login.html')  # Login page for unauthenticated users

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        users = load_users()
        
        if email in users and users[email] == password:
            session['email'] = email
            return redirect(url_for('home'))
        else:
            return "Invalid credentials", 401  # You can render a template for error instead

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    authorized_email = 'abc@test.com'  # Change this to the authorized email
    if 'email' not in session or session['email'] != authorized_email:
        return "Access denied", 403  # Forbidden access for unauthorized users

    if request.method == 'POST':
        # Handle file uploads here
        files = request.files.getlist('files')
        
        # Process the uploaded files
        all_text = []
        for file in files:
            try:
                if file.filename.endswith('.pdf'):
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        all_text.append(page.extract_text())
                elif file.filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                    all_text.append(df.to_string())
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 500

        # Preprocess and split the text
        try:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_text('\n'.join(all_text))

            # Create embeddings
            embeddings = OpenAIEmbeddings()

            # Create and save FAISS index
            db = FAISS.from_texts(texts, embeddings)
            db.save_local("faiss_index")

            return "Files processed and indexed successfully", 200
        except Exception as e:
            print(f"Error during text processing or indexing: {str(e)}")
            return jsonify({"error": f"Error during text processing or indexing: {str(e)}"}), 500

    return render_template('upload.html')  # Render the upload page

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == 'POST':
        user_message = request.json.get('message')
        
        try:
            # Load the FAISS index
            embeddings = OpenAIEmbeddings()
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            # Create a retriever
            retriever = db.as_retriever(search_kwargs={"k": 3})
            
            # Create a conversational chain
            llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.1)
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
            
            # Get the response
            result = qa_chain({"question": user_message, "chat_history": []})
            
            return jsonify({"response": result['answer']})
        # except openai.error.RateLimitError:  # Updated to use the correct exception handling
        #     return jsonify({"error": "OpenAI rate limit exceeded. Please try again later."}), 429
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    
    return render_template('index.html')  # Render the chat page for GET requests

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
