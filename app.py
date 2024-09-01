from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import json
import os
from PyPDF2 import PdfReader
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import openai  # Importing openai to handle the RateLimitError
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = ''

# Set up the LLM (Language Model)
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)


# Load authorized users from a file
def load_users():
    try:
        with open('users.json') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {str(e)}")
        return {}

def load_excel(file_path):
    """Load Excel file and return text content as Document objects."""
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Convert DataFrame to JSON
    json_data = df.to_json(orient='records')
    
    # Parse JSON string back to Python object
    data = json.loads(json_data)
    
    # Convert JSON objects to Document objects
    documents = [Document(page_content=json.dumps(item)) for item in data]

    # print(documents[1])
    
    return documents

def load_pdf(file_path):
    """Load PDF file and return text content as Document objects."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return pages
    except Exception as e:
        print(f"Error loading PDF file {file_path}: {str(e)}")
        return []

def load_files(list_of_files):
    """Load and process files into Document objects with error handling."""
    all_documents = []
    for file in list_of_files:
        if file.filename.endswith('.pdf'):
            # Save file temporarily and read it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                file_path = temp_file.name
            documents = load_pdf(file_path)
            all_documents.extend(documents)
            os.remove(file_path)  # Clean up temp file
        elif file.filename.endswith('.xlsx'):
            # Save file temporarily and read it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                file_path = temp_file.name
            documents = load_excel(file_path)
            all_documents.extend(documents)
            os.remove(file_path)  # Clean up temp file
        else:
            print(f"Unsupported file format: {file.filename}")
    
    return all_documents

# def create_or_load_faiss_index(list_of_files, faiss_index_path="openai_index"):
#     """Create a FAISS index if it doesn't exist; otherwise, load it."""
#     if os.path.exists(faiss_index_path):
#         print()
#         print("Loading existing FAISS index...")
#         print()
#         # db = FAISS.load_local(faiss_index_path, embedding_function, allow_dangerous_deserialization=True)
#         db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     else:
#         print("Creating new FAISS index...")
#         documents = load_files(list_of_files)
#         if documents:
#             db = FAISS.from_documents(documents, OpenAIEmbeddings())
#             db.save_local(faiss_index_path)
#         else:
#             raise ValueError("No valid documents were loaded for indexing.")

#         # db = FAISS.from_documents(documents, embedding_function)
#         # db.save_local(faiss_index_path)
    
#     return db

def create_or_load_faiss_index(list_of_files, faiss_index_path="openai_index"):
    """Create a FAISS index if it doesn't exist; otherwise, load and update it."""
    print("Loading or creating FAISS index...")
    documents = load_files(list_of_files)
    
    if not documents:
        raise ValueError("No valid documents were loaded for indexing.")
    
    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS index...")
        db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        print("Updating existing FAISS index with new documents...")
        db.add_documents(documents)
    else:
        print("Creating new FAISS index...")
        db = FAISS.from_documents(documents, OpenAIEmbeddings())
    
    print("Saving updated FAISS index...")
    db.save_local(faiss_index_path)
    
    return db

# def create_or_load_faiss_index(list_of_files, faiss_index_path="openai_index"):
#     """Create a FAISS index if it doesn't exist; otherwise, load it."""
#     if os.path.exists(faiss_index_path):
#         print()
#         print("Loading existing FAISS index...")
#         print()
#         db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     else:
#         print("Creating new FAISS index...")
#         db = None

#     documents = load_files(list_of_files)

#     # Chunk the documents
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunked_documents = text_splitter.split_documents(documents)
#     if chunked_documents:
#         if db:
#             db.add_documents(chunked_documents)
#         else:
#             db = FAISS.from_documents(chunked_documents, OpenAIEmbeddings())
#         db.save_local(faiss_index_path)
#     else:
#         raise ValueError("No valid documents were loaded for indexing.")

#         # db = FAISS.from_documents(documents, embedding_function)
#         # db.save_local(faiss_index_path)
    
#     return db


def query_faiss_index(user_input, files=None, faiss_index_path="openai_index"):
    """Answer questions using the FAISS index."""
    # Create or load FAISS index
    if files:
        db = create_or_load_faiss_index(files, faiss_index_path)
    else:
        db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    # Use the retriever from the FAISS index
    # retriever = db.as_retriever()
    retriever = db.as_retriever(search_kwargs={"k": 10})  # Change 10 to the desired number of documents
    # Create the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

    # Get the result for the user's query
    result = qa({"query": user_input})

    source_documents = result.get('source_documents', [])
    print('source documents:')
    for doc in source_documents:
        print(doc.metadata, doc.page_content[:30])
        break

    return result['result']

@app.route('/')
def home():
    if 'email' in session:
        users = load_users()
        return render_template('index.html', users=users)  # Pass users to the chat page for both admins and users
    return render_template('login.html')  # Login page for unauthenticated users

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        users = load_users()
        
        if (email in users['users'] and users['users'][email] == password) or (email in users['admins'] and users['admins'][email] == password):
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
    users = load_users()
    if 'email' not in session or session['email'] not in users['admins']:
        return "Access denied", 403  # Forbidden access for unauthorized users

    if request.method == 'POST':
        # Handle file uploads here
        files = request.files.getlist('files')
        try:
            # Create or load FAISS index
            db = create_or_load_faiss_index(files)
            return "Files processed and indexed successfully", 200  
        except Exception as e:
            print(f"Error during file processing or indexing: {str(e)}")
            return jsonify({"error": f"Error during file processing or indexing: {str(e)}"}), 500

    return render_template('upload.html') if 'email' in session and session['email'] in users['admins'] else redirect(url_for('home'))  # Render the upload page for admins only

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == 'POST':
        user_message = request.json.get('message')
        
        try:
            response = query_faiss_index(user_message)
            return jsonify({"response": response})
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    
    return render_template('index.html')  # Render the chat page for GET requests

@app.route('/add-users', methods=['GET', 'POST'])
def add_users():
    if 'email' not in session or session['email'] not in load_users()['admins']:
        return "Access denied", 403

    if request.method == 'GET':
        return render_template('add_users.html')

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            required_columns = ['email', 'password', 'role']
            if not all(col in df.columns for col in required_columns):
                return "File must contain email, password, and role columns", 400
            
            new_users = df.to_dict('records')
            users = load_users()
            
            for user in new_users:
                if user['role'] == 'admin':
                    users['admins'][user['email']] = user['password']
                elif user['role'] == 'user':
                    users['users'][user['email']] = user['password']
                else:
                    return f"Invalid role for user {user['email']}", 400
            
            with open('users.json', 'w') as f:
                json.dump(users, f)
            
            return "Users added successfully", 200
        except Exception as e:
            return f"Error processing file: {str(e)}", 500
    else:
        return "Invalid file format. Please upload a CSV or Excel file.", 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
