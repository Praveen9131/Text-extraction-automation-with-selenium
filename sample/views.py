import os
import openai
import logging
import mysql.connector
from dotenv import load_dotenv
from django.http import JsonResponse
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback  # For token tracking
from sample.utils.text_extraction import extract_text_from_multiple_pdfs
import json
import re
import tiktoken
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model_choice = "gpt-4o-mini"

# Initialize tokenizer corresponding to a specific model
tokenizer = tiktoken.encoding_for_model("gpt-4o")

# MySQL connection setup

def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",            # Replace with your MySQL host (e.g., "127.0.0.1" or your server IP)
        user="root",                 # Replace with your MySQL username
        password="9121564760Jp",      # Replace with your MySQL password
        database="my_database"    # Replace with your MySQL database name
    )

# Function to store details in MySQL
def store_details_in_mysql(published_id, persist_directory, merged_pdf_url, database_url):
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()

        # Create a table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_details (
                id INT AUTO_INCREMENT PRIMARY KEY,
                published_id VARCHAR(255),
                persist_directory VARCHAR(255),
                merged_pdf_url VARCHAR(2083),
                database_url VARCHAR(2083)
            )
        """)

        # Insert data into the table
        insert_query = """
            INSERT INTO pdf_details (published_id, persist_directory, merged_pdf_url, database_url)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (published_id, persist_directory, merged_pdf_url, database_url))

        # Commit the transaction
        connection.commit()

        logging.info(f"Details stored in MySQL: {published_id}, {persist_directory}, {merged_pdf_url}, {database_url}")

    except mysql.connector.Error as err:
        logging.error(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Simple function to calculate token count and cost
def calculate_tokens(text):
    if not text.strip():
        logging.error("No text provided for token calculation.")
        return

    try:
        # Calculate token count using tokenizer
        tokens = tokenizer.encode(text)
        total_tokens = len(tokens)
        cost_per_million_tokens = 0.130
        cost_estimate = (total_tokens / 1_000_000) * cost_per_million_tokens

        logging.info(f"Calculated Token Count: {total_tokens}")
        logging.info(f"Estimated Cost for Tokens: ${cost_estimate:.6f}")

        return {
            "total_tokens": total_tokens,
            "estimated_cost": cost_estimate
        }

    except Exception as e:
        logging.error(f"Error during token calculation: {e}")

# View for storing PDFs
def store_pdfs(request):
    if request.method == 'GET':
        logging.info("Received request to store PDFs.")
        pdf_paths = request.GET.get('pdf_path')  # Assuming it's a comma-separated list of paths
        language = request.GET.get('language')
        persist_directory = request.GET.get('persist_directory')
        published_id = request.GET.get('published_id')
        merged_pdf_url = request.GET.get('merged_pdf_url')
        database_url = request.GET.get('database_url')

        # Validate input parameters
        if not pdf_paths or not language or not persist_directory or not published_id or not merged_pdf_url or not database_url:
            missing_params = []
            if not pdf_paths:
                missing_params.append("'pdf_path'")
            if not language:
                missing_params.append("'language'")
            if not persist_directory:
                missing_params.append("'persist_directory'")
            if not published_id:
                missing_params.append("'published_id'")
            if not merged_pdf_url:
                missing_params.append("'merged_pdf_url'")
            if not database_url:
                missing_params.append("'database_url'")
            logging.error(f"Missing parameters: {', '.join(missing_params)} are required.")
            return JsonResponse({"error": f"Missing parameters: {', '.join(missing_params)} are required."}, status=400)

        # Convert pdf_paths from string to list
        pdf_paths = pdf_paths.split(',')

        # Validate PDF path existence
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logging.error(f"PDF file not found: '{pdf_path}'")
                return JsonResponse({"error": f"PDF file not found: '{pdf_path}'"}, status=400)

        # Extract text from the PDFs
        final_extracted_text = extract_text_from_multiple_pdfs(pdf_paths, language)

        # Calculate token count and cost
        token_info = calculate_tokens(final_extracted_text)
        if token_info:
            logging.info(f"Token count before embeddings: {token_info['total_tokens']}")
            logging.info(f"Estimated cost for tokens: ${token_info['estimated_cost']:.6f}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
        documents = text_splitter.split_text(final_extracted_text)

        if documents:
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")
            db = Chroma.from_texts(documents, embeddings, persist_directory=persist_directory)
            logging.info(f"Data stored successfully in {persist_directory}.")

            # Store the details in MySQL
            store_details_in_mysql(published_id, persist_directory, merged_pdf_url, database_url)

            return JsonResponse({"message": f"Data stored successfully in {persist_directory}.", "token_count": token_info['total_tokens'], "estimated_cost": token_info['estimated_cost']})
        else:
            logging.error("No text extracted, database creation skipped.")
            return JsonResponse({"error": "No text extracted, database creation skipped."}, status=400)
    else:
        logging.error("Invalid request method. Please use GET.")
        return JsonResponse({"error": "Invalid request method. Please use GET."}, status=405)
# Actual implementation of the initialize_llm_and_vectorstore function
def initialize_llm_and_vectorstore(splits=None, persist_directory=None):
    """Initialize the OpenAI LLM and set up the Chroma vector store with persistence."""
    
    # Retrieve the OpenAI API key from the environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please make sure it is set in the .env file.")
    
    if not persist_directory:
        raise ValueError("Persistent directory must be provided.")
    
    # Initialize the LLM
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini" , max_tokens=2000)
    embedding = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")  # Ensure consistent model
    
    # Check if the vector store already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        # Create the vector store if it doesn't exist
        if splits is None:
            raise ValueError("Document splits must be provided to create a new vector store.")
        
        print(f"Creating new vector store and saving to {persist_directory}")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    
    return llm, retriever

def run_retrieval_chain(llm, retriever):
    """Run the retrieval-based chain with a system prompt."""
    user_input = """Extract the complete Table of Contents from the provided document, maintaining:
    1. All main subject sections (e.g., History, Geography, Civics)
    2. All chapter numbers and titles under each section
    3. Exact page numbers with dotted line formatting
    4. Additional sections like 'Suggestions for Assessment' and 'Text Pages'
    5. Original language and formatting

    Return the Table of Contents exactly as structured in the document.

    Document content:
    {context}"""

    system_prompt = """You are an AI assistant specialized in extracting and formatting Table of Contents from textbooks and educational materials. Your task is to:

    1. Format the output to match the exact document structure:

    [Subject Section]
    1. [Chapter Title].[Page Number]
    2. [Chapter Title].[Page Number]
    [Additional Sections].[Page Number]

    Requirements:
    - Preserve all main subject divisions (like History, Geography, Civics)
    - Maintain sequential chapter numbering within each subject section
    - Include supplementary sections (like Suggestions for Assessment)
    - Use dotted lines between titles and page numbers
    - Keep exact page numbers as shown in original
    - Use consistent spacing and alignment
    - Present content in the original language
    - Preserve any special formatting or indentation
    - Include ALL chapters and sections with no omissions
    - Do not add explanatory text or modifications
    - Do not reorder or reorganize the content structure

    Important: Ensure that the output matches the document's original organization, including:
    - Main subject divisions
    - Chapter numbering style (1., 2., etc.)
    - Additional sections under each subject
    - Page number alignment
    - Section breaks and spacing
    -Important: make sure that ,you are expert and intelliget give all chapters from the tabel of content chapter1, chapter2,chapter3, chapter4, chapter5, chapter6, chapter 7, chapter8, chapter9, chapter10, chapter11, chapter 12 etc (if present)
    - Reflect the whole Tabel of contents page as it is without missing any word

    {context}"""

    # Combine system and human prompts
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Measure response time and cost
    start_time = time.time()
    with get_openai_callback() as callback:
        # Query the chain
        results = rag_chain.invoke({"input": user_input})
        total_tokens = callback.total_tokens
        total_cost = callback.total_cost
        prompt_tokens = callback.prompt_tokens
        completion_tokens = callback.completion_tokens

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Response Time: {response_time:.2f} seconds")
    print(f"Total Tokens: {total_tokens}")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Cost (USD): ${total_cost:.6f}")

    return {
        "response_time": response_time,
        "answer": results.get("answer", None),
        "token_count": total_tokens,
        "estimated_cost": total_cost
    }

# Post-processing to clean up duplicates and validate content
def clean_up_chapters_and_topics(answer):
    """Ensure no duplicate chapters or topics exist and remove irrelevant content."""
    lines = answer.splitlines()
    unique_lines = []
    seen = set()

    for line in lines:
        if line.strip() and line not in seen:
            unique_lines.append(line)
            seen.add(line)

    return "\n".join(unique_lines)

# Check for non-related or irrelevant content
def is_non_related_content(answer):
    """Check for non-related or irrelevant content in the answer."""
    non_related_keywords = ["irrelevant", "unrelated", "unwanted", "noise", "sorry", "editorial","Foreword", "Preface", "Acknowledgments","Appendix"]
    return any(keyword in answer.lower() for keyword in non_related_keywords)

def load_and_query(persist_directory, output_file="result.txt", retries=3):
    """Load an existing vector store, perform a query, and save the results in a text file."""
    
    # Initialize the language model and retriever
    llm, retriever = initialize_llm_and_vectorstore(persist_directory=persist_directory)

    attempt = 0
    while attempt < retries:
        attempt += 1
        print(f"Attempt {attempt}...")

        # Run the retrieval chain to get the results
        results = run_retrieval_chain(llm, retriever)

        # Extract the 'answer' portion which contains the chapters and topics
        answer = results.get('answer', '').strip()

        if answer:
            # Check for non-related content
            if is_non_related_content(answer):
                print(f"Non-related content detected on attempt {attempt}. Retrying...")
                continue  # Retry if non-related content is found

            # Clean up chapters and topics to ensure no duplicates
            final_result = clean_up_chapters_and_topics(answer)
            final_result = f"Chapters and Topics:\n{final_result}\n\nToken Count: {results['token_count']}\nEstimated Cost: ${results['estimated_cost']:.6f}"
            break
        else:
            final_result = "No content retrieved from the document."
    
    else:
        final_result = "Max retries reached without retrieving valid content."

    # Write the result to a text file
    with open(output_file, "w") as f:
        f.write(final_result)

    # Print the result to the console
    print(f"Results saved to {output_file}")
    print(final_result)

    # Return the final result as a string
    return final_result

def retrieve_answer(request):
    if request.method == 'GET':
        logging.info("Received request to retrieve answer.")
        persist_directory = request.GET.get('persist_directory')

        # Validate input parameters
        if not persist_directory:
            logging.error("Missing parameter: 'persist_directory' is required.")
            return JsonResponse({"error": "Missing parameter: 'persist_directory' is required."}, status=400)

        # Run the query
        try:
            answer = load_and_query(persist_directory=persist_directory)
            return JsonResponse({"answer": answer})
        except Exception as e:
            error_msg = f"Error retrieving answer: {str(e)}"
            logging.error(error_msg)
            return JsonResponse({"error": error_msg}, status=500)
    else:
        logging.error("Invalid request method. Please use GET.")
        return JsonResponse({"error": "Invalid request method. Please use GET."}, status=405)