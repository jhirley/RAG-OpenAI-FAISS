import re
import json
import requests
import streamlit as st
from datetime import datetime
from bs4 import BeautifulSoup
from src.keys.keys import get_api_key
from langchain.chains import RetrievalQA

# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
### Search
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

# Lets get the needed api keys
openai_api_key = get_api_key("OPENAI_API_KEY")
tavily_api_key = get_api_key("TAVILY_API_KEY")

# lets define the web search tool and k
web_search_tool = TavilySearchResults(k=3, tavily_api_key= tavily_api_key)


# Function to extract URLs from text
def fetch_url_content(url):
    """
    Fetches and returns the text content from all paragraph tags of a given URL.
    Args:
        url (str): The URL of the webpage to fetch content from.
    Returns:
        str: The concatenated text content of all paragraph tags if the request is successful.
             An error message if the request fails or an exception occurs.
    
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, verify=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content from all paragraph tags
            paragraphs = soup.find_all('p')

            # Concatenate the text content from all paragraphs
            text = '\n'.join([p.get_text() for p in paragraphs])

            # Return the concatenated text content
            return text
        else:
            return f"Error fetching content from {url}"
    except Exception as e:
        return f"Error fetching content: {e}"


def extract_urls_from_source_documents(source_documents):
    """
    Extracts URLs from the source documents' metadata.

    Args:
        source_documents (list): List of source documents as dictionaries.

    Returns:
        list: List of URLs extracted from the metadata.
    """
    urls = []
    for doc in source_documents:
        
        # If the document is a string, use regex to extract the URL
        if isinstance(doc, str):

            # Use regex to extract the URL from the metadata
            url_pattern = re.compile(r"metadata=\{'url': '([^']+)'")
            
            # Search for the URL pattern in the document
            match = url_pattern.search(doc)

            # If a match is found, extract the URL
            if match:
                urls.append(match.group(1))
        
        # If the document is a dictionary, extract the URL from the metadata
        elif isinstance(doc, Document):
        
            # Extract the URL from the metadata
            metadata = doc.metadata

            # Get the URL from the metadata
            url = metadata.get("url", "")  

            # Append the URL to the list of URLs 
            if url:
                urls.append(url)
    
    return urls

def process_and_store_documents(urls):
    """
    Processes a list of URLs, fetches their content, splits the content into smaller chunks,
    converts the chunks into embeddings, and stores them in a FAISS index with metadata.
    Args:
        urls (list of str): List of URLs to process.
    Returns:
        FAISS: A FAISS index containing the document embeddings and metadata.
    Steps:
        1. Fetch content from each URL.
        2. Split the content into smaller chunks.
        3. Convert the chunks into embeddings.
        4. Store the embeddings and metadata in a FAISS index.
    """

    # create an empty list to store the documents 
    documents = []

    # Get today's date
    today_date = datetime.today().strftime('%Y-%m-%d')
    
    # Fetch content from each URL
    for url in urls:
        content = fetch_url_content(url)

        # If content is not empty, add it to the list of documents
        if content:
            documents.append({
                "content": content, 
                "url": url, 
                "date": today_date
                }
            )
    
    # Text Splitting: Split documents into smaller chunks (useful for retrieval)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Create a list to store the chunks
    chunks = []

    # Split the content of each document into smaller chunks
    for document in documents:
        
        # Split the text content into smaller chunks
        for chunk in text_splitter.split_text(document["content"]):

            # Add the chunk to the list of chunks
            chunks.append({
                "content": chunk,
                "metadata": {
                    "url": document["url"],
                    "date": document["date"]
                }
            })
    
    # Convert documents to embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   

    # Convert chunks to a list of strings
    chunk_texts = [chunk["content"] for chunk in chunks]

    # Embed the chunks using the OpenAI embeddings
    # encoded_chunks = embeddings.embed_documents(chunk_texts)
    
    # Create FAISS index to store document embeddings with metadata
    vectorstore = FAISS.from_texts(
        texts=chunk_texts,
        embedding=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks]
    )
    
    return vectorstore

def grade_documents(documents, question):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search
    """

    # create an empty list to store the urls
    urls = []
    # Use OpenAI as the LLM
    for doc in documents:
        # Grader prompt
        try:
            # Grader prompt
            doc_grader_prompt = f"""Here is the retrieved document: \n\n {doc["content"]} \n\n Here is the user question: \n\n {question}. 

            Review the document and the question, carefully and objectively assess whether the document contain information that is relevant to the question.

            Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""


            # # Use OpenAI to determine if the question is suitable for the vectorstore or needs a web search
            llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)

            # Call the llm and wait for a response
            response = llm.invoke(doc_grader_prompt + f"\n\nQuestion: {question}")

            # Extract the answer (either 'yes' or 'no' ') from the response
            answer = json.loads(response.content)

            # If the document is relevant, add the URL to the list
            if answer["binary_score"] == "yes":
                urls.append(doc["url"])

        except Exception as e:
            return {"error": str(e)}
        
    # If no relevant documents are found, return an error message
    if urls:
        # Process and store the relevant documents
        vectorstore = process_and_store_documents(urls)
        # Return the vectorstore
        return vectorstore
    else:
        return {"error": "No relevant documents found"}

def web_search(question):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    st.write("---WEB SEARCH---")
    # Create an empty list to store the documents
    documents = []

    # Web search
    docs = web_search_tool.invoke({"query": f"From the United States {question}"})

    # Extract the content from the documents
    web_results = "\n".join([d["content"] for d in docs])

    # Add the web search results to the list of documents
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    # Return the updated state with the web search results
    return documents

def create_qa_chain(vectorstore):
    """
    Creates a question-answering (QA) chain using a vector store and OpenAI's language model.
    Args:
        vectorstore: The vector store to use for retrieving relevant documents.
    Returns:
        qa_chain: A RetrievalQA chain configured with a map-reduce or refine chain type, 
                  a customized prompt template, and a retriever from the provided vector store.
    """ 
    # Use OpenAI as the LLM
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Get today's date
    today_date = datetime.today().strftime('%Y-%m-%d')
    
    # Define the prompt template with today's date
    prompt_template = f"""
    Today's date is {today_date}. Based on this, answer the following question by considering the relevant context from the retrieved documents. 
    Provide any urls from the metadata for sources used to create the response.

    """

    # Create a ChatPromptTemplate that dynamically fills in today's date
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", 
            f"""You are a legal assitant researching and providing a clear and concise response to the question.
            The travel information is based on the current date: {today_date}.
            Provide any urls from the metadata for sources used to create the response.
            The Travel is from the united states to another country.
            Please try to use the information from the vectorstore to answer the question.
            If the information is not available in the vectorstore, use the web search for the morst recent information. 
            The travel is for a Non US citizen.
            """),
        ("user", prompt_template),
    ])


    # Set up the retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Create the RetrievalQA chain with the customized prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # Or "map_reduce", "refine" depending on the use case
        retriever=retriever,
        return_source_documents=True,
)

    return qa_chain

def route_question(question, vectorstore):
    """
    Routes a user question to either a vectorstore or a web search based on the content of the question.
    Args:
        question (str): The user's question that needs to be routed.
        vectorstore (str): The name or identifier of the vectorstore to be used for routing.
    Returns:
        str: A JSON string with a single key, 'datasource', that is either 'websearch' or 'vectorstore' depending on the question.
             If an error occurs, returns a JSON string with a single key, 'error', containing the error message.
    """
    # Use OpenAI as the LLM
    llm= OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Define the router instructions
    router_instructions = f"""
    You are a legal assitant responsible from routing a user questions to a vectorstore or web search.
    Review the vectorstore: {vectorstore}, If questions can be answered by the supporting facts in the vectorstore, 
    respond with 'vectorstore'.
    If the question can not be answered with the data contained in the vectorstore or for current events, use websearch.
    Return JSON with a single key, 'datasource', that is 'websearch' or 'vectorstore' depending on the question.
    """

    try:
        # Use OpenAI to determine if the question is suitable for the vectorstore or needs a web search
        response = llm.Completion.create(
            engine="gpt-3.5-turbo",  # Change to  if you prefer
            prompt=router_instructions + f"\n\nQuestion: {question}",
            max_tokens=100,
            temperature=0.5
        )

        # Extract the answer (either 'vectorstore' or 'websearch') from the response
        answer = response.choices[0].text.strip().lower()
    
    except Exception as e:
        return {"error": str(e)}
    
    return answer

def check_for_hallucination(answer, vectorstore):
    """
    Evaluates an AI-generated answer for hallucinations based on provided facts.
    This function uses an OpenAI language model to assess whether the given answer 
    is grounded in the provided facts (vectorstore) and does not contain any 
    hallucinated information outside the scope of those facts.
    
    Parameters:
    answer (str): The AI-generated answer to be evaluated.
    vectorstore (str): The factual information against which the answer will be checked.
    Returns:
    str: The graded answer with an explanation of the reasoning, or an error message if an exception occurs.
    """

    # Use OpenAI as the LLM
    llm= OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Define the hallucination grader instructions
    hallucination_grader_instructions = f"""

    you are an advaserial AI review the response from another AI.
    You will be given FACTS and an ANSWER. 
    Review the vectorstore, Use the vectorstore: {vectorstore} for FACTS: which support the  answer.
    Here is the grade criteria to follow:
    (1) Ensure the ANSWER is grounded in the FACTS. 
    (2) Ensure the ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
    Score:
    A score of yes means that the answer meets all of the criteria. This is the highest (best) score. 
    A score of no means that the answer does not meet all of the criteria. This is the lowest possible score you can give.
    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
    Avoid simply stating the correct answer at the outset."""

    try:
        # initate the llm 
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)

        # call the llm and wait for a response
        response = llm.invoke(hallucination_grader_instructions  + f"\n\n ANSWER: {answer}")

        # Extract the answer from the response
        graded_answer = (response.content)
       
    except Exception as e:
        return {"error": str(e)}
    
    return graded_answer

urls = [
    # "https://travel.state.gov/content/travel/en/international-travel.html",
    # "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html",
    # "https://www.osac.gov",
    # "https://www.osac.gov/Country",
    # "https://www.cbp.gov",
    # "https://www.iata.org",
    # "https://www.travel-advisory.info",
    # "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Canada.html", 
    "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Brazil.html",
    # "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Egypt.html",
    # "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/IsraeltheWestBankandGaza.html?wcmmode=disabled",
    # "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Thailand.html",
    # "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Ukraine.html",
    # "https://travel.gc.ca/destinations/united-states",
    # "https://travel.gc.ca/travelling/advisories",
    # "https://atlantic.caa.ca/travel/government-travel-advisories",
    "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories/brazil-travel-advisory.html",
    # "",
    # "",
    # "",
    # "",
    # "",
    # "",
]
# Create the the UI 
st.title("Langchain RAG OpenAI")
st.write("This is a Retrieval-Augmented Generation (RAG) with web search agent . The agent can answer questions using a vectorstore or web search.")
st.write("This work is based off of the paper: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks' Authors: Patrick Lewis, et al. Published: 2020. URL: https://arxiv.org/abs/2005.11401, 'SELF-RAG: Learning to Retrive, Generate, and Critique Through Self-Reflection' Authors: Akari Asai†, et al. Published: 2023. URL: https://arxiv.org/pdf/2310.11511 and on 'THE FAISS LIBRARY' Authors: Matthijs Douze, et al. Published 2025. URL: https://arxiv.org/pdf/2401.08281"  ) 
st.write('This work contributes to the #AIforSocalGood initiative time saving tool to research information about the restrictiong and and travel conditions.')
st.write("Helps travelers and attorneys create a travel plan based on safety, health, and environmental conditions and evaluate risk for the traveler. ")    

# Input box for the user to enter a description
the_question = st.text_area("Ask the AI", placeholder="Ask the AI about the country your client is considering travel?")

# Fetch and process documents from the URLs
vectorstore = process_and_store_documents(urls)


if st.button("Submit"):
    
    # User query input
    query = the_question
    
    # Get the response
    with st.spinner("Getting response..."):
        # Route the question to the appropriate data source
        question_route = route_question(query, vectorstore)

        # Use the appropriate data source to answer the question
        if question_route == "vectorstore":
            st.write("we have that answer stored from prior searches")
            
            # Create the QA Chain
            qa_chain = create_qa_chain(vectorstore)
            response = qa_chain.invoke(query)
        
        else:
            st.write("We need to search the web for your answer.")
            web_search_data = web_search_tool.run(query)
            #
            graded_web_search_data = grade_documents(web_search_data, query)
            #
            qa_chain = create_qa_chain(graded_web_search_data) 
            #
            response = qa_chain.invoke(query)

        # Display the answer
        st.write("Answer: ", response['result'])

        # Extract URLs from source documents
        source_urls = extract_urls_from_source_documents(response["source_documents"])

        # remove duplicates from the list
        source_urls = list(set(source_urls))

        # Display the source URLs
        st.write("Source URLs:")
        for url in source_urls:
            st.write("URL: ", url)

    # Hallucination Grader
        hallucination_review = check_for_hallucination(response['result'], vectorstore)
        st.write("Hallucination review: Acuracy ", hallucination_review)
else:
    st.write("Please ask the AI.")
