from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from IPython.display import display, Markdown

st.title("AI Chatbot ")

model_gpt = "gpt 4"
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model=model_gpt, 
    temperature=0.4, # Increase creativity
    max_tokens=2000, # Allow for longer responses
    frequency_penalty=0.5, # Reduce repetition
    presence_penalty=0.6, # Encourage new topics 
    api_key=api_key
    )
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=(
         "You are a professional financial advisor. Your job is to provide clear, actionable advice "
            "based on the client's financial situation and goals. Use available tools and data sources "
            "to ensure accurate answers.\n\n"
            "Client Question: {client_question}\n\n"
            "Agent's Scratchpad:\n{agent_scratchpad}"
        )
    )
folder_dir="C:/Users/igara/prediction_proyect_st/books"
all_documents=[]
    # docs = loader.load()
for file in os.listdir(folder_dir):
    print(file)
    file_path=os.path.join(folder_dir,file)
    print(file_path)
    loader=PyPDFLoader(file_path)
    docs = loader.load()
            # Split the document into chunks
    documents = RecursiveCharacterTextSplitter(
    chunk_size=10000, 
    chunk_overlap=0
    ).split_documents(docs)
    all_documents.extend(documents)

vector = FAISS.from_documents(all_documents, OpenAIEmbeddings(api_key=api_key))
retriever = vector.as_retriever()

search = DuckDuckGoSearchRun()

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for relevant information from the documents. Use this tool for any document-related questions of finance.",
    )
tools = [search, retriever_tool]

agent = create_tool_calling_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt)

def print_response(query, response):
    display(Markdown(f"""
    <div style="border: 2px solid #FF6347; padding: 10px; border-radius: 5px;">
            <h3 style="color: #FF6347;">Query:</h3>
            <p>{query}</p>
            <h3 style="color: #FF6347;">Response:</h3>
            <p>{response}</p>
        </div>
        """))
        
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
user_input = st.text_input("Type your prompt here:")

if st.button("Send"):
    if user_input.strip():  # Check if input is not empty
                
        response = agent_executor.invoke({"client_question": user_input})
            
                
        st.subheader("Your Query:")
        st.write(user_input)
        st.subheader("Chatbot's Response:")
        st.write(response)

st.subheader("Books Used to Train the Chatbot")
st.write("This chatbot has been trained using content from the following books:")
        
books = {
    "A Random Walk Down Wall Street": 
        "A comprehensive guide to investing that explains market theories, investment strategies, and the long-term benefits of diversified portfolios.",
    "Common Stocks and Uncommon Profits": 
        "A classic investing book that introduces the concept of 'scuttlebutt' and provides advice on evaluating growth companies.",
    "Economics in One Lesson": 
         "A concise and clear introduction to economic principles, focusing on the unseen consequences of policy decisions.",
    "Personal Financial Planning Guide": 
        "A guide to managing personal finances, covering budgeting, saving, investing, and planning for the future.",
    "Principles for Navigating Big Debt Crises": 
        "An analysis of past financial crises, offering insights into how economies recover and strategies to handle such situations.",
    "The Millionaire Next Door": 
        "A study of the habits and behaviors of millionaires, focusing on frugality, hard work, and smart financial decisions."
    }

for title, description in books.items():
    st.markdown(f"**{title}**: {description}")