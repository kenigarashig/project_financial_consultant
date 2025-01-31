import streamlit as st
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
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
import yfinance as yf
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Select a Page:", ["Index information", "Model Predictions", "financial Chatbot"])

# Route to different pages based on the selection
if page == "Index information":
    st.title("Index Information")
    def load_data():
        df=pd.read_csv("global_screener.csv")
        return df
    df_screener=load_data()
    def rank_stocks (df,column,ascending=False ,top_n=10):
        df["Rank"]=df[column].rank(ascending=ascending,method="min")
        ranked_df=df.sort_values("Rank").head(top_n)
        return ranked_df[["Rank","Ticker","Company Name",column,"Market","Industry","Sector"]]
    st.title("Global Stock Displayer")
    st.sidebar.header("Options")
    markets=df_screener["Market"].unique()
    selected_markets=st.sidebar.multiselect("Select markets:",
    markets,default=markets)
    filtered_df=df_screener[df_screener["Market"].isin(selected_markets)]

    ranking_options=["Market Cap USD","P/E Ratio","Dividend Yield"]
    selected_ranking=st.sidebar.selectbox("Select metric to rank ",ranking_options)

    sort_order =st.sidebar.radio("Sort order" , ["Descending", "Ascending"] )
    ascending=sort_order=="Ascending"

    top_n=st.sidebar.slider("Number of stocks to display",
    min_value=3,max_value=df_screener.shape[0],value=10,step=1
    )
    st.subheader(f"top{top_n} stocks by {selected_ranking} Acrross selected markets")
    ranked_stocks=rank_stocks(filtered_df,column=selected_ranking,ascending=ascending,top_n=top_n)
    st.dataframe(ranked_stocks)
    st.subheader("Industry distribution" )

    dist=ranked_stocks["Industry"].value_counts()
    st.bar_chart(dist)  

    dist_sect=ranked_stocks["Sector"].value_counts()
    st.bar_chart(dist_sect)  

    mark_distr = ranked_stocks["Market"].value_counts()

    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(mark_distr.values, labels=mark_distr.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Market Distribution")

    # Display the chart in Streamlit
    st.pyplot(fig)

elif page == "Model Predictions":
    st.title("Model Predictions")
    def input_transformation(ticker_symbol):
        columns_used = joblib.load("columns_used.pkl")
        st.write(columns_used)
        balance_sheets_input=[]
        income_statement_input=[]
        cash_flow_input=[]
        empresa=yf.Ticker(ticker_symbol)
    
        # Process balance sheet
        bs=empresa.balance_sheet.transpose().reset_index()
        bs=bs.rename(columns={"index":"Date"})
        bs["Company"]=ticker_symbol
        balance_sheets_input.append(bs)

        # Process income statement
        ist=empresa.financials.transpose().reset_index()
        ist=ist.rename(columns={"index":"Date"})
        ist["Company"]=ticker_symbol
        income_statement_input.append(ist)

        # Process cash flow
        cf=empresa.cash_flow.transpose().reset_index()
        cf=cf.rename(columns={"index":"Date"})
        cf["Company"]=ticker_symbol
        cash_flow_input.append(cf)


        df_balance_sheets=pd.concat(balance_sheets_input,axis=0,ignore_index=True)
        df_cash_flow=pd.concat(income_statement_input,axis=0,ignore_index=True)
        df_income_statement=pd.concat(cash_flow_input,axis=0,ignore_index=True)
        
        # Merge data on the 'Company' column
        df_all_input=pd.merge(df_balance_sheets,df_income_statement,on="Company",how="inner")
        df_all_input=pd.merge(df_all_input,df_cash_flow,on="Company",how="inner")
        # df_all_input.drop(columns=["Date1","Date2","Date3"], axis=1, inplace=True)
        threshold = 0.95
        missing_percentage=df_all_input.isna().mean() 
        columns_to_drop = missing_percentage[missing_percentage > threshold].index
        df_all_input = df_all_input.drop(columns=columns_to_drop)
        df_all_input=df_all_input.fillna(-1)    
        missing_columns = set(columns_used) - set(df_all_input.columns)
        for col in missing_columns:
            df_all_input[col] = -1
        
        df_all_input=df_all_input.select_dtypes(include="number")
        df_all_input = df_all_input[[col for col in df_all_input.columns if col in columns_used]]
        df_all_input = df_all_input[columns_used]
            
        return df_all_input
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)

        user_input = st.text_input("Enter the ticker you want to explore:")
        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("Please enter some text before predicting.")
            else:
                to_predict=input_transformation(user_input)
                prediction=model.predict(to_predict)
                
                total_predictions = len(prediction)
                good_count = np.sum(prediction)  
                weak_count = total_predictions - good_count 
                good_percentage = (good_count / total_predictions) * 100
                weak_percentage = 100 - good_percentage
                st.subheader(" Analysis Result:")
                st.write(f" **{good_percentage:.1f}%** of the financial data indicates **strong** Data.")
                st.write(f" **{weak_percentage:.1f}%** of the financial data shows **potential Weak data**.")



elif page == "financial Chatbot":
    st.title("financial Chatbot")


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