import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
import yfinance as yf
import os

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

