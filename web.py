import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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