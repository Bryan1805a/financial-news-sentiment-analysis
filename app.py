import streamlit as st
from transformers import pipeline
import yfinance as yf
import pandas as pd
import plotly.express as px

# Web config
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("Stock Market Sentiment Analysis (AI-Powered)")
st.markdown("This tool uses **FinBERT** to read financial news and forecast market sentiment.")

# Download model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

with st.spinner("Starting FinBERT..."):
    sentiment_pipeline = load_sentiment_model()

# Input sidebar
with st.sidebar:
    st.header("Enter stock code")
    ticker = st.text_input("Example: NVDA, AAPL, TSLA, GOOG", value="NVDA").upper()
    analyze_btn = st.button("Analyze now")

# Main processing function
def get_news_sentiment(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        news_list = stock.news
        
        if not news_list:
            return None, "Cannot found any news"
            
        headlines = []
        links = []
        
        for item in news_list:
            # Check key 'content'
            if 'content' in item:
                content = item['content']
                headlines.append(content.get('title', 'No Title'))
                
                # Get link
                link_data = content.get('clickThroughUrl')
                if link_data:
                    links.append(link_data.get('url', '#'))
                else:
                    links.append('#')
            else:
                # Fallback
                headlines.append(item.get('title', 'No Title'))
                links.append(item.get('link', '#'))
        
        if not headlines:
             return None, "Cannot get news title"

        # AI analyse
        results = sentiment_pipeline(headlines)
        
        # Get data
        data = []
        for i, (title, res) in enumerate(zip(headlines, results)):
            sentiment = res['label']
            score = res['score']
            
            data.append({
                "Title": title,
                "Emotion": sentiment,
                "Reliability": score,
                "Link": links[i]
            })
            
        return pd.DataFrame(data), None
        
    except Exception as e:
        return None, str(e)

# Main UI
if analyze_btn:
    st.divider()
    st.subheader(f"Analysis results for: {ticker}")

    df, error = get_news_sentiment(ticker)

    if error:
        st.error(f"Error: {error}")
    elif df is not None:
        col1, col2, col3 = st.columns(3)
        total_news = len(df)
        positive_news = len(df[df['Emotion'] == 'positive'])
        negative_news = len(df[df['Emotion'] == 'negative'])
        neutral_news = len(df[df['Emotion'] == 'neutral'])
    
    with col1:
        st.metric("Total number of news", total_news)
    with col2:
        sentiment_score = positive_news - negative_news
        if sentiment_score > 0:
            st.metric("Market score", f"+{sentiment_score}", delta="Positive")
        elif sentiment_score < 0:
            st.metric("Market score", f"{sentiment_score}", delta="Negative", delta_color='inverse')
        else:
            st.metric("Market score", "0", delta="Neutral", delta_color='off')
    
    # Visualization
    col_chart, col_list = st.columns([1, 2])

    with col_chart:
        st.markdown("### Emotion ratio")
        fig = px.pie(df, names='Emotion',
                     color='Emotion',
                     color_discrete_map={
                         'positive': '#2E8B57',
                         'negative': '#DC143C',
                         'neutral': '#808080'
                     },
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_list:
        st.markdown("### News detail")

        def colour_sentiment(val):
            colour = 'white'
            if val == 'positive': colour = '#90EE90'
            elif val == 'negative': colour = '#FFB6C1'
            return f'background-color: {colour}; color: black'
        
        st.dataframe(
            df[['Emotion', 'Title', 'Reliability']].style.applymap(colour_sentiment, subset=['Emotion']),
            use_container_width=True,
            height=400
        )