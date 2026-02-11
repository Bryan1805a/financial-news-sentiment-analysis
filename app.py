import streamlit as st
from transformers import pipeline
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Web config
st.set_page_config(page_title="Stock Sentiment & Price", layout="wide")
st.title("Market Sentiment & Stock Price Analysis")
st.markdown("Combine **AI FinBERT** (news reading) and **market data** (prices) to make decisions.")

# Download Model (Cache)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

with st.spinner('Starting AI FinBERT...'):
    sentiment_pipeline = load_sentiment_model()

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    ticker = st.text_input("Enter stock code:", value="NVDA").upper()
    period = st.selectbox("Timeframe:", ["1mo", "3mo", "6mo", "1y"], index=0)
    analyze_btn = st.button("Analyse now")

# Get price history
def get_stock_history(ticker_symbol, period="1mo"):
    try:
        stock = yf.Ticker(ticker_symbol)
        # Get price history (Date, Open, High, Low, Close, Volume)
        hist = stock.history(period=period)
        return hist, stock.info.get('longName', ticker_symbol)
    except Exception as e:
        return None, str(e)

# Data processing function
def get_news_sentiment(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        news_list = stock.news
        
        if not news_list: return None, "Cannot find any news"
            
        headlines = []
        links = []
        dates = []
        
        for item in news_list:
            if 'content' in item:
                content = item['content']
                title = content.get('title')
                if not title: continue
                headlines.append(title)
                
                # Lấy ngày đăng tin
                pub_date = content.get('pubDate')
                dates.append(pub_date)

                click_url = content.get('clickThroughUrl')
                if click_url and 'url' in click_url:
                    links.append(click_url['url'])
                else:
                    links.append('#')
            else:
                title = item.get('title')
                if not title: continue
                headlines.append(title)
                links.append(item.get('link', '#'))
                dates.append("N/A")
        
        if not headlines: return None, "Unable to extract the news title"

        # AI Analysis
        results = sentiment_pipeline(headlines)
        
        data = []
        for i, (title, res) in enumerate(zip(headlines, results)):
            data.append({
                "Date": dates[i],
                "Title": title,
                "Emotion": res['label'],
                "Reliability": res['score'],
                "Link": links[i]
            })
            
        return pd.DataFrame(data), None
        
    except Exception as e:
        return None, str(e)

# Main UI
if analyze_btn:
    st.divider()
    
    # Price data
    hist_df, company_name = get_stock_history(ticker, period)
    
    if hist_df is not None and not hist_df.empty:
        st.subheader(f"Price chart: {company_name}")
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=hist_df.index, 
            y=hist_df['Close'],
            mode='lines',
            name='Close price',
            fill='tozeroy',
            line=dict(color='#00CC96', width=2)
        ))
        
        fig_price.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Calculate changing %
        last_price = hist_df['Close'].iloc[-1]
        prev_price = hist_df['Close'].iloc[-2]
        change = last_price - prev_price
        pct_change = (change / prev_price) * 100
        
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Current price", f"${last_price:.2f}")
        col_p1.metric("Fluctuations", f"{change:.2f} ({pct_change:.2f}%)", 
                      delta_color="normal" if change > 0 else "inverse")

    # News data
    st.subheader(f"Analyze market emotion (AI FinBERT)")
    df_news, error = get_news_sentiment(ticker)
    
    if error:
        st.error(f"Error: {error}")
    elif df_news is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig_pie = px.pie(df_news, names='Emotion', 
                         color='Emotion',
                         color_discrete_map={'positive':'#00CC96', 'negative':'#EF553B', 'neutral':'#636EFA'},
                         hole=0.5, title="Emotional ratio")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Quick statistics
            pos_count = len(df_news[df_news['Emotion']=='positive'])
            neg_count = len(df_news[df_news['Emotion']=='negative'])
            
            if pos_count > neg_count:
                st.success("CONCLUSION: Market is positive.")
            elif neg_count > pos_count:
                st.warning("CONCLUSION: Market is negative")
            else:
                st.info("CONCLUSION: Market is neutral")

        with col2:
            # News list
            def color_sentiment(val):
                if val == 'positive': return 'background-color: #d4edda; color: black'
                elif val == 'negative': return 'background-color: #f8d7da; color: black'
                return ''

            st.dataframe(
                df_news[['Emotion', 'Title', 'Reliability']].style.applymap(color_sentiment, subset=['Emotion']),
                use_container_width=True,
                height=350,
                column_config={"Link": st.column_config.LinkColumn()}
            )