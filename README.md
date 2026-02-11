# Financial News Sentiment Analyzer (AI-Powered)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![AI Model](https://img.shields.io/badge/Model-FinBERT-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A real-time stock market sentiment analysis tool leveraging **FinBERT** (a state-of-the-art Large Language Model for Finance) to interpret financial news and forecast market trends.

**Live Demo:** https://financial-news-sentiment-analysis-app.streamlit.app/

---

## Key Features
-   **Multi-Dimensional Analysis:** Combines **Price Action** (historical data) and **News Sentiment** on a single interactive dashboard.
-   **Advanced AI Core:** Powered by `ProsusAI/finbert`, a BERT model pre-trained on a massive financial corpus for superior accuracy.
-   **Real-Time Data:** Automatically fetches the latest news headlines and stock prices via Yahoo Finance API.
-   **Interactive Visualization:** Dynamic candlestick/line charts and sentiment distribution donut charts using Plotly & Streamlit.

## Tech Stack
-   **Language:** Python 3.10+
-   **Frontend:** Streamlit
-   **AI/ML Core:** Hugging Face Transformers, PyTorch
-   **Data Source:** yfinance API
-   **Visualization:** Plotly Express, Plotly Graph Objects
-   **Data Processing:** Pandas, NumPy

## Screenshots

| Main UI | NVIDIA Stock Detail |
|:---:|:---:|
| ![Main UI](/stuff/main_ui.png) | ![NVIDIA Stock Detail](/stuff/main_ui_2.jpg) |

## Installation & Local Setup
To run this project locally on your machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Bryan1805a/financial-news-sentiment-analysis.git
    cd financial-news-sentiment-analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

4.  **Access the App:**
    Open your browser and go to `http://localhost:8501`