# Fake News Detection System

A comprehensive Streamlit application for detecting fake news with AI-powered explanations, designed for content moderation teams.

## Features

- **Deep Learning Classification**: Uses TF-IDF vectorization with Logistic Regression for news classification
- **RAG Integration**: Retrieval-Augmented Generation for fetching relevant context from verified sources
- **LLM Explanations**: OpenAI GPT-3.5-turbo generates detailed explanations for classifications
- **Visual Analytics**: Interactive pie charts showing classification confidence
- **URL Support**: Extract and analyze text directly from news article URLs
- **Report Generation**: Copy-ready explanations for internal moderation reports

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**: 
   Your `.env` file should contain:
   ```
   OPENAI_API_KEY=your_openai_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

3. **Dataset**: 
   The app uses `WELFake_Dataset_1000.csv` (first 1000 lines of your dataset) for training.

## Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## How to Use

1. **Train Model**: Click "Train/Retrain Model" in the sidebar to train the classification model
2. **Input Method**: Choose between text input or URL input
3. **Analyze**: Enter your news article text or URL and click "Analyze Article"
4. **Review Results**: 
   - View classification (Fake/Real) with confidence score
   - See pie chart visualization of prediction confidence
   - Review supporting sources from fact-checking databases
   - Read AI-generated explanation
5. **Generate Report**: Click "Copy Explanation for Report" to get formatted text for moderation reports

## Architecture

- **Frontend**: Streamlit with custom CSS styling
- **ML Model**: Scikit-learn with TF-IDF vectorization
- **LLM Integration**: LangChain + OpenAI GPT-3.5-turbo
- **Visualization**: Plotly for interactive charts
- **Web Scraping**: BeautifulSoup for URL text extraction

## Model Performance

The model is trained on the WELFake dataset and achieves good accuracy for binary classification of fake vs. real news articles.

## Security Notes

- API keys are loaded from environment variables
- URL extraction includes proper headers and timeout handling
- Text input is limited to prevent excessive API usage
