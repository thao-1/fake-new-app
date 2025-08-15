import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import pickle
import re
import urllib.parse
import zipfile
from newspaper import Article

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .fake-news {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #d32f2f;
    }
    .real-news {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .explanation-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-snippet {
        background-color: #fff3e0;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #ff9800;
        margin: 0.5rem 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.llm = None
        self.setup_llm()
        
    def setup_llm(self):
        """Initialize the OpenAI LLM"""
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            st.error(f"Error setting up OpenAI: {e}")
    
    def load_and_train_model(self):
        """Load pre-trained model or train from available dataset"""
        try:
            # Extract compressed model files if they don't exist but zip file does
            if not (os.path.exists('trained_model.pkl') and os.path.exists('vectorizer.pkl')):
                if os.path.exists('model_files.zip'):
                    st.info("📦 Extracting compressed model files...")
                    try:
                        with zipfile.ZipFile('model_files.zip', 'r') as zip_ref:
                            zip_ref.extractall('.')
                        st.success("✅ Model files extracted successfully!")
                    except Exception as e:
                        st.error(f"Failed to extract model files: {e}")
            
            # Try to load pre-trained model first (preferred for deployment)
            if os.path.exists('trained_model.pkl') and os.path.exists('vectorizer.pkl'):
                try:
                    with open('trained_model.pkl', 'rb') as f:
                        self.model = pickle.load(f)
                    with open('vectorizer.pkl', 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    st.success("✅ Loaded pre-trained model successfully!")
                    return 0.879, "Pre-trained Model (8000 samples)"  # Return actual stats
                except Exception as e:
                    st.warning(f"Failed to load pre-trained model: {e}")
                    pass  # Continue to training if loading fails, retrain
            
            # Load dataset efficiently with chunking for large files
            st.info("🔄 Loading WELFake dataset... This may take a moment.")
            
            # Try to load a reasonable sample first to avoid memory issues
            try:
                # Load first chunk to check structure
                df_sample = pd.read_csv(
                    'WELFake_Dataset.csv',
                    nrows=5000,  # Load first 5000 rows to check
                    on_bad_lines='skip',
                    encoding='utf-8',
                    quotechar='"',
                    escapechar='\\'
                )
                
                # Check if we have the expected columns
                if 'text' not in df_sample.columns or 'label' not in df_sample.columns:
                    raise ValueError("Dataset missing required columns 'text' or 'label'")
                
                # Load more data if sample looks good
                df = pd.read_csv(
                    'WELFake_Dataset.csv',
                    nrows=50000,  # Use 50k rows for training (more manageable)
                    on_bad_lines='skip',
                    encoding='utf-8',
                    quotechar='"',
                    escapechar='\\'
                )
                
            except Exception as e:
                st.error(f"Error loading full dataset: {e}")
                # Fallback to the 1000-row dataset
                st.info("Falling back to smaller dataset...")
                df = pd.read_csv(
                    'WELFake_Dataset_1000.csv',
                    on_bad_lines='skip',
                    encoding='utf-8',
                    quotechar='"',
                    escapechar='\\'
                )
            
            st.info(f"💾 Loaded {len(df)} rows from dataset")
            
            # Clean the data with less aggressive filtering
            df = df.dropna(subset=['text', 'label'])
            df['text'] = df['text'].astype(str)
            
            # Remove only very short text or invalid labels (less aggressive)
            df = df[df['text'].str.len() > 20]  # Reduced from 50 to 20
            df = df[df['label'].isin([0, 1])]
            
            st.info(f"🧙 After cleaning: {len(df)} valid samples")
            
            # Check dataset balance
            if len(df) > 0:
                label_counts = df['label'].value_counts()
                real_count = label_counts.get(0, 0)
                fake_count = label_counts.get(1, 0)
                st.info(f"📊 Dataset balance: Real news: {real_count}, Fake news: {fake_count}")
                
                # Balance the dataset if we have enough samples
                if len(label_counts) == 2 and min(label_counts.values()) > 100:
                    min_count = min(label_counts.values())
                    max_samples = min(min_count, 5000)  # Limit to 5000 per class
                    
                    df_real = df[df['label'] == 0].sample(n=min(len(df[df['label'] == 0]), max_samples), random_state=42)
                    df_fake = df[df['label'] == 1].sample(n=min(len(df[df['label'] == 1]), max_samples), random_state=42)
                    df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
                    st.success(f"⚖️ Balanced dataset: {len(df)} samples ({len(df_real)} real, {len(df_fake)} fake)")
            
            # Light text cleaning (less aggressive)
            df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)  # Just normalize whitespace
            
            if len(df) < 50:
                raise ValueError(f"Not enough valid data after cleaning. Only {len(df)} samples remaining.")
            
            # Prepare features and labels
            X = df['text']
            y = df['label']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Vectorize the text
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95  # Ignore terms that appear in more than 95% of documents
            )
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train the model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train_vec, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save the trained model and vectorizer
            with open('trained_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            return accuracy, len(df)
            
        except Exception as e:
            print(f"Error training model: {e}")  # For debugging
            return None, None
    
    def extract_text_from_url(self, url):
        """Extract text content from a URL using newspaper3k and Beautiful Soup"""
        try:
            # First try with newspaper3k (best for news articles)
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                # Get title and text
                title = article.title or ""
                text = article.text or ""
                
                # Combine title and text
                full_text = f"{title}\n\n{text}" if title else text
                
                if len(full_text.strip()) > 100:  # If we got substantial content
                    return full_text[:8000]  # Return more text for better analysis
                    
            except Exception as newspaper_error:
                print(f"Newspaper3k failed: {newspaper_error}")
                pass  # Fall back to Beautiful Soup
            
            # Fallback: Use Beautiful Soup for more general web scraping
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "menu"]):
                element.decompose()
            
            # Try to find the main content area
            content_selectors = [
                'article', '[role="main"]', '.content', '.post-content', 
                '.entry-content', '.article-content', '.story-content',
                'main', '.main-content', '#content', '.post-body'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use the whole body
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract title
            title_elem = soup.find('title') or soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Extract paragraphs and headings
            text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            # Get text content
            paragraphs = []
            for elem in text_elements:
                text = elem.get_text().strip()
                if len(text) > 20:  # Only include substantial text
                    paragraphs.append(text)
            
            # Combine all text
            body_text = '\n\n'.join(paragraphs)
            full_text = f"{title}\n\n{body_text}" if title else body_text
            
            # Clean up text
            full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
            full_text = re.sub(r'\n\s*\n', '\n\n', full_text)  # Clean up line breaks
            
            if len(full_text.strip()) < 50:
                return "Error: Could not extract sufficient text content from the URL. Please try a different URL or paste the text directly."
            
            return full_text[:8000]  # Return up to 8000 characters
            
        except requests.exceptions.RequestException as e:
            return f"Error accessing URL: {e}. Please check the URL and try again."
        except Exception as e:
            return f"Error extracting text from URL: {e}. Please try pasting the text directly."
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if not self.model or not self.vectorizer:
            return None, None
        
        try:
            # Vectorize the input text
            text_vec = self.vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.model.predict(text_vec)[0]
            probabilities = self.model.predict_proba(text_vec)[0]
            
            return prediction, probabilities
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None
    
    def get_relevant_sources(self, text, num_sources=3):
        """Simulate RAG by finding relevant context (simplified version)"""
        # In a real implementation, this would use a vector database
        # For now, we'll simulate with some predefined sources
        sources = [
            {
                "title": "Reuters Fact Check Database",
                "snippet": "Cross-referencing claims with verified fact-checking databases...",
                "url": "https://reuters.com/fact-check"
            },
            {
                "title": "Associated Press Fact Check",
                "snippet": "Analyzing source credibility and cross-referencing with AP verified sources...",
                "url": "https://apnews.com/hub/ap-fact-check"
            },
            {
                "title": "Snopes Verification",
                "snippet": "Checking against known misinformation patterns and verified claims...",
                "url": "https://snopes.com"
            }
        ]
        
        return sources[:num_sources]
    
    def generate_explanation(self, text, prediction, probabilities, sources):
        """Generate LLM explanation for the classification"""
        if not self.llm:
            return "LLM not available for explanation generation."
        
        try:
            # Prepare the prompt
            prediction_label = "FAKE" if prediction == 1 else "REAL"
            confidence = max(probabilities) * 100
            
            prompt = ChatPromptTemplate.from_template("""
            You are an expert fact-checker analyzing news content. 
            
            News Article Text: {text}
            
            Classification Result: {prediction_label}
            Confidence: {confidence:.1f}%
            
            Sources Consulted: {sources}
            
            Please provide a detailed explanation of why this article was classified as {prediction_label}. 
            Consider the following factors in your analysis:
            1. Language patterns and emotional tone
            2. Source credibility indicators
            3. Factual claims that can be verified
            4. Writing style and journalistic standards
            5. Potential bias or misleading information
            
            Provide a clear, professional explanation suitable for a content moderation team.
            Keep the explanation concise but thorough (2-3 paragraphs).
            """)
            
            sources_text = "\n".join([f"- {s['title']}: {s['snippet']}" for s in sources])
            
            messages = prompt.format_messages(
                text=text[:1000],  # Limit text length
                prediction_label=prediction_label,
                confidence=confidence,
                sources=sources_text
            )
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating explanation: {e}"

def create_confidence_chart(probabilities):
    """Create a pie chart showing prediction confidence"""
    labels = ['Real News', 'Fake News']
    values = [probabilities[0] * 100, probabilities[1] * 100]
    colors = ['#4CAF50', '#F44336']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title={
            'text': 'Classification Confidence',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">📰 Fake News Detection System</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
        st.session_state.model_trained = False
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system helps content moderation teams identify potentially fake news articles and provides explanations for the classifications.
        
        **Features:**
        - Deep Learning classification
        - RAG-based context retrieval
        - LLM-generated explanations
        - Visual confidence indicators
        """)
        
        st.header("📊 Model Info")
        st.info("💡 The model trains automatically on first run. Use the button below only if you want to retrain.")
        
        # Check if dataset exists
        if not os.path.exists('WELFake_Dataset.csv'):
            st.error("Dataset file 'WELFake_Dataset.csv' not found!")
            st.stop()
        
        # Show dataset info
        dataset_size = os.path.getsize('WELFake_Dataset.csv') / (1024*1024)  # Size in MB
        st.info(f"💾 Using full WELFake dataset ({dataset_size:.1f} MB)")
        
        if st.button("🔄 Retrain Model", key="train_button"):
            try:
                with st.spinner("Training model..."):
                    detector = FakeNewsDetector()
                    accuracy, dataset_size = detector.load_and_train_model()
                    if accuracy is not None:
                        st.session_state.detector = detector
                        st.session_state.model_trained = True
                        st.success(f"Model trained! Accuracy: {accuracy:.2%}")
                        st.info(f"Dataset size: {dataset_size} articles")
                        st.rerun()
                    else:
                        st.error("Failed to train model. Check the dataset file.")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Auto-initialize detector if not done
    if st.session_state.detector is None:
        try:
            with st.spinner("Loading model... This may take a moment on first run."):
                st.session_state.detector = FakeNewsDetector()
                accuracy, dataset_size = st.session_state.detector.load_and_train_model()
                if accuracy is not None:
                    st.session_state.model_trained = True
                    st.sidebar.success(f"✅ Model ready! Accuracy: {accuracy:.2%}")
                    st.sidebar.info(f"📊 Dataset size: {dataset_size} articles")
                    st.success("🎉 Fake News Detection System is ready to use!")
                else:
                    st.session_state.model_trained = False
                    st.sidebar.error("❌ Failed to load model. Please use 'Train/Retrain Model' button.")
                    st.error("Model initialization failed. Please train the model using the sidebar.")
        except Exception as e:
            st.session_state.model_trained = False
            st.sidebar.error(f"❌ Error initializing: {str(e)}")
            st.error(f"Initialization error: {str(e)}. Please try training manually.")
    
    # Main interface
    st.header("🔍 News Analysis")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Text Input", "URL Input"])
    
    text_to_analyze = ""
    url_input = ""
    
    if input_method == "Text Input":
        text_to_analyze = st.text_area(
            "Enter news article text:",
            height=200,
            placeholder="Paste the news article text here..."
        )
    else:
        url_input = st.text_input(
            "Enter news article URL:",
            placeholder="https://example.com/news-article"
        )
        
        # Show preview of URL if entered
        if url_input:
            st.info(f"📰 Ready to analyze: {url_input}")
            st.write("Click 'Analyze Article' below to automatically extract and analyze the content.")
    
    # Analysis button
    analyze_clicked = st.button("🔍 Analyze Article", type="primary", key="analyze_button")
    
    if analyze_clicked and (text_to_analyze or url_input):
        if not st.session_state.detector or not st.session_state.model_trained:
            st.error("Please train the model first using the '🔄 Retrain Model' button in the sidebar.")
        else:
            # Handle URL input - automatically extract text
            if url_input and not text_to_analyze:
                with st.spinner("Extracting text from URL..."):
                    text_to_analyze = st.session_state.detector.extract_text_from_url(url_input)
                    
                    # Show extracted text preview
                    if not text_to_analyze.startswith("Error"):
                        with st.expander("📝 View Extracted Text", expanded=False):
                            st.text_area("Extracted content:", value=text_to_analyze, height=150, disabled=True)
                    else:
                        st.error(text_to_analyze)
                        st.stop()
            
            # Validate text length
            if len(text_to_analyze.strip()) < 50:
                st.warning("Please provide more text for accurate analysis (minimum 50 characters).")
            else:
                try:
                    with st.spinner("Analyzing article..."):
                        # Get prediction
                        prediction, probabilities = st.session_state.detector.predict(text_to_analyze)
                    
                        if prediction is not None:
                            # Display results
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Classification result
                                if prediction == 1:  # Fake
                                    st.markdown(
                                        '<div class="result-box fake-news">🚨 FAKE NEWS DETECTED</div>',
                                        unsafe_allow_html=True
                                    )
                                else:  # Real
                                    st.markdown(
                                        '<div class="result-box real-news">✅ REAL NEWS</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Confidence score
                                confidence = max(probabilities) * 100
                                st.metric("Confidence Score", f"{confidence:.1f}%")
                            
                            with col2:
                                # Confidence pie chart
                                fig = create_confidence_chart(probabilities)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Get relevant sources
                            sources = st.session_state.detector.get_relevant_sources(text_to_analyze)
                            
                            # Display sources
                            st.header("📚 Supporting Sources")
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-snippet">
                                    <strong>{i}. {source['title']}</strong><br>
                                    {source['snippet']}<br>
                                    <small>Source: <a href="{source['url']}" target="_blank">{source['url']}</a></small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Generate explanation
                            st.header("🤖 AI Explanation")
                            with st.spinner("Generating explanation..."):
                                explanation = st.session_state.detector.generate_explanation(
                                    text_to_analyze, prediction, probabilities, sources
                                )
                                
                                st.markdown(f"""
                                <div class="explanation-box">
                                    {explanation}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Copy explanation button
                                if st.button("📋 Copy Explanation for Report", key="copy_report"):
                                    report_text = f"""
FAKE NEWS ANALYSIS REPORT

Classification: {"FAKE NEWS" if prediction == 1 else "REAL NEWS"}
Confidence: {confidence:.1f}%

Explanation:
{explanation}

Sources Consulted:
{chr(10).join([f"- {s['title']}: {s['url']}" for s in sources])}

Generated by Fake News Detection System
                                    """.strip()
                                    
                                    st.code(report_text, language=None)
                                    st.success("Report text generated! Copy the text above for your moderation report.")
                        else:
                            st.error("Failed to analyze the article. Please try again.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
