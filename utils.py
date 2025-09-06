"""
Utility functions for the Fake News Classifier Streamlit application.
Handles model loading, preprocessing, web scraping, and classification.
"""

import re
import pickle
import joblib
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Try to import newspaper3k, fallback to None if not available
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    Article = None
    NEWSPAPER_AVAILABLE = False


# ------------------- Model Loading -------------------

@st.cache_resource
def load_models():
    """
    Load all models and vectorizer from the ./model/ directory.
    Uses Streamlit caching to avoid reloading models on every interaction.
    
    Returns:
        dict: Dictionary containing vectorizer, logistic_regression, and random_forest models
    """
    try:
        models = {}
        
        # Load TF-IDF vectorizer
        with open('./vectorizer.pkl', 'rb') as f:
            models['vectorizer'] = pickle.load(f)
        
        # Load Logistic Regression model
        with open('./logistic_regression.pkl', 'rb') as f:
            models['logistic_regression'] = pickle.load(f)
        
        # Load Random Forest model
        # with open('./model/random_forest_classifier.pkl', 'rb') as f:
        #     models['random_forest'] = pickle.load(f)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


# ------------------- Text Preprocessing -------------------

def clean_text(text: str) -> str:
    """
    Clean and preprocess text to match training preprocessing.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove content within brackets [...]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove newlines and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove tokens with digits
    words = text.split()
    words = [word for word in words if not any(char.isdigit() for char in word)]
    
    # Join and trim whitespace
    text = ' '.join(words).strip()
    
    return text


# ------------------- Web Scraping -------------------

def scrape_article(url: str) -> Dict[str, str]:
    """
    Scrape article content from a URL using newspaper3k with BeautifulSoup fallback.
    
    Args:
        url (str): URL to scrape
        
    Returns:
        dict: Dictionary with url, title, text, and source keys
    """
    result = {
        "url": url,
        "title": "No Title Found",
        "text": "",
        "source": urlparse(url).netloc
    }
    
    # Try newspaper3k first if available
    if NEWSPAPER_AVAILABLE:
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and article.text.strip():
                result["title"] = article.title or "No Title Found"
                result["text"] = article.text.strip()
                return result
            else:
                raise Exception("Empty article text from newspaper3k")
                
        except Exception as e:
            # Continue to BeautifulSoup fallback
            pass
    
    # Fallback to requests + BeautifulSoup
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find title
        title_tag = soup.find('title')
        if title_tag:
            result["title"] = title_tag.get_text().strip()
        
        # Try to find main content
        # First try <article> tags
        article_tags = soup.find_all('article')
        if article_tags:
            text_parts = []
            for article_tag in article_tags:
                paragraphs = article_tag.find_all('p')
                text_parts.extend([p.get_text().strip() for p in paragraphs])
            result["text"] = ' '.join(text_parts)
        else:
            # Fallback to all <p> tags
            paragraphs = soup.find_all('p')
            text_parts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            result["text"] = ' '.join(text_parts)
        
        # Clean up the text
        result["text"] = result["text"].strip()
        
        if not result["text"]:
            raise Exception("No text content found")
            
        return result
        
    except Exception as e2:
        # If all methods fail, return empty result
        result["text"] = ""
        return result


def scrape_multiple_articles(urls: List[str], max_workers: int = 5) -> List[Dict[str, str]]:
    """
    Scrape multiple articles concurrently using ThreadPoolExecutor.
    
    Args:
        urls (List[str]): List of URLs to scrape
        max_workers (int): Maximum number of concurrent threads
        
    Returns:
        List[Dict[str, str]]: List of scraped article dictionaries
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(scrape_article, url): url for url in urls}
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Add failed result
                results.append({
                    "url": url,
                    "title": "Scraping Failed",
                    "text": "",
                    "source": urlparse(url).netloc
                })
    
    return results


# ------------------- Classification -------------------

def classify_text(text: str, models: Dict, model_type: str = "logistic_regression") -> Tuple[str, float]:
    """
    Classify text using the specified model.
    
    Args:
        text (str): Text to classify
        models (Dict): Dictionary containing loaded models
        model_type (str): Type of model to use ("logistic_regression", "random_forest", or "both")
        
    Returns:
        Tuple[str, float]: Prediction ("True" or "Fake") and probability
    """
    if not text.strip():
        return "Fake", 0.0
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return "Fake", 0.0
    
    try:
        # Transform text using TF-IDF vectorizer
        vectorizer = models['vectorizer']
        text_vector = vectorizer.transform([cleaned_text])
        
        if model_type == "logistic_regression":
            model = models['logistic_regression']
            probabilities = model.predict_proba(text_vector)[0]
            prediction = model.predict(text_vector)[0]
            
        elif model_type == "random_forest":
            model = models['random_forest']
            probabilities = model.predict_proba(text_vector)[0]
            prediction = model.predict(text_vector)[0]
            
        elif model_type == "both":
            # Ensemble prediction - average probabilities
            lr_model = models['logistic_regression']
            rf_model = models['random_forest']
            
            lr_proba = lr_model.predict_proba(text_vector)[0]
            rf_proba = rf_model.predict_proba(text_vector)[0]
            
            # Average the probabilities
            probabilities = (lr_proba + rf_proba) / 2
            prediction = 1 if probabilities[1] >= 0.5 else 0
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Convert prediction to label
        # Assuming 1 = True, 0 = Fake
        label = "True" if prediction == 1 else "Fake"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        return label, confidence
        
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return "Fake", 0.0


def classify_multiple_texts(texts: List[str], models: Dict, model_type: str = "logistic_regression") -> List[Tuple[str, float]]:
    """
    Classify multiple texts concurrently.
    
    Args:
        texts (List[str]): List of texts to classify
        models (Dict): Dictionary containing loaded models
        model_type (str): Type of model to use
        
    Returns:
        List[Tuple[str, float]]: List of (prediction, confidence) tuples
    """
    results = []
    for text in texts:
        result = classify_text(text, models, model_type)
        results.append(result)
    return results


# ------------------- Data Processing -------------------

def process_csv_upload(uploaded_file) -> List[Dict[str, str]]:
    """
    Process uploaded CSV file to extract URLs and text.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with url, title, text, source
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Look for URL column (case insensitive)
        url_column = None
        for col in df.columns:
            if 'url' in col.lower():
                url_column = col
                break
        
        if url_column is None:
            st.error("No URL column found in CSV. Please ensure your CSV has a column containing URLs.")
            return []
        
        # Extract URLs
        urls = df[url_column].dropna().tolist()
        
        # Convert to strings and filter valid URLs
        valid_urls = []
        for url in urls:
            url_str = str(url).strip()
            if url_str.startswith(('http://', 'https://')):
                valid_urls.append(url_str)
        
        if not valid_urls:
            st.error("No valid URLs found in the CSV file.")
            return []
        
        return scrape_multiple_articles(valid_urls)
        
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return []


def create_results_dataframe(results: List[Dict], predictions: List[Tuple[str, float]], model_type: str) -> pd.DataFrame:
    """
    Create a DataFrame with classification results.
    
    Args:
        results (List[Dict]): List of scraped article dictionaries
        predictions (List[Tuple[str, float]]): List of (prediction, confidence) tuples
        model_type (str): Type of model used
        
    Returns:
        pd.DataFrame: DataFrame with results
    """
    data = []
    for i, (result, (prediction, confidence)) in enumerate(zip(results, predictions)):
        data.append({
            'Index': i + 1,
            'URL': result['url'],
            'Title': result['title'],
            'Source': result['source'],
            'Text_Length': len(result['text']),
            'Prediction': prediction,
            'Confidence': f"{confidence:.2%}",
            'Model': model_type
        })
    
    return pd.DataFrame(data)


def separate_results(results: List[Dict], predictions: List[Tuple[str, float]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Separate results into True and Fake lists.
    
    Args:
        results (List[Dict]): List of scraped article dictionaries
        predictions (List[Tuple[str, float]]): List of (prediction, confidence) tuples
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (true_list, fake_list)
    """
    true_list = []
    fake_list = []
    
    for result, (prediction, confidence) in zip(results, predictions):
        result_with_prediction = {
            **result,
            'prediction': prediction,
            'confidence': confidence
        }
        
        if prediction == "True":
            true_list.append(result_with_prediction)
        else:
            fake_list.append(result_with_prediction)
    
    return true_list, fake_list


# ------------------- Utility Functions -------------------

def get_text_excerpt(text: str, max_length: int = 200) -> str:
    """
    Get a text excerpt for display purposes.
    
    Args:
        text (str): Full text
        max_length (int): Maximum length of excerpt
        
    Returns:
        str: Text excerpt
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
