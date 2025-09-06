"""
Fake News Classifier - Streamlit Application
Main application file with UI components and functionality.
"""

import streamlit as st
import pandas as pd
from utils import (
    load_models, scrape_article, scrape_multiple_articles, classify_text,
    classify_multiple_texts, process_csv_upload, create_results_dataframe,
    separate_results, get_text_excerpt, validate_url
)


# ------------------- Page Configuration -------------------

st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------- Header -------------------

st.title("📰 Fake News Classifier — Streamlit")
st.markdown("Classify news articles as Fake or True using machine learning models")
st.divider()


# ------------------- Sidebar Configuration -------------------

st.sidebar.header("⚙️ Configuration")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Model:",
    ["Logistic Regression", "Random Forest", "Both (ensemble)"],
    help="Choose the machine learning model for classification"
)

# Map display names to internal names
model_mapping = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest", 
    "Both (ensemble)": "both"
}
selected_model = model_mapping[model_option]

# Probability threshold
threshold = st.sidebar.slider(
    "Probability Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum probability threshold for classification"
)

# Thread configuration
max_threads = st.sidebar.slider(
    "Max Concurrent Threads:",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of concurrent threads for processing multiple URLs"
)


# ------------------- Model Loading -------------------

@st.cache_resource
def get_models():
    """Load models with caching"""
    return load_models()

models = get_models()

if models is None:
    st.error("❌ Failed to load models. Please ensure the model files exist in the ./model/ directory.")
    st.stop()


# ------------------- Main Content -------------------

# Input method selection
st.subheader("📝 Input Method")
input_method = st.radio(
    "Choose how to input your content:",
    ["Single URL", "Multiple URLs", "CSV Upload", "Raw Text"],
    horizontal=True
)

st.divider()

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []


# ------------------- Input Processing -------------------

if input_method == "Single URL":
    st.subheader("🔗 Single URL Analysis")
    
    url = st.text_input(
        "Enter news article URL:",
        placeholder="https://example.com/news-article",
        help="Paste the URL of the news article you want to analyze"
    )
    
    if st.button("🔍 Analyze Article", type="primary"):
        if not url.strip():
            st.warning("Please enter a URL")
        elif not validate_url(url):
            st.error("Please enter a valid URL")
        else:
            with st.spinner("Scraping and analyzing article..."):
                # Scrape article
                article_data = scrape_article(url)
                
                if not article_data['text']:
                    st.error("❌ Failed to extract article content. Please check the URL or try another.")
                else:
                    # Classify
                    prediction, confidence = classify_text(article_data['text'], models, selected_model)
                    
                    # Display results
                    st.success(f"✅ Analysis Complete!")
                    
                    # Article info card
                    with st.container():
                        st.markdown("### 📄 Article Information")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Title:** {article_data['title']}")
                            st.markdown(f"**Source:** {article_data['source']}")
                            st.markdown(f"**Text Length:** {len(article_data['text'])} characters")
                        
                        with col2:
                            st.markdown(f"**Prediction:** {prediction}")
                            st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Text excerpt
                    st.markdown("### 📖 Article Excerpt")
                    excerpt = get_text_excerpt(article_data['text'], 500)
                    st.text_area("Content:", excerpt, height=200, disabled=True)


elif input_method == "Multiple URLs":
    st.subheader("🔗 Multiple URLs Analysis")
    
    urls_text = st.text_area(
        "Enter URLs (one per line):",
        placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://example.com/article3",
        height=150,
        help="Enter multiple URLs, one per line"
    )
    
    if st.button("🔍 Analyze All Articles", type="primary"):
        if not urls_text.strip():
            st.warning("Please enter at least one URL")
        else:
            # Parse URLs
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            valid_urls = [url for url in urls if validate_url(url)]
            
            if not valid_urls:
                st.error("No valid URLs found. Please check your input.")
            else:
                with st.spinner(f"Processing {len(valid_urls)} articles..."):
                    # Scrape articles
                    results = scrape_multiple_articles(valid_urls, max_threads)
                    
                    # Filter successful scrapes
                    successful_results = [r for r in results if r['text']]
                    failed_count = len(results) - len(successful_results)
                    
                    if successful_results:
                        # Classify articles
                        texts = [r['text'] for r in successful_results]
                        predictions = classify_multiple_texts(texts, models, selected_model)
                        
                        # Store results
                        st.session_state.results = successful_results
                        st.session_state.predictions = predictions
                        
                        st.success(f"✅ Processed {len(successful_results)} articles successfully!")
                        if failed_count > 0:
                            st.warning(f"⚠️ Failed to process {failed_count} articles")
                    else:
                        st.error("❌ Failed to process all articles. Please check the URLs.")


elif input_method == "CSV Upload":
    st.subheader("📊 CSV Upload Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file:",
        type=['csv'],
        help="CSV file should contain a column with URLs (e.g., 'url', 'URL', 'link')"
    )
    
    if uploaded_file is not None:
        if st.button("🔍 Analyze CSV Articles", type="primary"):
            with st.spinner("Processing CSV file..."):
                # Process CSV
                results = process_csv_upload(uploaded_file)
                
                if results:
                    # Classify articles
                    texts = [r['text'] for r in results]
                    predictions = classify_multiple_texts(texts, models, selected_model)
                    
                    # Store results
                    st.session_state.results = results
                    st.session_state.predictions = predictions
                    
                    st.success(f"✅ Processed {len(results)} articles from CSV!")
                else:
                    st.error("❌ No valid articles found in CSV file.")


elif input_method == "Raw Text":
    st.subheader("✍️ Raw Text Analysis")
    
    title = st.text_input("Article Title (optional):", placeholder="Enter article title")
    text_content = st.text_area(
        "Article Content:",
        placeholder="Paste the article content here...",
        height=200
    )
    
    if st.button("🔍 Analyze Text", type="primary"):
        if not text_content.strip():
            st.warning("Please enter some text content")
        else:
            with st.spinner("Analyzing text..."):
                # Classify text
                prediction, confidence = classify_text(text_content, models, selected_model)
                
                # Display results
                st.success(f"✅ Analysis Complete!")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Title:** {title or 'No title provided'}")
                    st.markdown(f"**Text Length:** {len(text_content)} characters")
                
                with col2:
                    st.markdown(f"**Prediction:** {prediction}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")


# ------------------- Results Display -------------------

if st.session_state.results and st.session_state.predictions:
    st.divider()
    st.subheader("📊 Classification Results")
    
    # Separate results into True and Fake
    true_list, fake_list = separate_results(st.session_state.results, st.session_state.predictions)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", len(st.session_state.results))
    with col2:
        st.metric("True Articles", len(true_list))
    with col3:
        st.metric("Fake Articles", len(fake_list))
    with col4:
        st.metric("Model Used", model_option)
    
    # Results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"✅ True Articles ({len(true_list)})")
        if true_list:
            for i, article in enumerate(true_list):
                with st.expander(f"{i+1}. {article['title'][:50]}..."):
                    st.markdown(f"**URL:** {article['url']}")
                    st.markdown(f"**Source:** {article['source']}")
                    st.markdown(f"**Confidence:** {article['confidence']:.2%}")
                    st.markdown(f"**Excerpt:** {get_text_excerpt(article['text'], 300)}")
        else:
            st.info("No articles classified as True")
    
    with col2:
        st.subheader(f"❌ Fake Articles ({len(fake_list)})")
        if fake_list:
            for i, article in enumerate(fake_list):
                with st.expander(f"{i+1}. {article['title'][:50]}..."):
                    st.markdown(f"**URL:** {article['url']}")
                    st.markdown(f"**Source:** {article['source']}")
                    st.markdown(f"**Confidence:** {article['confidence']:.2%}")
                    st.markdown(f"**Excerpt:** {get_text_excerpt(article['text'], 300)}")
        else:
            st.info("No articles classified as Fake")
    
    # Results DataFrame
    st.subheader("📋 Detailed Results")
    results_df = create_results_dataframe(st.session_state.results, st.session_state.predictions, model_option)
    st.dataframe(results_df, use_container_width=True)
    
    # CSV Download
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=f"fake_news_classification_results_{model_option.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )


# ------------------- Footer -------------------

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit, scikit-learn, and newspaper3k | For educational purposes only</p>
    </div>
    """,
    unsafe_allow_html=True
)