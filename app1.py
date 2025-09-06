"""
Fake News Classifier - Modern Streamlit Application
Enhanced version with modern UI/UX, animations, and visual appeal.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from utils import (
    load_models, scrape_article, scrape_multiple_articles, classify_text,
    classify_multiple_texts, process_csv_upload, create_results_dataframe,
    separate_results, get_text_excerpt, validate_url
)


# ------------------- Page Configuration -------------------

st.set_page_config(
    page_title="🔍 AI News Verifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# AI-Powered News Verification Tool\nBuilt with Streamlit and Machine Learning"
    }
)


# ------------------- Custom CSS Styling -------------------

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --warning-color: #ff7f0e;
        --info-color: #17a2b8;
        --light-color: #f8f9fa;
        --dark-color: #343a40;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --gradient-danger: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: var(--gradient-primary);
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status indicators */
    .status-true {
        background: var(--gradient-success);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 3px 10px rgba(44, 160, 44, 0.3);
    }
    
    .status-fake {
        background: var(--gradient-danger);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 3px 10px rgba(214, 39, 40, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(31, 119, 180, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: var(--gradient-primary);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-primary);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)


# ------------------- Header Section -------------------

st.markdown("""
<div class="main-header fade-in">
    <h1>🔍 AI News Verifier</h1>
    <p>Advanced Machine Learning-Powered News Authenticity Detection</p>
</div>
""", unsafe_allow_html=True)


# ------------------- Sidebar Configuration -------------------

st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")

# Model selection with icons
model_option = st.sidebar.selectbox(
    "🤖 AI Model Selection:",
    ["🧠 Logistic Regression", "🌲 Random Forest", "🎯 Ensemble (Both)"],
    help="Choose the machine learning model for classification"
)

# Map display names to internal names
model_mapping = {
    "🧠 Logistic Regression": "logistic_regression",
    "🌲 Random Forest": "random_forest", 
    "🎯 Ensemble (Both)": "both"
}
selected_model = model_mapping[model_option]

st.sidebar.markdown("---")

# Probability threshold with visual indicator
threshold = st.sidebar.slider(
    "🎯 Confidence Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum probability threshold for classification"
)

# Visual threshold indicator
threshold_color = "🟢" if threshold >= 0.7 else "🟡" if threshold >= 0.5 else "🔴"
st.sidebar.markdown(f"**Threshold Level:** {threshold_color} {'High' if threshold >= 0.7 else 'Medium' if threshold >= 0.5 else 'Low'}")

st.sidebar.markdown("---")

# Thread configuration
max_threads = st.sidebar.slider(
    "⚡ Concurrent Processing:",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of concurrent threads for processing multiple URLs"
)

st.sidebar.markdown("---")

# Performance metrics
st.sidebar.markdown("### 📊 Performance")
st.sidebar.markdown(f"**Processing Speed:** {'🚀 Fast' if max_threads >= 7 else '⚡ Medium' if max_threads >= 4 else '🐌 Slow'}")
st.sidebar.markdown(f"**Model Type:** {model_option.split()[1] if len(model_option.split()) > 1 else 'Ensemble'}")


# ------------------- Model Loading -------------------

@st.cache_resource
def get_models():
    """Load models with caching"""
    return load_models()

models = get_models()

if models is None:
    st.error("❌ **Model Loading Failed**\n\nPlease ensure the model files exist in the `./model/` directory:\n- `vectorizer.pkl`\n- `logistic_regression.pkl`\n- `random_forest_classifier.pkl`")
    st.stop()

# Success indicator
st.sidebar.markdown("### ✅ System Status")
st.sidebar.success("**Models Loaded Successfully**")


# ------------------- Main Content -------------------

# Input method selection with modern styling
st.markdown("## 📝 Input Method")
input_method = st.radio(
    "Choose your input method:",
    ["🔗 Single URL", "📋 Multiple URLs", "📊 CSV Upload", "✍️ Raw Text"],
    horizontal=True,
    help="Select how you want to provide content for analysis"
)

st.markdown("---")

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []


# ------------------- Input Processing -------------------

if input_method == "🔗 Single URL":
    st.markdown("### 🔗 Single Article Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input(
            "Enter news article URL:",
            placeholder="https://example.com/news-article",
            help="Paste the URL of the news article you want to analyze",
            key="single_url"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        analyze_btn = st.button("🔍 Analyze", type="primary", key="analyze_single")
    
    if analyze_btn:
        if not url.strip():
            st.warning("⚠️ Please enter a URL")
        elif not validate_url(url):
            st.error("❌ Please enter a valid URL")
        else:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Scraping phase
            status_text.text("🌐 Scraping article content...")
            progress_bar.progress(25)
            
            article_data = scrape_article(url)
            progress_bar.progress(50)
            
            if not article_data['text']:
                progress_bar.progress(100)
                st.error("❌ **Scraping Failed**\n\nUnable to extract article content. Please check the URL or try another.")
            else:
                # Analysis phase
                status_text.text("🤖 Analyzing with AI...")
                progress_bar.progress(75)
                
                prediction, confidence = classify_text(article_data['text'], models, selected_model)
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                # Results display
                st.markdown("### 📊 Analysis Results")
                
                # Main result card
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card fade-in">
                        <h3 style="margin: 0; color: #2c3e50;">{article_data['title']}</h3>
                        <p style="margin: 0.5rem 0; color: #7f8c8d;">Source: {article_data['source']}</p>
                        <p style="margin: 0; color: #95a5a6;">Length: {len(article_data['text']):,} characters</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    status_class = "status-true" if prediction == "True" else "status-fake"
                    st.markdown(f"""
                    <div class="metric-card fade-in">
                        <h2 class="metric-value" style="color: {'#2ca02c' if prediction == 'True' else '#d62728'};">
                            {prediction}
                        </h2>
                        <p class="metric-label">Prediction</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card fade-in">
                        <h2 class="metric-value" style="color: #1f77b4;">{confidence:.1%}</h2>
                        <p class="metric-label">Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                
                fig.update_layout(height=300, font={'family': 'Inter'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Article excerpt
                st.markdown("### 📖 Article Preview")
                excerpt = get_text_excerpt(article_data['text'], 500)
                st.markdown(f"""
                <div class="metric-card fade-in">
                    <p style="line-height: 1.6; color: #2c3e50;">{excerpt}</p>
                </div>
                """, unsafe_allow_html=True)


elif input_method == "📋 Multiple URLs":
    st.markdown("### 📋 Batch Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        urls_text = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://example.com/article3",
            height=150,
            help="Enter multiple URLs, one per line",
            key="multiple_urls"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        analyze_all_btn = st.button("🚀 Analyze All", type="primary", key="analyze_multiple")
    
    if analyze_all_btn:
        if not urls_text.strip():
            st.warning("⚠️ Please enter at least one URL")
        else:
            # Parse URLs
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            valid_urls = [url for url in urls if validate_url(url)]
            
            if not valid_urls:
                st.error("❌ No valid URLs found. Please check your input.")
            else:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"🔄 Processing {len(valid_urls)} articles...")
                
                # Scrape articles
                results = scrape_multiple_articles(valid_urls, max_threads)
                progress_bar.progress(50)
                
                # Filter successful scrapes
                successful_results = [r for r in results if r['text']]
                failed_count = len(results) - len(successful_results)
                
                if successful_results:
                    status_text.text("🤖 Running AI analysis...")
                    
                    # Classify articles
                    texts = [r['text'] for r in successful_results]
                    predictions = classify_multiple_texts(texts, models, selected_model)
                    progress_bar.progress(100)
                    
                    # Store results
                    st.session_state.results = successful_results
                    st.session_state.predictions = predictions
                    
                    status_text.text("✅ Batch analysis complete!")
                    
                    # Success message with stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"✅ **{len(successful_results)}** articles processed")
                    with col2:
                        if failed_count > 0:
                            st.warning(f"⚠️ **{failed_count}** failed")
                        else:
                            st.info("🎯 **100%** success rate")
                    with col3:
                        st.info(f"⚡ **{max_threads}** threads used")
                else:
                    progress_bar.progress(100)
                    st.error("❌ Failed to process all articles. Please check the URLs.")


elif input_method == "📊 CSV Upload":
    st.markdown("### 📊 CSV Batch Processing")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file:",
        type=['csv'],
        help="CSV file should contain a column with URLs (e.g., 'url', 'URL', 'link')",
        key="csv_upload"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_analyze_btn = st.button("📊 Process CSV", type="primary", key="analyze_csv")
        
        with col2:
            if st.button("📋 Preview CSV"):
                try:
                    df_preview = pd.read_csv(uploaded_file)
                    st.dataframe(df_preview.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
        
        if csv_analyze_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📊 Processing CSV file...")
            progress_bar.progress(25)
            
            # Process CSV
            results = process_csv_upload(uploaded_file)
            progress_bar.progress(50)
            
            if results:
                status_text.text("🤖 Running AI analysis...")
                
                # Classify articles
                texts = [r['text'] for r in results]
                predictions = classify_multiple_texts(texts, models, selected_model)
                progress_bar.progress(100)
                
                # Store results
                st.session_state.results = results
                st.session_state.predictions = predictions
                
                status_text.text("✅ CSV processing complete!")
                st.success(f"✅ **{len(results)}** articles processed from CSV!")
            else:
                progress_bar.progress(100)
                st.error("❌ No valid articles found in CSV file.")


elif input_method == "✍️ Raw Text":
    st.markdown("### ✍️ Direct Text Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        title = st.text_input("Article Title (optional):", placeholder="Enter article title", key="raw_title")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        text_analyze_btn = st.button("🔍 Analyze Text", type="primary", key="analyze_text")
    
    text_content = st.text_area(
        "Article Content:",
        placeholder="Paste the article content here...",
        height=200,
        key="raw_content"
    )
    
    if text_analyze_btn:
        if not text_content.strip():
            st.warning("⚠️ Please enter some text content")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🤖 Analyzing text with AI...")
            progress_bar.progress(50)
            
            # Classify text
            prediction, confidence = classify_text(text_content, models, selected_model)
            progress_bar.progress(100)
            
            status_text.text("✅ Analysis complete!")
            
            # Results display
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card fade-in">
                    <h3 style="margin: 0; color: #2c3e50;">{title or 'No title provided'}</h3>
                    <p style="margin: 0.5rem 0; color: #7f8c8d;">Text Length: {len(text_content):,} characters</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                status_class = "status-true" if prediction == "True" else "status-fake"
                st.markdown(f"""
                <div class="metric-card fade-in">
                    <h2 class="metric-value" style="color: {'#2ca02c' if prediction == 'True' else '#d62728'};">
                        {prediction}
                    </h2>
                    <p class="metric-label">Prediction</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card fade-in">
                    <h2 class="metric-value" style="color: #1f77b4;">{confidence:.1%}</h2>
                    <p class="metric-label">Confidence</p>
                </div>
                """, unsafe_allow_html=True)


# ------------------- Results Display -------------------

if st.session_state.results and st.session_state.predictions:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results Dashboard")
    
    # Separate results into True and Fake
    true_list, fake_list = separate_results(st.session_state.results, st.session_state.predictions)
    
    # Summary metrics with modern cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h2 class="metric-value" style="color: #1f77b4;">{len(st.session_state.results)}</h2>
            <p class="metric-label">Total Articles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h2 class="metric-value" style="color: #2ca02c;">{len(true_list)}</h2>
            <p class="metric-label">True Articles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h2 class="metric-value" style="color: #d62728;">{len(fake_list)}</h2>
            <p class="metric-label">Fake Articles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h2 class="metric-value" style="color: #ff7f0e;">{model_option.split()[1] if len(model_option.split()) > 1 else 'Ensemble'}</h2>
            <p class="metric-label">Model Used</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    if len(st.session_state.results) > 0:
        # Pie chart
        fig_pie = px.pie(
            values=[len(true_list), len(fake_list)],
            names=['True', 'Fake'],
            title="Classification Distribution",
            color_discrete_map={'True': '#2ca02c', 'Fake': '#d62728'}
        )
        fig_pie.update_layout(
            font={'family': 'Inter'},
            title_font_size=20,
            showlegend=True
        )
        
        # Bar chart
        fig_bar = px.bar(
            x=['True', 'Fake'],
            y=[len(true_list), len(fake_list)],
            title="Article Count by Classification",
            color=['True', 'Fake'],
            color_discrete_map={'True': '#2ca02c', 'Fake': '#d62728'}
        )
        fig_bar.update_layout(
            font={'family': 'Inter'},
            title_font_size=20,
            showlegend=False
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Results in two columns with modern styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### ✅ True Articles ({len(true_list)})")
        if true_list:
            for i, article in enumerate(true_list):
                with st.expander(f"📰 {i+1}. {article['title'][:60]}...", expanded=False):
                    st.markdown(f"""
                    <div class="metric-card slide-in">
                        <p><strong>🔗 URL:</strong> <a href="{article['url']}" target="_blank">{article['url']}</a></p>
                        <p><strong>🌐 Source:</strong> {article['source']}</p>
                        <p><strong>🎯 Confidence:</strong> <span class="status-true">{article['confidence']:.1%}</span></p>
                        <p><strong>📖 Excerpt:</strong></p>
                        <p style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #2ca02c;">
                            {get_text_excerpt(article['text'], 300)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("🎯 No articles classified as True")
    
    with col2:
        st.markdown(f"### ❌ Fake Articles ({len(fake_list)})")
        if fake_list:
            for i, article in enumerate(fake_list):
                with st.expander(f"📰 {i+1}. {article['title'][:60]}...", expanded=False):
                    st.markdown(f"""
                    <div class="metric-card slide-in">
                        <p><strong>🔗 URL:</strong> <a href="{article['url']}" target="_blank">{article['url']}</a></p>
                        <p><strong>🌐 Source:</strong> {article['source']}</p>
                        <p><strong>🎯 Confidence:</strong> <span class="status-fake">{article['confidence']:.1%}</span></p>
                        <p><strong>📖 Excerpt:</strong></p>
                        <p style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #d62728;">
                            {get_text_excerpt(article['text'], 300)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("🎯 No articles classified as Fake")
    
    # Results DataFrame with modern styling
    st.markdown("### 📋 Detailed Results Table")
    results_df = create_results_dataframe(st.session_state.results, st.session_state.predictions, model_option)
    
    # Style the dataframe
    styled_df = results_df.style.apply(
        lambda x: ['background-color: #e8f5e8' if x['Prediction'] == 'True' else 'background-color: #ffeaea' for _ in x], 
        axis=1
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # CSV Download with modern button
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=f"ai_news_verification_results_{model_option.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        help="Download the complete analysis results"
    )


# ------------------- Footer -------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #6c757d;'>
    <p style='font-size: 1.1rem; margin: 0; font-weight: 500;'>🔍 AI News Verifier</p>
    <p style='font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Powered by Advanced Machine Learning & Streamlit</p>
    <p style='font-size: 0.8rem; margin: 0.5rem 0 0 0; opacity: 0.7;'>Built for educational and research purposes</p>
</div>
""", unsafe_allow_html=True)
