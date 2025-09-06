# Fake News Classifier - Streamlit Application

A comprehensive Streamlit application that classifies news articles as Fake or True using machine learning models and TF-IDF vectorization.

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Required model files in `./model/` directory:
  - `vectorizer.pkl`
  - `logistic_regression.pkl`
  - `random_forest_classifier.pkl`

### Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Features

### Multiple Input Methods
- **Single URL**: Analyze one news article by URL
- **Multiple URLs**: Process multiple articles simultaneously
- **CSV Upload**: Upload a CSV file with URLs for batch processing
- **Raw Text**: Direct text input for analysis

### Machine Learning Models
- **Logistic Regression**: Fast and interpretable classification
- **Random Forest**: Robust ensemble method
- **Ensemble (Both)**: Combines both models for improved accuracy

### Advanced Features
- **Concurrent Processing**: Multi-threaded scraping and classification
- **Web Scraping**: Automatic article extraction using newspaper3k and BeautifulSoup
- **Results Export**: Download classification results as CSV
- **Interactive UI**: Clean, responsive interface with expandable results

## 🛠️ Usage Instructions

### 1. Single URL Analysis
1. Select "Single URL" input method
2. Paste the news article URL
3. Click "Analyze Article"
4. View prediction, confidence, and article excerpt

### 2. Multiple URLs Analysis
1. Select "Multiple URLs" input method
2. Enter URLs (one per line) in the text area
3. Adjust thread count in sidebar if needed
4. Click "Analyze All Articles"
5. Review results in True/Fake columns

### 3. CSV Upload
1. Select "CSV Upload" input method
2. Upload a CSV file with a URL column
3. Click "Analyze CSV Articles"
4. Download results as CSV

### 4. Raw Text Analysis
1. Select "Raw Text" input method
2. Enter article title (optional) and content
3. Click "Analyze Text"
4. View classification results

## ⚙️ Configuration Options

### Sidebar Settings
- **Model Selection**: Choose between Logistic Regression, Random Forest, or Both
- **Probability Threshold**: Adjust classification threshold (default: 0.5)
- **Max Concurrent Threads**: Control parallel processing (default: 5)

### Classification Rules
- If probability for class 1 ≥ threshold → "True"
- Otherwise → "Fake"
- Confidence score shows the probability of the predicted class

## 📊 Output Format

### Results Display
- **Summary Metrics**: Total articles, True count, Fake count
- **Two-Column Layout**: Separate True and Fake articles
- **Expandable Cards**: Detailed view of each article
- **Results Table**: Comprehensive DataFrame with all details
- **CSV Download**: Export results for further analysis

### Article Information
Each result includes:
- URL
- Title
- Source domain
- Text length
- Prediction (True/Fake)
- Confidence percentage
- Text excerpt

## 🔧 Technical Details

### Preprocessing Pipeline
The application uses the same preprocessing as the training data:
- Convert to lowercase
- Remove brackets [...]
- Remove URLs and HTML tags
- Remove punctuation and newlines
- Remove tokens with digits
- Trim whitespace

### Web Scraping Strategy
1. **Primary**: newspaper3k library for reliable article extraction
2. **Fallback**: requests + BeautifulSoup for difficult pages
3. **Error Handling**: Graceful failure with informative messages
4. **Timeout Protection**: 10-second timeout for requests

### Model Architecture
- **TF-IDF Vectorizer**: Text feature extraction
- **Logistic Regression**: Linear classification model
- **Random Forest**: Ensemble tree-based model
- **Ensemble Method**: Average probability from both models

## 🚨 Error Handling

The application handles various error scenarios:
- Invalid URLs
- Scraping failures
- Empty articles
- Non-HTML pages
- Model loading errors
- Network timeouts

Failed articles are reported but don't crash the application.

## 📁 Project Structure

```
roshan/
├── app.py                 # Main Streamlit application
├── utils.py              # Helper functions and utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── model/               # Model files directory
    ├── vectorizer.pkl
    ├── logistic_regression.pkl
    └── random_forest_classifier.pkl
```

## 🔍 Troubleshooting

### Common Issues

1. **"Failed to load models" error**
   - Ensure model files exist in `./model/` directory
   - Check file permissions

2. **"Failed to extract article content"**
   - URL might be invalid or inaccessible
   - Try a different news article URL
   - Check internet connection

3. **Slow performance**
   - Reduce max concurrent threads
   - Check network connection
   - Some websites may be slow to respond

4. **CSV upload issues**
   - Ensure CSV has a URL column
   - URLs must start with http:// or https://
   - Check CSV format and encoding

### Performance Tips
- Use fewer threads for slower connections
- Process articles in smaller batches
- Ensure stable internet connection

## 📝 Notes

- This application is for educational and research purposes
- Classification accuracy depends on the quality of training data
- Results should be used as a starting point for further verification
- The application respects website terms of service and implements reasonable delays

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## 📄 License

This project is for educational purposes. Please respect the terms of service of websites you scrape.

