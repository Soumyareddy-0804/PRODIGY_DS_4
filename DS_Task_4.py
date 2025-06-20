# 🚀 COMPLETE TWITTER SENTIMENT ANALYSIS PROJECT
# 📊 Task-04: Analyze and visualize sentiment patterns in social media data
# 🎯 Platform: Google Colab
# 📈 Goal: Understand public opinion and attitudes towards specific topics/brands

# ===================================================================
# SECTION 1: SETUP AND INSTALLATIONS
# ===================================================================

# Install required packages (removed kaggle and opendatasets since using Google Drive)
!pip install pandas numpy matplotlib seaborn plotly wordcloud textblob scikit-learn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Text processing libraries
import re
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter

# Machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ===================================================================
# SECTION 2: DATA LOADING AND EXPLORATION
# ===================================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("🔄 Loading dataset from Google Drive...")
print("📁 Using your uploaded file: twitter_training.csv")

try:
    # Load the dataset from your Google Drive path
    file_path = '/content/drive/MyDrive/Projects/Task4/twitter_training.csv'
    df = pd.read_csv(file_path, encoding='latin-1', header=None)

    # Define column names based on typical Twitter sentiment dataset structure
    # The dataset typically has: [tweet_id, entity, sentiment, tweet_text]
    df.columns = ['tweet_id', 'entity', 'sentiment', 'tweet_text']

    print(f"✅ Dataset loaded successfully from Google Drive!")
    print(f"📊 Loaded {len(df)} tweets for analysis")

except FileNotFoundError:
    print(f"❌ File not found at: {file_path}")
    print("🔍 Please check if the file exists in your Google Drive")
    print("📝 Make sure the file is named 'twitter_training.csv'")

    # List available files in the directory to help user
    import os
    try:
        files = os.listdir('/content/drive/MyDrive/Projects/Task4/')
        print(f"📂 Files found in your Task4 folder: {files}")
    except:
        print("📂 Could not access the Task4 folder")

    # Exit if file not found
    raise Exception("Dataset file not found. Please check your file path.")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("🔄 Please check your file format and try again")

# Display basic information about the dataset
print("\n📊 DATASET OVERVIEW")
print("=" * 50)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\n📈 DATASET STATISTICS")
print("=" * 50)
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# ===================================================================
# SECTION 3: DATA PREPROCESSING
# ===================================================================

print("\n🔧 DATA PREPROCESSING")
print("=" * 50)

# Clean the data
df = df.dropna()  # Remove rows with missing values
df = df.drop_duplicates()  # Remove duplicate rows

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess tweet text"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

# Apply text preprocessing
df['cleaned_text'] = df['tweet_text'].apply(preprocess_text)

# Remove empty texts after cleaning
df = df[df['cleaned_text'].str.len() > 0]

print(f"✅ Data cleaned. Final dataset shape: {df.shape}")

# ===================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ===================================================================

print("\n📊 EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create subplots for comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Twitter Sentiment Analysis - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts()
axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Overall Sentiment Distribution')

# 2. Entity Distribution
entity_counts = df['entity'].value_counts().head(10)
axes[0, 1].bar(entity_counts.index, entity_counts.values)
axes[0, 1].set_title('Top 10 Entities by Tweet Count')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Sentiment by Entity (Top 5 entities)
top_entities = df['entity'].value_counts().head(5).index
entity_sentiment = df[df['entity'].isin(top_entities)].groupby(['entity', 'sentiment']).size().unstack(fill_value=0)
entity_sentiment.plot(kind='bar', stacked=True, ax=axes[1, 0])
axes[1, 0].set_title('Sentiment Distribution by Top 5 Entities')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend(title='Sentiment')

# 4. Tweet Length Distribution
df['text_length'] = df['cleaned_text'].str.len()
axes[1, 1].hist(df['text_length'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 1].set_title('Distribution of Tweet Lengths')
axes[1, 1].set_xlabel('Character Count')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n📈 SUMMARY STATISTICS")
print("=" * 30)
print("Sentiment Distribution:")
print(df['sentiment'].value_counts())
print(f"\nAverage tweet length: {df['text_length'].mean():.1f} characters")
print(f"Total unique entities: {df['entity'].nunique()}")

# ===================================================================
# SECTION 5: ADVANCED SENTIMENT ANALYSIS
# ===================================================================

print("\n🤖 ADVANCED SENTIMENT ANALYSIS")
print("=" * 50)

# Sentiment Analysis using TextBlob
def get_textblob_sentiment(text):
    """Get sentiment polarity using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Apply TextBlob sentiment analysis
df['textblob_sentiment'] = df['cleaned_text'].apply(get_textblob_sentiment)

# Compare original labels with TextBlob predictions
comparison = pd.crosstab(df['sentiment'], df['textblob_sentiment'], margins=True)
print("Comparison: Original Labels vs TextBlob Predictions")
print(comparison)

# ===================================================================
# SECTION 6: INTERACTIVE VISUALIZATIONS
# ===================================================================

print("\n📊 CREATING INTERACTIVE VISUALIZATIONS")
print("=" * 50)

# 1. Interactive Sentiment Distribution by Entity
fig1 = px.sunburst(
    df,
    path=['sentiment', 'entity'],
    title='Interactive Sentiment Distribution by Entity',
    color='sentiment',
    color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#4682B4'}
)
fig1.show()

# 2. Sentiment Timeline (if timestamp available, otherwise skip)
try:
    # Try to create a timeline if date information is available
    df['date'] = pd.to_datetime('2024-01-01')  # Placeholder date
    daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

    fig2 = px.line(
        daily_sentiment.reset_index(),
        x='date',
        y=['Positive', 'Negative', 'Neutral'],
        title='Sentiment Trends Over Time',
        labels={'value': 'Tweet Count', 'variable': 'Sentiment'}
    )
    fig2.show()
except:
    print("Timeline visualization skipped (no date information)")

# 3. Entity Sentiment Heatmap
entity_sentiment_matrix = pd.crosstab(df['entity'], df['sentiment'])
fig3 = px.imshow(
    entity_sentiment_matrix.values,
    x=entity_sentiment_matrix.columns,
    y=entity_sentiment_matrix.index,
    title='Entity vs Sentiment Heatmap',
    color_continuous_scale='RdYlBu',
    aspect='auto'
)
fig3.update_layout(height=600)
fig3.show()

# ===================================================================
# SECTION 7: WORD CLOUD ANALYSIS
# ===================================================================

print("\n☁️ WORD CLOUD ANALYSIS")
print("=" * 50)

# Create word clouds for each sentiment
sentiments = df['sentiment'].unique()
fig, axes = plt.subplots(1, len(sentiments), figsize=(18, 6))

for idx, sentiment in enumerate(sentiments):
    # Get text for specific sentiment
    sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])

    # Create word cloud
    wordcloud = WordCloud(
        width=400,
        height=300,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(sentiment_text)

    # Plot
    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].set_title(f'{sentiment} Sentiment - Word Cloud', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# ===================================================================
# SECTION 8: MACHINE LEARNING MODEL
# ===================================================================

print("\n🤖 BUILDING MACHINE LEARNING MODEL")
print("=" * 50)

# Prepare data for machine learning
X = df['cleaned_text']
y = df['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("📊 MODEL PERFORMANCE")
print("=" * 30)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ===================================================================
# SECTION 9: BUSINESS INSIGHTS AND RECOMMENDATIONS
# ===================================================================

print("\n💡 BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("=" * 50)

# Calculate key metrics
total_tweets = len(df)
positive_ratio = (df['sentiment'] == 'Positive').sum() / total_tweets * 100
negative_ratio = (df['sentiment'] == 'Negative').sum() / total_tweets * 100
neutral_ratio = (df['sentiment'] == 'Neutral').sum() / total_tweets * 100

print(f"📊 OVERALL SENTIMENT SUMMARY")
print(f"   • Total Tweets Analyzed: {total_tweets:,}")
print(f"   • Positive Sentiment: {positive_ratio:.1f}%")
print(f"   • Negative Sentiment: {negative_ratio:.1f}%")
print(f"   • Neutral Sentiment: {neutral_ratio:.1f}%")

# Top entities analysis
print(f"\n🏢 TOP ENTITIES ANALYSIS")
top_5_entities = df['entity'].value_counts().head(5)
for entity in top_5_entities.index:
    entity_data = df[df['entity'] == entity]
    pos_pct = (entity_data['sentiment'] == 'Positive').sum() / len(entity_data) * 100
    neg_pct = (entity_data['sentiment'] == 'Negative').sum() / len(entity_data) * 100
    print(f"   • {entity}: {pos_pct:.1f}% Positive, {neg_pct:.1f}% Negative")

print(f"\n🎯 KEY RECOMMENDATIONS")
print("   1. Focus on entities with high negative sentiment for reputation management")
print("   2. Leverage positive sentiment entities for marketing campaigns")
print("   3. Monitor neutral sentiment entities for engagement opportunities")
print("   4. Implement real-time monitoring for sentiment tracking")
print("   5. Develop targeted strategies based on entity-specific sentiment patterns")

# ===================================================================
# SECTION 10: EXPORT RESULTS
# ===================================================================

print("\n💾 EXPORTING RESULTS")
print("=" * 50)

# Create summary report
summary_df = df.groupby(['entity', 'sentiment']).size().unstack(fill_value=0)
summary_df['total'] = summary_df.sum(axis=1)
summary_df = summary_df.sort_values('total', ascending=False)

# Save to CSV
summary_df.to_csv('sentiment_analysis_summary.csv')
df.to_csv('processed_twitter_data.csv', index=False)

print("✅ Results exported successfully!")
print("   • sentiment_analysis_summary.csv - Entity-wise sentiment summary")
print("   • processed_twitter_data.csv - Complete processed dataset")

print(f"\n🎉 ANALYSIS COMPLETE!")
print("=" * 50)
print("Your sentiment analysis project is ready!")
print("You can now use these insights for business decision-making.")
