# Importing necessary libraries for data manipulation, visualization, and analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm.notebook import tqdm
from scipy.special import softmax

# Set the style for matplotlib
plt.style.use('ggplot')

# Load and display a preview of the dataset
df = pd.read_csv('DataSet/Reviews.csv').head(500)
df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of reviews by stars', figsize=(10,5)).set_xlabel('Review Stars')
plt.show()

# Example text for sentiment analysis
example = df['Text'].values[50]

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on example texts
sia.polarity_scores('I am very happy')
sia.polarity_scores('I am sad')
sia.polarity_scores(example)

# Apply sentiment analysis across the dataset
results = {}
for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    results[row['Id']] = sia.polarity_scores(text)

# Merge sentiment analysis results with original data
vaders = pd.DataFrame(results).T.rename_axis('Id').reset_index()
merged_df = vaders.merge(df, how='left')

# Plot compound sentiment scores by Amazon star review
sb.barplot(data=merged_df, x='Score', y='compound').set_title('Compound score by Amazon Star Review')
plt.show()

# Visualize positive, neutral, and negative sentiment scores
fig, axs = plt.subplots(1, 3, figsize=(15,5))
for i, sentiment in enumerate(['pos', 'neu', 'neg']):
    sb.barplot(data=merged_df, x='Score', y=sentiment, ax=axs[i]).set_title(sentiment.capitalize())
plt.tight_layout()
plt.show()

# Initialize tokenizer and model for advanced sentiment analysis
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to calculate sentiment scores using RoBERTa model
def polarity_scores_roberta(text):
    # Adjusting the tokenizer call to truncate the text to the model's max input size
    encoded_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# Apply both VADER and RoBERTa sentiment analysis
combined_results = {}
for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    vader_result = {f'vader_{k}': v for k, v in sia.polarity_scores(text).items()}
    combined_results[row['Id']] = {**vader_result, **polarity_scores_roberta(text)}

# Create a DataFrame from the combined results and merge with the original data
results_df = pd.DataFrame(combined_results).T.rename_axis('Id').reset_index().merge(df, how='left')

# Example sentiment analysis with Hugging Face pipeline
sentiment_analysis_pipeline = pipeline('sentiment-analysis')
print(sentiment_analysis_pipeline('I love redbull'), sentiment_analysis_pipeline('I hate Monster'))
