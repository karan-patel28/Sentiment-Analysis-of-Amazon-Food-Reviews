# Sentiment Analysis on Amazon Food Reviews

#### An insightful exploration into the sentiments expressed in Amazon food reviews, utilizing both traditional and cutting-edge natural language processing techniques.

## Project Overview
- This project aims to dissect the vast array of consumer feedback contained within Amazon's food review dataset. By applying sentiment analysis, we seek to classify these reviews into positive, negative, and neutral categories, offering a granular view of customer sentiments. The project leverages Python, NLTK for foundational NLP tasks, and dives deep with the Transformers library for advanced sentiment analysis using pre-trained models.

## Features
- Sentiment analysis using NLTK's SentimentIntensityAnalyzer
- Advanced sentiment classification with RoBERTa model from Hugging Face's - Transformers
- Comprehensive data visualization for intuitive insights
- Comparative analysis of sentiment scores

## Technologies Used
- Python
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for visualization
- NLTK for basic sentiment analysis
- Transformers & PyTorch for advanced NLP tasks
- tqdm for progress tracking

#### Detailed versions of these libraries can be found in the requirements.txt file.

## Installation
- Ensure you have Python installed on your machine. 
- Clone this repository and install the required Python packages: 
#### Install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Usage
- Run the Jupyter notebooks or Python scripts provided in the repository to perform the sentiment analysis. Ensure you have the dataset 'DataSet/Reviews.csv' placed in the appropriate directory.

## Dataset
- This project uses the Amazon Food Review dataset available publicly. It consists of reviews that have been classified into various sentiment categories.
- Link: (https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## Models
- Utilizes NLTK's SentimentIntensityAnalyzer and Hugging Face's Transformers library for advanced sentiment analysis with deep learning models.