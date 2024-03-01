#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

plt.style.use('ggplot')

import nltk


# In[7]:


# Reading Data from csv
df = pd.read_csv('DataSet/Reviews.csv')
df = df.head(500)

ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of reviews by stars', figsize=(10,5))
ax.set_xlabel('Review Stars')

plt.show()


# In[8]:


example = df['Text'].values[50]
print(example)


# In[9]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I am very happy')


# In[10]:


sia.polarity_scores('I am sad')


# In[11]:


sia.polarity_scores(example)


# In[12]:


df


# In[13]:


import time


# In[14]:


res = {}
total = len(df)
for i, row in df.iterrows():
  text = row['Text']
  myId = row['Id']
  res[myId] = sia.polarity_scores(text)
  


# In[15]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[16]:


# Sentiment score and metadata
ax = sb.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound score by Amazon Star Review')
plt.show()


# In[17]:


fig, axs = plt.subplots(1, 3, figsize=(15,5))
sb.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sb.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sb.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[18]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[19]:


import torch
print(torch.__version__)


# In[20]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[21]:


example


# In[23]:


sia.polarity_scores(example)


# In[26]:


encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
  'roberta_neg': scores[0],
  'roberta_neu': scores[1],
  'roberta_pos': scores[2]
}
print(scores_dict)


# In[27]:


def polarity_scores_roberta(example):
  encoded_text = tokenizer(example, return_tensors='pt')
  output = model(**encoded_text)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)
  scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
  }
  return scores_dict


# In[30]:


res = {}
for i, row in df.iterrows():
  try:
    text = row['Text']
    myId = row['Id']
    vader_result = sia.polarity_scores(text)
    vader_result_rename = {}
    for key, value in vader_result.items():
      vader_result_rename[f'vader_{key}'] = value
    roberta_result = polarity_scores_roberta(text)
    both = {**vader_result_rename, **roberta_result}
    res[myId] = both
  except RuntimeError:
    print(f'Broke for id {myId}')


# In[29]:


roberta_result


# In[31]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# In[32]:


results_df.head()


# In[33]:


results_df.columns


# In[34]:


sb.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'], hue='Score', palette='tab10')
plt.show()


# In[35]:


results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[36]:


results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[39]:


results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0]


# In[40]:


results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0]


# In[42]:


from transformers import pipeline

sentiment_pipeline = pipeline('sentiment-analysis')


# In[43]:


sentiment_pipeline('I love redbull')


# In[44]:


sentiment_pipeline('I hate Monster')


# In[46]:


import torch
print(torch.__version__)



# In[ ]:




