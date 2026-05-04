# TASK 4
# Load the Dataset
import pandas as pd
df = pd.read_csv(r"C:\Users\Jebisha\Downloads\twitter_training.csv")
df.head()
df.columns
df.head()

df = pd.read_csv(r"C:\Users\Jebisha\Downloads\twitter_training.csv", header=None)
df.head()

# Give Proper Column Names
df.columns = ['ID', 'Topic', 'Sentiment', 'Text']
df.head()

# Check Data
df.info()
df['Sentiment'].value_counts()

# Clean Text
df['Text'] = df['Text'].astype(str)
df.dropna(inplace=True)

# Sentiment Distribution
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

# Sentiment by Topic
sns.countplot(x='Topic', hue='Sentiment', data=df)
plt.xticks(rotation=45)
plt.title("Sentiment by Topic")
plt.show()

# Fix Code
top_topics = df['Topic'].value_counts().head(10).index

filtered_df = df[df['Topic'].isin(top_topics)]

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
sns.countplot(x='Topic', hue='Sentiment', data=filtered_df)
plt.xticks(rotation=45)
plt.title("Sentiment by Top 10 Topics")
plt.show()

# Word Cloud
from wordcloud import WordCloud
text = " ".join(df['Text'])
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud")
plt.show()
