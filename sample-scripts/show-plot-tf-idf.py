from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

import plotly.express as px


# Sample corpus
corpus_path = "../data_in/articles/"
corpus_sample = []
# counter = 19

# Specify the main doc for comparison
main_doc = "2023-11-17_2262.txt"
main_doc_path = os.path.join(corpus_path, main_doc)
with open(main_doc_path, 'r', encoding='utf-8') as f:
    text = f.read()
corpus_sample.append(text)

for article in tqdm(os.listdir(corpus_path)):
    if article != main_doc:
        article_path = os.path.join(corpus_path, article)
        with open(article_path, 'r', encoding='utf-8') as f:
            text = f.read()
        corpus_sample.append(text)

# Select target document
target_doc_idx = 0
target_doc = corpus_sample[target_doc_idx]

# Vectorize with TF-IDF to get scores
tfidf_vectorizer = TfidfVectorizer(stop_words='english', norm=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_sample)
tfidf_scores = tfidf_matrix[target_doc_idx].toarray().flatten()
vocab = tfidf_vectorizer.get_feature_names_out()

# Vectorize with raw counts to get DF and TF
count_vectorizer = CountVectorizer(stop_words='english')
X = count_vectorizer.fit_transform(corpus_sample)
doc_tf = X[target_doc_idx].toarray().flatten()
df = (X > 0).sum(axis=0).A1
count_vocab = count_vectorizer.get_feature_names_out()

# Align vocabularies (TF-IDF and CountVectorizer may tokenize slightly differently)
word_to_index = {word: i for i, word in enumerate(vocab)}
common_words = [word for word in count_vocab if word in word_to_index]

# Gather values for plotting
tf_values = [doc_tf[np.where(count_vocab == word)[0][0]] for word in common_words]
df_values = [df[np.where(count_vocab == word)[0][0]] for word in common_words]
tfidf_values = [tfidf_scores[word_to_index[word]] for word in common_words]

import pandas as pd

# Create a DataFrame with words and their scores
data = {
    'word': common_words,
    'term_frequency': tf_values,
    'document_frequency': df_values,
    'tfidf_score': tfidf_values
}

df = pd.DataFrame(data)

# Sort by TF-IDF score (descending)
df_sorted = df.sort_values(by='tfidf_score', ascending=False)

# # Save to CSV
csv_path = "tfidf_scores_doc1-non-norm.csv"
df_sorted.to_csv(csv_path, index=False)

# df_sorted = pd.read_csv(csv_path)

df_top = df_sorted.head(50)




# Plot
# Create a Plotly scatter plot
fig = px.scatter(
    df_top,
    x="document_frequency",
    y="term_frequency",
    
    text="word",
    
    color="tfidf_score",
    color_continuous_scale="blues",
    title="TF vs DF (Point Size = TF-IDF Score) - Top 50 TF-IDF scores",
    labels={
        "document_frequency": "Document Frequency (across corpus)",
        "term_frequency": "Term Frequency (in target document)",
        "tfidf_score": "TF-IDF Score"
    }
)

# Update layout for better readability
fig.update_traces(textposition='top center')
fig.update_layout(
    showlegend=False,
    plot_bgcolor='white'
)

fig.write_html("tf-idf-scatter-non-norm.html")
