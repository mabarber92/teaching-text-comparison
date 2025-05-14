from sentence_transformers import SentenceTransformer, models
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import re
from torch import nn

from torch import cuda
import os
import random



articles_dir = "../data_in/articles/"
chosen_articles = []

# Randomly select articles
no_articles = 100
articles = os.listdir(articles_dir)
for i in range(no_articles):
    chosen_articles.append(random.choice(articles))

input_sequences = []
input_titles = []

for article in chosen_articles:
    article_path = os.path.join(articles_dir, article)
    with open(article_path, "r", encoding='utf-8') as f:
        text = f.read()
    splits = text.split("-----")
    input_titles.append(splits[0])
    text = splits[1]
    tokens = text.split()
    if len(tokens) > 512:
        sequence = " ".join(tokens[:512])
    else:
        sequence = " ".join(tokens)
    input_sequences.append(sequence)



device = "cuda:0" if cuda.is_available() else "cpu"
print(device)
word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                            pooling_mode_mean_tokens=True)
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                        out_features=512, 
                        activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device = device)

# Get word embeddings using a masked language model or similar technique
# This is simplified: real tokenization needed for proper alignment
embeddings = model.encode(input_sequences)



# Reduce to 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Plot

dict_list = []
for i, sequence in enumerate(input_sequences):
    dict_list.append({"x": reduced[i, 0], "y": reduced[i, 1], "title": str(i) + "-" + input_titles[i], "no": i })

df = pd.DataFrame(dict_list)
df.to_csv("seq_embedding_space-50-arts.csv")
fig = px.scatter(df, x="x", y="y", text="no", color="title", title="100 random articles from the Gaza corpus clustered according to their sequence embeddings")
fig.update_traces(textposition='top center')
fig.update_layout(
    font=dict(        
        size=8
    )
)


fig.write_html("sequence-random-100-arts.html")