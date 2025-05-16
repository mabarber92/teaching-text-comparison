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

# Use topic model to select articles
topics = [15, 18, 33, 28]
topic_df = pd.read_csv("../data_out/articles-data/topic-model/topic-model.csv")
chosen_articles = topic_df[topic_df["Topic"].isin(topics)][["file", "Topic", "topic_1", "topic_2", "topic_3", "topic_4"]].values.tolist()


# # Randomly select articles
# no_articles = 100
# articles = os.listdir(articles_dir)
# for i in range(no_articles):
#     chosen_articles.append(random.choice(articles))

input_sequences = []
input_titles = []
article_topic = []

for article in chosen_articles:
    article_path = os.path.join("..", article[0])
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
    topic_labels = ", ".join(article[2:])
    article_topic.append(f"{article[1]}: {topic_labels}")



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
    dict_list.append({"x": reduced[i, 0], "y": reduced[i, 1], "title": str(i) + "-" + input_titles[i], "no": i , "topic_no": str(article_topic[i])})

df = pd.DataFrame(dict_list)
df.to_csv("seq_embedding_space-4-tops4.csv")
fig = px.scatter(df, x="x", y="y", text="no", color="title", title=f"{len(df)} articles from the Gaza corpus clustered according to their sequence embeddings")
fig.update_traces(textposition='top center')
fig.update_layout(
    font=dict(        
        size=8
    )
)


fig.write_html("sequence-4-topics-arts4.html")

fig = px.scatter(df, x="x", y="y", text="no", color="topic_no", title=f"{len(df)} articles from the Gaza corpus clustered according to their sequence embeddings, coloured according to BERTopic")
fig.update_traces(textposition='top center')
fig.update_layout(
    font=dict(        
        size=8
    )
)
fig.write_html("sequence-4-topics-labelled4.html")