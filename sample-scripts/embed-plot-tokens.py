from sentence_transformers import SentenceTransformer, models
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import re
from torch import nn

from torch import cuda





article_path = "../data_in/articles/2023-11-17_2262.txt"
with open(article_path, "r", encoding='utf-8') as f:
    text = f.read()

text = text.split("-----")[-1]
# Take only first six sentences
text = ".".join(text.split(".")[:10])
# article_tokens = []
# for sentence in sentences:
#     normalised = re.sub(""""\W+|[-"']+""", " ", sentence)
#     article_tokens.extend(normalised.split() + ["."])

# print(" ".join(article_tokens))
print(text)

# # Normalise text
article_tokens = re.split("\s", text)

article_tokens = [x for x in article_tokens if x != " "]


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
embeddings = model.encode(article_tokens)



# Reduce to 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Plot

dict_list = []
for i, word in enumerate(article_tokens):
    dict_list.append({"x": reduced[i, 0], "y": reduced[i, 1], "word": word })

df = pd.DataFrame(dict_list)
df.to_csv("token_embedding_space.csv")
fig = px.scatter(df, x="x", y="y", text="word", title="The tokens in the first 10 sentences of the article 2023-11-17_2262.txt clustered according to their position in the article's embedding space")
fig.update_traces(textposition='top center')
fig.update_layout(
    font=dict(        
        size=8
    )
)


fig.write_html("token-embedding-space.html")