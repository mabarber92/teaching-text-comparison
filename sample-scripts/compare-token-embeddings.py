import torch
import numpy as np
from sentence_transformers import models
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

# Set up transformer model
word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
model = word_embedding_model.auto_model
tokenizer = word_embedding_model.tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Input sentences
sentence1 = "Mark Regev, Netanyahu’s former spokesperson, for the country’s overall approach to its war on Gaza, in which, he has said, Israel is “trying to be as surgical as humanly possible”"
sentence2 = "It added that the raid, which began at 2am (00:00 GMT), has resulted in a “number of martyrs and wounded”. Al Jazeera Arabic reported that the hospital’s surgical building was on fire following the Israeli bombing."
sentence3 = "She recalled MSF staff working without some of the most “basic surgical supplies”, dressing the wounds of babies who had had limbs amputated, and women with second-degree burns."
sentence4 = "But critics have pointed out that the Trump administration’s decision to assassinate Iranian General Qassem Soleimani in Iraq on January 3, 2020, brought the two countries to the brink of war. Republican presidential candidate Nikki Haley has also called for “surgical strikes” on Iranian assets and officials outside Iran."

# Keywords to highlight
highlight_keywords = {
    "surgical": "Surgical",
    "raid": "Warfare",
    "war": "Warfare",
    "strikes": "Warfare"
}

def get_embeddings_with_labels(sentence, sentence_label):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    token_embeddings = output.last_hidden_state[0]  # shape: [seq_len, hidden_dim]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    data = []
    for i, token in enumerate(tokens):
        embedding = token_embeddings[i].cpu().numpy()
        word = token.replace("▁", "").strip("Ġ")  # cleanup for subword tokens
        label = highlight_keywords.get(word.lower(), "Other")
        data.append({"embedding": embedding, "token": word, "sentence": sentence_label, "keyword": label})
    return data

# Process both sentences
data1 = get_embeddings_with_labels(sentence1, "Sentence 1")
data2 = get_embeddings_with_labels(sentence2, "Sentence 2")
data3 = get_embeddings_with_labels(sentence3, "Sentence 3")
data4 = get_embeddings_with_labels(sentence4, "Sentence 4")
all_data = data1 + data2 + data3 + data4

# PCA reduction
all_embeddings = np.array([d["embedding"] for d in all_data])
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_embeddings)

# Create DataFrame
df = pd.DataFrame({
    "x": reduced[:, 0],
    "y": reduced[:, 1],
    "token": [d["token"] for d in all_data],
    "sentence": [d["sentence"] for d in all_data],
    "highlight": [d["keyword"] for d in all_data]
})

# Plot
fig = px.scatter(
    df,
    x="x",
    y="y",
    text="token",
    color="highlight",
    symbol="sentence",
    title="Contextual Token Embedding Space with Highlighted Keywords"
)
fig.update_traces(textposition="top center")
fig.update_layout(font=dict(size=9), showlegend=True)

fig.write_html("token-embedding-compared.html")
