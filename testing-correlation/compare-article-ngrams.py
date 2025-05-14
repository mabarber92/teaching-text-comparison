import pandas as pd
from collections import Counter
import os
import plotly.express as px

def count_ngrams(text, suffix):
    tokens = text.split()
    list_dict = []
    for key, value in Counter(tokens).items():
        list_dict.append({"ngram": key, f"count-{suffix}": value})
    return pd.DataFrame(list_dict)

    

# Use the similarity scores to fetch three document pairs with different similarities
tfidf_similarities = pd.read_csv("../data_out/articles-data/tfidf/tfidf-over-0.3-len100.csv")

similarity_thres = [0.3, 0.6, 0.9]
filtered_df = tfidf_similarities.copy()
selected_pairs = []

for similarity in similarity_thres:
    filtered_df = filtered_df[filtered_df["similarity"] > similarity]
    filtered_df = filtered_df.sort_values(by=["similarity"])
    selected_pair = filtered_df.iloc[0][["filename-1", "filename-2", "similarity"]].values.tolist()
    selected_pairs.append(selected_pair)

print(selected_pairs)

# Use selected pairs to identify the two texts and create the input for the scatter by counting n-grams
base_path = "../data_in/articles/"

full_df = pd.DataFrame()

for pair in selected_pairs:    
    file_path = os.path.join(base_path, pair[0])
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    ngrams_df = count_ngrams(text, "text-1")
    file_path = os.path.join(base_path, pair[1])
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    ngrams2_df = count_ngrams(text, "text-2")
    merged_df = pd.merge(ngrams_df, ngrams2_df, on="ngram", how="inner")
    merged_df.fillna(0)
    merged_df["similarity"] = pair[2]
    full_df = pd.concat([full_df, merged_df])

print(full_df)

fig = px.scatter(full_df, x='count-text-1', y='count-text-2', facet_col = 'similarity', hover_data =['ngram'])
fig.show()





# Plot the scatter with the df