import pandas as pd
import os
import re
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = list(ENGLISH_STOP_WORDS)

# Specify Articles path directory
article_dir = "../data_in/articles/"

# Loop through articles, find one that is long and then cut the first three sentences
# Find another for later comparison
count = 0
for article in tqdm(os.listdir(article_dir)):
    article_path = os.path.join(article_dir, article)
    with open(article_path, "r", encoding='utf-8') as f:
        text = f.read()
    if len(text.split()) > 300:
        count += 1
        if count > 3:
            break

# text = text.split("-----")[-1]
# text = ".".join(text.split(".")[:3])
main_doc = "2023-11-17_2262.txt"
article_path = os.path.join(article_dir, main_doc)
with open(article_path, "r", encoding='utf-8') as f:
    text = f.read()

print(text)

# Normalise text
text = re.sub("\W+", " ", text).lower()

# Calculate n-grams
n_grams = Counter(text.split())

# Create output excluding stopwords
list_dict = []
for word, count in n_grams.items():
    if word not in stop_words:
        list_dict.append({"n-gram": word, "count": count})

df = pd.DataFrame(list_dict)
df = df.sort_values(by="count", ascending=False)

df.to_csv(article + ".csv", index=False)
