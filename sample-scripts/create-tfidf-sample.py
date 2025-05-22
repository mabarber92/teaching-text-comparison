import pandas as pd

csv_path = "../data_out/articles-data/tfidf/tfidf-over-0.3.csv"
df = pd.read_csv(csv_path)

df = df[df["year-1"] == 2024]
df = df[df["year-2"] == 2024]

print(df[["year-1", "year-2"]])

df.to_csv("tfidf-2024.csv", index=False)
