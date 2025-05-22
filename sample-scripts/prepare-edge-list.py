import pandas as pd

pairs_df = pd.read_csv("tfidf-2024.csv")
edges = pairs_df[["filename-1", "filename-2", "similarity"]]

edges["id1"] = edges["filename-1"].str.split("[_.]").str[-2]
edges["id2"] = edges["filename-2"].str.split("[_.]").str[-2]

nodes1 = pairs_df[["filename-1", "title-1", "year-1", "month-1"]]
nodes2 = pairs_df[["filename-2", "title-2", "year-2", "month-2"]]
nodes1 = nodes1.rename(columns={"filename-1": "Id", "title-1": "Label", "year-1": "year", "month-1": "month"})
nodes2 = nodes2.rename(columns={"filename-2": "Id", "title-2": "Label", "year-2": "year", "month-2": "month"})
nodes = pd.concat([nodes1, nodes2]).drop_duplicates()

title_df = pd.read_csv("../data_out/articles-data/title/title.csv")[["length", "file"]]
title_df["Id"] = title_df["file"].str.split("\\").str[-1]
print(title_df)
nodes = pd.merge(nodes, title_df, on="Id")

edges = edges.rename(columns = {"filename-1": "Source", "filename-2": "Target", "similarity": "Weight"})

edges.to_csv("tfidf-2024-edges-list.csv", encoding='utf-8-sig', index=False)
nodes.to_csv("tfidf-2024-nodes-list.csv", encoding='utf-8-sig', index=False)
