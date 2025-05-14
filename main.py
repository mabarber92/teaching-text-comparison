from pipelines.corpusAnalytics import corpusAnalytics
from pipelines.ComputeTFIDF import computeTFIDF
import os

if __name__ == "__main__":
    path_in = "data_in/articles"
    path_out = "data_out/articles-data/"
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    corpusAnalytics(path_in, path_out, pipeline_names=['article-titles'])

    # tfidf = computeTFIDF(path_in, article_length_min=200)
    # csv_out = os.path.join(path_out, "tfidf-over-0.3-len200.csv")
    # tfidf.run_tfidf(csv_out)