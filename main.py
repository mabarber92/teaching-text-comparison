from pipelines.corpusAnalytics import corpusAnalytics
import os

if __name__ == "__main__":
    path_in = "data_in/articles"
    path_out = "data_out/articles-data/"
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    corpusAnalytics(path_in, path_out, pipeline_names=['topic-model'])