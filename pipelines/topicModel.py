import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, models
from torch import nn
from torch import cuda
from umap import UMAP
import os


class buildTopicModel:
    def __init__ (self, text_field, df=None, csv=None, run_full_pipeline=False, seq_length = 512, embed_model = "sentence-transformers/all-MiniLM-L6-v2", min_topic_size =None, n_neighbours = 15):
        if csv is not None:
            self.df = pd.read_csv(csv, encoding='utf-8-sig')
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Provide either a df or csv as input")
        self._text_field = text_field
        self.seq_length = seq_length
        self.model_name = embed_model
        self.min_topic_size = min_topic_size
        self.n_neighbours = n_neighbours

        if run_full_pipeline:
            self.berttopic_pipeline()
    
    def initialise_embed_model(self):
        print("loading model...")       
        word_embedding_model = models.Transformer(self.model_name, self.seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True)
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                                out_features=self.seq_length, 
                                activation_function=nn.Tanh())
        # It seems we may not need the device set up
        device = "cuda:0" if cuda.is_available() else "cpu"
        print(device)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device = device)
        print("model loaded")
        return model

    def embed_sentences(self, model):
        sentences = self.df[self._text_field].to_list()
        embeds = model.encode(sentences, show_progress_bar=True)
        print("Sentences embedded")
        return embeds
    
    def create_berttopic_model(self, embeds):
        print("Creating Topic Model")
        umapModel = UMAP(n_neighbors=self.n_neighbours, n_components=5, min_dist=0.0, metric='cosine')
        topic_model = BERTopic(language = 'multilingual',umap_model=umapModel)
        self.df['Topic'], probabilities = topic_model.fit_transform(self.df[self._text_field], embeds)
        
        print("Adding topic info")
        topic_info = topic_model.get_topic_info()
        self.df = self.df.merge(topic_info, left_on='Topic', right_on = "Topic", how='left')
    
    def clean_topic_df(self):
        if "Name" in self.df.columns:
            self.df[["topic_no", "topic_1", "topic_2", "topic_3", "topic_4"]] = self.df["Name"].str.split("_", expand=True)
        self.df.drop(columns=[self._text_field, "topic_no", "Name"], inplace=True, errors="ignore")
        self.df = self.df.sort_values(by=["Topic"])


    def berttopic_pipeline(self):
        model = self.initialise_embed_model()
        embeds = self.embed_sentences(model)
        self.create_berttopic_model(embeds)
        self.clean_topic_df()
        return self.df
