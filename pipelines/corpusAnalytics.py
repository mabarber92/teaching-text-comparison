from pipelines.utilities import ArticleProcessor
from pipelines.topicModel import buildTopicModel
from collections import Counter
from tqdm import tqdm
import os
import re
import math
import gc
class corpusAnalytics(ArticleProcessor):
    


    def __init__(self, dir, out_dir, ngram_range = [1,4], initialise_only = False, pipeline_names = [], embedding_seq_len = 512):
        super().__init__(dir)
        self.out_dir = out_dir
        self.ngram_range = range(ngram_range[0], ngram_range[1])
        self.embedding_seq_len = embedding_seq_len

        self._processors = []
        self._pipeline_dict = {
            "article-titles" : {"func": self.article_titles, "group_processor": False},
            "article-lengths" : {"func": self.article_lengths, "group_processor": True},
            "article-content" : {"func": self.article_content, "group_processor": False}
            
        }
        self._group_by_cols = ["count", "length"]
        self._date_cols = list(self._date_patterns.keys())
        self._inc_cols = []
        self._active_group_by = []
        self._non_group_processors = []

        self.build_processors(pipeline_names)

        if not initialise_only:
            self.run_pipeline()

    
    def build_ngram_processors(self):
        """Create a series of processors based on the specified ngram ranges"""        
        for n in self.ngram_range:
            self._processors.append((f"{n}-gram", self.ngram_processor(n)))
        

    def build_processors(self, pipeline_names = []):
        """Create a processor list to pass to the article processor. If an empty list is passed (default behaviour), then all pipelines are loaded"""
        if len(pipeline_names) == 0:
            pipeline_names = list(self._pipeline_dict.keys()) + ["ngram", "topic-model"]            
        for pipeline in pipeline_names:
            if pipeline == "ngram":
                self.build_ngram_processors()
            if pipeline == "topic-model":
                self._processors.append((pipeline, self.article_content(token_limit=self.embedding_seq_len, title_col=True)))
                self._non_group_processors.append(pipeline)
            if pipeline in list(self._pipeline_dict.keys()):
                self._processors.append((pipeline, self._pipeline_dict[pipeline]["func"]()))
                if not self._pipeline_dict[pipeline]["group_processor"]:
                    self._non_group_processors.append(pipeline)
            else:
                print(f"{pipeline} not found, pipeline options:")
                print(self._pipeline_dict)
        print("Pipeline loaded:")
        print(self._processors)


    def text_cleaner(self, text):
        text = re.sub(r'\W+|"+', " ", text).lower()
        return text

    def article_titles(self, splitter = "\n+-----"):
        def _titles_processor(text, file):
            splits = re.split(splitter, text)            
            length = len(splits[1].split())
            return {"title": splits[0], "length": length, "file": file}
        return _titles_processor
    
    def article_lengths(self):
        def _length_processor(text, metadata):
            length = len(text.split())
            return {"length": length}
        return _length_processor
        
    def article_content(self, splitter = "\n+-----", token_limit = 512, split_evenly = True, title_col = False, include_file = True):
        def _content_processor(text, file):
            parts = re.split(splitter, text)
            article = parts[1] if len(parts) > 1 else parts[0]
            tokens = article.split()
            length = len(tokens)
            if length > token_limit:
                out = []
                split_count = int(math.ceil(length/token_limit))                
                if split_evenly:
                    split_size = int(math.ceil(length/split_count))
                else:
                    split_size = token_limit
                for i in range(0, length, split_size):
                    end = i + split_size
                    if end > length:
                        end = -1
                if title_col:
                    out_dict = {"text" : " ".join(tokens[i:end]), "title": parts[0]}
                else:
                    out_dict = {"text" : " ".join(tokens[i:end])}
                if include_file:
                    out_dict["file"] = file
                out.append(out_dict)
            else:
                if title_col:
                    out = {"text" : article, "title": parts[0]}
                else:
                    out = {"text" : article}
                if include_file:
                    out["file"] = file
            return out
        return _content_processor

    def ngram_processor(self, n=1):
        def _processor(text, metadata):
            text = self.text_cleaner(text)
            tokens = text.split()
            grams = zip(*[tokens[i:] for i in range(n)])
            counts = Counter([' '.join(g) for g in grams])
            return [{f"{n}-gram": g, "count": c} for g, c in counts.items()]
        return _processor

    
    def find_main_col(self, df):
        column_names = df.columns.to_list()
        

        self._active_group_by = None

        for col in self._group_by_cols:
            if col in column_names:                
                inc_col = col
                column_names.remove(col)
                self._active_group_by = inc_col
                break
        for col in self._date_patterns:
            if col in column_names:
                column_names.remove(col)

        if "file" in column_names:
            column_names.remove("file")      
        
        if len(column_names) == 1:
            return column_names[0]
        elif "text" in column_names and "title" in column_names:
            return "text"
        else:
            return self._active_group_by
    
    def group_df_by(self, df, main_col, group_by = "year"):
        
        if isinstance(group_by, str):
            group_cols = [group_by, main_col]
        else:
            group_cols = list(group_by) + [main_col]
        
        if self._active_group_by in group_cols:
            group_cols.remove(self._active_group_by)

        grouped = (
        df.groupby(group_cols)
            .agg(
                **{
                    f"{self._active_group_by}-sum": (self._active_group_by, 'sum'),
                    f"{self._active_group_by}-mean": (self._active_group_by, 'mean')
                }
            )
            .reset_index()
        )

        grouped = grouped.sort_values(by = f"{self._active_group_by}-sum", ascending=False)

        return grouped
    
    def export_df(self, df, main_col, group_by=None):
        if group_by:
            if isinstance(group_by, list):
                file_name = f"{main_col}-{'-'.join(group_by)}.csv"
            else:
                file_name = f"{main_col}-{group_by}.csv"
        else:
            file_name = f"{main_col}.csv"
        out_dir =  os.path.join(self.out_dir, main_col)
        os.makedirs(out_dir, exist_ok = True)
        out_path = os.path.join(out_dir, file_name)
        
        df.to_csv(out_path, index=False)
        

    def run_pipeline(self, group_bys = ["year", ["year", "month"]]):
        # NOTE (from Chat-GPT):
        # This method processes all processors at once, keeping outputs in RAM.
        # For very large corpora, consider a memory-safe version:
        #    - Process one processor at a time
        #    - Save output to disk immediately
        #    - Reload as needed
        # Trade-off: higher disk I/O, slightly slower, but much lower memory usage.

        print("Running the specified pipeline")

        dfs = self.process_all_articles(self._processors)

        for df_name in tqdm(dfs):
            df = dfs[df_name]
            main_col = self.find_main_col(df)
            
            
            if df_name == "topic-model":
                topic_df = buildTopicModel(main_col, df=df, seq_length = self.embedding_seq_len).berttopic_pipeline()
                self.export_df(topic_df, "topic-model")
                del topic_df
                gc.collect()

            else:
                if self._active_group_by is not None:
                    df = df.sort_values(by=[self._active_group_by])
                
                self.export_df(df, main_col)
            
            
            
            if df_name not in self._non_group_processors:
                for group_by in group_bys:                
                    grouped_df = self.group_df_by(df, main_col, group_by = group_by)
                    self.export_df(grouped_df, main_col, group_by)
                    del grouped_df
                    gc.collect()
            
