from pipelines.utilities import ArticleProcessor
from collections import Counter
from tqdm import tqdm
import os

class corpusAnalytics(ArticleProcessor):
    def __init__(self, dir, out_dir, full_pipeline = False):
        super().__init__(dir)
        self.out_dir = out_dir
        self._reset_cols()
        if full_pipeline:
            self.full_pipeline()

    def _reset_cols(self):
        self._add_col = None
        self._active_col = None

    def article_lengths(self):
        self._active_col = "length"
        def _length_func(text):
            return {self._active_col : len(text.split())}
        return self._process_articles(_length_func, df_out = True)
    
    def count_ngrams(self, n=1):
        def _ngrams_func(text):
            tokens = text.split()
            grams = zip(*[tokens[i:] for i in range(n)])
            counts = Counter([' '.join(g) for g in grams])
            self._add_col = "{}-gram".format(n)
            self._active_col = "count"
            return [{self._add_col: g,
                    self._active_col: c} for g, c in counts.items()]
        return self._process_articles(_ngrams_func, df_out=True)
    
    def analysis_by(self, *args, df = None, func = None, group_by = "year"):
        if func:
            df = func(*args)
        elif df is None:
            print("analysis_by needs a function or a df")
            exit()
        
        

        group_cols = ([group_by] if isinstance(group_by, str) else group_by)
        group_cols = [col for col in (group_cols + [self._add_col]) if col is not None]
        
        grouped = df.groupby(group_cols)[self._active_col].sum().reset_index()
        grouped = grouped.sort_values(by = [self._active_col], ascending=False)

        self._reset_cols()

        return grouped
    
    def ngram_processor(self, n=1):
        def _processor(text, metadata):
            tokens = text.split()
            grams = zip(*[tokens[i:] for i in range(n)])
            counts = Counter([' '.join(g) for g in grams])
            return [{f"{n}-gram": g, "count": c} for g, c in counts.items()]
        return _processor
    
    def full_pipeline(self, group_bys = ["year", ["year", "month"]], ngrams=range(1,4)):
        
        print("Running a full pipeline...")

        processors = []
        for n in ngrams:
            processors.append((f"{n}-gram", self.ngram_processor(n)))
        
        dfs = self.process_all_articles(processors)
        
        updated_dfs = {}
        for df in tqdm(dfs):
            if df.split("-")[1] == "gram":
                cols = [df, "count"]
            elif df == "article-length":
                cols = [None, "length"]
            new_dict = {"df": dfs[df], "cols": cols}
            updated_dfs[df] = new_dict

        
        for df in tqdm(updated_dfs.keys()):
           
            for group_by in tqdm(group_bys):
                if isinstance(group_by, list):
                    file_end = "-".join(group_by)
                else:
                    file_end = group_by

                self._add_col = updated_dfs[df]["cols"][0]
                self._active_col = updated_dfs[df]["cols"][1]
                
                path_out = os.path.join(self.out_dir, "{}-by-{}.csv".format(df, file_end))
                self.analysis_by(df = updated_dfs[df]["df"], group_by = group_by).to_csv(path_out, index=False)



        # article_lens_path = os.path.join(self.out_dir, "article_lengths.csv")
        # article_lens_df = self.article_lengths()
        # article_lens_df.to_csv(article_lens_path, index=False)

        
        # for group_by in tqdm(group_bys, desc="Grouping article lengths and ngrams"):
        #     if isinstance(group_by, list):
        #         file_end = "-".join(group_by)
        #     else:
        #         file_end = group_by
            
        #     article_path_out = os.path.join(self.out_dir, "article_len-{}.csv".format(file_end))
        #     article_grouped = self.analysis_by(func = self.article_lengths, group_by = group_by)
        #     article_grouped.to_csv(article_path_out, index=False)

        #     for n in ngrams:
        #         ngram_path_out = os.path.join(self.out_dir, "ngrams-{}-by-{}.csv".format(n, file_end))
        #         ngram_grouped = self.analysis_by(n, func = self.count_ngrams, group_by=group_by)
        #         ngram_grouped.to_csv(ngram_path_out, index=False)

        
        