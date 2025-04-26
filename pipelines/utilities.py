import os
import re
import pandas as pd
from tqdm import tqdm

class ArticleProcessor():
    def __init__ (self, dir):
        self.data = []
        self.files = []
        for root, dirs, files in os.walk(dir):
            for name in files:
                self.files.append(os.path.join(root, name))
    
    def publication_date(self, filename, date_patterns = {"year": "\d{4}", "month": "-(\d{2})-", "day": "-(\d{2})_"}):
        out = {}
        for key, pattern in date_patterns.items():
            match = re.findall(pattern, filename)
            out[key] = match[0] if match else None
        return out

    def _process_articles(self, func, df_out = False):
        out = []
        for file in tqdm(self.files):
            
            

            # Run process on article
            with open(file, "r", encoding='utf-8') as f:
                text = f.read()

            # Clean text
            text = re.sub("\W+", " ", text).lower()
            
            _process_results = func(text)

            # Loop through results of that process and add them to row
            if isinstance(_process_results, dict):
                # Start row by producing metadata columns
                row = self.publication_date(file)
                row.update(_process_results)
            elif isinstance(_process_results, list):
                for result in _process_results:
                    row = self.publication_date(file)
                    row.update(result)
            
            else:
                print("Function needs to produce a dictionary")
                exit()

            # Add row to out
            out.append(row)
        
        if df_out:
            out = pd.DataFrame(out)
        
        return out
    
    def process_all_articles(self, processors, df_out=True):
        """
        Process all articles using multiple processor functions at once.
        
        Args:
            processors: A list of (name, function) tuples
                        where each function takes (text, metadata) and returns dict or list of dicts.
            df_out: Whether to return outputs as DataFrames (default True).
        
        Returns:
            A dict mapping processor names to lists of outputs or DataFrames.
        """
        outputs = {name: [] for name, _ in processors}

        for file in tqdm(self.files):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()

            metadata = self.publication_date(file)

            for name, func in processors:
                result = func(text, metadata)

                if isinstance(result, dict):
                    row = metadata.copy()
                    row.update(result)
                    outputs[name].append(row)

                elif isinstance(result, list):
                    for r in result:
                        row = metadata.copy()
                        row.update(r)
                        outputs[name].append(row)

                else:
                    raise ValueError(f"Processor '{name}' must return dict or list of dicts.")

        if df_out:
            outputs = {name: pd.DataFrame(rows) for name, rows in outputs.items()}

        return outputs

        



