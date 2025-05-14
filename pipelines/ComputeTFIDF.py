from pipelines.utilities import ArticleProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import os
from tqdm import tqdm

class computeTFIDF(ArticleProcessor):
    def __init__(self, article_dir, ngram_range= (2,3), create_metadata=True, max_features = 10000, stop_words="english", articles = None, test_run=False, similarities_above=0.3, article_length_min = 0):
        """If not using stopwords, set stop_words to None. If an output_path is provided then the class will run the pipeline in full and export to the outputpath
        similarities_above - take only similarities with a cosine above the set value - to avoid very large outputs containing materials that are not similar"""
        # Initialise the parent class
        super().__init__(article_dir)

        self.article_dir = article_dir
        if test_run:
            print("Testing with 50 articles")
            self.article_paths = os.listdir(article_dir)[:50]
        else:
            self.article_paths = os.listdir(article_dir)
        self.ngram_range = ngram_range
        self.metadata_list = []
        self.create_metadata = create_metadata
        self.max_features = max_features
        self.stop_words = stop_words
        if articles is not None:
            self.articles = articles
        else:
            self.articles = []
        self.main_fieldname = "filename"
        self.similarities_above = similarities_above
        self.article_length_min = article_length_min

    
    def process_input(self):
        """Create list of articles for tfidf input. If needed, create a metadata, path dictionary
        for merging with final output [{"articlepath": path, "date": date, "title": title}]"""
        new_file_list = []
        for path in tqdm(self.article_paths, desc="Building the input"):
            filepath = os.path.join(self.article_dir, path)
            with open(filepath, "r", encoding='utf-8') as f:
                text = f.read()
            if len(text.split()) > self.article_length_min:
                self.articles.append(text)
                new_file_list.append(path)
            if self.create_metadata:
                metadata_dict = {self.main_fieldname: path}
                metadata_dict.update(self.fetch_title(text))
                metadata_dict.update(self.publication_date(filepath))
                self.metadata_list.append(metadata_dict)
            self.article_paths = new_file_list                
    
    def create_tfidf_matrix(self):
        """Use sklearn to create tfidf representations for each article and then use cosine_similarity
        to align the articles in a matrix. stop_words takes a none type - so if you do not want to use
        stop_words, set stop_words to None"""

        # Initialise vectoriser
        print("Initialising vectorizer...")
        vectorizer = TfidfVectorizer(ngram_range = self.ngram_range, max_features = self.max_features, stop_words = self.stop_words)
        # Compute tfidf matrix
        print("Computing tfidf matrix...")
        tfidf_matrix = vectorizer.fit_transform(self.articles)

        # Calculate similarities with cosine
        print("Caclulating cosine similarity...")
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Suggested ChatGPT update:
        # c. Optional: Save TF-IDF matrix or vectorizer
        # Let user save/load the fitted vectorizer or raw matrix (can be used for more advanced visualizations or clustering).

        return similarity_matrix

    def create_column_name(self, key_no, new_base_string = None):
        """For uniformity and to control key names - we always create the fields for article-1, article-2 using this function
        Function is also used to build new column names if needed using new_base_string"""
        if new_base_string is None:
            return f"{self.main_fieldname}-{key_no}"
        else:
            return f"{new_base_string}-{key_no}"

    def check_merge_error(self, df_left, df_right, left_key, right_key):
        missing_keys = set(df_left[left_key]) - set(df_right[right_key])
        if missing_keys:
            raise ValueError(f"Missing metadata for articles: {missing_keys}")

    def perform_merge(self, df_left, df_right, key_no):
        """key_no is the side of the pairwise relationship - e.g. article-{1} vs. article-{2}"""
        rename_dict = {}
        for column_name in df_right.columns:
            if column_name != self.main_fieldname:
                new_col = self.create_column_name(key_no, new_base_string = column_name)
                rename_dict[column_name] = new_col
        df_right = df_right.rename(columns=rename_dict) 
        left_key = self.create_column_name(key_no)
        self.check_merge_error(df_left, df_right, left_key, self.main_fieldname)
        merged = pd.merge(df_left, df_right, left_on = left_key, right_on = self.main_fieldname)
        # Drop duplicated main_fieldname column
        
        if self.main_fieldname in merged.columns:
            merged = merged.drop(columns=[self.main_fieldname])
        return merged

    def merge_metadata(self, pairwise_df):
        print("Merging metadata...")
        # Transform list of dictionaries into metadata df
        metadata_df = pd.DataFrame(self.metadata_list)

        # Merge on filename-1 and filename2
        for i in range(1,3):
            pairwise_df = self.perform_merge(pairwise_df, metadata_df, i)


        # Return merged df
        return pairwise_df


    def matrix_to_df(self, similarity_matrix):
        """Take the tfidf matrix created by sklearn and convert it to a pairwise df. Add metadata if it
        is not an empty list"""
        
        print("Converting matrix to pairwise file...")
        # Turn similarity_matrix into a df
        df_sim = pd.DataFrame(similarity_matrix, index=self.article_paths, columns=self.article_paths)

        # Turn df_sim into a pairwise_df
        
        # Create the column list using the defined functions - this helps us avoid column mismatches at the merge point
        column_list = []
        for i in range(1,3):
            column_list.append(self.create_column_name(i))
        column_list.append("similarity")
        
        # Transform matrix into pairwise
        pairwise_df = df_sim.stack().reset_index()

        # Add new columns to pairwise
        pairwise_df.columns = column_list

        # Remove articles that are the same and make uni-directional
        # This line only allows cases where filename1 is less than filename2 - removes cases where filename1 and filename2 are the same and will make
        # the relationship chronological
        pairwise_df = pairwise_df[pairwise_df[column_list[0]] < pairwise_df[column_list[1]]]

        # Keep only similarities above a certain amount - to reduce file size
        print(f"Removing similarity scores below {self.similarities_above}")
        print(f"Length before pruning: {len(pairwise_df)}")
        pairwise_df = pairwise_df[pairwise_df["similarity"] > self.similarities_above]
        print(f"Length after pruning: {len(pairwise_df)}")

        # Sort values by similarity - largest first
        pairwise_df = pairwise_df.sort_values(by=["similarity"], ascending=False)

        # If metadata exists - merge that
        if len(self.metadata_list) > 0:
            pairwise_df = self.merge_metadata(pairwise_df)
        
        # Return the final pairwise_df
        return pairwise_df
    
    def run_tfidf(self, csv_path = None):
        """Function to run full tfidf pipeline and return the df or export to csv"""
        self.process_input()
        similarity_matrix = self.create_tfidf_matrix()
        pairwise_df = self.matrix_to_df(similarity_matrix)
        pairwise_df.to_csv(csv_path, index=False)
