import pandas as pd
import plotly.express as px
import os
from tqdm import tqdm

articles_path = "../data_in/articles/"
comp_month = "8"
comp_year = "2023"

year_month_count = {}
all_articles = []

for article_path in tqdm(os.listdir(articles_path)):
    file_split = article_path.split("-")
    year = file_split[0]
    month = file_split[1]
    year_month = year + "-" + month

    full_path = os.path.join(articles_path, article_path)

    with open(full_path, "r", encoding='utf-8') as f:
        text = f.read()
    
    word_count = len(text.split())

    if year_month in year_month_count.keys():
        year_month_count[year_month]["word_count"] += word_count
        year_month_count[year_month]["article_count"] += 1
    else:
        year_month_count[year_month] = {"word_count": word_count, "article_count": 1}
    distance = (int(year)-int(comp_year))*12 + (int(month)-int(comp_month))
    all_articles.append({"months_from_october_2023": distance, "article_word_count": word_count})

# out_dicts =[]    
# for key, value in year_month_count.items():
#     split = key.split("-")
#     year = split[0]
#     month = split[1]
#     distance = (int(year)-int(comp_year))*12 + (int(month)-int(comp_month))
#     out_dicts.append({"months_from_october_2023": distance, "year-month" : key, "article_word_count": value["word_count"], "article_count": value["article_count"]})

# out_df = pd.DataFrame(out_dicts)

# out_df.to_csv("months_from_october.csv", index=False)

all_articles_df = pd.DataFrame(all_articles)

fig = px.scatter(all_articles_df, x="months_from_october_2023", y="article_word_count")
fig.show()


