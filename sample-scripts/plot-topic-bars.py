import pandas as pd
import plotly.express as px

# Load in the topic model dataset
csv = "../data_out/articles-data/topic-model/topic-model.csv"

df = pd.read_csv(csv)

# Remove outlier topic
df = df[df["Topic"] != -1]
print(df.columns)


df["Topic"] = df["Topic"].astype(str)
df["year"] = df["year"].astype(str)
df["month"] = df["month"].astype(str)
df["Topic label"] = df[["topic_1", "topic_2", "topic_3", "topic_4"]].apply(", ".join, axis=1)


# Get top 10 topics
top_10 = df[["Topic", "Count"]].drop_duplicates().sort_values(by = ["Count"], ascending = False).iloc[:10]
# Convert to list of topic ids
top_10 = top_10["Topic"].tolist()


top_10_df = df[df["Topic"].isin(top_10)].drop_duplicates()

# Count years using group_by - get years_top_10 and months_top_10
years_top_10 = top_10_df.groupby(['Topic', 'Topic label', 'year']).size().reset_index(name='Count')
months_top_10 = top_10_df.groupby(['Topic', 'Topic label', 'year', 'month']).size().reset_index(name='Count')

years_top_10 = years_top_10.sort_values(by=["Topic", "year"])
months_top_10 = months_top_10.sort_values(by=["Topic", "year", "month"])

top_df = top_10_df[["Topic label", "Count"]].drop_duplicates()

# Plot a bar
fig = px.bar(top_df, x="Topic label", y="Count")
fig.write_html("top_10_topic_bar.html")

# Plot a stacked bar
fig = px.bar(years_top_10, x="Topic label", y="Count", color="year")
fig.write_html("top_10_topic_years_bar.html")

# Plot a line chart
fig = px.line(years_top_10, x="year", y="Count", color="Topic label")
fig.write_html("top_10_topic_years_line.html")

# Plot year subplots
fig = px.bar(years_top_10, x="Topic label", y="Count",
             facet_row="year")
fig.write_html("top_10_topic_years_facet_bar.html")

# Plot month facet columns - for 2023
months_2023 = months_top_10[months_top_10["year"] == '2023'].drop_duplicates()
fig = px.bar(months_2023, x="Topic", y="Count",
             facet_col="month")
fig.write_html("top_10_topic_2023_month_facet.html")

# Plot year month subplots
fig = px.bar(months_top_10, x="Topic", y="Count",
             facet_row="year", facet_col="month")
fig.write_html("top_10_topic_year_month_facet.html")

