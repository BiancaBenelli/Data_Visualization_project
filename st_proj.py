# py -m streamlit run st_proj.py

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import iplot
from bubbly.bubbly import bubbleplot 
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Create a dynamic page
st.set_page_config(layout= 'centered')
st.header('Data Visualization project')
st.subheader('Movies throughout the years, from the early nineteenth century to current day.')

film_df = pd.read_csv('filmtv_movies - ENG.csv')
st.write('Data about movies available on IMDb and other websites has been combined together to create this dataset.')
st.write('The file in the English version contains 40.047 movies and 19 attributes, while the Italian version contains one extra-attribute for the local title used when the movie was published in Italy; I used the English version to be coherent with the language of the course.')

st.write('The dataframe:')
st.write(film_df)
st.write('Rows and columns:', film_df.shape)
st.write('Dataframe tail:')
st.write(film_df.tail())

st.write('Some informations:')
st.write(film_df.describe())

# Drop columns that are not needed
film_df.drop(['directors','actors', 'notes', 'rhythm', 'humor', 'effort', 'tension', 'erotism'], axis=1, inplace=True)
# Drop the rows with null values 
film_df = film_df.dropna(subset=['genre'])
film_df = film_df.dropna(subset=['country'])

st.title('The plots')

st.subheader('Word clouds')
film_df['description'] = film_df['description'].astype(str)
description_column = film_df['description']
# Combine all the descriptions into a single string
all_descriptions = ' '.join(description_column)
# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

fig = plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud from all Film descriptions")
st.write(fig)

# Group the DataFrame by genre
grouped_by_genre = film_df.groupby('genre')
num_plots = len(grouped_by_genre)
num_cols, num_rows = 3, 9

fig, axs = plt.subplots(nrows = 9, ncols = 3, figsize = (20, 15), constrained_layout = True)
axs = axs.flatten()
# Loop through each genre group and plot the word cloud
for i, (genre, group_df) in enumerate(grouped_by_genre):
    all_descriptions = ' '.join(group_df['description'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

    # Display the word cloud in the current subplot
    axs[i].imshow(wordcloud, interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(f"Word Cloud for {genre} Films", fontsize=15)

# Hide any remaining empty subplots
for i in range(num_plots, num_cols * num_rows):
    fig.delaxes(axs[i])

# Adjust space between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.tight_layout()
st.write(fig)

st.subheader('Movies per decade')
# Create a new column for the decade
film_df['decade'] = (film_df['year'] // 10) * 10
# Group the DataFrame by decade and count the number of movies
# I also group by genre in order to later use it to color my bar chart
films_per_decade = film_df.groupby(['decade', 'genre']).size().reset_index(name='count')

fig = px.bar(films_per_decade, x='decade', y='count', color='genre', labels={'count': 'Number of Movies'},
title='Number of Movies by decade')
fig.update_layout(xaxis_title='Decade', yaxis_title='Number of Movies', xaxis=dict(type='category'),  # Use categorical x-axis for decades
showlegend=False, height=600)
st.write(fig)

st.subheader('Movies produced in the United States')
# Group the filtered DataFrame by year, country, and genre, and count the number of films
film_count = film_df.groupby(['year', 'country', 'genre']).size().reset_index(name='count')
# Find the top country by number of films produced
top_country = film_count.groupby('country')['count'].sum().idxmax()
# Filter the data for the top country
film_count_top_country = film_count[(film_count['country'] == top_country)]

fig = px.bar(film_count_top_country, x='year', y='count', color='genre',labels={'count': 'Number of Films'},
title=f'Number of Films per Year in {top_country}')
fig.update_layout(xaxis_title='Year', yaxis_title='Number of Films', xaxis=dict(type='category'),  # Use categorical x-axis for years
showlegend=True, legend_title='Genre', height=600)
st.write(fig)

st.subheader('Films by country')
# Group the DataFrame by year, country and count the number of films
films_by_year_country = film_df.groupby(['year', 'country']).size().reset_index(name='count')
# Create an animated choropleth map using Plotly Express
fig = px.choropleth( films_by_year_country, locations='country', locationmode='country names', color='count', hover_name='country',
color_continuous_scale= px.colors.sequential.PuRd, labels={'count': 'Number of Films'}, title='Number of Films by Country', animation_frame='year')
fig.update_layout(geo=dict(showcoastlines=True), height=600)
st.write(fig)

st.subheader('Genre by country')
# Focus on the top 5 countries by number of films produced
top_countries = film_df['country'].value_counts().head(5).index
# Define the number of top genres to consider (otherwise I would have too many small slices)
top_genre_count = 10
# Create pie charts for each of the top countries and their top 10 genres
for country in top_countries:
    country_df = film_df[film_df['country'] == country]
    top_genres = country_df['genre'].value_counts().head(top_genre_count)
    fig = go.Figure(data=[go.Pie(labels=top_genres.index, values=top_genres.values)])
    fig.update_layout(title=f'Top {top_genre_count} Genres in {country}', showlegend=True)
    st.write(fig)

st.subheader('Popularity of genres through the years')
genre_decade_counts = film_df.groupby(['genre', 'decade']).size().reset_index(name='film_count')
fig = px.line(genre_decade_counts, x='decade', y='film_count', color='genre', labels={'film_count': 'Number of Films', 'decade': 'Decade'},
title='Number of Films Produced in Each Genre Each Decade')
fig.update_traces(mode="markers+lines", hovertemplate=None)
fig.update_layout(xaxis_title='Decade', yaxis_title='Number of Films', height=600)
st.write(fig)

filtered_films = film_df[(film_df['country'] == top_country) & (film_df['year'] >= 2000) & (film_df['year'] <= 2022)]
genre_counts = filtered_films['genre'].value_counts().reset_index()
genre_counts.columns = ['genre', 'count']
# Filter out genres 
filtered_genres = genre_counts[genre_counts['count'] > 10]['genre']
filtered_films = filtered_films[filtered_films['genre'].isin(filtered_genres)]
fig = px.density_heatmap(filtered_films, x='year', y='genre', labels={'count': 'Number of Films'},
title=f'Density Heatmap of Films by Year and Genre in {top_country}', height=600,
color_continuous_scale='Agsunset')
fig.update_layout( xaxis_title='Year', yaxis_title='Genre', xaxis=dict(type='category'), showlegend=True,
legend_title='Number of Films')
st.write(fig)
 
st.subheader('Movies by their critics vote')
# Filter the DataFrame for the top_country
top_country_df = film_df[film_df['country'] == top_country].copy()
# Convert the 'year' column to datetime using .loc
top_country_df.loc[:, 'year'] = pd.to_datetime(top_country_df['year'], format='%Y')
fig = px.scatter(top_country_df, x='year', y='critics_vote', color='genre', hover_name='title', labels={'rating': 'Rating'},
title=f'Rating of Films in {top_country}')
fig.update_layout(xaxis_title='Year', yaxis_title='Rating', showlegend=True, legend_title='Genre', height=600)
st.write(fig)

fig = px.scatter(top_country_df, x='year', y='public_vote', color='genre', hover_name='title', labels={'rating': 'Rating'},
title=f'Rating of Films in {top_country}')
fig.update_layout(xaxis_title='Year', yaxis_title='Rating', showlegend=True, legend_title='Genre', height=600)
st.write(fig)

st.subheader('Critics VS public votes')
top_10_critics_films = film_df.sort_values(by='critics_vote', ascending=False).head(25)
fig = px.bar( top_10_critics_films, x='title', y=['critics_vote', 'public_vote'], labels={'title': 'Movie Title', 'value': 'Vote'},
title='Critics VS Public Vote for top 10 Films by Critics Vote')
fig.update_layout( xaxis_title='Movie Title', yaxis_title='Vote', legend_title='Vote Type', barmode='group', xaxis_tickangle= 45)
st.write(fig)
top_10_critics_films = film_df.sort_values(by='public_vote', ascending=False).head(25)
fig = px.bar( top_10_critics_films, x='title', y=['public_vote','critics_vote'], labels={'title': 'Movie Title', 'value': 'Vote'},
title='Critics VS Public Vote for top 10 Films by Public Vote')
fig.update_layout( xaxis_title='Movie Title', yaxis_title='Vote', legend_title='Vote Type', barmode='group', xaxis_tickangle= 45)
st.write(fig)

mean_votes_per_year = film_df.groupby('year')[['public_vote', 'critics_vote']].mean().reset_index()
fig = px.line(mean_votes_per_year, x='year', y=['public_vote', 'critics_vote'], labels={'value': 'Mean Rating', 'year': 'Year'},
title='Mean Ratings Over the Years')
fig.update_layout(yaxis_title='Mean Rating', height=600)
st.write(fig)

st.subheader('Average rating per year')
top_5_genres = film_df['genre'].value_counts().head(5).index
filtered_df = film_df[film_df['genre'].isin(top_5_genres)]
filter_films = filtered_df[(filtered_df['year'] >= 1980) & (filtered_df['year'] <= 2022)]
mean_critics_vote_per_genre_per_year = filter_films.groupby(['year', 'genre'])['critics_vote'].mean().reset_index()

fig = px.scatter_3d(mean_critics_vote_per_genre_per_year, x='year', y='genre', z='critics_vote', color='critics_vote',
labels={'critics_vote': 'Mean Critics Vote'}, title='Mean Critics Vote per Genre per Year')
fig.update_layout(scene=dict(yaxis_title='Genre', zaxis_title='Mean Critics Vote'), height=800)
st.write(fig)

mean_public_vote_per_genre_per_year = filter_films.groupby(['year', 'genre'])['public_vote'].mean().reset_index()

fig = px.scatter_3d(mean_public_vote_per_genre_per_year, x='year', y='genre', z='public_vote', color='public_vote',
labels={'public_vote': 'Mean Public Vote'}, title='Mean Public Vote per Genre per Year')
fig.update_layout(scene=dict(yaxis_title='Genre', zaxis_title='Mean public Vote'), height=800)
st.write(fig)

st.subheader('Films by duration')
sorted_df = film_df.sort_values(by='duration', ascending=False)
top_10_longest = sorted_df.head(10)

fig = px.bar( top_10_longest, x='duration', y='title', orientation='h', color='year', title='Top 10 Longest Films by Duration',
labels={'duration': 'Duration (minutes)'}, color_continuous_scale='Rainbow')
st.write(fig)

duration_bins = [0, 90, 120, 150, float('inf')]
duration_labels = ['Short (<90 mins)', 'Medium (90-120 mins)', 'Long (120-150 mins)', 'Very Long (>150 mins)']
film_df['duration_category'] = pd.cut(film_df['duration'], bins=duration_bins, labels=duration_labels, right=False)
duration_counts = film_df['duration_category'].value_counts()

fig = go.Figure(data=[go.Pie(labels=duration_counts.index, values=duration_counts.values)])
fig.update_layout( title='Film Duration Distribution',showlegend=True,)
fig.update_traces(hoverinfo='label+percent', pull=[0.1, 0, 0.2, 0, 0, 0])
st.write(fig)