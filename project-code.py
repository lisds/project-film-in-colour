#IMPORT LIBRARIES

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('mode.copy_on_write', True)
import numpy as np

#READ THE DATA
df = pd.read_csv("movie_dataset.csv")

#START REPLICATION

#converting str to datetime for 'release_date' column
df['release_date'] = pd.to_datetime(df['release_date'])

#adding 'year' column to original df
df['year'] = df['release_date'].dt.year

#creating a new filtered df with only the years 2006-2017
filtered_df = df[(df['year'] >= 2007) & (df['year'] <= 2016)]

#drop duplicate films
filtered_df.drop_duplicates(subset='title', inplace = True)

#drop animated films
filtered_df.drop(filtered_df[filtered_df['genres'].str.contains('Animation', na=False)].index, inplace=True)

#dropna
filtered_df.dropna(subset=['title', 'revenue'], inplace=True)

#drop values without english as the original language
filtered_df.drop(filtered_df[filtered_df['original_language'] != 'en'].index, inplace=True)

#drop all rows without revenue data
filtered_df.drop(filtered_df[filtered_df['revenue'] == 0].index, inplace=True)

#create a df with only films that have been produced in the UK, US or Canada
# western countries list
target_countries = ['United Kingdom', 'United States of America', 'Canada']

#function to check if any target country is present in the row
def has_target_country(row):
    countries = [entry['name'] for entry in eval(row)]
    return any(target_country in countries for target_country in target_countries)

#apply function to'production countries' column and create new df
western_df = filtered_df[filtered_df['production_countries'].apply(has_target_country)]

# create a function to find films with English as the first lang
#function to filter rows based on the first language being English
def has_english_first_language(row):
    languages = eval(row) 
    return languages[0]['name'] == 'English' if languages else False

#apply function to 'spoken languages' and create new df
english_df = western_df[western_df['spoken_languages'].apply(has_english_first_language)]

#finds films with 0 budget
zero_budget = english_df[english_df['budget']==0]

#create new df for films with 0 budget
zero_budget_movies = zero_budget[[ 'year', 'id', 'title']]

#manually add budget using data from TheNumbers.com
#create a dictionary with movie titles as keys and research budget values as values
budget_dict = {
    'The Campaign': 86907746,
    'Carriers': 120866,
    'Wild Hogs': 168213584,
    'Semi-Pro': 33479698,
    'Lucky You': 5755286}

#create a new column for 'estimate budget' and map the dictionary to fill the missing values
zero_budget_movies['estimate budget'] = zero_budget_movies['title'].map(budget_dict)

print(zero_budget_movies)

#merge with english_df


#groupby year and find top 30 films with highest budget
#top30_by_year = english_df.groupby('year').apply(lambda group: group.nlargest(30, 'budget'))

