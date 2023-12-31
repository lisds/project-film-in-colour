#IMPORT LIBRARIES

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('mode.copy_on_write', True)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from scipy.stats import pearsonr


#READ THE DATA
df = pd.read_csv("data/movie_metadata.csv")

#START REPLICATION
#creating a new filtered df with only the years 2006-2017
filtered_df = df[(df['title_year'] >= 2007) & (df['title_year'] <= 2016)]

#drop duplicate films
filtered_df.drop_duplicates(subset='movie_title', inplace = True)

#drop animated films
filtered_df.drop(filtered_df[filtered_df['genres'].str.contains('Animation', na=False)].index, inplace=True)

#dropna for gross and movie title
filtered_df.dropna(subset=['movie_title', 'gross'], inplace=True)

#drop values for films not in the USA
filtered_df.drop(filtered_df[filtered_df['country'] != 'USA'].index, inplace=True)

#drop values without english as the original language
filtered_df.drop(filtered_df[filtered_df['language'] != 'English'].index, inplace=True)

#drop all rows without revenue data
filtered_df.drop(filtered_df[filtered_df['gross'] == 0].index, inplace=True)

#finds films with no budget values
zero_budget = filtered_df[pd.isna(filtered_df['budget'])]

#create new df for films with no budget
zero_budget_df = zero_budget[[ 'movie_title', 'title_year']].copy()

#strip white space
zero_budget_df['movie_title'] = zero_budget_df['movie_title'].str.strip().astype(str)

#budget dictionary
#manually add budget using data from TheNumbers.com
#create a dictionary with movie titles as keys and research budget values as values
budget_dict = {
    'The Campaign': 86907746,
    'Carriers': 120866,
    'Wild Hogs': 168213584,
    'Semi-Pro': 33479698,
    'Lucky You': 5755286,
    'Anna Karenina': 12816367,
    'Here Comes the Boom': 45290318,
    'The Odd Life of Timothy Green': 51853450,
    'Because I Said So': 42674040,
    'Get on Up': 30569935,
    'Dead Man Down': 10895295,
    'Hot Rod': 13938332,
    'Cirque du Soleil: Worlds Away': 12512862,
    'For Colored Girls': 37729698,
    'Strange Wilderness': 6575282,
    'Regression': 55039,
    'The Book Thief': 21488481,
    "Won't Back Down": 5310554,
    'Observe and Report': 24007324,
    'Madea Goes to Jail': 90508336,
    'McFarland, USA': 44480275,
    'Mr. Turner': 3958546,
    'Top Five': 25317379,
    'Just Wright': 21540363,
    'About Time': 15323921,
    'Cedar Rapids' : 6861102,
    'The Collection': 6810754,	
    'Welcome to the Rileys': 152857,
    'Free Style': 141108,
    'Repo! The Genetic Opera': 140244,
    'Enough Said': 17550872,
    'When Did You Last See Your Father?': 1071240,
    'Definitely, Maybe':32241649,
    'Dom Hemingway' : 523511,
    'Control' : 871577,
    'The Art of Getting By': 1430241,
    'Adam' : 2283291,
    'Slow West' : 2290194,
    'Frances Ha' : 4067398,
    'Kevin Hart: Let Me Explain': 32244051,
    "Meek's Cutoff" : 977772,
    'Eden Lake' : 7321,
    'Submarine': 467602,
    'Man on Wire' : 2962242,
    'Snow Angels': 402858,
    'The Greatest Movie Ever Sold': 638476,
    'Mooz-lum' : 362239,
    'Eddie: The Sleepwalking Cannibal': 1632,
    'Chernobyl Diaries' : 18119640,
    'October Baby' : 5355847,
    'The Skeleton Twins' : 5284309,
    'Martha Marcy May Marlene': 2981038,
    'Obvious Child' : 3122616,
    'I Origins' : 336472,
    'The Perfect Host' : 48764,
    'Fruitvale Station' : 16098998,
    'Celeste & Jesse Forever': 3103407,
    'Drinking Buddies' : 343706,
    'Good Kill' : 317072,
    'Blue Ruin' : 258384,
    'Compliance' :319285,
    'The Finest Hours': 27569558,
    'Dinner for Schmucks': 73026337,
    'State of Play': 37017955,
    'Trouble with the Curve': 35763137,
    'We Bought a Zoo': 75624550,
    'Get Him to the Greek' : 61153526,
    'Music and Lyrics': 50572589,
    'Soul Men' : 12082391,
    'Because I Said So': 42674040,
    'Wanderlust': 17288155,
    'The Three Stooges'	: 44338224,
    'The Sitter': 30542576,
    'Hot Rod': 13938332, 
    'Cirque du Soleil: Worlds Away': 12512862,
    'In the Valley of Elah': 6777741,
    'For Colored Girls': 37729698,
    'Orphan': 41596251,
    'The Stepfather':29062561,
    'Appaloosa': 20211394,
    'Capitalism: A Love Story': 14363397,
    'All Good Things':582024,
    'Metallica Through the Never': 3419967,
    'Lottery Ticket' :24719879,
    'U2 3D' : 12898847,
    'Blood and Chocolate': 3526588,
    'I Think I Love My Wife': 12559771,
    'Rachel Getting Married' :12769861,
    'Darling Companion': 793352,
    'The Goods: Live Hard, Sell Hard': 15122676,
    "World's Greatest Dad": 221805,	
    'Blood Done Sign My Name':109383,
    'Adventureland': 16044025,	
    'Gentlemen Broncos': 115155,	
    'The Savages' : 6623082,
    'I Hope They Serve Beer in Hell': 1429299,	
    'Trade of Innocents': 15091,	
    'Standing Ovation': 531806,	
    'Standard Operating Procedure': 228830,	
    'Life During Wartime': 281447, 
    'Please Give': 4033574,	
    'Damsels in Distress': 1007535,	
    'Redemption Road': 29384,
    'Inside Job': 4311834,	
    'Teeth': 347578,
    'The Last Five Years': 147299,	
    'Plush': 3080,
    'Supercapitalist': 15919,
    'All Is Bright': 4556,
    'Higher Ground': 841056,	
    'Grace Unplugged': 2507159,	
    'N-Secure': 2595644,	
    'Shine a Light': 5505267,	
    'Food, Inc.': 4417674,	
    'Palo Alto': 767732,	
    'Kevin Hart: Laugh at My Pain': 7706436,	
    'Escape from Tomorrow': 171962,
    'Water & Power': 42557,	
    'The Looking Glass': 1711,
    'Shotgun Stories':  46026,
    'Good Dick':28835,	
    'Sound of My Voice': 408015,	
    'The Exploding Girl': 25572,	
    'Shanghai Calling': 10443,
    }


# Create a new column for 'estimate budget' and map the dictionary to fill the missing values
zero_budget_df['estimate budget'] = zero_budget_df['movie_title'].map(budget_dict)

#map onto the cleaned dataframe
filtered_df['movie_title'] = filtered_df['movie_title'].str.strip().astype(str)
filtered_df['budget'] = filtered_df['budget'].fillna(filtered_df['movie_title'].map(budget_dict))

#drop the Nan value
filtered_df = filtered_df.dropna(subset=['budget'])

#groupby year and find top 10 films with highest budget
top10_by_year = filtered_df.groupby('title_year').apply(lambda group: group.nlargest(10, 'budget'))

#create top actors df with top 10 films
top_actors = top10_by_year.loc[:,['movie_title','title_year','content_rating','actor_1_name','actor_2_name','actor_3_name', 'budget','gross', 'genres']]
top_actors.reset_index(drop=True, inplace=True)
top_actors['actor_1_name'].str.strip()
top_actors['actor_2_name'].str.strip()
top_actors['actor_3_name'].str.strip()
top_actors['movie_title'].str.strip()

# Manually find the diversity data and create a dict
# Unique actors dictionary
# Name: Gender(F/N/M), Race (W/B/H/A/O), Birth Year, STARPower 10 weeks before release
diversity_dict = {
    'Liam James': ('M', 'W', 1996),
    'Tom McCarthy': ('M', 'W', 1966),
    'Cary-Hiroyuki Tagawa': ('M', 'A', 1950),
    'Keanu Reeves': ('M', 'O', 1964),
    'Jin Akanishi': ('M', 'A', 1984),
    'Ayelet Zurer': ('F', 'O', 1969),
    'Tom Hanks': ('M', 'W', 1956),
    'Armin Mueller-Stahl': ('M', 'W', 1930),
    'T.I.': ('M', 'B', 1980),
    'CCH Pounder': ('F', 'B', 1952),
    'Wes Studi': ('M', 'A', 1947),
    'Joel David Moore': ('M', 'W', 1977),
    'Alan D. Purwin': ('M', 'W', 1961),
    'Lauren Cohan': ('F', 'W', 1982),
    'Tadanobu Asano': ('M', 'A', 1973),
    'Gary Oldman': ('M', 'W', 1958),
    'Kodi Smit-McPhee': ('M', 'W', 1996),
    'Lara Pulver': ('F', 'W', 1980),
    'Noah Taylor': ('M', 'W', 1969),
    'Steve Carell': ('M', 'W', 1962),
    'Jimmy Bennett': ('M', 'W', 1996),
    'Andre Braugher': ('M', 'B', 1962),
    'Ioan Gruffudd': ('M', 'W', 1973),
    'Dwayne Johnson': ('M', 'O', 1972),
    'Jason Statham': ('M', 'W', 1967),
    'Dennis Quaid': ('M', 'W', 1954),
    'Leo Howard': ('M', 'W', 1997),
    'Peter Fonda': ('M', 'W', 1940),
    'Matt Long': ('M', 'W', 1980),
    'Kate McKinnon': ('F', 'W', 1984),
    'Ed Begley Jr.': ('M', 'W', 1949),
    'Zach Woods': ('M', 'O', 1984),
    'Temuera Morrison': ('M', 'O', 1960),
    'Taika Waititi': ('M', 'O', 1975),
    'Djimon Hounsou': ('M', 'B', 1964),
    'Bradley Cooper': ('M', 'W', 1975),
    'Chloë Grace Moretz': ('F', 'W', 1997),
    'Willow Smith': ('F', 'B', 2000),
    'Alice Braga': ('F', 'H', 1983),
    'Vivica A. Fox': ('F', 'B', 1964),
    'Judd Hirsch': ('M', 'W', 1935),
    'Sela Ward': ('F', 'W', 1956),
    'Harrison Ford': ('M', 'W', 1942),
    'Jim Broadbent': ('M', 'W', 1949),
    'Mackenzie Foy': ('F', 'W', 2000),
    'Matthew McConaughey': ('M', 'W', 1969),
    'Don Cheadle': ('M', 'B', 1964),
    'Ewen Bremner': ('M', 'W', 1972),
    'Ralph Brown': ('M', 'W', 1957),
    'Daryl Sabara': ('M', 'W', 1992),
    'Samantha Morton': ('F', 'W', 1977),
    'Polly Walker': ('F', 'W', 1966),
    'Eddie Redmayne': ('M', 'W', 1982),
    'Channing Tatum': ('M', 'W', 1980),
    'Omar Sy': ('M', 'B', 1978),
    'Sharlto Copley': ('M', 'W', 1973),
    'Angelina Jolie Pitt': ('F', 'W', 1975),
    'Sam Riley': ('M', 'W', 1980),
    'Christopher Meloni': ('M', 'W', 1961),
    'Harry Lennix': ('M', 'B', 1964),
    'Nicole Scherzinger': ('F', 'O', 1978),
    'Michael Stuhlbarg': ('M', 'W', 1968),
    'Michael Nyqvist': ('M', 'W', 1960),
    'Jeremy Renner': ('M', 'W', 1971),
    'Tim Holmes': ('M', 'W', 1967),
    'Larry Joe Campbell': ('M', 'W', 1970),
    'Clifton Collins Jr.': ('M', 'H', 1970),
    'Charlie Hunnam': ('M', 'W', 1980),
    'Nonso Anozie': ('M', 'B', 1978),
    'Cara Delevingne': ('F', 'W', 1992),
    'Orlando Bloom': ('M', 'W', 1977),
    'Jack Davenport': ('M', 'W', 1973),
    'Stephen Graham': ('M', 'W', 1973),
    'Jake Gyllenhaal': ('M', 'W', 1980),
    'Richard Coyle': ('M', 'W', 1972),
    'Reece Ritchie': ('M', 'W', 1986),
    'Sean Harris': ('M', 'W', 1966),
    'Scott Grimes': ('M', 'W', 1971),
    'Mark Addy': ('M', 'W', 1964),
    'Tzi Ma': ('M', 'A', 1962),
    'Noémie Lenoir': ('F', 'O', 1979),
    'Dana Ivey': ('F', 'W', 1941),
    'Kristen Stewart': ('F', 'W', 1990),
    'Nicholas Elia': ('M', 'W', 1997),
    'Scott Porter': ('M', 'W', 1979),
    'Kick Gurry': ('M', 'W', 1978),
    'Kirsten Dunst': ('F', 'W', 1982),
    'Lydia Wilson': ('F', 'W', 1984),
    'Sofia Boutella': ('F', 'O', 1982),
    'Melissa Roxburgh': ('F', 'W', 1992),
    'Noel Clarke': ('M', 'B', 1975),
    'Benedict Cumberbatch': ('M', 'W', 1976),
    'Leonard Nimoy': ('M', 'W', 1931),
    'Robin Atkin Downes': ('M', 'W', 1976),
    'Ike Barinholtz': ('M', 'W', 1977),
    'James Frain': ('M', 'W', 1968),
    'Olivia Wilde': ('F', 'W', 1984),
    'Emilia Clarke': ('F', 'W', 1986),
    'Matt Smith': ('M', 'W', 1982),
    'Common': ('M', 'B', 1972),
    'Tony Curran': ('M', 'W', 1969),
    'Mackenzie Crook': ('M', 'W', 1971),
    'Toby Jones': ('M', 'W', 1966),
    'B.J. Novak': ('M', 'W', 1979),
    'Chris Zylka': ('M', 'W', 1985),
    'Matt Damon': ('M', 'W', 1970),
    'Albert Finney': ('M', 'W', 1936),
    'Pierfrancesco Favino': ('M', 'W', 1969),
    'Damián Alcázar': ('M', 'H', 1953),
    'Shane Rangi': ('M', 'O', 1969),
    'Laura Brent': ('F', 'W', 1988),
    'Bruce Spence': ('M', 'W', 1945),
    'Jason Flemyng': ('M', 'W', 1966),
    'Julia Ormond': ('F', 'W', 1965),
    'Heath Ledger': ('M', 'W', 1979),
    'Kristin Scott Thomas': ('F', 'W', 1960),
    'Eva Green': ('F', 'W', 1980),
    'Josh Hutcherson': ('M', 'W', 1992),
    'Philip Seymour Hoffman': ('M', 'W', 2014),
    'Peter Mensah': ('M', 'B', 1959),
    'Ty Burrell': ('M', 'W', 1967),
    'Aasif Mandvi': ('M', 'A', 1966),
    'Seychelle Gabriel': ('F', 'H', 1991),
    'Noah Ringer': ('M', 'W', 1996),
    'Christoph Waltz': ('M', 'W', 1956),
    'Casper Crump': ('M', 'W', 1977),
    'Ruth Wilson': ('F', 'W', 1982),
    'Tom Wilkinson': ('M', 'W', 1948),
    'Jet Li': ('M', 'A', 1963),
    'Brendan Fraser': ('M', 'W', 1968),
    'Russell Wong': ('M', 'A', 1963),
    'Lukas Haas': ('M', 'W', 1976),
    'Omar Benson Miller': ('M', 'B', 1978),
    'Robert Capron': ('M', 'W', 1998),
    'Art Malik': ('M', 'A', 1952),
    'Simon Merrells': ('M', 'W', 1965),
    'Natalie Portman': ('F', 'O', 1981),
    'Thomas Robinson': ('M', 'W', 2002),
    'Chris Bauer': ('M', 'W', 1966),
    'Sophia Myles': ('F','W',1980),
    'Bingbing Li': ('F','A',1973),
    'Kelsey Grammer': ('M','W',1955),
    'Lester Speight': ('M','B',1963),
    'Ramon Rodriguez': ('M','H',1979),
    "Michael O'Neill": ('M','W',1951),
    'Zack Ward': ('M','W',1970),
    'Brandon T. Jackson': ('M','B',1984),
    'Callum Rennie': ('M','W',1960),
    'Ruth Negga': ('F','O',1982),
    'Stephen McHattie': ('M','W',1947),
    'Billy Crudup': ('M','W',1968),
    'Matt Frewer': ('M','W',1958),
    'Peter Capaldi': ('M','W',1958),
    'Mireille Enos': ('F','W',1975),
    'Lily James': ('F','W',1989),
    'Dominic Monaghan': ('M','W',1976),
    'Tye Sheridan': ('M','W',1996),
    'Jill Hennessy': ('F', 'W', 1968),
    'Tichina Arnold': ('F', 'B', 1969),
    'Drew Sidora': ('F', 'B', 1985),
    'Johnny Depp': ('M', 'W', 1963),
    'J.K. Simmons': ('M', 'W', 1955),
    'Christopher Lee': ('M', 'W', 1922),
    'Will Smith': ('M', 'B', 1968),
    'Chris Evans': ('M', 'W', 1981),
    'Nicolas Cage': ('M', 'W', 1964),
    'Peter Dinklage': ('M', 'W', 1969),
    'Christian Bale': ('M', 'W', 1974),
    'Brad Pitt': ('M', 'W', 1963),
    'Robert Downey Jr.': ('M', 'W', 1965),
    'Glenn Morshower': ('M', 'W', 1959),
    'Oliver Platt': ('M', 'W', 1960),
    'Joseph Gordon-Levitt': ('M', 'W', 1981),
    'Robin Williams': ('M', 'W', 1951),
    'Hugh Jackman': ('M', 'W', 1968),
    'Chris Hemsworth':('M', 'W', 1983),
    'Jeff Bridges': ('M', 'W', 1949),
    'Leonardo DiCaprio': ('M', 'W', 1974),
    'Anthony Hopkins': ('M', 'W', 1937),
    'Ryan Reynolds': ('M', 'W', 1937),
    'Jennifer Lawrence': ('F', 'W',1990),
    'Tom Cruise': ('M', 'W', 1962),
    'Paul Walker': ('M', 'W', 1973),
    'Tom Hardy': ('M', 'W', 1977),
    'Emma Stone':('F', 'W', 1988),
    'Liam Neeson': ('M', 'W', 1952),
    'Aidan Turner': ('M', 'W', 1977),
    'Michael Fassbender':('M', 'W', 1977),
    'Henry Cavill': ('M', 'W', 1983),
    'Eddie Marsan':('M', 'W', 1968),
    'Scarlett Johansson': ('F', 'W', 1984),
    'Judy Greer': ('F', 'W', 1975),
    'Bryce Dallas Howard': ('F', 'W', 1981),
    'Dominic Cooper':('M', 'W', 1978),
    'James Franco':('M', 'W', 1978),
    'Morgan Freeman': ('M', 'B', 1937),
    'Edgar Ramírez':('M', 'H', 1977),
    'Ray Winstone':('M', 'W', 1957),
    'Charlize Theron': ('F', 'W', 1975),
    'Steve Coogan': ('M', 'W', 1965),
    'Kevin Dunn':('M', 'W', 1956),
    'Rami Malek': ('M', 'W', 1981),
    'William Hurt':('M', 'W', 1950),
    'Alan Rickman': ('M', 'W', 1946),
    'Sam Claflin':('M', 'W', 1986),
    'Vin Diesel': ('M', 'W', 1967),
    'Andrew Garfield': ('M', 'W', 1983),
    'Alexander Skarsgård': ('M', 'W', 1976),
    'Adam Brown':('M', 'W', 1980),
    'Mila Kunis': ('F', 'W', 1983),
    'Jon Favreau': ('M', 'W', 1966),
    'Bruce Greenwood':('M', 'W', 1956),
    'Anne Hathaway': ('F', 'W', 1982),
    'Hayley Atwell':('F', 'W', 1982),
    'James Nesbitt':('M', 'W', 1965)
}

#populate the diversity data from the dict to top_actors df

#function to find the actors diversity data according to row
def get_combined_info(row, info_index):
    actor_list = [row['actor_1_name'], row['actor_2_name'], row['actor_3_name']]
    info_list = []
    for actor in actor_list:
        if actor in diversity_dict:
            info_list.append(diversity_dict[actor][info_index])
        else:
            info_list.append('Unknown')
    return info_list

#apply function to new columns in top_actors df
top_actors['genders'] = top_actors.apply(lambda row: get_combined_info(row, 0), axis=1)
top_actors['race'] = top_actors.apply(lambda row: get_combined_info(row, 1), axis=1)
top_actors['birth_year'] = top_actors.apply(lambda row: get_combined_info(row, 2), axis=1)

#Assertation test to ensure all the data has populated and there are no unknown values
assert not any(top_actors['genders'].astype(str).eq('Unknown'))
assert not any(top_actors['race'].astype(str).eq('Unknown'))
assert not any(top_actors['birth_year'].astype(str).eq('Unknown'))


#convert gender and race to proportions

#Convert gender into numeric proportion
genders_column = top_actors['genders'] #Only male or female

#The female proportion of actors is calculated as the number of 'F' in each movie divided by three.
Fproportions = top_actors.genders.apply(lambda cast: cast.count('F') / 3)

#put proportions into a new column
top_actors['female_proportion'] = Fproportions

#calculate race proportion using similar lambda function
Wproportions = top_actors.race.apply(lambda cast: cast.count('W')/3)

#put proportions into a new column
top_actors['white_proportion'] = Wproportions

#Calculate age at the time of movie then average age

#function to calculate age at time of movie
def calculate_age(row):
    return [row['title_year'] - year for year in row['birth_year']]

#apply calculate_age to new column
top_actors['age_at_time_of_movie'] = top_actors.apply(calculate_age, axis=1)

#convert from float to int
top_actors['age_at_time_of_movie'] = top_actors['age_at_time_of_movie'].apply(lambda x: [int(age) for age in x])

# Apply a lambda function to create the 'average_age' column
top_actors['average_age'] = top_actors['age_at_time_of_movie'].apply(lambda age_list: sum(age_list) / len(age_list) )

#calculate attendance from average ticket price

av_ticket_price = {
    2017: 8.97,
    2016: 8.65,
    2015: 8.43,
    2014: 8.17,
    2013: 8.13,
    2012: 7.96,
    2011: 7.93,
    2010: 7.89,
    2009: 7.50,
    2008: 7.18,
    2007: 6.88,
    2006: 6.55 }


top_actors['attendance'] = top_actors['title_year'].map(av_ticket_price)
top_actors['attendance'] = top_actors['budget'] / top_actors['attendance']

#Converting content rating using dummy encoding

# Use get_dummies to create dummy columns for 'content_rating'
content_rating_dummies = pd.get_dummies(top_actors['content_rating'], prefix='content_rating (Dummy)')

# Concatenate the dummy columns to the original DataFrame
top_actors = pd.concat([top_actors, content_rating_dummies], axis=1)

#Create Genre dummy data
# Create binary columns for 'comedy', 'action', 'drama', and 'other'
top_actors['genre_comedy (Dummy)'] = top_actors['genres'].apply(lambda x: 1 if 'Comedy' in x else 0)
top_actors['genre_action (Dummy)'] = top_actors['genres'].apply(lambda x: 1 if 'Action' in x else 0)
top_actors['genre_drama (Dummy)'] = top_actors['genres'].apply(lambda x: 1 if 'Drama' in x else 0)
top_actors['genre_other (Dummy)'] = top_actors['genres'].apply(lambda x: 0 if any(genre in x for genre in ['Comedy', 'Action', 'Drama']) else 1)

#Populate starpower values
#Manually find values using IMDB pro for each actor 10 weeks before the movie release date

#dict needs this format to account for actors that appear in more than one movie
# the -- enables the string to be split later on

starpower_dict = {
    "Pirates of the Caribbean: At World's End--Johnny Depp": 1,
    'Alice in Wonderland--Johnny Depp': 3,
    'Pirates of the Caribbean: On Stranger Tides--Johnny Depp': 4,
    'The Lone Ranger--Johnny Depp': 7,
    'Alice Through the Looking Glass--Johnny Depp': 20,
    'Spider-Man 3--J.K. Simmons': 79,
    'Terminator Genisys--J.K. Simmons': 13,
    'The Golden Compass--Christopher Lee': 30,
    'Hugo--Christopher Lee': 568,
    'I Am Legend--Will Smith': 2,
    'Hancock--Will Smith': 2,
    'Men in Black 3--Will Smith': 11,
    'Suicide Squad--Will Smith': 83,
    'Fantastic 4: Rise of the Silver Surfer--Chris Evans': 12,
    'Captain America: The First Avenger--Chris Evans': 1,
    'Captain America: The Winter Soldier--Chris Evans': 6,
    'Captain America: Civil War--Chris Evans': 5,
    'Ghost Rider--Nicolas Cage': 2,
    "The Sorcerer's Apprentice--Nicolas Cage": 15,
    'The Chronicles of Narnia: Prince Caspian--Peter Dinklage': 549,
    'X-Men: Days of Future Past--Peter Dinklage': 16,
    'The Dark Knight--Christian Bale': 1,
    'Terminator Salvation--Christian Bale': 3,
    'The Dark Knight Rises--Christian Bale': 2,
    'The Curious Case of Benjamin Button--Brad Pitt': 4,
    'World War Z--Brad Pitt': 7,
    'Iron Man--Robert Downey Jr.': 1,
    'Tropic Thunder--Robert Downey Jr.': 1,
    'Iron Man 2--Robert Downey Jr.': 1,
    'The Avengers--Robert Downey Jr.': 5,
    'Iron Man 3--Robert Downey Jr.': 1,
    'Avengers: Age of Ultron--Robert Downey Jr.': 47,
    'Captain America: Civil War--Robert Downey Jr.': 10,
    'Transformers: Revenge of the Fallen--Glenn Morshower': 849,
    'Transformers: Dark of the Moon--Glenn Morshower': 1466,
    '2012--Oliver Platt': 689,
    'X-Men: First Class--Oliver Platt': 1031,
    'G.I. Joe: The Rise of Cobra--Joseph Gordon-Levitt': 15,
    'Inception--Joseph Gordon-Levitt': 6,
    'The Dark Knight Rises--Joseph Gordon-Levitt': 3,
    'Night at the Museum: Battle of the Smithsonian--Robin Williams': 71,
    'Night at the Museum: Secret of the Tomb--Robin Williams': 1,
    'X-Men Origins: Wolverine--Hugh Jackman': 2,
    'X-Men: Days of Future Past--Hugh Jackman': 8,
    'Pan--Hugh Jackman': 59,
    'Star Trek--Chris Hemsworth': 17,
    'Thor--Chris Hemsworth': 1,
    'The Avengers--Chris Hemsworth': 1,
    'Snow White and the Huntsman--Chris Hemsworth': 1,
    'Avengers: Age of Ultron--Chris Hemsworth': 11,
    'Iron Man--Jeff Bridges': 5,
    'TRON: Legacy--Jeff Bridges': 4,
    'Inception--Leonardo DiCaprio': 2,
    'The Revenant--Leonardo DiCaprio': 1,
    'The Wolfman--Anthony Hopkins': 102,
    'Thor--Anthony Hopkins': 102,
    'X-Men Origins: Wolverine--Ryan Reynolds': 5,
    'Green Lantern--Ryan Reynolds': 4,
    'X-Men: First Class--Jennifer Lawrence': 1,
    'X-Men: Days of Future Past--Jennifer Lawrence': 1,
    'The Hunger Games: Mockingjay - Part 2--Jennifer Lawrence': 2,
    'X-Men: Apocalypse--Jennifer Lawrence': 7,
    'Mission: Impossible - Ghost Protocol--Tom Cruise': 26,
    'Edge of Tomorrow--Tom Cruise': 21,
    'Fast Five--Paul Walker': 7,
    'Furious 7--Paul Walker': 1,
    'Inception--Tom Hardy': 764,
    'The Dark Knight Rises--Tom Hardy': 1,
    'The Revenant--Tom Hardy': 1,
    'The Amazing Spider-Man--Emma Stone': 1,
    'The Amazing Spider-Man 2--Emma Stone': 1,
    'Battleship--Liam Neeson': 2,
    'Wrath of the Titans--Liam Neeson': 2,
    'The Hobbit: An Unexpected Journey--Aidan Turner': 4317,
    'The Hobbit: The Desolation of Smaug--Aidan Turner': 7614,
    'X-Men: First Class--Michael Fassbender': 2,
    'Prometheus--Michael Fassbender': 4,
    'X-Men: Apocalypse--Michael Fassbender': 24,
    'Man of Steel--Henry Cavill': 1,
    'Batman v Superman: Dawn of Justice--Henry Cavill': 2,
    'Hancock--Eddie Marsan': 701,
    'Jack the Giant Slayer--Eddie Marsan': 694,
    'Iron Man 2--Scarlett Johansson': 2,
    'The Avengers--Scarlett Johansson': 2,
    'Captain America: The Winter Soldier--Scarlett Johansson': 1,
    'Avengers: Age of Ultron--Scarlett Johansson': 4,
    'Captain America: Civil War--Scarlett Johansson': 6,
    'Dawn of the Planet of the Apes--Judy Greer': 36,
    'Tomorrowland--Judy Greer': 27,
    'Jurassic World--Judy Greer': 27,
    'Ant-Man--Judy Greer': 27,
    'Terminator Salvation--Bryce Dallas Howard': 7,
    'Jurassic World--Bryce Dallas Howard': 1,
    'Captain America: The First Avenger--Dominic Cooper': 47,
    'Warcraft--Dominic Cooper': 28,
    'Spider-Man 3--James Franco': 4,
    'Oz the Great and Powerful--James Franco': 2,
    'Evan Almighty--Morgan Freeman': 20,
    'The Dark Knight--Morgan Freeman': 3,
    'The Bourne Ultimatum--Edgar Ramírez': 324,
    'Wrath of the Titans--Edgar Ramírez': 246,
    'Indiana Jones and the Kingdom of the Crystal Skull--Ray Winstone': 98,
    'Hugo--Ray Winstone': 821,
    'Hancock--Charlize Theron': 6,
    'Prometheus--Charlize Theron': 2,
    'Tropic Thunder--Steve Coogan': 44,
    'Night at the Museum: Battle of the Smithsonian--Steve Coogan': 170,
    'Night at the Museum: Secret of the Tomb--Steve Coogan': 667,
    'Transformers--Kevin Dunn': 688,
    'Transformers: Revenge of the Fallen--Kevin Dunn': 1087,
    'Transformers: Dark of the Moon--Kevin Dunn': 1087,
    'Night at the Museum: Battle of the Smithsonian--Rami Malek': 236,
    'Night at the Museum: Secret of the Tomb--Rami Malek': 282,
    'The Incredible Hulk--William Hurt': 50,
    'Robin Hood--William Hurt': 191,
    'Alice in Wonderland--Alan Rickman': 24,
    'Alice Through the Looking Glass--Alan Rickman': 1,
    'Pirates of the Caribbean: On Stranger Tides--Sam Claflin': 13,
    'Snow White and the Huntsman--Sam Claflin': 6,
    'Fast Five--Vin Diesel': 3,
    'Guardians of the Galaxy--Vin Diesel': 42,
    'Furious 7--Vin Diesel': 6,
    'The Amazing Spider-Man--Andrew Garfield': 2,
    'The Amazing Spider-Man 2--Andrew Garfield': 3,
    'Battleship--Alexander Skarsgård': 34,
    'The Legend of Tarzan--Alexander Skarsgård': 3,
    'The Hobbit: An Unexpected Journey--Adam Brown': 193,
    'The Hobbit: The Desolation of Smaug--Adam Brown': 754,
    'Oz the Great and Powerful--Mila Kunis': 2,
    'Jupiter Ascending--Mila Kunis': 6,
    'Iron Man--Jon Favreau': 12,
    'Iron Man 2--Jon Favreau': 18,
    'Iron Man 3--Jon Favreau': 69,
    'Star Trek--Bruce Greenwood': 23,
    'Star Trek Into Darkness--Bruce Greenwood': 365,
    'Alice in Wonderland--Anne Hathaway': 11,
    'Interstellar--Anne Hathaway': 6,
    'Alice Through the Looking Glass--Anne Hathaway': 97,
    'Captain America: The First Avenger--Hayley Atwell': 3,
    'Captain America: The Winter Soldier--Hayley Atwell': 14,
    'Ant-Man--Hayley Atwell': 1,
    'The Hobbit: An Unexpected Journey--James Nesbitt': 73,
    'The Hobbit: The Desolation of Smaug--James Nesbitt': 576,
    'Evan Almighty--Jimmy Bennett': 1002,
    'Wild Hogs--Jill Hennessy': 229,
    'Transformers--Zack Ward': 345171,
    'Rush Hour 3--Tzi Ma':1296,
    'The Bourne Ultimatum--Matt Damon':2,
    'Indiana Jones and the Kingdom of the Crystal Skull--Harrison Ford': 1,
    'The Incredible Hulk--Ty Burrell': 253,
    'The Mummy: Tomb of the Dragon Emperor--Jet Li':33,
    'Speed Racer--Scott Porter': 170,
    'Avatar--CCH Pounder': 17,
    'Angels & Demons--Tom Hanks':12,
    'Watchmen--Matt Frewer': 105,
    'Robin Hood--Mark Addy': 216,
    'Prince of Persia: The Sands of Time--Jake Gyllenhaal': 3,
    'The Chronicles of Narnia: The Voyage of the Dawn Treader--Bruce Spence': 2631,
    'The Last Airbender--Seychelle Gabriel': 25,
    'Hugo--Chloë Grace Moretz':16,
    'The Adventures of Tintin--Toby Jones': 224,
    'John Carter--Daryl Sabara': 775,
    'Oz the Great and Powerful--Tim Holmes': 272,
    'World War Z--Peter Capaldi': 2750,
    'Star Trek Into Darkness--Benedict Cumberbatch': 1,
    'Pacific Rim--Charlie Hunnam': 1,
    '47 Ronin--Keanu Reeves': 49,
    'Transformers: Age of Extinction--Bingbing Li': 150,
    'Maleficent--Angelina Jolie Pitt': 1,
    'Dawn of the Planet of the Apes--Gary Oldman': 182,
    'Guardians of the Galaxy--Bradley Cooper': 3,
    'Interstellar--Matthew McConaughey':3,
    'Furious 7--Jason Statham':22,
    'Jupiter Ascending--Channing Tatum':11,
    'Star Trek Beyond--Sofia Boutella':3,
    'The Legend of Tarzan--Christoph Waltz': 176,
    'Independence Day: Resurgence--Vivica A. Fox': 164,
    'Ghostbusters--Ed Begley Jr.': 2809,
    "Pirates of the Caribbean: At World's End--Orlando Bloom": 27,
    'The Golden Compass--Eva Green': 33,
    'Wild Hogs--Tichina Arnold': 4337,
    "Transformers--Michael O'Neill": 1872,
    'I Am Legend--Alice Braga': 66,
    'Rush Hour 3--Dana Ivey':5081,
    'Fantastic 4: Rise of the Silver Surfer--Ioan Gruffudd':32,
    'The Chronicles of Narnia: Prince Caspian--Pierfrancesco Favino':1587,
    'The Dark Knight--Heath Ledger':1,
    'The Curious Case of Benjamin Button--Jason Flemyng':407,
    'The Incredible Hulk--Peter Mensah':1498,
    'The Mummy: Tomb of the Dragon Emperor--Brendan Fraser': 10,
    'Speed Racer--Kick Gurry': 233,
    'Avatar--Joel David Moore': 19,
    '2012--Liam James':471,
    'G.I. Joe: The Rise of Cobra--Dennis Quaid':103,
    'Angels & Demons--Ayelet Zurer': 20,
    'Star Trek--Leonard Nimoy':173,
    'Watchmen--Billy Crudup': 4,
    'Prince of Persia: The Sands of Time--Richard Coyle':120,
    'TRON: Legacy--Olivia Wilde':420,
    'The Chronicles of Narnia: The Voyage of the Dawn Treader--Shane Rangi':4944,
    "The Sorcerer's Apprentice--Omar Benson Miller": 2596,
    'The Wolfman--Simon Merrells':2356,
    'The Last Airbender--Noah Ringer': 17,
    'Green Lantern--Temuera Morrison': 555,
    'Transformers: Dark of the Moon--Lester Speight': 1294,
    'Thor--Natalie Portman': 1,
    'Mission: Impossible - Ghost Protocol--Jeremy Renner': 16,
    'The Adventures of Tintin--Mackenzie Crook':4159,
    'John Carter--Samantha Morton':86,
    'Men in Black 3--Michael Stuhlbarg': 75,
    'Snow White and the Huntsman--Kristen Stewart': 6,
    'Man of Steel--Christopher Meloni':327,
    'The Lone Ranger--Ruth Wilson': 142,
    'Jack the Giant Slayer--Ewen Bremner': 1546,
    'Pacific Rim--Clifton Collins Jr.': 447,
    '47 Ronin--Cary-Hiroyuki Tagawa': 1963,
    'Transformers: Age of Extinction--Sophia Myles':301,
    'Maleficent--Sharlto Copley':58,
    'Edge of Tomorrow--Lara Pulver':552,
    'Tomorrowland--Chris Bauer':4263,
    'The Hunger Games: Mockingjay - Part 2--Philip Seymour Hoffman':62,
    'Terminator Genisys--Emilia Clarke':1,
    'Pan--Cara Delevingne':1,
    'Batman v Superman: Dawn of Justice--Lauren Cohan': 10,
    'Star Trek Beyond--Melissa Roxburgh':558,
    'Suicide Squad--Robin Atkin Downes':658,
    'Independence Day: Resurgence--Sela Ward': 80,
    'Warcraft--Callum Rennie':2605,
    'Ghostbusters--Kate McKinnon': 4,
    "Pirates of the Caribbean: At World's End--Jack Davenport":12,
    'Spider-Man 3--Kirsten Dunst':1,
    'The Golden Compass--Kristin Scott Thomas':410,
    'Evan Almighty--Steve Carell':20,
    'Wild Hogs--Drew Sidora':441,
    'I Am Legend--Willow Smith':50,
    'Rush Hour 3--Noémie Lenoir':57,
    'Fantastic 4: Rise of the Silver Surfer--Andre Braugher':1248,
    'The Bourne Ultimatum--Albert Finney':341,
    'The Chronicles of Narnia: Prince Caspian--Damián Alcázar':2151,
    'Indiana Jones and the Kingdom of the Crystal Skull--Jim Broadbent': 117,
    'The Curious Case of Benjamin Button--Julia Ormond':56,
    'The Mummy: Tomb of the Dragon Emperor--Russell Wong':706,
    'Speed Racer--Nicholas Elia': 805,
    'Tropic Thunder--Brandon T. Jackson': 115,
    'Avatar--Wes Studi':19,
    'Transformers: Revenge of the Fallen--Ramon Rodriguez':96,
    'Terminator Salvation--Common': 153,
    '2012--Tom McCarthy':609,
    'G.I. Joe: The Rise of Cobra--Leo Howard':132,
    'X-Men Origins: Wolverine--Dominic Monaghan': 22,
    'Angels & Demons--Armin Mueller-Stahl': 451,
    'Watchmen--Stephen McHattie': 899673,
    'Robin Hood--Scott Grimes':151,
    'Prince of Persia: The Sands of Time--Reece Ritchie':698,
    'TRON: Legacy--James Frain':90,
    'The Chronicles of Narnia: The Voyage of the Dawn Treader--Laura Brent': 2380,
    "The Sorcerer's Apprentice--Robert Capron":2399,
    'The Wolfman--Art Malik': 2565,
    'The Last Airbender--Aasif Mandvi':210,
    'Pirates of the Caribbean: On Stranger Tides--Stephen Graham': 404,
    'Green Lantern--Taika Waititi':2352,
    'Mission: Impossible - Ghost Protocol--Michael Nyqvist':195,
    'The Adventures of Tintin--Tony Curran':1244,
    'Fast Five--Dwayne Johnson':6,
    'John Carter--Polly Walker':82,
    'The Amazing Spider-Man--Chris Zylka':191,
    'Men in Black 3--Nicole Scherzinger':126,
    'Battleship--Tadanobu Asano':837,
    'Wrath of the Titans--Lily James':1206,
    'Prometheus--Sean Harris':168,
    'Man of Steel--Harry Lennix':1067,
    'The Lone Ranger--Tom Wilkinson':1201,
    'Iron Man 3--Don Cheadle':148,
    'Jack the Giant Slayer--Ralph Brown':2506,
    'World War Z--Mireille Enos':28,
    'Star Trek Into Darkness--Noel Clarke':726,
    'Pacific Rim--Larry Joe Campbell':3673,
    '47 Ronin--Jin Akanishi':3114,
    'Transformers: Age of Extinction--Kelsey Grammer':620,
    'The Amazing Spider-Man 2--B.J. Novak':1569,
    'Maleficent--Sam Riley':66,
    'Edge of Tomorrow--Noah Taylor':611,
    'Dawn of the Planet of the Apes--Kodi Smit-McPhee':233,
    'Guardians of the Galaxy--Djimon Hounsou':301,
    'Interstellar--Mackenzie Foy':7,
    'Tomorrowland--Thomas Robinson':1117,
    'Jupiter Ascending--Eddie Redmayne':3,
    'The Hunger Games: Mockingjay - Part 2--Josh Hutcherson':24,
    'Terminator Genisys--Matt Smith':363,
    'Jurassic World--Omar Sy':445,
    'Pan--Nonso Anozie':932,
    'The Revenant--Lukas Haas':858,
    'Ant-Man--T.I.':1124,
    'Batman v Superman: Dawn of Justice--Alan D. Purwin':21538,
    'Star Trek Beyond--Lydia Wilson':91,
    'The Legend of Tarzan--Casper Crump':1579,
    'X-Men: Apocalypse--Tye Sheridan':66,
    'Suicide Squad--Ike Barinholtz':97,
    'Independence Day: Resurgence--Judd Hirsch':825,
    'Warcraft--Ruth Negga':825,
    'Ghostbusters--Zach Woods':560,
}

#create the unique movie--actor values to map the correct starpower values for actors that appear more than once
movie_actor_1_list = (top_actors['movie_title'] + '--' + top_actors['actor_1_name']).tolist()
movie_actor_2_list = (top_actors['movie_title'] + '--' + top_actors['actor_2_name']).tolist()
movie_actor_3_list = (top_actors['movie_title'] + '--' + top_actors['actor_3_name']).tolist()

#combine lists so that it can be put into one column
combined_list = movie_actor_1_list + movie_actor_2_list + movie_actor_3_list

#put list into a df
starpower_df = pd.DataFrame({'movie_actor': combined_list})

#function to get the dictionary value based on the 'movie_actor' column
def get_dict_value(row):
    key = row['movie_actor']
    return starpower_dict.get(key, None)

#apply function to create a new column 'starpower_value'
starpower_df['starpower_value'] = starpower_df.apply(get_dict_value, axis=1)

#this is for an assertation test
starpower_dict_nan_count = starpower_df['starpower_value'].isna().sum()

assert starpower_dict_nan_count == 0

#Merge with the top_actors df
#split at the -- to get movie and actor separated
starpower_df[['movie_name', 'actor']] = starpower_df['movie_actor'].str.split('--', expand=True)

#group by movie title
grouped_starpower = starpower_df.groupby('movie_name')['starpower_value'].agg(list).reset_index()

#merge with top actors
merged_df = pd.merge(top_actors, grouped_starpower, left_on='movie_title', right_on='movie_name', how='left')

#drop 'movie_name' column
merged_df.drop(columns=['movie_name'], inplace=True)

#calculate highest starpower values
# Apply a lambda function to create the 'highest_starpower' column
merged_df['highest_starpower'] = merged_df['starpower_value'].apply(lambda values: min(values))

#start analysis

#create a df with only necessary data for analysis
analysis_df = merged_df[[
    'white_proportion',
    'female_proportion',
    'average_age',
    'attendance',
    'highest_starpower',
    'gross',
    'budget',
    'content_rating (Dummy)_PG',
    'content_rating (Dummy)_PG-13',
    'content_rating (Dummy)_R',
    'genre_comedy (Dummy)',
    'genre_action (Dummy)',
    'genre_drama (Dummy)',
    'genre_other (Dummy)'
]]
#overview

#calculate summary statistics
summary_stats = analysis_df.describe()

#pairplot
full_analysis_pairplot = sns.pairplot(analysis_df)

#correlation matrix
correlation_matrix = analysis_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.savefig('images/correlation_plot.png')


#investigating white proportion

#white pairplot
white_pairplot = sns.pairplot(analysis_df, hue = 'white_proportion')

#white regplot with gross revenue
sns.regplot(x='white_proportion', y='gross', data=analysis_df)
plt.savefig('images/white_regplot.png')

#correlation coefficient white proportion and gross
white_correlation_coefficient, white_p_value = pearsonr(analysis_df['white_proportion'], analysis_df['gross'])
print(f"White Pearson Correlation Coefficient: {white_correlation_coefficient:.2f}, p-value: {white_p_value:.4f}")

#boxplots to see errors
sns.boxplot(x='white_proportion', y='gross', data=analysis_df)
plt.savefig('images/white_boxplot.png')

#investigating female proportion

#female pairplot
female_pairplot = sns.pairplot(analysis_df, hue = 'female_proportion')

#female regplot with gross revenue
sns.regplot(x='female_proportion', y='gross', data=analysis_df)
plt.savefig('images/female_regplot.png')

#correlation coefficient female proportion and gross
female_correlation_coefficient, female_p_value = pearsonr(analysis_df['female_proportion'], analysis_df['gross'])
print(f"Female Pearson Correlation Coefficient: {female_correlation_coefficient:.2f}, p-value: {female_p_value:.4f}")

#boxplots to see errors
sns.boxplot(x='female_proportion', y='gross', data=analysis_df)
plt.savefig('images/female_boxplot.png')




