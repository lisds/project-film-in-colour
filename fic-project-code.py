#IMPORT LIBRARIES

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('mode.copy_on_write', True)
import numpy as np
import matplotlib.pyplot as plt


#READ THE DATA
df = pd.read_csv("C:/Users/Summe/Documents/dsip-work/film-in-colour/project-film-in-colour/movie_metadata.csv")


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

#display
print(top_actors.head())