# Project Film in Colour

# Further Analysis

**Part 1:** Multi-level regression using nesting

**Part 2:** Investigating movie streaming 


```python
#IMPORT LIBRARIES

import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('mode.copy_on_write', True)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr


#READ THE DATA
clean_df_2 = pd.read_csv("data/clean_data_2.csv")
clean_df_2.head()
```

    Requirement already satisfied: statsmodels in /opt/conda/lib/python3.11/site-packages (0.14.0)
    Requirement already satisfied: numpy>=1.18 in /opt/conda/lib/python3.11/site-packages (from statsmodels) (1.26.2)
    Requirement already satisfied: scipy!=1.9.2,>=1.4 in /opt/conda/lib/python3.11/site-packages (from statsmodels) (1.11.3)
    Requirement already satisfied: pandas>=1.0 in /opt/conda/lib/python3.11/site-packages (from statsmodels) (2.1.3)
    Requirement already satisfied: patsy>=0.5.2 in /opt/conda/lib/python3.11/site-packages (from statsmodels) (0.5.3)
    Requirement already satisfied: packaging>=21.3 in /opt/conda/lib/python3.11/site-packages (from statsmodels) (23.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.11/site-packages (from pandas>=1.0->statsmodels) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.11/site-packages (from pandas>=1.0->statsmodels) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in /opt/conda/lib/python3.11/site-packages (from pandas>=1.0->statsmodels) (2023.3)
    Requirement already satisfied: six in /opt/conda/lib/python3.11/site-packages (from patsy>=0.5.2->statsmodels) (1.16.0)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>white_proportion</th>
      <th>female_proportion</th>
      <th>genres</th>
      <th>average_age</th>
      <th>attendance_in_mills</th>
      <th>budget_in_mills</th>
      <th>highest_starpower</th>
      <th>genre_action (Dummy)</th>
      <th>genre_comedy (Dummy)</th>
      <th>genre_drama (Dummy)</th>
      <th>genre_other (Dummy)</th>
      <th>content_rating (Dummy)_PG</th>
      <th>content_rating (Dummy)_PG-13</th>
      <th>content_rating (Dummy)_R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>Action|Adventure|Fantasy</td>
      <td>36.000000</td>
      <td>45.0</td>
      <td>300.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>Action|Adventure|Romance</td>
      <td>35.333333</td>
      <td>49.0</td>
      <td>258.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>0.666667</td>
      <td>Adventure|Family|Fantasy</td>
      <td>53.000000</td>
      <td>10.0</td>
      <td>180.0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>Comedy|Family|Fantasy</td>
      <td>42.000000</td>
      <td>15.0</td>
      <td>175.0</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>Action|Adventure|Comedy</td>
      <td>33.000000</td>
      <td>24.0</td>
      <td>168.0</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Part 1

## Mixed Effects Model, Grouping by Genre

#### Introduction

Xi Fang's research highlights the audience's preference for White leading actors in movies, but it raises the question of whether this preference stems from the actors' race or the predominantly White-centric themes in the stories. Fang suggests future research should focus on the impact of movie storylines on market performance, especially in films featuring predominantly White or male narratives, to better understand what drives movie success.

In the intersectional data science literature we found using mixed-effects models is particularly beneficial in this context. (Scott, 2017) These models can disentangle whether the audience preference is for White actors themselves or for stories about White people, by incorporating movie genre as a random effect. This allows for an analysis of audience preferences across different genres, acknowledging that genres like historical dramas or fantasy may inherently feature more White-led narratives.

Mixed-effects models are adept at handling the effects of race and gender on movie market performance, offering a nuanced understanding of these relationships. By grouping movies by genre, they enable an examination of how intersectional factors such as race and gender play out across various genres. This approach is effective because movies fall into hierarchical categories like genres, where movies within the same genre might share certain characteristics, and mixed-effects models are well-suited to capture both within-genre and between-genre variations.

### White Proportion MLR


```python
ml_model = smf.mixedlm('attendance_in_mills ~ white_proportion', 
                                clean_df_2, groups=clean_df_2['genres'])
mdf = ml_model.fit()
print(mdf.summary())  
```

                  Mixed Linear Model Regression Results
    =================================================================
    Model:            MixedLM Dependent Variable: attendance_in_mills
    No. Observations: 100     Method:             REML               
    No. Groups:       45      Scale:              297.4077           
    Min. group size:  1       Log-Likelihood:     -422.4623          
    Max. group size:  29      Converged:          Yes                
    Mean group size:  2.2                                            
    ------------------------------------------------------------------
                       Coef.   Std.Err.    z    P>|z|   [0.025  0.975]
    ------------------------------------------------------------------
    Intercept          25.290     6.348  3.984  0.000   12.849  37.732
    white_proportion    0.210     7.357  0.029  0.977  -14.209  14.630
    Group Var           8.598     1.271                               
    =================================================================
    


### Female Proportion MLR


```python
multilevel_model = smf.mixedlm('attendance_in_mills ~ female_proportion', 
                                clean_df_2, groups=clean_df_2['genres'])
mdf = multilevel_model.fit()
print(mdf.summary()) 
```

                  Mixed Linear Model Regression Results
    =================================================================
    Model:            MixedLM Dependent Variable: attendance_in_mills
    No. Observations: 100     Method:             REML               
    No. Groups:       45      Scale:              297.1723           
    Min. group size:  1       Log-Likelihood:     -422.4620          
    Max. group size:  29      Converged:          Yes                
    Mean group size:  2.2                                            
    -----------------------------------------------------------------
                          Coef.  Std.Err.   z    P>|z|  [0.025 0.975]
    -----------------------------------------------------------------
    Intercept             25.871    2.659  9.730 0.000  20.660 31.082
    female_proportion     -1.864    7.105 -0.262 0.793 -15.790 12.061
    Group Var              8.629    1.264                            
    =================================================================
    


### White*Female Interaction MLR


```python
clean_df_2['white_female_interaction'] = (clean_df_2['white_proportion']* clean_df_2['female_proportion'])

# Mixed Linear Model
multilevel_model = smf.mixedlm('attendance_in_mills ~ white_proportion + female_proportion + white_female_interaction', 
                                clean_df_2, groups=clean_df_2['genres'])
mdf = multilevel_model.fit()
print(mdf.summary())   
```

                    Mixed Linear Model Regression Results
    =====================================================================
    Model:              MixedLM  Dependent Variable:  attendance_in_mills
    No. Observations:   100      Method:              REML               
    No. Groups:         45       Scale:               293.0980           
    Min. group size:    1        Log-Likelihood:      -414.4247          
    Max. group size:    29       Converged:           Yes                
    Mean group size:    2.2                                              
    ---------------------------------------------------------------------
                              Coef.  Std.Err.   z    P>|z|  [0.025 0.975]
    ---------------------------------------------------------------------
    Intercept                 17.834    9.011  1.979 0.048   0.172 35.496
    white_proportion           9.956   10.474  0.951 0.342 -10.572 30.484
    female_proportion         24.631   20.881  1.180 0.238 -16.294 65.557
    white_female_interaction -36.697   27.174 -1.350 0.177 -89.957 16.563
    Group Var                 14.666    1.486                            
    =====================================================================
    


### MLR Results

The mixed linear model suggests no significant effect of cast demographics on movie attendance in the sample analyzed. The presence of White leading actors (coefficient = 0.419, p = 0.815) and the interaction between White and female leading actors (coefficient = -2.649, p = 0.177) were not statistically significant predictors of movie attendance. Similarly, the proportion of female leading actors was associated with a negative, yet non-significant, impact on attendance (coefficient = -1.369, p = 0.467).

A notable aspect of the model is the observed variability in attendance across movie genres, reflected by a group variance of 14.666. This indicates that genre-specific factors significantly influence attendance and merit further exploration.

The findings underscore the complexity of audience preferences, with no clear pattern emerging on the influence of racial and gender diversity in leading roles on movie attendance within this dataset.



## Part 2

## Streaming

#### Introduction

Another method for our further analysis is investigating if consumer discrimination is present in streaming data. Netflix released their first ever viewing data in December 2023-What We Watched: A Netflix Engagement Report including raw data: https://about.netflix.com/en/news/what-we-watched-a-netflix-engagement-report. They have been critiqued in the past for a lack of transparency when it comes to data sharing, yet have recently released viewing data on 18,000 titles representing 99% of all viewing on Netflix — and nearly 100 billion hours viewed. In the current era of streaming, moving away from cinematic shows, this is more relevant to today's consumer culture.

The Netflix data set contains limited variables and as such we chose to investigate shows that are streamed globally. Due to the limited time frame of this analysis, the top five shows with the highest viewing hours for the previous five years (2019-2023) were analysed. The same hypotheses as in our replication were investigated.



```python
#READ THE DATA AND CLEAN
df = pd.read_csv("data/What_We_Watched_A_Netflix_Engagement_Report_2023Jan-Jun.csv", encoding='latin1')
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 5'])
df = df.iloc[4:]
df = df.reset_index(drop=True)
new_column_labels = {'Unnamed: 1': 'Title', 'Unnamed: 2': 'Available Globally?', 'Unnamed: 3': 'Release Date', 'Unnamed: 4': 'Hours Viewed'}
df = df.rename(columns=new_column_labels)
df = df.drop(0)
df['Release Date'] = pd.to_datetime(df['Release Date'])
df['Year'] = df['Release Date'].dt.year
df['Title'] = df['Title'].astype(str)
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Available Globally?</th>
      <th>Release Date</th>
      <th>Hours Viewed</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>The Night Agent: Season 1</td>
      <td>Yes</td>
      <td>2023-03-23</td>
      <td>812,100,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ginny &amp; Georgia: Season 2</td>
      <td>Yes</td>
      <td>2023-01-05</td>
      <td>665,100,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Glory: Season 1 // ? ???: ?? 1</td>
      <td>Yes</td>
      <td>2022-12-30</td>
      <td>622,800,000</td>
      <td>2022.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wednesday: Season 1</td>
      <td>Yes</td>
      <td>2022-11-23</td>
      <td>507,700,000</td>
      <td>2022.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Queen Charlotte: A Bridgerton Story</td>
      <td>Yes</td>
      <td>2023-05-04</td>
      <td>503,000,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>You: Season 4</td>
      <td>Yes</td>
      <td>2023-02-09</td>
      <td>440,600,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>La Reina del Sur: Season 3</td>
      <td>No</td>
      <td>2022-12-30</td>
      <td>429,600,000</td>
      <td>2022.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Outer Banks: Season 3</td>
      <td>Yes</td>
      <td>2023-02-23</td>
      <td>402,500,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ginny &amp; Georgia: Season 1</td>
      <td>Yes</td>
      <td>2021-02-24</td>
      <td>302,100,000</td>
      <td>2021.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FUBAR: Season 1</td>
      <td>Yes</td>
      <td>2023-05-25</td>
      <td>266,200,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Manifest: Season 4</td>
      <td>Yes</td>
      <td>2022-11-04</td>
      <td>262,600,000</td>
      <td>2022.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kaleidoscope: Limited Series</td>
      <td>Yes</td>
      <td>2023-01-01</td>
      <td>252,500,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Firefly Lane: Season 2</td>
      <td>Yes</td>
      <td>2022-12-02</td>
      <td>251,500,000</td>
      <td>2022.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>The Mother</td>
      <td>Yes</td>
      <td>2023-05-12</td>
      <td>249,900,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Physical: 100: Season 1 // ???: 100: ?? 1</td>
      <td>Yes</td>
      <td>2023-01-24</td>
      <td>235,000,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Crash Course in Romance: Limited Series // ?? ...</td>
      <td>Yes</td>
      <td>2023-01-14</td>
      <td>234,800,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Love Is Blind: Season 4</td>
      <td>Yes</td>
      <td>2023-03-24</td>
      <td>229,700,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BEEF: Season 1</td>
      <td>Yes</td>
      <td>2023-04-06</td>
      <td>221,100,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>The Diplomat: Season 1</td>
      <td>Yes</td>
      <td>2023-04-20</td>
      <td>214,100,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Luther: The Fallen Sun</td>
      <td>Yes</td>
      <td>2023-03-10</td>
      <td>209,700,000</td>
      <td>2023.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sorted = df.sort_values(by='Year')
df_sorted = df_sorted.dropna(subset=['Year'])
df_sorted['Available Globally?'] = df_sorted['Available Globally?'].astype(str)
global_df = df_sorted[df_sorted['Available Globally?'] == 'Yes']
global_df.reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Available Globally?</th>
      <th>Release Date</th>
      <th>Hours Viewed</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trailer Park Boys: Season 4</td>
      <td>Yes</td>
      <td>2010-09-22</td>
      <td>6,800,000</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trailer Park Boys: Season 3</td>
      <td>Yes</td>
      <td>2010-09-22</td>
      <td>6,800,000</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trailer Park Boys: Season 1</td>
      <td>Yes</td>
      <td>2010-09-22</td>
      <td>5,400,000</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trailer Park Boys: Season 2</td>
      <td>Yes</td>
      <td>2010-09-22</td>
      <td>5,800,000</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trailer Park Boys: Season 5</td>
      <td>Yes</td>
      <td>2010-09-22</td>
      <td>8,400,000</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3910</th>
      <td>We Have a Ghost</td>
      <td>Yes</td>
      <td>2023-02-24</td>
      <td>124,400,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>3911</th>
      <td>Bloodhounds: Season 1 // ????: ?? 1</td>
      <td>Yes</td>
      <td>2023-06-09</td>
      <td>146,700,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>3912</th>
      <td>XO, Kitty: Season 1</td>
      <td>Yes</td>
      <td>2023-05-18</td>
      <td>200,700,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>3913</th>
      <td>Ginny &amp; Georgia: Season 2</td>
      <td>Yes</td>
      <td>2023-01-05</td>
      <td>665,100,000</td>
      <td>2023.0</td>
    </tr>
    <tr>
      <th>3914</th>
      <td>Extraction 2</td>
      <td>Yes</td>
      <td>2023-06-16</td>
      <td>201,800,000</td>
      <td>2023.0</td>
    </tr>
  </tbody>
</table>
<p>3915 rows × 5 columns</p>
</div>




```python
recent_years_df = global_df[(global_df['Year']>=2019)]
recent_years_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Available Globally?</th>
      <th>Release Date</th>
      <th>Hours Viewed</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>The Rain: Season 2</td>
      <td>Yes</td>
      <td>2019-05-17</td>
      <td>10,200,000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>Kingdom: Season 1 // ??: ?? 1</td>
      <td>Yes</td>
      <td>2019-01-25</td>
      <td>10,200,000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>Rim of the World</td>
      <td>Yes</td>
      <td>2019-05-24</td>
      <td>10,200,000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>17970</th>
      <td>Upstarts // ?????????</td>
      <td>Yes</td>
      <td>2019-10-18</td>
      <td>100,000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>17833</th>
      <td>Tiffany Haddish: Black Mitzvah</td>
      <td>Yes</td>
      <td>2019-12-03</td>
      <td>100,000</td>
      <td>2019.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
recent_years_df['Hours Viewed'] = recent_years_df['Hours Viewed'].str.replace(',', '')
recent_years_df['Hours Viewed'] = pd.to_numeric(recent_years_df['Hours Viewed'])
```


```python
top_5_per_year = recent_years_df.groupby('Year').apply(lambda group: group.nlargest(5, 'Hours Viewed'))
top_5_per_year.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Title</th>
      <th>Available Globally?</th>
      <th>Release Date</th>
      <th>Hours Viewed</th>
      <th>Year</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2019.0</th>
      <th>73</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>Yes</td>
      <td>2019-12-14</td>
      <td>120300000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>You: Season 2</td>
      <td>Yes</td>
      <td>2019-12-26</td>
      <td>95000000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>Murder Mystery</td>
      <td>Yes</td>
      <td>2019-06-14</td>
      <td>87900000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>165</th>
      <td>The Witcher: Season 1</td>
      <td>Yes</td>
      <td>2019-12-20</td>
      <td>72200000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>191</th>
      <td>Stranger Things 3</td>
      <td>Yes</td>
      <td>2019-07-04</td>
      <td>67000000</td>
      <td>2019.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Manually build the netflix diversity dictionary

Using IMDb Pro


```python
# race = B, W, A, H, O
# show -- actor : ('Gender', 'Race', 'Age')
netflix_dict_maybe = {
    'Crash Landing on You: Season 1 // ??? ???: ?? 1--Hyun Bin': ('M', 'A', 1982),
    'Crash Landing on You: Season 1 // ??? ???: ?? 1--Son Ye-jin': ('F', 'A', 1982),
    'Crash Landing on You: Season 1 // ??? ???: ?? 1--Seo Ji-hye': ('F', 'A', 1984),
    'You: Season 2--Penn Badgley': ('M', 'W', 1986),
    'You: Season 2--Victoria Pedretti': ('F', 'W', 1995),
    'You: Season 2--Ambyr Childers': ('F', 'W', 1988),
    'Murder Mystery--Adam Sandler': ('M', 'W', 1966 ),
    'Murder Mystery--Jennifer Aniston': ('F', 'W', 1969),
    'Murder Mystery--Luke Evans': ('M', 'W',1979 ),
    'The Witcher: Season 1--Freya Allan': ('F', 'W', 2001),
    'The Witcher: Season 1--Henry Cavill': ('M', 'W', 1983),
    'The Witcher: Season 1--Anya Chalotra': ('F', 'A', 1995),
    'Outer Banks: Season 1--Chase Stokes': ('M', 'W', 1992),
    'Outer Banks: Season 1--Madelyn Cline': ('F', 'W', 1997),
    'Outer Banks: Season 1--Madison Bailey': ('F', 'W', 1999),
    'Bridgerton: Season 1--Regé-Jean Page': ('M', 'B', 1988 ),
    'Bridgerton: Season 1--Jonathan Bailey': ('M', 'W', 1988 ),
    'Bridgerton: Season 1--Phoebe Dynevor': ('M', 'W', 1995 ),
    'Alice in Borderland: Season 1 // ????????: ????1--Kento Yamazaki': ('M', 'A', 1994),
    'Alice in Borderland: Season 1 // ????????: ????1--Tao Tsuchiya': ('F', 'A', 1995),
    'Alice in Borderland: Season 1 // ????????: ????1--Nijiro Murakami': ('M', 'A', 1997),
    'Extraction--Chris Hemsworth':  ('M', 'W', 1983),
    'Extraction--Rudhraksh Jaiswal':  ('M', 'A',2003),
    'Extraction--Golshifteh Farahani':  ('F', 'A', 1983),
    'Never Have I Ever: Season 1--Maitreyi Ramakrishnan':  ('F', 'A', 2001),
    'Never Have I Ever: Season 1--Poorna Jagannathan':  ('F', 'A', 1972),
    'Never Have I Ever: Season 1--Darren Barnet':  ('M', 'O', 1991),
    'Ginny & Georgia: Season 1--Brianne Howey': ('F', 'W', 1989),
    'Ginny & Georgia: Season 1--Antonia Gentry': ('F', 'B', 1997),
    'Ginny & Georgia: Season 1--Felix Mallard': ('M', 'W', 1998),
    'Outer Banks: Season 2--Chase Stokes': ('M', 'W', 1992),
    'Outer Banks: Season 2--Madelyn Cline': ('F', 'W', 1997),
    'Outer Banks: Season 2--Madison Bailey': ('F', 'W', 1999),
    'Sex/Life: Season 1--Sarah Shahi': ('F', 'O', 1980),
    'Sex/Life: Season 1--Mike Vogel': ('M', 'W', 1979),
    'Sex/Life: Season 1--Adam Demos': ('M', 'W', 1985),
    'You: Season 3--Penn Badgley': ('M', 'W', 1986),
    'You: Season 3--Victoria Pedretti': ('F', 'W', 1995),
    'You: Season 3--Saffron Burrows': ('F', 'W', 1972),
    'Shadow and Bone: Season 1--Jessie Mei Li':  ('F', 'A', 1995),
    'Shadow and Bone: Season 1--Archie Renaux':  ('M', 'W', 1997),
    'Shadow and Bone: Season 1--Ben Barnes':  ('M', 'W', 1981),
    'The Glory: Season 1 // ? ???: ?? 1--Song Hye-kyo': ('F', 'A', 1981),
    'The Glory: Season 1 // ? ???: ?? 1--Lee Do-hyun': ('M', 'A', 1995),
    'The Glory: Season 1 // ? ???: ?? 1--Lim Ji-yeon': ('F', 'A', 1990),
    'Wednesday: Season 1--Jenna Ortega': ('F', 'H', 2002),
    'Wednesday: Season 1--Emma Myers': ('F', 'W', 2002),
    'Wednesday: Season 1--Hunter Doohan': ('M', 'W', 1994),
    'Manifest: Season 4--Melissa Roxburgh': ('F', 'W', 1992),
    'Manifest: Season 4--Josh Dallas': ('M', 'W', 1978),
    'Manifest: Season 4--J.R. Ramirez': ('M', 'H', 1980),
    'Firefly Lane: Season 2--Katherine Heigl':  ('F', 'W', 1978),
    'Firefly Lane: Season 2--Ben Lawson':  ('M', 'W', 1980),
    'Firefly Lane: Season 2--Sarah Chalke':  ('F', 'W', 1976),
    'Emily in Paris: Season 3--Lily Collins': ('F', 'W', 1989),
    'Emily in Paris: Season 3--Philippine Leroy-Beaulieu': ('F', 'W', 1963),
    'Emily in Paris: Season 3--Ashley Park': ('F', 'A', 1991),
    'The Night Agent: Season 1--Gabriel Basso': ('M', 'W', 1994),
    'The Night Agent: Season 1--Luciane Buchanan': ('F', 'O', 1993),
    'The Night Agent: Season 1--Fola Evans-Akingbola': ('F', 'B', 1994),
    'Ginny & Georgia: Season 2--Brianne Howey': ('F', 'W', 1989),
    'Ginny & Georgia: Season 2--Antonia Gentry': ('F', 'B', 1997),
    'Ginny & Georgia: Season 2--Diesel La Torraca': ('M', 'W', 2011),
    'Queen Charlotte: A Bridgerton Story--India Amarteifio': ('F', 'B', 2001),
    'Queen Charlotte: A Bridgerton Story--Golda Rosheuvel': ('F', 'B', 1970),
    'Queen Charlotte: A Bridgerton Story--Corey Mylchreest': ('M', 'W', 1998),
    'You: Season 4--Penn Badgley': ('M', 'W', 1986),
    'You: Season 4--Tati Gabrielle': ('F', 'B', 1996),
    'You: Season 4--Charlotte Ritchie': ('F', 'W', 1989),
    'Outer Banks: Season 3--Chase Stokes': ('M', 'W', 1992),
    'Outer Banks: Season 3--Madelyn Cline': ('F', 'W', 1997),
    'Outer Banks: Season 3--Rudy Pankow': ('M', 'W', 1997),
}
```


```python
# Create a DataFrame
diversity_df = pd.DataFrame.from_dict(netflix_dict_maybe, orient='index', columns=['Gender', 'Race', 'Year of Birth'])

# Reset index to get a column with the dictionary keys
diversity_df.reset_index(inplace=True)
diversity_df.rename(columns={'index': 'Title'}, inplace=True)

# Iterate through the 'Title' column and split the string at '--'
diversity_df['Actor'] = diversity_df['Title'].apply(lambda x: x.split('--')[1])
diversity_df['Title'] = diversity_df['Title'].apply(lambda x: x.split('--')[0])

# Display the updated DataFrame
diversity_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Gender</th>
      <th>Race</th>
      <th>Year of Birth</th>
      <th>Actor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>M</td>
      <td>A</td>
      <td>1982</td>
      <td>Hyun Bin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>F</td>
      <td>A</td>
      <td>1982</td>
      <td>Son Ye-jin</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>F</td>
      <td>A</td>
      <td>1984</td>
      <td>Seo Ji-hye</td>
    </tr>
    <tr>
      <th>3</th>
      <td>You: Season 2</td>
      <td>M</td>
      <td>W</td>
      <td>1986</td>
      <td>Penn Badgley</td>
    </tr>
    <tr>
      <th>4</th>
      <td>You: Season 2</td>
      <td>F</td>
      <td>W</td>
      <td>1995</td>
      <td>Victoria Pedretti</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group by 'Title' and aggregate values into lists
grouped_df = diversity_df.groupby('Title').agg({
    'Gender': list,
    'Race': list,
    'Year of Birth': list
}).reset_index()

# Display the grouped DataFrame
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Gender</th>
      <th>Race</th>
      <th>Year of Birth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alice in Borderland: Season 1 // ????????: ????1</td>
      <td>[M, F, M]</td>
      <td>[A, A, A]</td>
      <td>[1994, 1995, 1997]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bridgerton: Season 1</td>
      <td>[M, M, M]</td>
      <td>[B, W, W]</td>
      <td>[1988, 1988, 1995]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>[M, F, F]</td>
      <td>[A, A, A]</td>
      <td>[1982, 1982, 1984]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emily in Paris: Season 3</td>
      <td>[F, F, F]</td>
      <td>[W, W, A]</td>
      <td>[1989, 1963, 1991]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Extraction</td>
      <td>[M, M, F]</td>
      <td>[W, A, A]</td>
      <td>[1983, 2003, 1983]</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_df = pd.merge(grouped_df, top_5_per_year, on='Title', how='left')

# Display the merged DataFrame
merged_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Gender</th>
      <th>Race</th>
      <th>Year of Birth</th>
      <th>Available Globally?</th>
      <th>Release Date</th>
      <th>Hours Viewed</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alice in Borderland: Season 1 // ????????: ????1</td>
      <td>[M, F, M]</td>
      <td>[A, A, A]</td>
      <td>[1994, 1995, 1997]</td>
      <td>Yes</td>
      <td>2020-12-10</td>
      <td>92200000</td>
      <td>2020.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bridgerton: Season 1</td>
      <td>[M, M, M]</td>
      <td>[B, W, W]</td>
      <td>[1988, 1988, 1995]</td>
      <td>Yes</td>
      <td>2020-12-25</td>
      <td>136600000</td>
      <td>2020.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>[M, F, F]</td>
      <td>[A, A, A]</td>
      <td>[1982, 1982, 1984]</td>
      <td>Yes</td>
      <td>2019-12-14</td>
      <td>120300000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emily in Paris: Season 3</td>
      <td>[F, F, F]</td>
      <td>[W, W, A]</td>
      <td>[1989, 1963, 1991]</td>
      <td>Yes</td>
      <td>2022-12-21</td>
      <td>161100000</td>
      <td>2022.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Extraction</td>
      <td>[M, M, F]</td>
      <td>[W, A, A]</td>
      <td>[1983, 2003, 1983]</td>
      <td>Yes</td>
      <td>2020-04-24</td>
      <td>86100000</td>
      <td>2020.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Calculate Gender and race proportions


```python
#The female proportion of actors is calculated as the number of 'F' in each movie divided by three.
Fproportions = merged_df.Gender.apply(lambda cast: cast.count('F') / 3)

#put proportions into a new column
merged_df['female_proportion'] = Fproportions

#calculate race proportion using similar lambda function
Wproportions = merged_df.Race.apply(lambda cast: cast.count('W')/3)

#put proportions into a new column
merged_df['white_proportion'] = Wproportions

#display
merged_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Gender</th>
      <th>Race</th>
      <th>Year of Birth</th>
      <th>Available Globally?</th>
      <th>Release Date</th>
      <th>Hours Viewed</th>
      <th>Year</th>
      <th>female_proportion</th>
      <th>white_proportion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alice in Borderland: Season 1 // ????????: ????1</td>
      <td>[M, F, M]</td>
      <td>[A, A, A]</td>
      <td>[1994, 1995, 1997]</td>
      <td>Yes</td>
      <td>2020-12-10</td>
      <td>92200000</td>
      <td>2020.0</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bridgerton: Season 1</td>
      <td>[M, M, M]</td>
      <td>[B, W, W]</td>
      <td>[1988, 1988, 1995]</td>
      <td>Yes</td>
      <td>2020-12-25</td>
      <td>136600000</td>
      <td>2020.0</td>
      <td>0.000000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>[M, F, F]</td>
      <td>[A, A, A]</td>
      <td>[1982, 1982, 1984]</td>
      <td>Yes</td>
      <td>2019-12-14</td>
      <td>120300000</td>
      <td>2019.0</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emily in Paris: Season 3</td>
      <td>[F, F, F]</td>
      <td>[W, W, A]</td>
      <td>[1989, 1963, 1991]</td>
      <td>Yes</td>
      <td>2022-12-21</td>
      <td>161100000</td>
      <td>2022.0</td>
      <td>1.000000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Extraction</td>
      <td>[M, M, F]</td>
      <td>[W, A, A]</td>
      <td>[1983, 2003, 1983]</td>
      <td>Yes</td>
      <td>2020-04-24</td>
      <td>86100000</td>
      <td>2020.0</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>The Witcher: Season 1</td>
      <td>[F, M, F]</td>
      <td>[W, W, A]</td>
      <td>[2001, 1983, 1995]</td>
      <td>Yes</td>
      <td>2019-12-20</td>
      <td>72200000</td>
      <td>2019.0</td>
      <td>0.666667</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Wednesday: Season 1</td>
      <td>[F, F, M]</td>
      <td>[H, W, W]</td>
      <td>[2002, 2002, 1994]</td>
      <td>Yes</td>
      <td>2022-11-23</td>
      <td>507700000</td>
      <td>2022.0</td>
      <td>0.666667</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>21</th>
      <td>You: Season 2</td>
      <td>[M, F, F]</td>
      <td>[W, W, W]</td>
      <td>[1986, 1995, 1988]</td>
      <td>Yes</td>
      <td>2019-12-26</td>
      <td>95000000</td>
      <td>2019.0</td>
      <td>0.666667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>You: Season 3</td>
      <td>[M, F, F]</td>
      <td>[W, W, W]</td>
      <td>[1986, 1995, 1972]</td>
      <td>Yes</td>
      <td>2021-10-15</td>
      <td>107200000</td>
      <td>2021.0</td>
      <td>0.666667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>You: Season 4</td>
      <td>[M, F, F]</td>
      <td>[W, B, W]</td>
      <td>[1986, 1996, 1989]</td>
      <td>Yes</td>
      <td>2023-02-09</td>
      <td>440600000</td>
      <td>2023.0</td>
      <td>0.666667</td>
      <td>0.666667</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 10 columns</p>
</div>



#### Age


```python
# Function to calculate age at the time of the movie
def calculate_age(row):
    return [row['Year'] - year for year in row['Year of Birth']]

# Apply calculate_age to the new column
merged_df['age_at_time_of_movie'] = merged_df.apply(calculate_age, axis=1)

# Convert from float to int
merged_df['age_at_time_of_movie'] = merged_df['age_at_time_of_movie'].apply(lambda x: [int(age) for age in x])

# Apply a lambda function to create the 'average_age' column
merged_df['average_age'] = merged_df['age_at_time_of_movie'].apply(lambda age_list: round(sum(age_list) / len(age_list), 0))

# Display the updated DataFrame
merged_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Gender</th>
      <th>Race</th>
      <th>Year of Birth</th>
      <th>Available Globally?</th>
      <th>Release Date</th>
      <th>Hours Viewed</th>
      <th>Year</th>
      <th>female_proportion</th>
      <th>white_proportion</th>
      <th>age_at_time_of_movie</th>
      <th>average_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alice in Borderland: Season 1 // ????????: ????1</td>
      <td>[M, F, M]</td>
      <td>[A, A, A]</td>
      <td>[1994, 1995, 1997]</td>
      <td>Yes</td>
      <td>2020-12-10</td>
      <td>92200000</td>
      <td>2020.0</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>[26, 25, 23]</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bridgerton: Season 1</td>
      <td>[M, M, M]</td>
      <td>[B, W, W]</td>
      <td>[1988, 1988, 1995]</td>
      <td>Yes</td>
      <td>2020-12-25</td>
      <td>136600000</td>
      <td>2020.0</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>[32, 32, 25]</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Crash Landing on You: Season 1 // ??? ???: ?? 1</td>
      <td>[M, F, F]</td>
      <td>[A, A, A]</td>
      <td>[1982, 1982, 1984]</td>
      <td>Yes</td>
      <td>2019-12-14</td>
      <td>120300000</td>
      <td>2019.0</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>[37, 37, 35]</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emily in Paris: Season 3</td>
      <td>[F, F, F]</td>
      <td>[W, W, A]</td>
      <td>[1989, 1963, 1991]</td>
      <td>Yes</td>
      <td>2022-12-21</td>
      <td>161100000</td>
      <td>2022.0</td>
      <td>1.000000</td>
      <td>0.666667</td>
      <td>[33, 59, 31]</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Extraction</td>
      <td>[M, M, F]</td>
      <td>[W, A, A]</td>
      <td>[1983, 2003, 1983]</td>
      <td>Yes</td>
      <td>2020-04-24</td>
      <td>86100000</td>
      <td>2020.0</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>[37, 17, 37]</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>The Witcher: Season 1</td>
      <td>[F, M, F]</td>
      <td>[W, W, A]</td>
      <td>[2001, 1983, 1995]</td>
      <td>Yes</td>
      <td>2019-12-20</td>
      <td>72200000</td>
      <td>2019.0</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>[18, 36, 24]</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Wednesday: Season 1</td>
      <td>[F, F, M]</td>
      <td>[H, W, W]</td>
      <td>[2002, 2002, 1994]</td>
      <td>Yes</td>
      <td>2022-11-23</td>
      <td>507700000</td>
      <td>2022.0</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>[20, 20, 28]</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>You: Season 2</td>
      <td>[M, F, F]</td>
      <td>[W, W, W]</td>
      <td>[1986, 1995, 1988]</td>
      <td>Yes</td>
      <td>2019-12-26</td>
      <td>95000000</td>
      <td>2019.0</td>
      <td>0.666667</td>
      <td>1.000000</td>
      <td>[33, 24, 31]</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>You: Season 3</td>
      <td>[M, F, F]</td>
      <td>[W, W, W]</td>
      <td>[1986, 1995, 1972]</td>
      <td>Yes</td>
      <td>2021-10-15</td>
      <td>107200000</td>
      <td>2021.0</td>
      <td>0.666667</td>
      <td>1.000000</td>
      <td>[35, 26, 49]</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>You: Season 4</td>
      <td>[M, F, F]</td>
      <td>[W, B, W]</td>
      <td>[1986, 1996, 1989]</td>
      <td>Yes</td>
      <td>2023-02-09</td>
      <td>440600000</td>
      <td>2023.0</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>[37, 27, 34]</td>
      <td>33.0</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 12 columns</p>
</div>



### Analysis

#### White


```python
from scipy.stats import pearsonr
correlation_coefficient, p_value = pearsonr(merged_df['white_proportion'], merged_df['Hours Viewed'])
print(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}, p-value: {p_value:.4f}")
```

    Pearson Correlation Coefficient: -0.16, p-value: 0.4494


#### Visualise with regplot


```python
sns.regplot(x='white_proportion', y='Hours Viewed', data=merged_df)
plt.title('Regression Plot')
plt.xlabel('White Proportion')
plt.ylabel('Hours Viewed')
plt.show()
```


    
![png](output_27_0.png)
    


**Explanation:** The Correlation Coefficient between the 'white_proportion' and 'Hours Viewed' is -0.16. The negative correlation indicates that as white proportion increases, hours viewed decreases however the magnitude of the correlation is relatively weak suggesting a low linear relationship between the two variables. The P value is greater than the common significance level of 0.05, therefore there is no correlation between the 'white_proportion' and 'Hours Viewed' variables and there is not enough evidence to reject the null hypothesis. Based on these results, we cannot confidently assert a meaningful relationship between the proportion of white individuals and the number of hours viewed.



#### Female


```python
correlation_coefficient, p_value = pearsonr(merged_df['female_proportion'], merged_df['Hours Viewed'])
print(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}, p-value: {p_value:.4f}")
```

    Pearson Correlation Coefficient: 0.27, p-value: 0.2098


#### Visualise with Regplot


```python
sns.regplot(x='female_proportion', y='Hours Viewed', data=merged_df)
plt.title('Regression Plot')
plt.xlabel('Female Proportion')
plt.ylabel('Hours Viewed')
plt.show()
```


    
![png](output_32_0.png)
    


**Explanation:** The Correlation Coefficient between the 'female_proportion' and 'Hours Viewed' is 0.27. The positive correlation indicates that as female proportion increases the number of hours views also tends to increase. The magnitude of the correlation suggests a moderate relationship between the two variables however the P value is greater than 0.05 so there is not enough evidence to reject the null hypothesis. The null hypothesis in this case is that there is no correlation between 'female_proportion' and 'Hours Viewed.' Therefore, based on the p-value, we do not have sufficient statistical evidence to conclude that there is a significant correlation between these two variables and we cannot confidently assert a meaningful relationship between the proportion of females and the number of hours viewed based on these results.

### White and Female Interaction

Although there is limited significance, here is a visualisation of the relationship between female proportion, white proportion and hours viewed of streaming.


```python
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Extract relevant columns from merged_df
X1 = merged_df['white_proportion']
X2 = merged_df['female_proportion']
y = merged_df['Hours Viewed']

# Reshape data for sklearn
X = np.column_stack((X1, X2))

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Create a meshgrid for 3D plot
x1_range = np.linspace(X1.min(), X1.max(), 20)
x2_range = np.linspace(X2.min(), X2.max(), 20)
X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
X_mesh = np.column_stack((X1_mesh.ravel(), X2_mesh.ravel()))

# Predict the dependent variable for the meshgrid
y_pred_mesh = model.predict(X_mesh)
Y_pred_mesh = y_pred_mesh.reshape(X1_mesh.shape)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the original data points
ax.scatter(X1, X2, y, c='r', marker='o', label='Actual Data')

# Plot the regression surface
ax.plot_surface(X1_mesh, X2_mesh, Y_pred_mesh, color='b', alpha=0.5, label='Regression Surface')

ax.set_xlabel('White Proportion')
ax.set_ylabel('Female Proportion')
ax.set_zlabel('Hours Viewed')
ax.set_title('3D Regression Plot')

plt.show()
```


    
![png](output_35_0.png)
    


### Conclusion:
We cannot confidently state that there is any consumer discrimination for netflix streaming. However, the sample under analysis is small and therefore limited, and only included shows available globally. If the data allowed us to investigate US or western-only films, results may differ as the audience reach is likely to be less diverse.

### Future Analysis

The observed group variance of 14.666 in our mixed linear model is indicative of significant differences in movie attendance across genres, suggesting that genre may play a pivotal role in shaping audience turnout. This finding highlights the potential for genre-specific audience preferences and behaviors that could be crucial for industry stakeholders and content creators.


Considering the substantial variance attributed to genres, future research should prioritize:


Dissecting Genre-Specific Dynamics: Delving into the unique audience preferences within each genre could elucidate distinct factors that drive attendance, informing targeted production and marketing strategies.


Evaluating Contextual Influences: Investigating the historical and cultural underpinnings that characterize different genres may provide insight into their varying impact on audience engagement.


Analyzing Marketing Efficacy: Assessing the success of genre-specific marketing campaigns in relation to cast diversity could reveal strategic insights for optimizing promotional efforts.


Diversity and Genre Receptivity: Exploring how changes in cast diversity are received in traditionally homogeneous genres could uncover new opportunities for inclusive storytelling.


Intersectional Portrayals: It is crucial to examine how intersectional identities are presented across genres and their influence on market performance.

Future investigations should employ robust statistical methods capable of disentangling the complex interplay between genre, audience preferences, and diversity factors. Such research is essential for a comprehensive understanding of the determinants of success in the movie industry.






```python

```
