# Film in Colour

This project explores the impact of top cast member diversity on Hollywood movie box office performance, aiming to replicate the findings of Xing Fan's (2021) master's thesis and conducting additional analyses.

### Introduction

Research consistently shows that minorities, including racial and ethnic groups, women, and disabled groups, are underrepresented in the film industry (Iasiello, 2017; Eschholz, 2002; Karniouchina, 2023). This underrepresentation is evident in both leading roles and directorial positions, where employers often attribute the root of bias to consumer attitudes rather than organisational practices, explained by individual preference for racial and ethnic similarity within networks, as well as systemic issues such as lack of opportunities and outcomes for women and racial/ethnic minorities (Iasiello, 2017; Iasiello, 2020; Erigha, 2015). 


This underrepresentation carries substantial social and economic implications, potentially perpetuating discrimination and harmful stereotypes. It restricts the range of narratives available to society about diverse groups, thereby shaping perceptions over time.The empirical challenge of testing the influence and scope of consumer racial attitudes on purchasing decisions has constrained prior efforts to disentangle managerial bias from consumer bias. (Kuppaswamy, 2018.)


In this study, we expand upon research regarding consumer reactions to workforce diversity in the film industry, utilizing an intersectional data analysis approach. We replicate and build upon Xi Fang's 2021 study from The University of Guelph, which evaluated U.S. film performance in relation to race and gender diversity of casts (Fang, 2021). Our work further investigates the impact of diversity on viewing hours on Netflix. Moreover, we explore consumer discrimination nuances by examining the variability of responses to main cast diversity across different movie genres, using mixed effects modeling. This approach broadens our understanding of the social contexts in which consumer discrimination operates.

### Hypotheses
Fang used the Ordinary Least Squared (OLS) Linear Regression technique utilizing a control function, which separately models all control variables, to explore five different hypothesis within his research, (see below), his results rejected all null hypotheses. 

_H1: The ratio of White actors in movie leading cast is positively related to the movie budget._


_H2: The movie market performance is negatively related to the ratio of White actors in the
movie leading cast_


_H3: The ratio of women in the movie leading cast is negatively related to the average age of the leading cast._


_H4: The movie market performance is negatively related to the ratio of female actors in the
movie leading cast_


_H5: The interaction of the ratio of actors from different ethnicities and the ratio of women has
a significant impact on movie market performance_

![Hypotheses Diagram](images/hypotheses_image.png)


**Given the unsurprising implications of H1 and H3, we chose explore the direct effects of the leading casts diversity on movie market performance, H2, H4 and H5.**


### Data Replication

**Movie Data:** Fang used the IMDB 5000 dataset from the Kaggle website. It contains 28 variables for the top rated 5043 movies, spanning across 100 years in 66 countries for the year 2007-2016. He used the gross values from this dataset to calculate attendance by dividing gross revenue by average ticket price in each year, using U.S. average movie ticket price data from 2007 to 2016 in Statista (https://www.statista.com).


**Demographic Data:** Fang collected race, gender, and age information of actors using NNDB (https://www.nndb.com), where celebrities’ gender, race, birthday, and nationality information is aggregated as well as a Kairos, a facial recognition ML API (https://www.kairos.com) to identify casts’ race, gender, and age by their photos on their IMDb personal page.

**Differences in our data collection process:** Fang selected the top 30 films per year from 2006-17, due to time constraints we selected the top 10 films per year. We chose not to use the Kairos ML facial recognition due to low confidence levels and the bias of the black-box algorithm, preferring as researchers to make categorising decisions and reflect on our biases.


### Analysis Replication Process & Bootstrapping:
We used the OLS and control function approach to replicate his results. Given our smaller sample size (n=100 vs n=300), we implemented a bootstrapping procedure to assess the variability and robustness of our Ordinary Least Squares (OLS) regression coefficients. In general, we found a high variability of coefficients for all of our models which suggest our estimates are sensitive to the specific sample we used, explaining a divergence between our results and Xi Fang's results.

### Installation instructions for user
```bash
pip3 install -r build_requirements.txt
```

This will install all the prerequisites.

To fetch the data, run:

```bash
python3 fetch_data.py
```

To build the book, run:

```
jupyter-book build .
```

The book build appears in the `_build/html` directory.  You can open it with your browser.


### Guide for users
- Start: Relication_Cleaning --> Replication_Results --> Further_Analysis

- Find a link to the Fan, X., 2021 study here: [Fan, X., 2021](https://atrium.lib.uoguelph.ca/server/api/core/bitstreams/6c82a2c1-57ba-4963-b09e-3942e3410421/content)
- Download the dataset from kaggle here: [IMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

### Suggestions for contributors and future analyses
The points below are suggestions for contributors to enhance this research:
-Analyze audience preferences by genre to identify key attendance drivers.
-Investigate how historical and cultural contexts within genres affect audience engagement.
-Assess the impact of genre-specific marketing on the success of movies with diverse casts.
-Study genre receptivity to diverse casts to find potential for broader storytelling.
- Improve the quality of the sample size from the top 10 highest budgeting movies per year to the highest 30 budgeting movies per year
- Incorporate further control variables such as release season (cite), movie duration, and IMDB movie score
- Investigate ethnicity as the independent variable as a more nuance substitute for race
- Investigate sexuality as an independent variable

### List of known issues
- The sample size used is one third of the sample used by Xing Fan, leading to the inability to replicate exact results as highlighted in our bootstrapping.

### References
Eschholz, Sarah et al. “SYMBOLIC REALITY BITES: WOMEN AND RACIAL/ETHNIC MINORITIES IN MODERN FILM.” Sociological Spectrum 22 (2002): 299 - 334.

Fan, X., (2021). The Influence of Movie Main Cast’s Diversity on Attendance (Doctoral dissertation, University of Guelph). Available at: https://atrium.lib.uoguelph.ca/server/api/core/bitstreams/6c82a2c1-57ba-4963-b09e-3942e3410421/content (Accessed: 27/12/2023)

Harris, A. (2016). Industry Folks Are Really Trying to Make the “Diversity Doesn’t Sell Overseas” Mantra Happen. [online] Available at: http://www.slate.com/blogs/browbeat/2016/03/30/the_hollywood_reporter_on_empire_s_global_ratings_is_the_latest_attempt.html (Accessed: 27/12/2023)

Iasiello, Carmen. “Underrepresentation of minorities in hollywood films: An agent based modeling approach to explanations.” 2017 Winter Simulation Conference (WSC) (2017): 4582-4583.

Karniouchina, Ekaterina V. et al. “Women and Minority Film Directors in Hollywood: Performance Implications of Product Development and Distribution Biases.” Journal of Marketing Research 60 (2023): 25 - 51.

Moore, E. E., & Coleman, C. (2015). Starving for diversity: Ideological implications of race representations in The Hunger Games. _The Journal of Popular Culture_, 48(5), 948. Available at: https://digitalcommons.tacoma.uw.edu/cgi/viewcontent.cgi?article=1785&context=ias_pub (Accessed: 27/12/2023)

Roxborough, S. (2016). America’s TV Exports Too Diverse for Overseas. [online] Available at: http://www.hollywoodreporter.com/news/americas-tv-exports-diverse-overseas-879109 (Accessed: 27/12/2023)

Scott, Nicholas A., and Janet Siltanen. "Intersectionality and quantitative methods: Assessing regression from a feminist perspective." International Journal of Social Research Methodology 20.4 (2017): 373-385.





