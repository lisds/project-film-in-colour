# Project Film in Colour

## Replication Results

#### Read the data


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


#READ THE DATA
clean_df = pd.read_csv("data/clean_data.csv")
clean_df.head()
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



### Replicating Data Analysis: Ordinary Least Squared (OLS) Regression including Control Function

First, investigate the data for any stand-out statistics

#### Summary Statistics


```python
data_cleaned_summary_stats = clean_df.describe()
data_cleaned_summary_stats
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
      <th>white_proportion</th>
      <th>female_proportion</th>
      <th>average_age</th>
      <th>attendance_in_mills</th>
      <th>budget_in_mills</th>
      <th>highest_starpower</th>
      <th>genre_action (Dummy)</th>
      <th>genre_comedy (Dummy)</th>
      <th>genre_drama (Dummy)</th>
      <th>genre_other (Dummy)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.823333</td>
      <td>0.223333</td>
      <td>40.796667</td>
      <td>26.08000</td>
      <td>178.110000</td>
      <td>85.650000</td>
      <td>0.79000</td>
      <td>0.100000</td>
      <td>0.140000</td>
      <td>0.120000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.239036</td>
      <td>0.246434</td>
      <td>8.426602</td>
      <td>17.38784</td>
      <td>37.617129</td>
      <td>287.178415</td>
      <td>0.40936</td>
      <td>0.301511</td>
      <td>0.348735</td>
      <td>0.326599</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.333333</td>
      <td>4.00000</td>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>35.250000</td>
      <td>14.00000</td>
      <td>150.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>40.666667</td>
      <td>22.00000</td>
      <td>175.000000</td>
      <td>3.500000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>46.333333</td>
      <td>33.25000</td>
      <td>200.000000</td>
      <td>27.250000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>64.333333</td>
      <td>101.00000</td>
      <td>300.000000</td>
      <td>2380.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Explanation:** 

#### Correlation Matrix


```python
#identifying initial correlations
correlation_matrix = clean_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

#code below saves image to our images folder in our directory as 'correlation_plot.png'
plt.savefig('images/correlation_plot.png')
```


    
![png](output_6_0.png)
    



    <Figure size 640x480 with 0 Axes>


**Explanation:** There are no strong correlations either positive or negative

### Pairplots

#### White proportion and attendance


```python
pairplot_data = clean_df[['white_proportion', 'attendance_in_mills']]

# Create a pairplot
sns.pairplot(pairplot_data)
plt.show()

#code below saves image to our images folder in our directory as 'white_pairplot.png'
plt.savefig('images/white_pairplot.png')
```


    
![png](output_9_0.png)
    



    <Figure size 640x480 with 0 Axes>


#### Female proportion and attendance


```python
pairplot_data = clean_df[['female_proportion', 'attendance_in_mills']]

# Create a pairplot
sns.pairplot(pairplot_data)
plt.show()

#code below saves image to our images folder in our directory as 'female_pairplot.png'
plt.savefig('images/female_pairplot.png')
```


    
![png](output_11_0.png)
    



    <Figure size 640x480 with 0 Axes>


## OLS Regression

### Standardising results for comparability


```python
from sklearn.preprocessing import StandardScaler
# Create a scaler object
scaler = StandardScaler()

clean_df['white_female_interaction'] = (clean_df['white_proportion']* clean_df['female_proportion'])

# List of columns to standardize (all except the dependent variable)
columns_to_standardize = ['white_proportion',
    'female_proportion',
    'white_female_interaction',
    'budget_in_mills',
    'average_age',
    'highest_starpower',
    'genre_action (Dummy)',
    'genre_comedy (Dummy)',
    'genre_drama (Dummy)',
    'genre_other (Dummy)',
    'content_rating (Dummy)_PG',
    'content_rating (Dummy)_PG-13',
    'content_rating (Dummy)_R']  

# Standardize these columns
clean_df[columns_to_standardize] = scaler.fit_transform(clean_df[columns_to_standardize])
```


```python
from sklearn.preprocessing import StandardScaler
# Create a scaler object
scaler = StandardScaler()

clean_df['white_female_interaction'] = (clean_df['white_proportion']* clean_df['female_proportion'])

# List of columns to standardize (all except the dependent variable)
columns_to_standardize = ['white_proportion',
    'female_proportion',
    'white_female_interaction',
    'budget_in_mills',
    'average_age',
    'highest_starpower',
    'genre_action (Dummy)',
    'genre_comedy (Dummy)',
    'genre_drama (Dummy)',
    'genre_other (Dummy)',
    'content_rating (Dummy)_PG',
    'content_rating (Dummy)_PG-13',
    'content_rating (Dummy)_R']  

# Standardize these columns
clean_df[columns_to_standardize] = scaler.fit_transform(clean_df[columns_to_standardize])
```


```python
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Extract relevant columns from clean_df
X1 = clean_df['white_proportion']
X2 = clean_df['female_proportion']
y = clean_df['attendance_in_mills']

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
ax.set_zlabel('Attendance in Millions')
ax.set_title('3D Regression Plot')

plt.show()

```


    
![png](output_15_0.png)
    


#### H2: White Proportion

Fang White_P Coef: (β = -0.6342, p = 0.0433) WhiteProportion has a statistically significant negative effect on LnAttendance, which indicates that low racial diversity in leading acting roles will damage movie market sales. 

Our White_P Coef : (β = 1.89, p = 0.687

#### H4: Female Proportion

Fang Female_P Coef: (β = -2.2104, p = 0.0030) FemaleProportion has a statistically significant negative effect (β = -2.2104, p = 0.0030) on LnAttendance.

Our Female_P Coef: (β = 7.69, p = 0.440)

#### H5: White*Female Proportion

Fang: (β = 2.6801, p = 0.0017) of the two-way interaction term on LnAttendance.

Our White*Female interaction coef: (β = -9.32, p = 0.351)


#### Comparing Results

There is significant divergence from Fangs results and ou results for each coefficient. We predict this is as a result of our smaller sample size. We will implement a bootstrapping procedure to assess the variability and robustness of the Ordinary Least Squares (OLS) regression coefficients. 


### Bootstrapping to assess coefficient variance for our sample



```python
n_bootstrap = 1000
n_coefficients = 4  # Adjust based on the number of predictors + 1 for the intercept
bootstrapped_coefs = np.zeros((n_bootstrap, n_coefficients))

for i in range(n_bootstrap):
    # Resample with replacement
    boot_sample = clean_df.sample(n=len(clean_df), replace=True)

    # Fit OLS model
    X = sm.add_constant(boot_sample[['white_proportion','female_proportion','white_female_interaction']])  # Add a constant for the intercept
    Y = boot_sample['attendance_in_mills']
    model = sm.OLS(Y, X).fit()
    
    # Store coefficients
    bootstrapped_coefs[i] = model.params

# Analyze the results
mean_coefs = np.mean(bootstrapped_coefs, axis=0)
std_coefs = np.std(bootstrapped_coefs, axis=0)

print(mean_coefs)
print(std_coefs)
```

    [26.12719072  0.25716368 -1.14677957 -2.23122629]
    [1.75732041 2.39424879 1.7331912  1.66897813]



```python
coef_labels = ['Intercept', 'White_P Coefficient','Female_P Coefficient','White*Female_P Coefficient']
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(bootstrapped_coefs, columns=coef_labels))
plt.title('H5: Bootstrapped Coefficients Boxplot')
plt.ylabel('Coefficient Value')
plt.show()
```


    
![png](output_18_0.png)
    


## Limitations

- We used a sample size 1/3 the size of Xing Fan's
- Difficult to interpret the 'control function' methodology used by Xing Fan
- The term race is relatively ambigious and should be defined better
- Due to scope, we could not ontrol for all control variables stated in the paper
- Our results are quite different to Xing Fan's, so we cannot be confident in our attempt to replicate despite limitations


## Conclusions

In our replication study, we attempted to validate the findings of Fang's research on the impact of racial and gender diversity in leading acting roles on movie market sales. Our results, however, showed significant deviations from Fang's original findings.

H2: White Proportion Fang's study found a statistically significant negative effect of White Proportion on LnAttendance (β = -0.6342, p = 0.0433), suggesting that lower racial diversity in leading roles negatively impacts movie market sales. Contrarily, our replication yielded a positive coefficient (β = 1.89) but with a non-significant p-value (p = 0.687), indicating a lack of statistical evidence to support the same conclusion.

4Female Proportion In Fang's study, Female Proportion also had a significant negative effect on LnAttendance (β = -2.2104, p = 0.0030). Our study, however, showed a large positive coefficient (β = 7.69) but again, without statistical significance (p = 0.440).

In Fang's study, Female Proportion also had a significant negative effect on LnAttendance (β = -2.2104, p = 0.0030). Our study, however, showed a large positive coefficient (β = 7.69) but again, without statistical significance (p = 0.440).

H5: White*Female Proportion Interaction Fang reported a significant positive effect (β = 2.6801, p = 0.0017) of the interaction term on LnAttendance. Our findings were contrary, indicating a negative interaction effect (β = -9.32), but with a non-significant p-value (p = 0.351).




To address these discrepancies and better understand the robustness of our findings, we implemented a bootstrapping procedure. We found very high variance for our coefficents (σ = 2.58 , 5.04, 4.74)  This allowed us to explain the variability and stability of the OLS regression coefficients.



The notable discrepancies between Fang's results and ours, particularly in the direction and significance of the coefficients, may be attributed to our smaller sample size. This limitation underscores the potential influence of sample size on the reliability and generalizability of study outcomes.











```python

```
