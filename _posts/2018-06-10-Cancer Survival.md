---
layout: post
title: Project- How long will you survive after breast cancer surgery?
subtitle: Analysis of data related to breast cancer
bigimg:
    - /img/path.jpg
    
published: true  
author : Rishabh garg
tags: [project,data analysis]
---

# This is the case study done by my regarding 306 patients who were suffered  from breast cancer.

** The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer. **

 Number of Instances: 306 

 Number of Attributes: 4 (including the class attribute) 

 Attribute Information:
- Age of patient at time of operation (numerical) <br/>
- Patient's year of operation (year - 1900, numerical) <br/>
- Number of positive axillary nodes detected (numerical) <br/>
- Survival status (class attribute) 1 => the patient survived 5 years or longer, 2 = the patient died within 5 year <br/><br/>
Information from Kaggle -https://www.kaggle.com/gilsousa/habermans-survival-data-set/data

## Domain Information-
- Axiliary nodes are those nodes which are present in underarm area and  a normal body have about 20-40 axiliary nodes.The role they play is to drain the lymph (a clear or white fluid made up of white blood cells) produced from the breasts and surrounding areas, including the neck, the upper arms, and the underarm area.
- Positive axiliary nodes signifies the number of axiliary nodes which are damaged due to  breast cancer and they have to removed by surgeon during surgery.

## The connection-
- The axillary lymph nodes are usually the first set of lymph nodes where breast cancer will spread.

- And because the breast and armpit are close to each other, the lymph nodes are a common place where this type of cancer spreads.

- As a general rule, the more a cancer has spread from its starting point, the worse the prognosis may be for a person.

- Also, if the cancer has spread to the axillary lymph nodes, a doctor will usually recommend removing the lymph nodes during the surgery to remove the originating tumor.

- Lymph nodes are responsible for draining lymph fluid, so their removal can cause some side effects after surgery


```python
#importing required libraries for doing EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")
```


```python
#Let's load the dataset into Pandas Dataframe
data = pd.read_csv("C:\\Users\\Prince\\datasets\\haberman.csv",header=None,names=["Age","Year of operation","Number of positive axiliary nodes detected","Survival status"])
```


```python
# Let's analyse the shape of the data and as well as first five rows of data to know about the structure of dataset
print(data.shape)
data.head()
```

    (306, 4)
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Year of operation</th>
      <th>Number of positive axiliary nodes detected</th>
      <th>Survival status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>64</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>62</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>65</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>59</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>65</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Questions that can be answered :-
<br/>

**  Q1 - Does the success of surgery depends upon age of patient or not ? ** <br/>

** Q2 -  The number of positive axiliary nodes detected have anything related to the remaining life of patient ? ** <br/>

** Q3 - Does number of positive axiliary nodes detected directly depend upon age ? ** <br/>

** Q4 - Given this information can you classify whether the patient will more than 5 years or not ? ** <br/>



```python
#As column "Year of operation" of our dataset has value equal to "year -1900"
data["Year of operation"] = data["Year of operation"] + 1900    #Adding 1900 to make it more understandable
```


```python
#Let's have a look at descriptive statistics of the data
data.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Year of operation</th>
      <th>Number of positive axiliary nodes detected</th>
      <th>Survival status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>306.000000</td>
      <td>306.000000</td>
      <td>306.000000</td>
      <td>306.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.457516</td>
      <td>1962.852941</td>
      <td>4.026144</td>
      <td>1.264706</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.803452</td>
      <td>3.249405</td>
      <td>7.189654</td>
      <td>0.441899</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>1958.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44.000000</td>
      <td>1960.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>52.000000</td>
      <td>1963.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.750000</td>
      <td>1965.750000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83.000000</td>
      <td>1969.000000</td>
      <td>52.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>



<br/>
<br/>

### Takeaways -
- Our data has not any missing value in any of the column. Cheers!
- The age of the patients is centred around 52 with standard deviation equals to 10.83.
- Our data is uniformly distributed with respect to "Year of operations" observation as  we have almost same number of observations or rows for each year and which can be concluded by looking at Percentile values.
- The number of positive axiliary nodes detected are centred around 4 with standard deviation equals to 7 and with maximum value equal to 52.
- Survival staus has just 2 values 1 and 2
<br/>


```python
# Let's have a look at how many patients had survived more than 5 years out of 306 patients ?
data["Survival status"].value_counts()
```




    1    225
    2     81
    Name: Survival status, dtype: int64



<br/>
### 225 patients had survived more than 5 years out of 306 patients .
- It's great to see the survival rate of living more than 5 years after surgery is ***73.5%*** but there is a lot more to improve as it will be better if we can save more and more lives !
<br/>


```python
# Let's visualize the same by plotting count plot
sns.countplot(x="Survival status",data=data)
plt.show()
```


![png](/img/Cancer+Survival/output_13_0.png)


### Distribution of Number of positive axiliary nodes detected - Histogram with KDE


```python
fig = plt.figure(figsize=(12,12))
g = sns.FacetGrid(data,hue="Survival status",size=6)
g.map(sns.distplot,"Number of positive axiliary nodes detected",kde=True)
g.add_legend()
plt.show()
```


    <matplotlib.figure.Figure at 0x44061bf4a8>



![png](/img/Cancer+Survival/output_15_1.png)


<br/>
### Takeaways-
- We can not conclude anything about survival status from the Number of positive axiliary nodes detected by observing this distribution.
<br/>

### Correlation matrix -


```python
correlation = data.corr() #returns correlation between two columns for each set of two columns
sns.set()
sns.heatmap(correlation,annot=True,cmap="coolwarm") #visualizing it with the help of Heatmap
plt.show()
```


![png](/img/Cancer+Survival/output_18_0.png)


<br/>
### Takeaways-
- It is confirming our belief that there is no correlation between two columns of our data except between number of positive axiliary nodes detected and Survival status of 0.2

** It is also answering our Q3 - Does number of positive axiliary nodes detected directly depend upon age ? **
- As there is no correlation between the two hence, number of positive axiliary nodes detected  does not depend upon age of patient
<br/>

### Exploring the data further -

#### Let's look at the average values of Age and Number of positive axiliary nodes detected for different status of Survival after surgery to get our answer


```python
# Using pivot table for this purpose
data.pivot_table(index="Survival status",values=["Number of positive axiliary nodes detected","Age"],aggfunc=np.mean)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Number of positive axiliary nodes detected</th>
    </tr>
    <tr>
      <th>Survival status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>52.017778</td>
      <td>2.791111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53.679012</td>
      <td>7.456790</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualizing the answer given by pivot table using Boxplot
sns.boxplot(x="Survival status",y="Age",data = data)
plt.show()
```


![png](/img/Cancer+Survival/output_23_0.png)



```python
sns.boxplot(x="Survival status",y="Number of positive axiliary nodes detected",data = data)
plt.show()
```


![png](/img/Cancer+Survival/output_24_0.png)


<br/>
<br/>
# Conclusions :

<br/>
- It is clearly showing us that **Number of positive axiliary nodes detected is the most important feature for classification **  as there is a very large difference between the central value of Number of positive axiliary nodes detected for those survived more than 5 years and those who had not which is signifying that more number of positive axiliary nodes means that the cancer has extended to large level and thus there is very maximum probability that it will not let live the patient more than 5 years and thus answering our **Q2 -  The number of positive axiliary nodes detected have anything related to the remaining life of patient ? **
<br/>
- It is also confirming our initial belief of ** Q1 - Does the success of surgery depends upon age of patient or not ? ** <br/>
     No,the success of surgery does not depends upon the age of patient as there is very less difference between the  central value of age  for those who had survived for more than 5 years and those who hadn't.
<br/>
- A number of positive axiliary nodes detected does not depend upon age
<br/>
