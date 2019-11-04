#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Movie Data Analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>
# 
# <a id='intro'></a>
# ## Introduction
# 
# In this data analysis we will be invistigate the trends and properties of movies. 
# 
# 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
#  In this section of the report, Data will be loaded, check for cleanliness, and then trim and clean the dataset for analysis. 
# 
# 

# In[4]:



df = pd.read_csv('tmdb-movies.csv')
df.head()


# In[5]:


df.shape


# So there are 10866 movies and 21 columns

# In[6]:


df.info()


# from the above result we can see that there are some null values in the columns: cast, homepage, director, tagline, keyword,overview,  genre and production_companies.
# 

# In[7]:


df.describe()


# From the above table we can see:
# the average runtime for a movie is 102.07 Miniutes. 
# the average budget for a movie was 1.46 and the maximum was 4.250000e+08$
# 
# we can also notice alot of zeros in revenue, budget and runtime,
# so we have to decide if we can just drop them or replace them with null values so the integrity of the data wont 
# be harmed if there were alot of them.
# So I decided that I was going to count how many zeros exactly I have in these columns:

# In[8]:


#counting zero values in budget column:
rows, col = df.query('budget == 0').shape
print('There are {} rows and {} columns'.format(rows, col))


# In[9]:


#counting zero values in revenue column:
rows, col = df.query('revenue == 0').shape
print('There are {} rows and {} columns'.format(rows, col))


# In[10]:


#counting zero values in runtime column:
rows, col = df.query('runtime == 0').shape
print('There are {} rows and {} columns'.format(rows, col))


# Because there are too much data in the budget and the revenue columns, I decided that I will keep them and replace
# the Zero values with Null values.

# 
# ### Data Cleaning 

# In[11]:


#First we will drop the columns that we will not be using in our analysis:
col = ['imdb_id', 'homepage', 'tagline', 'overview', 'budget_adj', 'revenue_adj']
df.drop(col, axis=1, inplace=True)

#checking to see if the columns have been deleted
df.head()


# In[12]:


#Second we will drop any duplicate rows in the data (if any):
df.drop_duplicates(inplace=True)

rows, col = df.shape
print('There are now {} columns and {} entries of movie data'.format(col, rows-1))


# In[13]:


#Third we will handle the zero vlues in the budget, revnue and rubtime column by replacing them with 
#Null values

zero_col = ['budget', 'revenue', 'runtime']
df[zero_col] = df[zero_col].replace(0, np.NAN)


# In[14]:


#lastly we will Drop all the NaN values
# Subset helps to define in which columns to look for missing values
df.dropna(subset = zero_col, inplace = True)
rows, col = df.shape

print('Now there are only {} entries'.format(rows-1))


# In[15]:


#lets take a final look after cleaning the data:
df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### What are the average popularity of movies by the Release year?

# In[16]:


df.groupby('release_year')['popularity'].mean()


# In[17]:


df.groupby('release_year')['popularity'].mean().plot(kind='line', figsize = (18, 8), color = 'r')
plt.title('Average Popularity of Movies over the Years', fontsize = 20)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Popularity', fontsize = 18);


# From the above graph, We can see that on average the popularity of movies is increasing year by year until it reached its peak in 2015. 

# ### Which Movies are the most popular of all time? 

# In[18]:


df.describe()


# from the table above we can see that the mean of the popularity is 1.19, so we can assume that any movie that got more than 1.19 in popularity is considered as omne of the most popular movies of all the time

# In[19]:


# creating a list of columns that will be viewed
col = ['original_title', 'cast', 'director', 'budget', 'revenue','popularity']

# Using query function to show records of movies which have a popularity of more than 1.19
# Also using sort_values function to make sure it is sorted according to the popularity column

df.query('popularity>1.16')[col].sort_values('popularity', ascending = False).head(5)


# ### What is the average profit of movies over the years ?

# In[16]:


# We will create a column called profit and add it to the table
df.insert(0, 'profit', df['revenue'] - df['budget'])
df.head()


# In[17]:


df.groupby('release_year')['profit'].mean().plot(kind = 'line', figsize = (18, 8), color = 'b')
plt.title('Average Profit of movies through the years', fontsize = 20)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Profit', fontsize = 18)


# We can see from the graph that the average profit of a movie reached its peak in th70s and in the 90s, on the contrary it reached its lowest point in the 60s.

# ### What variables affect the revinue of a movie the most?

# In[18]:


#To know what variables have an effect on the revenue we can find the correlation between the variables:
df.corr()


# From the above table, we can notice a strong correlation with popularity, budget and vote count.
# also we can see there is a Weak correlation between the revenue and the runtime of the movie

# In[19]:


#We can visulaize the above correlation relationship using a scatter plot:
sns.regplot(x = df['revenue'], y = df['popularity'], fit_reg = False)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16
fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size
plt.title('Relationship between Revenue and popularity', fontsize = 20)
plt.xlabel('Revenue', fontsize = 18)
plt.ylabel('Popularity', fontsize = 18);


# In[20]:


sns.regplot(x = df['budget'], y = df['revenue'], fit_reg = False)
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 16
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

plt.title('Relationship between budget and revenue', fontsize = 20)
plt.xlabel('Budget', fontsize = 18)
plt.ylabel('Revenue', fontsize = 18);


# In[21]:


sns.regplot(x = df['vote_count'], y = df['revenue'], fit_reg = False)
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 16
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

plt.title('Relationship between vote_count and revenue', fontsize = 20)
plt.xlabel('Vote Count', fontsize = 18)
plt.ylabel('Revenue', fontsize = 18);


# In[22]:


sns.regplot(x = df['runtime'], y = df['revenue'], fit_reg = False)
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 16
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

plt.title('Relationship between runtime and revenue', fontsize = 20)
plt.xlabel('Runtime', fontsize = 18)
plt.ylabel('Revenue', fontsize = 18);


# <a id='conclusions'></a>
# ## Conclusions
# 
# From our findings we can say that:
# The average popularity of movies is increasing year by year and it reached its peak in 2015.
# Also the average profit of a movie reached its peak in the 70s and in the 90s, on the contrary it reached its lowest point in the 60s. We also found out that the movie Jurrasic world was the most popular movie of all time.
# At last we concluded that the popularity, budget and vote count were the most affective variables on the revnue of the movies and the runtime of a movie had a very low impact on the movie's revnue .
# 
# 

# ## Limitations

# The data was not very  clear , some columns looked ambiguous to me, such as the budget and revenue columns did not have a currency specified so there may be some differences due to fluctuating exchange rates.
# There might have been more variables attainable from the movies that can help give a more precise analysis such as the who was the producer or who was the writer of the movie for example.
# In the data wrangling phace: Rows with NaN values were dropped, hence a lot of key data might have been lost in the process, and rows with zero values were checked and because there were too much missing data in the budget and the revenue columns, I decided that I will keep them and replace the Zero values with Null values, which may have affected the analysis result.
# 
# 
# 
# 
# 

# In[23]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




