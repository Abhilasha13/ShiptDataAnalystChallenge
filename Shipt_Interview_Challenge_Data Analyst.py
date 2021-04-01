#!/usr/bin/env python
# coding: utf-8

# ## Shipt – Interview Challenge – Data Analyst

# ### Required Questions

# In[321]:


# Importing the libraries
import pandas as pd
import numpy as np

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# In[322]:


# Reading the InterviewData_Cost.csv file
InterviewData_Cost = pd.read_csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Cost.csv')


# In[323]:


# Data in InterviewData_Cost
InterviewData_Cost


# In[324]:


# Reading the InterviewData_Rev.csv file
InterviewData_Rev = pd.read_csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Rev.csv')


# In[325]:


# Data in InterviewData_Rev 
InterviewData_Rev


# ### Q1. Join these two data sets by “date” and “source_id”, returning all rows from both regardless of whether there is a match between the two data sets
# 
# <b> Ans: Since we need all rows from both InterviewData_Cost and InterviewData_Rev regardless of whether there is a match between the two data sets, I have joined them using outer join on 'data' and 'source_id'. </b>

# In[326]:


# Join the two tables using outer join
data_all = pd.merge(InterviewData_Cost,InterviewData_Rev,on=['date','source_id'],how='outer')
data_all


# In[327]:


data_all['source_id'].unique()


# ### Q2. Join these two data sets by “date” and “source_id”, returning only the rows from the “Cost” file that have no corresponding date in the “Revenue” file.
# 
# <b> Ans: Here, I have used left join to join the two data sets. As we need all the rows from InterviewData_Cost, it is placed on the left side of the query. </b>

# In[328]:


# Join the two tables using left join
pd.merge(InterviewData_Cost,InterviewData_Rev,on=['date','source_id'],how='left')


# ### Q3. Using your result from #1:
# 
# ### a. What are the Top 4 sources (“source_id” values) in terms of total revenue generation across this data set?
# 
# ### b. How would you visualize the monthly revenue for those Top 4 sources?

# ### a. What are the Top 4 sources (“source_id” values) in terms of total revenue generation across this data set?

# In[329]:


# data_all_total contains the aggregate of revenue for each source_id 
data_all_total = data_all.groupby(data_all['source_id']).agg({'revenue':'sum'}).reset_index()
data_all_total


# In[330]:


#Sorting the data obtained from previous query in descending order of total revenue
data_all_total.sort_values(by=['revenue'], inplace=True, ascending=False)
data_all_total


# In[331]:


#Top 4 source_ids with highest revenue
data_all_total_top4 = data_all_total.head(4)
data_all_total_top4


# In[332]:


# Creating a list with only the top 4 source_ids
data_source_id4 = data_all_total_top4['source_id'].tolist()


# Top 4 sources (“source_id” values) in terms of total revenue generation across this data set

# In[333]:


# Data in the list data_source_id4 showing top 4 “source_id” values in terms of total revenue generation across data set
data_source_id4


# ### b. How would you visualize the monthly revenue for those Top 4 sources?

# In order to visualize the monthly revenue, I have created a dataframe having the top 4 source_ids and the aggregated revenue generated in each month. 

# In[341]:


# Selecting records which belongs to the top 4 source_ids having highest revenue
data_all_filtered = data_all[data_all['source_id'].isin(data_source_id4)]


# In[342]:


data_all_filtered


# In[343]:


# Unique source_ids
data_all_filtered.source_id.unique()


# In[344]:


# Checking the data types
data_all_filtered.dtypes


# In[345]:


# Converting date from object to date format 
data_all_filtered['date'] = pd.to_datetime(data_all_filtered['date'], format = '%m/%d/%y')


# According to the question, we need the monthly revenue, so I have selected the sum of revenue for each source_id in each month in year 2014.

# In[346]:


# data_all_date contains the monthly aggregate of revenue for each source_id 
data_all_date = data_all_filtered.groupby([data_all_filtered['date'].dt.month,data_all_filtered['source_id']]).agg({'revenue':'sum'})


# In[347]:


data_all_date


# Visualizing the monthly revenue for those Top 4 sources

# In[348]:


data_all_vis = data_all_date.reset_index()


# In[349]:


# Plot to visualize the data
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

data_all_vis.plot()


# In[350]:


# Monthly revenue for Top 4 sources plotted together
pv = pd.pivot_table(data_all_vis, index=data_all_vis.date, columns=data_all_vis.source_id,
                    values='revenue')
pv.plot()


# In[351]:


# Single plot for each source_id to understand the revenue pattern of each source
data_all_vis.groupby('source_id').plot(y='revenue')


# ### Questions 4 and 5 deal with “InterviewData_Activity.csv”.
# 
# ### 4.Assuming you’ve read the data into an R object called activity_data, run the following code to build a basic logistic regression model:

# In[352]:


# Reading the InterviewData_Activity.csv file
activity_data = pd.read_csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Activity.csv')


# In[353]:


# Data in activity_data
activity_data


# In[354]:


# Importing the library for statistical model
import statsmodels.api as sm


# In[355]:


dummy_genders = pd.get_dummies(activity_data['gender'], prefix = 'gender')
dummy_metro = pd.get_dummies(activity_data['metropolitan_area'], prefix = 'metro_area')
dummy_device = pd.get_dummies(activity_data['device_type'], prefix = 'device')


# In[356]:


activity_data


# In[357]:


cols_to_keep = ['active', 'age']


# In[358]:


activity_data = activity_data[cols_to_keep].join(dummy_genders.ix[:, 'gender_M':])
activity_data = activity_data.join(dummy_metro.ix[:, 'metro_area_Birmingham':])
activity_data = activity_data.join(dummy_device.ix[:, 'device_Mobile':])
activity_data = sm.add_constant(activity_data, prepend=False)


# In[359]:


explanatory_cols = activity_data.columns[1:]
full_logit_model = sm.GLM(activity_data['active'],activity_data[explanatory_cols], family=sm.families.Binomial())


# In[360]:


result = full_logit_model.fit()


# In[361]:


activity_data


# Apply this model to the same data that the model was trained on and assess the prediction accuracy.

# In[362]:


# Applying the model on activity_data to assess the prediction accuracy
predictions = result.predict(activity_data[explanatory_cols])
predictions


# In[363]:


# Converting probability to binary category active = 1 if x >=0.5, active = 0 if x < 0.5 
predictions_nominal = [1 if x > 0.5 else 0 for x in predictions]


# In[364]:


# Calculating the accuracy of the model
from sklearn import metrics
print('Accuracy: ',metrics.accuracy_score(activity_data['active'], predictions_nominal))


# The accuracy of the model is 58%

# ### 5. Split the data into training and test samples, and build a model over the training data using the following Python code:

# In[365]:


# Splitting the data into training and test set
training_data = activity_data[1:4000]
test_data = activity_data[4001:].copy()

training_logit_model = sm.GLM(training_data['active'],training_data[explanatory_cols],family=sm.families.Binomial())

training_result = training_logit_model.fit()


# In[366]:


training_result.summary()


# In[367]:


# Applying the model on test_data to assess the prediction accuracy
predictions_test = training_result.predict(test_data[explanatory_cols])


# In[368]:


# Converting probability to binary category active = 1 if x >=0.5, active = 0 if x < 0.5 
predictions_test_nominal = [1 if x > 0.5 else 0 for x in predictions_test]


# In[369]:


# Calculating and printing the accuracy 
print('Accuracy: ',metrics.accuracy_score(test_data['active'], predictions_test_nominal))


# The accuracy of the model was 21%, which was less than the accuracy in ques.4. This is because in the previous question, the model was evaluated on the same data on which the model was trained i.e. the activity_data. So when we evaluate the model that we trained we get high scores, this shows how well our model learnt from our training data.
# 
# However, in question 5, the model is evaluated on a new data set and so the accuracy of the model is reduced.
# 
# One of the major reason for low accuracy is overfitting. Overfitting models the training data very well.
# 
# It takes place when a model learns the detail and noise in the training data well and negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. But, these concepts do not apply to new data.

# ### 6.This data comes from a subset of userdata JSON blobs stored in our database. Parse out the values (stored in the “data_to_parse” column) into four separate columns. So for example, the four additional columns for the first entry would have values of “N”, “U”, “A7”, and “W”. You can use any R functions/packages you want for this.

# In[370]:


# Reading the csv file
InterviewData_Parsing = pd.read_csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Parsing.csv')


# In[371]:


# Data in InterviewData_Parsing
InterviewData_Parsing


# In[372]:


# Importing library 're' and using it to remove the special characters from the column data_to_parse
import re

InterviewData_Parsing.data_to_parse = InterviewData_Parsing.data_to_parse.apply(lambda x:' '.join(re.findall(r'\w+', x)))


# In[373]:


# Splitting the column data_to_parse into the 4 additional columns
InterviewData_Parsing[['data_to_parse_val','col1','col2','col3','col4']] = pd.DataFrame([x.split(' ') for x in InterviewData_Parsing['data_to_parse'].tolist()]) 


# In[374]:


# Displaying the data after parsing
InterviewData_Parsing


# ## Additional Questions – Pick One

# A) Within our web and mobile apps, members can generally find items through search and/or the product
# category tree (note that you can also search after clicking into a product category, in which case the
# search is filtered by the chosen category). Let's say that we decide to test a different product category
# tree. The Product team asks for your help in setting up the test and calling the results. How would you
# help them: (i) figure out how long we should run this test; (ii) decide what metric to measure; (iii) and
# then evaluate the test?

# <b> How would you help them: </b>
# The new product category tree can be tested with the help of A/B testing. A/B testing refers to a randomized experimentation process where two or more versions of a web page or page elements are shown to different segments of website visitors at the same time to determine which version leaves the maximum impact and drives business metrics.
#  
# This is one of the simplest ways to understand the performance of any website using statistical analysis while spending less time and money.
#  
# <b> i. Figure out how long we should run this test: </b>
# The test should run at least for one complete week. This is because for a few websites the conversion rates can be low during weekdays and can increase over the weekends and vice-versa. Considering that the web and mobile app is for Shipt, it can be possible that working people visit the website during weekends, or stay-at-home mothers can visit the website during weekdays. So, to get valid test data, tests should run throughout the week so as to include all possible fluctuations. The duration will also depend on the website traffic. If the traffic is lower, the test will have to run for a longer time.
#  
# <b> ii. Decide what metric to measure: </b>
# One of the most important metrics will be click-through rate. This will give the percentage of people that clicked on the search product category. This helps to measure the success of marketing efforts. Other metrics would be the bounce rate i.e. the percentage of visitors who clicked on the product category but did not stay there and left, the conversion rate, and the number of people who added their products into the cart after searching.
#  
# <b> iii. Evaluate the test?: </b>
# In order to evaluate the test, the different metrics that have been measured through the Website Optimizer should be considered. For instance, a higher click-through rate shows that the search category is engaging and people are interested in clicking and navigating through the category. A 2% click-through-rate is usually considered good. Similarly, if the bounce rate is high then it shows that the visitors did not find the page or content attractive and so they did not stay for a long time.

# In[ ]:




