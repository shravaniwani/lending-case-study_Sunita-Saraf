#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[32]:


#Importing Necessary Library

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.core.display import HTML
import plotly.express as px #plotting
warnings.filterwarnings('ignore') # if there are any warning due to version mismatch, it will be ignored


# In[33]:


#data loading
loan= pd.read_csv('loan.csv') 


# In[34]:


# Printing the data(first 5 rows)

loan.head() 


# In[22]:


#Basic information about the data



# In[35]:


loan.info()


# In[36]:


loan.shape


# In[37]:


#Number of rows and columns
print('Number of Columns:',loan.shape[1])
print('Number of Rows:',loan.shape[0])



# In[38]:


#Number of missing values
print('Number of missing values:',loan.isnull().sum().sum())


# In[39]:


#Number of unique values
print('Number of unique values:',loan.nunique().sum())


# In[40]:


#Number of duplicates
print('Number of duplicates:',loan.duplicated().sum())


# In[30]:


#Missing Value Check
loan.isnull().sum()


# In[41]:


#removing null values

loan.dropna(axis = 1, how = 'all', inplace = True)
loan.head()


# In[42]:


# Checking column with large amount of null values(in percentage) and irrelevant columns
print((loan.isnull().sum()/loan.shape[0]*100).round(2).sort_values(ascending=False))


# In[43]:


#Removing column with 50% or more null values as it will reduce the impact on analysis
loan = loan.loc[:,loan.isnull().sum()/loan.shape[0]*100<50]
# Shape of the dataframe after removing columns
print(loan.shape)


# In[75]:


# Checking columns again for null value percentage
print((loan.isnull().sum()/loan.shape[0]*100).round(2).sort_values(ascending=False))


# In[76]:


#We have removed the column which contain more than 50% missing values
#this will reduce impact on analyis and imporve the accuracy of the analysis.


# In[77]:


# Columns in the dataframe
print(loan.columns)


# In[78]:


# Checking for missing values across the rows
print((loan.isnull().sum(axis=1)).max())


# In[61]:


#we have removed missing value of column, similarly missing value in row are checked and it is very less
#hence we will move ahead with further process

loan.shape


# In[83]:


#now we will remove irrelevant column, means column that are can not contribute to the analysis

loan=loan.drop(['id','zip_code','funded_amnt_inv','url','desc','delinq_2yrs','earliest_cr_line','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','last_credit_pull_d','application_type'],axis = 1)
print(loan.shape)
#we have removed columns which are not useful in analysis, seen shape of dataframe after removing irrilivant columns


# In[86]:


loan.columns


# In[87]:


# Checking columns for 1 unqiue values and removing them

print(loan.nunique().sort_values(ascending=True))

loan = loan.loc[:,loan.nunique()>1]

#shape of datafram after removing
print(loan.shape)


# In[88]:


#checking columns in dataframe
print(loan.columns)


# In[40]:


# Checking for missing values across the dataframe
print(loan.isnull().sum().sort_values(ascending=False))


# In[ ]:


#emp_length is having 1075 
#pub_rec_bankruptcies haing 697 
#both have null values these can be removed of fixed depending upon objective of the analysis


# In[89]:


#Checking 'emp_length' columns for feasibility of inserting null values
print(loan.emp_length.value_counts())


# In[90]:


# Removing null values in 'emp_length' columns
loan = loan.dropna(subset=['emp_length'])

#checking Shape of the dataframe
print(loan.shape)


# In[91]:


#Checking 'pub_rec_bankruptcies' columns for feasibility of inserting null values
print(loan.pub_rec_bankruptcies.value_counts())


# In[92]:


#adding 0 for null values in 'pub_rec_bankruptcies'
loan.pub_rec_bankruptcies.fillna(0,inplace=True)


# In[48]:


#checking for missing values across the dataframe
print(loan.isnull().sum())


# In[93]:


#checking shape of dataframe
print(loan.shape)


# In[ ]:


#We are done with fixing and removing null values to improve the accuracy of the analysis


# In[94]:


#Number of duplicates
print('Number of duplicates:',loan.duplicated().sum())

#No duplicates found


# In[51]:


# Checking information about the dataframe
print(loan.info())


# In[ ]:


# Correcting data type and format for columns in the dataframe



# In[95]:


# Setting decimal point limit for all data 
for x in loan.columns:
    if(loan[x].dtype=='float64'):
      loan[x]=loan[x].round(2)
      
loan.head()


# In[96]:


#we are removing records with current loan as we will required only Completed loan or Defaulted loan so filtering it

loan = loan[loan.loan_status != "Current"]
loan.loan_status.unique()



# In[97]:


#checking shape of dataframs
print(loan.shape)


# In[ ]:


#Now data is cleaned
#also fixed accourding to the requurement 
#now its time to select columns which we feel require for analysis
#we have to divide column into categorical and numerical


# In[99]:


#divding the column as per categorical and numerical
cat_cols = ['term','grade','sub_grade','emp_length','home_ownership','verification_status','purpose','addr_state']
cont_cols=['loan_amnt','int_rate','annual_inc','dti','pub_rec_bankruptcies','issue_year','issue_month']
id_cols=['id']
result_cols=['loan_status']


# In[100]:


# Checking outlier values in continuous columns via box plot
#The continuous columns are loan_amnt, int_rate, annual_inc, dti whereas the categorical columns are term, grade, sub_grade, emp_length, home_ownership, verification_status, purpose, addr_state, issue_month, issue_year, pub_rec_bankruptcies
#We will check outlier using box plot and then remove the outliers as per requirement



# In[69]:


#starting with loan_amnt
#using plotly for interactive interaction and value retrival from chart for upper fence.
px.box(loan,x='loan_amnt',width=750,height=350,title='Distribution of Loan Amount',labels={'loan_amnt':'Loan Amount'}).show()

#upper fence turns out to be 29.175k =29175 whereas max is 35k=35000 which is not much more then upper fence thus will not have much impact on the analysis.


# In[101]:


#starting with int_rate
#using plotly for interactive interaction and value retrival from chart for upper fence.
px.box(loan,x='int_rate',width=750,height=350,title='Distribution of Interest Rate',labels={'int_rate':'Interest Rate'}).show()
#upper fence turns out to be 22.64 whereas max is 24.4 which is not much more then upper fence thus will not have much impact on the analysis.


# In[71]:


#starting with annual_inc
#using plotly for interactive interaction and value retrival from chart for upper fence.
px.box(loan,x='annual_inc',width=750,height=350,title='Distribution of Annual Income of the Burrower',labels={'annual_inc':'Annual Income'}).show()
#upper fence turns out to be 146k whereas max is 6000k which is much from upper fence thus we will remove the outliers in column annual_inc.


# In[102]:


## checking trend of values in annual_inc using line chart to find the appropriate quantile to use to remove outliers
px.line(sorted(loan.annual_inc),width=750,height=350,title='Trend of Annual Income',labels={'value':'Annual Income','index':'Position in Data'}).show()


# In[ ]:


#As it can be observed from the line chart, the annual_inc is increasing in expontntial format around 99th percentile. Thus we can remove values greater than 99th percentile.


# In[103]:


#removing outliers in annual_inc greater than 99th percentile

loan = loan[loan.annual_inc<=np.percentile(loan.annual_inc,99)]


# In[78]:


## checking trend of values in annual_inc using line chart to find the appropriate quantile to use to remove outliers
px.line(sorted(loan.annual_inc),width=750,height=350,title='Trend of Annual Income',labels={'value':'Annual Income','index':'Position in Data'}).show()

#As the trend is more compatible with the analysis, we can proceed.


# In[104]:


#starting with dti
#using plotly for interactive interaction and value retrival from chart for upper fence.
px.box(loan,x='dti',width=750,height=350,title='Distribution of Debt To Income Ratio',labels={'dti':'DTI ratio'}).show()
#there are no outliers in dti hence we can move ahead with analysis.


# In[81]:


#Univariate Analysis

# Loan status 
print(loan.loan_status.value_counts()*100/loan.loan_status.count())



# In[105]:


sns.countplot(x = 'loan_status', data = loan)


# In[106]:


loan.sub_grade = pd.to_numeric(loan.sub_grade.apply(lambda x : x[-1]))
loan.sub_grade.head()


# In[107]:


fig, ax = plt.subplots(figsize=(12,7))
sns.set_palette('colorblind')
sns.countplot(x = 'grade', order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] , hue = 'sub_grade',data = loan[loan.loan_status == 'Charged Off'])


# In[108]:


sns.countplot(x = 'grade', data = loan[loan.loan_status == 'Charged Off'], order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])


# In[109]:


#checking unique values for home_ownership
loan['home_ownership'].unique()


# In[110]:


#replacing 'NONE' with 'OTHERS'
loan['home_ownership'].replace(to_replace = ['NONE'],value='OTHER',inplace = True)


# In[111]:


#checking unique values for home_ownership again
loan['home_ownership'].unique()


# In[92]:


fig, ax = plt.subplots(figsize = (6,4))
ax.set(yscale = 'log')
sns.countplot(x='home_ownership', data=loan[loan['loan_status']=='Charged Off'])


# In[112]:


#analyzing purpose

fig, ax = plt.subplots(figsize = (12,8))
ax.set(xscale = 'log')
sns.countplot(y ='purpose', data=loan[loan.loan_status == 'Charged Off'])


# In[116]:


loan.emp_length=loan.emp_length.apply(lambda x: x.replace('years','').replace('+','').replace('< 1','0.5').replace('year','')).astype(float)


# In[119]:


# Distribution of emp_length based on loan_status
plt.figure(figsize=(10,5))
sns.countplot(data=loan,x='emp_length',hue='loan_status')
plt.xlabel('Employment Length in years')
plt.ylabel('Count')
plt.title('Distribution of Employment Length For Loan Status',fontsize=12)
plt.show()


# In[ ]:


#the Employees with 10+ years of experience are likely to default and have higher chance of fully paying the loan.


# In[ ]:


#Bivariate Analysis


# In[121]:


loan.int_rate=loan.int_rate.apply(lambda x:str(x).replace('%','')).astype('float').round(2)


# In[122]:


# Comparison of interest rate based on grade
plt.figure(figsize=(10,5))
sns.boxplot(data=loan,x='int_rate',y='grade')
plt.xlabel('Interest Rate')
plt.ylabel('Grade')
plt.title('Comparison of Interest Rate Based On Grade',fontsize=12)
plt.show()


# In[ ]:


#the Grade represent risk factor thus we can say interst rate increases with the risk.


# In[123]:


# Comparison of DTI over grade for loan status
plt.figure(figsize=(10,5))
sns.barplot(data=loan,x='dti',y='grade',hue='loan_status')
plt.xlabel('DTI')
plt.ylabel('Grade')
plt.title('Comparison of DTI Based On Grade For Loan Status',fontsize=12)
plt.show()


# In[ ]:


#the Grade A which is lowest risk also has lowest DTI ratio which we can say that higher grade has lower rate of default.


# In[125]:


# Comparison of annual income to public record bankruptcy over loan status
plt.figure(figsize=(10,5))
sns.displot(y=loan.pub_rec_bankruptcies.astype('category'),x=loan.annual_inc,hue=loan.loan_status)
plt.xlabel('Annual Income')
plt.ylabel('Public Record Bankruptcies')
plt.title('Public Record Bankruptcies Vs Annual Income',fontsize=12)
plt.show()


# In[ ]:


#The brrowers are mostly having no record of Public Recorded Bankruptcy and are safe choice for loan issue.


# In[131]:


plt.figure(figsize=(20,20))
plt.subplot(221)
sns.barplot(data =loan,y='loan_amnt', x='emp_length', hue ='loan_status',palette="pastel")
plt.subplot(222)
sns.barplot(data =loan,y='loan_amnt', x='verification_status', hue ='loan_status',palette="pastel")


# In[ ]:


#employees with longer working history got the loan approved for a higher amount. 
#Looking at the verification status data, verified loan applications tend to have higher loan amount. 
#Which might indicate that the firms are first verifying the loans with higher values.


# In[136]:


# Distribution of Term based on loan_status
plt.figure(figsize=(10,5))
sns.countplot(data=loan,x='term',hue='loan_status')
plt.xlabel('Term')
plt.ylabel('Count')
plt.title('Distribution of Term For Loan Status',fontsize=12)
plt.show()


# In[ ]:


# 60 month term has higher chance of defaulting than 36 month term whereas the 36 month term has higher chance of fully paid loan.


# In[138]:


# Distribution of pub_rec_bankruptcies
plt.figure(figsize=(10,5))
sns.countplot(loan.pub_rec_bankruptcies)
plt.xlabel('Public Record Bankruptcies')
plt.ylabel('Density')
plt.title('Distribution of Public Record Bankruptcies',fontsize=12)
plt.show()


# In[ ]:


#No record of Bankruptcy and are safe choice for loan issue.


# In[134]:


# Distribution of annual_inc based on loan_status
plt.figure(figsize=(10,5))
sns.histplot(data=loan,x='annual_inc',hue='loan_status',bins=20,kde=True)
plt.xlabel('Annual Income')
plt.ylabel('Count')
plt.title('Distribution of Annual Income For Loan Status',fontsize=12)
plt.show()


# In[ ]:


#Borrowers with less 50000 annual income are more likely to default and higher annual income are less likely to default.


# In[135]:


# Distribution of DTI based on Grade
plt.figure(figsize=(10,5))
sns.histplot(data=loan,x='dti',hue='loan_status',bins=10)
plt.xlabel('DTI')
plt.ylabel('Count')
plt.title('Distribution of DTI For Loan Status',fontsize=12)
plt.show()


# In[ ]:


#the Loan Status varies with DTI ratio, we can see that the loans in DTI ratio 10-15 have higher number of defaulted loan but higher dti has higher chance of defaulting.


# In[141]:


sns.heatmap(loan[['loan_amnt','dti','annual_inc','int_rate','term','pub_rec_bankruptcies']].corr(),annot=True)
plt.show()


# In[ ]:


#Observations

# The Employees with 10+ years of experience are likely to default and have higher chance of fully paying the loan.
# we can say that higher grade has lower rate of default.
# Borrowers with less 50000 annual income are more likely to default and higher annual income are less likely to default.
# 60 month term has higher chance of defaulting than 36 month term whereas the 36 month term has higher chance of fully paid loan.
# The annual income of customers who have fully paid the loan is higher than the defaulters.
# DTI is slightly higher in defaulters than the fully paid.
# Large percentage of loans are taken for debt consolidation followed by credit card and other.
# The borrowers have no record of Public Recorded Bankruptcy and are safe choice for loan issue.


# In[ ]:




