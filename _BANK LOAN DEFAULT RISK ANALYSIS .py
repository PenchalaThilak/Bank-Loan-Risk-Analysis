#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings


# In[4]:


application_data = pd.read_csv(r"C:\Users\thila\Downloads\NARESH IT  PYTHON FSDS NOTES\PYTHON\PROJECTS\BANK RISK ANALYSIS\application_data.csv")
previous_data    = pd.read_csv(r"C:\Users\thila\Downloads\NARESH IT  PYTHON FSDS NOTES\PYTHON\PROJECTS\BANK RISK ANALYSIS\previous_application.csv\previous_application.csv")


# In[5]:


application_data


# In[6]:


application_data.head()


# In[7]:


application_data.shape


# In[8]:


application_data.shape[0]
# shape[0] function will return the no of rows in 
# the given data frame


# In[9]:


application_data.shape[1]
# shape[1] frunction will return the no of columns 
# in the given data frame


# In[10]:


previous_data


# In[11]:


previous_data.head()


# In[12]:


previous_data.shape


# In[13]:


application_data.info(verbose=True)


# In[14]:


previous_data.info(verbose=True)
# here pandas has the attribute DataFrame
# <class 'pandas.core.frame.DataFrame'>


# In[15]:


application_data.describe()


# In[16]:


previous_data.describe()


# In[17]:


print("application_data:"  ,application_data.shape)
print("previous_data:" ,previous_data.shape )


# # DATA CLEANING AND MANIPULATION

# # Null Value Calculation

# In[18]:


pip install missingno
# here we are installing missingno packages in python


# In[19]:


import missingno as mn
mn.matrix(application_data)
# from the output we can see the graphical representation
# of the data
# all the blank spaces we see are the missing
# values in the data set


# In[21]:


application_data.isnull()
# .isnull() function is used to detect missing or NaN (Not a Number) values in a DataFrame or Series
# this function will give us the missing values in python whether there are missing values in the dataset or not
# if there are any  missing values in dataset(columns) then it will show true at the respective place of the missing value
# otherwise false


# In[22]:


application_data.isnull().sum()
# this function will give us the number of missing values in the dataset
# suppose if there is a missing values in the column and if we use .isnull().sum() then it will give us the number of
# missing values in the dataset


# In[23]:


application_data.shape[0]
# this command will give us the number of rows present in the dataset(application_data)


# In[24]:


# to calculate the missing value(%percentage) in each column

k=round(application_data.isnull().sum()/application_data.shape[0]*100.00,2)
print(k)

# the round() function is used for rounding off the
# number to the nearest number
# the round() function will take the 2 arguments
# Example-->(3.1459 ,2) it will give the
# output--->3.14 after decimal it will print 2 digits
# bcz we passed 2 as the argument
# Example-->(3.1459,3) it will give the 
# output--->3.145 after decimal it will print 3 digits
# bcz we passed 3 as the arguments


# In[25]:


null_application_data = pd.DataFrame(k).reset_index()


# In[26]:


#null_application_data = pd.DataFrame((application_data.isnull().sum())*100/application_data.shape[0]).reset_index()

# reset_index(): This method is used to reset the index of the DataFrame. 
# When you perform operations like calculating null percentages, 
# the resulting DataFrame may have a modified index. 
# reset_index() is used to bring the index back to 
# the default integer-based index and add the previous 
# index as a new column in the DataFrame.


# In[27]:


null_application_data
#this will give us the dataset(application_data) with column wise missing values percentage


# In[28]:


null_application_data.columns = ['Column Name', 'Null Value Percentage']
#.columns this command is used to manipulate the columns in the dataset
# here we have added the extra column name called 'Null Values Percentage' to the dataset of null_application_data


# In[29]:


null_application_data


# In[30]:


null_application_data = pd.DataFrame(k).reset_index()
null_application_data.columns = ['Column Name', 'Null Value Percentage']
fig = plt.figure(figsize=(18,6))
ax = sns.pointplot(x="Column Name",y="Null Value Percentage",data=null_application_data,color='blue')
plt.xticks(rotation =90,fontsize =8)
ax.axhline(40, ls='--',color='red')
plt.title("PERCENTAGE OF MISSING VALUES IN APPLICATION DATA")
plt.ylabel("NULL VALUE PERCENTAGE")
plt.xlabel("COLUMN NAMES")
plt.show()
# in the above figsize is used to determine the size of the figure(the out image or graph) size
# sns.pointplot is used to point the graph for plotting the graph we need to provide the 
# x-axis information i.e; on x-axis what information we must keep
# here we kept the Column Name in the x-axis
# and in the y-axis we keep the Null value Percentage and we need to provide the data and the data is'null_application_data'
# because the data points like 'Column Name' and  "Null Value Percentage" are from that dataset
# .xticks command is used to write the names on x-axis
# axhline means horizontal line it is used to plot the horizontal line parallel to x-axis


# In[31]:


nullcolumn_40_application_data= null_application_data[null_application_data['Null Value Percentage']>=40]
# this is used to filter the percentage of empty(missing)
# values greater than or equal to 40% in  respective columns


# In[32]:


nullcolumn_40_application_data


# In[33]:


len(nullcolumn_40_application_data)
# from this we got to know that the missing values percentage with >=40% is there totally of 49 columns
# so we can drop these columns so that these columns will not effect our data


# # Calculations of Previousdata missing values

# In[34]:


previous_data


# In[35]:


mn.matrix(previous_data)


# In[36]:


round(previous_data.isnull().sum()/previous_data.shape[0]*100.00,2)


# In[37]:


null_previous_data=pd.DataFrame((previous_data.isnull().sum()*100)/previous_data.shape[0]).reset_index()


# In[38]:


null_previous_data


# In[39]:


null_previous_data.columns=['Column Name','Null Value Percentage']


# In[40]:


null_previous_data


# In[41]:


null_previous_data=((previous_data.isnull().sum()*100)/previous_data.shape[0]).reset_index()
null_previous_data.columns=['Column Name','Null Value Percentage']
fig=plt.figure(figsize=(18,6))
ax =sns.barplot(x='Column Name',y='Null Value Percentage',data=null_previous_data,color='blue')
plt.xticks(rotation=90,fontsize=8)
ax.axhline(40,ls='--',color='red')
plt.title('PERCENTAGE OF MISSING VALUES IN PREVIOUS DATA')
plt.ylabel('NULL VALUE PERCENTAGE')
plt.xlabel('COLUMN NAMES')
plt.show()


# In[42]:


nullcolumn_40_previous_data = null_previous_data[null_previous_data['Null Value Percentage']>=40]
nullcolumn_40_previous_data


# In[43]:


len(nullcolumn_40_previous_data)
# in the previous data there 11 columns with the 40% missing values


# # ANALYZING & REMOVING UNNECESSARY COLUMNS IN THE APPLICATION DATA SET

# #Checking the correlation between the EXT_SOURCE columns and the TARGET column
# 

# In[44]:


source_column = application_data[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","TARGET"]]
source_column_correlation = source_column.corr()
ax = sns.heatmap(source_column_correlation,xticklabels=source_column_correlation.columns,yticklabels=source_column_correlation.columns,annot = True,cmap = "RdYlGn")


# In[45]:


# Based on the above heatmap visualization we can conclude that there is almost
# no correlation between the EXT_SOURCE_1 and EXT_SOURCE_2 and EXT_SOURCE_3 and TARGET 
# so we can drop these EXT_SOURCE columns from the dataset


# In[46]:


len(nullcolumn_40_application_data)


# In[47]:


Unwantedcolumns_application_data = nullcolumn_40_application_data["Column Name"].tolist()+["EXT_SOURCE_2","EXT_SOURCE_3"]
len(Unwantedcolumns_application_data)
# Initially we have created a dataset with the 40% null values which contains 
# all the columns with the missing value percentage >=40% 
# By using this command 'nullcolumn_40_application_data["Column Name"].tolist()' it will
# convert the columns present in the null_40_application_data dataset into the list format
# now we need to add the extra columns called '"EXT_SOURCE_2","SOURCE_SOURCE_3"' to that list
# and we named that list as 'Unwantedcolumns_application_data' so that tese columns can be ignored
# initillay the len of the 40% missing values dataset is 49 
# anbd after adding the two morer columns the length is 51 
# as we can see that EXT_SOURCE_1 column also  not show any kind of correlation 
# as it was already present in the nullcolumn_40_previous_data it was not mentioned here


# In[48]:


import warnings


# FLAG DOCUMENTS

# In[49]:


flag_document = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
dataset_flag_target = application_data[flag_document+["TARGET"]]
length = len(flag_document)
dataset_flag_target["TARGET"] = dataset_flag_target["TARGET"].replace({1:"Defaulter",0:"Repayer"})
fig = plt.figure(figsize=(21,24))

for i,j in itertools.zip_longest(flag_document,range(length)):
    plt.subplot(5,4,j+1)
    ax = sns.countplot(x=i,hue = "TARGET",data =dataset_flag_target,palette=["r","g"])
    plt.yticks(fontsize=8)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)
# In this code we have created a list called 'flag_document' in which it has the 
# flag document columns from flag_document_2 to flag_document_21 
# after that we have created another list called  dataset_flag_target in which
# it has the columns including the flag document and target column
# the count of flag_document from flag_document_2 to flag_document_21 is 20
# and the range of flag_document will give us the values from 0 to 19
# in the dataset the target column has the values 1 and o
# the values of the target column in the list we have created dataset_flag_document
# and these values will not change in the application_data they will be as it is like 0 and 1
# Example lets take i,j values as 2,3 then
# the for loop we used gives us i=2 means it will check the value present at the second index  flag_document 
# and the value present at the second index in flag_document is 'FLAG_DOCUMENT_4'  and j=3 is  and it 
# will present in the range of 20 (0,19) so that it will enters the for loop
# subplots(5,4) specifies that the subplots will have 5 rows and 4 columns and the 
# j value starts from 0 and it will continue to 19
# in countplot x =i means according the example it is FLAG_DOCUMENT_4 and the data we provided is dataset_flag_document
# in the column description of target variable it is specified that:
# 


# In[52]:


# from the above graph we can say that in most of the applicant cases,clients who applied
# for loans has not submitted FLAG_DOCUMENT_X except FLAG_DOCUMENT_3
# sog flag document 3 we can delete rest of the columns


# In[53]:


flag_document.remove('FLAG_DOCUMENT_3')
Unwantedcolumns_application_data = Unwantedcolumns_application_data +flag_document


# In[54]:


flag_document


# In[55]:


Unwantedcolumns_application_data


# In[56]:


len(Unwantedcolumns_application_data)


# CHECKING THE CONTACT NUMBER PARAMETERS

# In[57]:


contact_column = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','TARGET']
contact_correlation = application_data[contact_column].corr()
fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(contact_correlation,xticklabels=contact_correlation.columns,yticklabels=contact_correlation.columns,annot=True,cmap="RdYlGn",linewidth=1)


# In[58]:


# from the above graph we can say that there is no correlation 
# between the contact numbers and target column
# so we can also ignore the contact details columns from the 
# dataset


# In[59]:


contact_column.remove("TARGET")


# In[60]:


contact_column


# In[61]:


Unwantedcolumns_application_data = Unwantedcolumns_application_data+contact_column


# In[62]:


len(Unwantedcolumns_application_data)


# In[63]:


# so 76 columns can be dropped from the application_data


# In[64]:


application_data.drop(labels=Unwantedcolumns_application_data,axis = 1,inplace = True)


# In[65]:


application_data.shape
# initially application_data has 112 columns and after removing the unwantedcolumns
# it was reduced to 46


# In[66]:


application_data.info()


# ANALYZING AND REMOVING UNWANTED COLUMNS FROM THE PREVIOUS DATASET

# In[67]:


nullcolumn_40_previous_data


# In[68]:


Unwantedcolumns_previous_data = nullcolumn_40_previous_data["Column Name"].tolist()
Unwantedcolumns_previous_data


# In[69]:


len(Unwantedcolumns_previous_data)


# In[70]:


Unnecessarycolumns_previous_data = ['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']


# In[71]:


Unwantedcolumns_previous_data = Unwantedcolumns_previous_data+Unnecessarycolumns_previous_data


# In[72]:


len(Unwantedcolumns_previous_data)


# In[73]:


# so totally 15 columns can be dropped from the previous_data


# In[74]:


previous_data.drop(labels = Unwantedcolumns_previous_data,axis = 1,inplace = True)


# In[75]:


previous_data.shape


# In[76]:


previous_data.info()


#  STANDARDIZING THE VALUES :
#  STANDARDIZING THE VALUES IN APPLICATION DATA

# In[77]:


application_data.info()


# In[78]:


#Strategy for applicationDF:
#Convert DAYS_DECISION,DAYS_EMPLOYED, DAYS_REGISTRATION,DAYS_ID_PUBLISH from negative to positive as days cannot be negative.
#Convert DAYS_BIRTH from negative to positive values and calculate age and create categorical bins columns
#Categorize the amount variables into bins
#Convert region rating column and few other columns to categorical


# In[79]:


date_columns = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']
for columns in date_columns:
    application_data[columns] = abs(application_data[columns])


# In[80]:


# Binning numerical column to create a categorical column


# In[81]:


application_data['AMT_INCOME_TOTAL'] = pd.to_numeric(application_data['AMT_INCOME_TOTAL'])


# In[82]:


application_data.info()


# In[83]:


application_data['AMT_INCOME_TOTAL'] = application_data['AMT_INCOME_TOTAL'] / 100000
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
slot = ['0-100k', '100k-200k', '200k-300k', '300k-400k', '400k-500k', '500k-600k', '600k-700k', '700k-800k', '800k-900k', '900k-1m', '1m-Above']
application_data['AMT_INCOME_RANGE'] = pd.cut(application_data['AMT_INCOME_TOTAL'], bins, labels=slot)


# In[84]:


application_data['AMT_INCOME_RANGE'].value_counts(normalize=True)*100


# In[85]:


#More than 50% loan applicants have income amount in the range of 100K-200K. 
#Almost 92% loan applicants have income less than 300K


# In[86]:


# Creating bins for Credit amount


# In[87]:


application_data['AMT_CREDIT']=application_data['AMT_CREDIT']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,100]
slots = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k',
       '800k-900k','900k-1M', '1M Above']

application_data['AMT_CREDIT_RANGE']=pd.cut(application_data['AMT_CREDIT'],bins=bins,labels=slots)


# In[88]:


application_data['AMT_CREDIT_RANGE'].value_counts(normalize=True)*100


# In[89]:


#More Than 16% loan applicants have taken loan which amounts to more than 1M.


# In[90]:


# Creating bins for age
application_data['AGE'] = application_data['DAYS_BIRTH'] // 365
bins = [0,20,30,40,50,100]
slots = ['0-20','20-30','30-40','40-50','50 above']

application_data['AGE_GROUP']=pd.cut(application_data['AGE'],bins=bins,labels=slots)


# In[91]:


application_data['AGE_GROUP'].value_counts(normalize=True)*100


# In[92]:


#31% loan applicants have age above 50 years. 
#More than 55% of loan applicants have age over 40 years.


# In[93]:


# Creating bins for Employement Time
application_data['YEARS_EMPLOYED'] = application_data['DAYS_EMPLOYED'] // 365
bins = [0,5,10,20,30,40,50,60,150]
slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

application_data['EMPLOYMENT_YEAR']=pd.cut(application_data['YEARS_EMPLOYED'],bins=bins,labels=slots)


# In[94]:


application_data['EMPLOYMENT_YEAR'].value_counts(normalize=True)*100


# In[95]:


application_data.nunique().sort_values()


# DATA TYPE CONVERSIONS

# In[96]:


application_data.info()


# In[97]:


categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',
                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',
                       'REGION_RATING_CLIENT_W_CITY'
                      ]
for col in categorical_columns:
    application_data[col] =pd.Categorical(application_data[col])


# In[98]:


application_data.info()


# STANDARDIZING THE VALUES OF PREVIOUS DATA

# In[99]:


#Strategy for previous DATA:
#Convert DAYS_DECISION from negative to positive values and create categorical bins columns.
#Convert loan purpose and few other columns to categorical.


# In[100]:


previous_data['DAYS_DECISION'] = abs(previous_data['DAYS_DECISION'])


# In[101]:


previous_data.nunique().sort_values()


# In[102]:


previous_data['DAYS_DECISION_GROUP'] = (previous_data['DAYS_DECISION']-(previous_data['DAYS_DECISION'] % 400)).astype(str)+'-'+ ((previous_data['DAYS_DECISION'] - (previous_data['DAYS_DECISION'] % 400)) + (previous_data['DAYS_DECISION'] % 400) + (400 - (previous_data['DAYS_DECISION'] % 400))).astype(str)


# In[103]:


previous_data['DAYS_DECISION_GROUP'].value_counts(normalize=True)*100


# In[104]:


Catgorical_columns_previous= ['NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                   'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',
                    'NAME_CONTRACT_TYPE','DAYS_DECISION_GROUP']

for column in Catgorical_columns_previous:
    previous_data[col] =pd.Categorical(previous_data[column])


# In[105]:


previous_data.info()


# NULL VALUE IMPUTATION

#  Imputing Null Values in application data
#  #To impute null values in categorical variables which has lower null percentage, mode() is used to impute the most frequent items.
# #To impute null values in categorical variables which has higher null percentage, a new category is created.
# #To impute null values in numerical variables which has lower null percentage, median() is used as
# #There are no outliers in the columns
# #Mean returned decimal values and median returned whole numbers and the columns were number of requests
#  checking the null value % of each column in ap

# In[106]:


application_data.isnull().sum()


# In[107]:


application_data.shape[0]


# In[108]:


round(application_data.isnull().sum() / application_data.shape[0] * 100.00,2)


# In[109]:


#Impute categorical variable 'NAME_TYPE_SUITE' which has lower 
#null percentage(0.42%) with the most frequent category using mode()[0]:


# In[110]:


application_data['NAME_TYPE_SUITE'].describe()


# In[111]:


application_data['NAME_TYPE_SUITE'].fillna((application_data['NAME_TYPE_SUITE'].mode()[0]),inplace = True)


# In[112]:


#Impute categorical variable 'OCCUPATION_TYPE' which has higher null percentage(31.35%) 
#with a new category as assigning to any existing category might influence the analysis:


# In[113]:


application_data['OCCUPATION_TYPE'] = application_data['OCCUPATION_TYPE'].cat.add_categories('Unknown')
application_data['OCCUPATION_TYPE'].fillna('Unknown', inplace =True) 


# In[114]:


application_data[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
               'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()


# In[115]:


#Impute with median as mean has decimals


# In[116]:


amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR',]

for col in amount:
    application_data[col].fillna(application_data[col].median(),inplace = True)


# In[117]:


round(application_data.isnull().sum() / application_data.shape[0] * 100.00,2)


# In[118]:


application_data.info()


# In[119]:


application_data['EMPLOYMENT_YEAR'] = pd.to_numeric(application_data['EMPLOYMENT_YEAR'])


# In[120]:


application_data['AMT_INCOME_TOTAL'] = pd.to_numeric(application_data['AMT_INCOME_TOTAL'])


# In[121]:


application_data['YEARS_EMPLOYED'] = application_data['DAYS_EMPLOYED'] // 365
bins = [0,5,10,20,30,40,50,60,150]
slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

application_data['EMPLOYMENT_YEAR']=pd.cut(application_data['YEARS_EMPLOYED'],bins=bins,labels=slots)


# In[122]:


application_data['EMPLOYMENT_YEAR'].value_counts(normalize=True)*100


# IMPUTING NULL VALUES IN PREVIOUS DATA

# In[123]:


previous_data.isnull().sum()


# In[124]:


round(previous_data.isnull().sum()/previous_data.shape[0]*100.2)


# In[125]:


# from the above graph we can say that 'AMT_ANNUITY' 'AMT_GOODS_PRICE' 'CNT_PAYMENT'
# has the missing value percentage is almost 22 and 23 percentages
# and now we need to fill that missing values


# In[126]:


# For imputing the missing values in the 'AMT_ANNUITY' first we need to check the
# skewness of the graph


# In[127]:


plt.figure(figsize=(6,6))
sns.kdeplot(previous_data['AMT_ANNUITY'])
plt.show()


# In[128]:


#There is a single peak at the left side of the distribution and it indicates the presence of
#outliers and hence imputing with mean would not be the right approach and hence imputing with median.


# In[129]:


previous_data['AMT_ANNUITY'].fillna(previous_data['AMT_ANNUITY'].median(),inplace = True)


# In[130]:


#Impute AMT_GOODS_PRICE with 
#mode as the distribution is closely similar:


# In[131]:


plt.figure(figsize=(6,6))
sns.kdeplot(previous_data['AMT_GOODS_PRICE'][pd.notnull(previous_data['AMT_GOODS_PRICE'])])
plt.show()


# In[132]:


#There are several peaks along the distribution. Let's impute using the mode, mean and median and 
#see if the distribution is still about the same.


# In[ ]:


statsDF = pd.DataFrame() # new dataframe with columns imputed with mode, median and mean
statsDF['AMT_GOODS_PRICE_mode'] = previous_data['AMT_GOODS_PRICE'].fillna(previous_data['AMT_GOODS_PRICE'].mode()[0])
statsDF['AMT_GOODS_PRICE_median'] = previous_data['AMT_GOODS_PRICE'].fillna(previous_data['AMT_GOODS_PRICE'].median())
statsDF['AMT_GOODS_PRICE_mean'] = previous_data['AMT_GOODS_PRICE'].fillna(previous_data['AMT_GOODS_PRICE'].mean())

cols = ['AMT_GOODS_PRICE_mode', 'AMT_GOODS_PRICE_median','AMT_GOODS_PRICE_mean']

plt.figure(figsize=(18,10))
plt.suptitle('Distribution of Original data vs imputed data')
plt.subplot(221)
sns.distplot(previous_data['AMT_GOODS_PRICE'][pd.notnull(previous_data['AMT_GOODS_PRICE'])]);
for i in enumerate(cols): 
    plt.subplot(2,2,i[0]+2)
    sns.distplot(statsDF[i[1]])


# In[ ]:


#Insight:
#The original distribution is closer with the distribution 
#of data imputed with mode in this case


# In[ ]:


previous_data['AMT_GOODS_PRICE'].fillna(previous_data['AMT_GOODS_PRICE'].mode()[0], inplace=True)


# In[ ]:


#Impute CNT_PAYMENT with 0 as the NAME_CONTRACT_STATUS 
#for these indicate that most of these loans were not started:


# In[ ]:


previous_data.loc[previous_data['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()


# In[ ]:


previous_data['CNT_PAYMENT'].fillna(0,inplace = True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




