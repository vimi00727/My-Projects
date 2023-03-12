#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:

# Employees in new-age insurance company perform multiple outbound efforts to sell term insurance to the customers. 
# Telephonic marketing campaigns still remain one of the most effective ways to reach out to people however they incur a lot of cost. Hence, it is important to identify the customers that are most likely to convert beforehand so that they can be specifically targeted via call.
# 
# It is expected to build a model which will help the organisation by predicting the potential customers.

# # Variables in the Dataset:

# ● age: Age of the customer (Int)
# ● job : Type of job(Object)
# ● marital : Marital status(Object)
# ● educational_qual : Education status(Object)
# ● call_type : Contact communication type(Object)
# ● day: Last contact day of the month (Int)
# ● mon: Last contact month of year(Object)
# ● dur: Last contact duration, in seconds (Int)
# ● num_calls: Number of contacts performed during this campaign and for this client(Int)
# ● prev_outcome: Outcome of the previous marketing campaign(Object)
# ● y - Client subscribtion status.(Object)

# # Importing necessary dependencies:

# In[4]:


# for data reading and data manipulation
import numpy as np
import pandas as pd
import statistics as st
from imblearn.over_sampling import SMOTE

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# for model creation and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# In[5]:


df=pd.read_excel(r'C:\Users\vimkumar\Downloads\Customer Conversion Prediction (1).xlsx')


# In[7]:


df.head(10)


# In[8]:


df.shape


# In[9]:


df.columns


# # Checking and Handling the dateset
# 
# 1. Checking for Missing Values

# In[10]:


df.isnull().sum()


# There is no null values in the dataset. Next we can check for the duplicates due to spelling mistakes.

# 2. Checking for spelling mistakes

# In[11]:


#df.job.unique()
#df.marital.unique()
#df.education_qual.unique()
#df.call_type.unique()
#df.mon.unique()
#df.prev_outcome.unique()

for i in df.columns:
    if i=="age":
        pass
    else:
        print("the unique values for {} is {}".format(i,df[i].unique()))
        


# After the unique check for all the columns, we could derive that there is no duplicates due to spelling mistakes.
# 

# 3. Cross-checking the data-types of all the columns in the dataframe

# In[12]:


df.dtypes


# No data type corrections needed

# 4. Checking whether the data is balanced or not

# In[13]:


df["y"].value_counts()


# Data is not balanced, will have to do oversampling if the model doesn't perform good.

# # Elimination of Unnecessary Columns
# 
# As we are going to categorize the data based on the Month column, we are eliminating the Day column from the dataset.
# 

# In[14]:


df.drop(columns='day',inplace=True)


# In[15]:


df.head(10)


# # Exploratory Data Analysis

# 1. Plotting count plots for all the categorical columns

# In[16]:


sns.set_theme(style='darkgrid',palette='pastel')
plt.figure(figsize=(20,25))
plt.subplot(431)
sns.countplot(x=df['job'],hue=df['y'])
plt.xticks(rotation=90)
plt.xlabel('Job')
plt.ylabel('Converison Yes/No')
plt.title('Customer Conversion Rate based on Job')



# Intepretation
# 
# Customer Conversion rate is high for Retired and Students group. Ratio between the outbound efforts and positive response is higher for these two groups.
# Employees have majorly focused on Blue-Collar Job group.
# 

# In[17]:


plt.figure(figsize=(5,8))
sns.countplot(x=df['marital'],hue=df['y'])
plt.xlabel('Marital')
plt.ylabel('Converison Yes/No')
plt.title('Customer Conversion Rate based on Marital Status')
plt.show()


# Intepretation
# 
# Conversion rate is high among Divorce group and employees have concentrated more on married group.

# In[142]:


plt.figure(figsize=(5,8))
sns.countplot(x=df['education_qual'],hue=df['y'])
plt.xlabel('education_qual')
plt.ylabel('Converison Yes/No')
plt.title('Customer Conversion Rate based on education_qual')
plt.show()


# Interpretation
# 
# High concentration of Outbound efforts has been made to the customer having secondary level education and the customer conversion ratio is relatively higher with customer having primary level education.

# In[143]:


plt.figure(figsize=(5,8))
sns.countplot(x=df['call_type'],hue=df['y'])
plt.xlabel('call_type')
plt.ylabel('Converison Yes/No')
plt.title('Customer Conversion Rate based on call_type')
plt.show()


# Intepretation-
# 
# Cellular Type is preferred type of communication and conversion rate is high for telephonic reach out.

# In[20]:


y1=df["mon"].value_counts().index


# In[144]:


plt.figure(figsize=(5,8))
sns.countplot(x=df['mon'],hue=df['y'])
plt.xlabel('mon')
plt.ylabel('Converison Yes/No')
plt.title('Customer Conversion Rate based on mon')
plt.show()


# Intepretation-
# 
# Outbound reach outs are high during the mid-year time and in during the year-end it is very low. But as per the conversion ratio in Dec, organisation can also increase their Outbonds during year end also.
# 

# In[22]:


df.columns


# In[23]:


plt.figure(figsize=(5,6))
sns.countplot(x=df['prev_outcome'],hue=df['y'])
plt.xlabel('prev_outcome')
plt.ylabel('Converison Yes/No')
plt.title('Customer Conversion Rate based on prev_outcome')
plt.show()


# In[24]:


df["age"].min()


# In[25]:


df["age"].max()


# In[145]:


sns.set_theme(style='darkgrid',palette='pastel')
plt.figure(figsize=(20,25))
plt.subplot(431)
sns.histplot(x="age",hue="y",bins=8 ,data=df,multiple="stack")
plt.xlabel("age")
plt.title("Age vs Conversion Rate")


# Intepretation-
# 
# Organisation has concentrated the age group 20-60years. Conversion ratio is high in 20-25 and 60+.

# In[141]:


sns.set_theme(style='darkgrid',palette='pastel')
plt.figure(figsize=(20,25))
plt.subplot(431)
sns.histplot(x="num_calls",hue="y",bins=40 ,data=df,multiple="stack")
plt.xlabel("Num_calls")
plt.title("Num_calls vs Conversion Rate")
plt.show()


# Inpretation-
# 
# Most of the attempts are less than 5times and mostly customer conversion is acheived within 4 attempts.

# # Checking for Correlation

# In[28]:


sns.heatmap(df.corr())
plt.show()


# No correlation between continous variables. Hence there is no need for removing any feature out of model.

# # Checking for Outliers
# 
# Checking for outliers using the box plot and removing them from the dataframe.

# In[136]:


sns.boxplot(data=df,x="age")


# Considered Outliers for Age column is 75+.

# In[30]:


y = df[df["age"]>75]
y.head()


# In[31]:


for i in y.index:
     df = df.drop(i,axis=0)


# In[32]:


df.shape


# In[137]:


sns.boxplot(data=df,x="dur")


# Considered Outliers for Call_Duration are the calls with duration greater than 1000sec.

# In[34]:


dur_out = df.loc[df["dur"]>1000]


# In[35]:


len(dur_out.loc[dur_out["y"]=="yes"])


# In[36]:


dur_out.shape


# In[37]:


dur_out


# In[38]:


df["y"].value_counts()


# In[39]:


sns.boxplot(data=df,x="num_calls")


# In[40]:


numcalls_out = df.loc[df["num_calls"]>20]


# In[41]:


numcalls_out.shape


# In[42]:


len(numcalls_out.loc[numcalls_out["y"]=="yes"])


# In[43]:


for i in numcalls_out.index:
     df = df.drop(i,axis=0)


# In[44]:


df.columns


# # Encoding
# 
# Encoding the categorical columns using One Hot Encoder.
# 

# In[46]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
ohe_list =['job', 'marital', 'education_qual', 'call_type', 'mon','prev_outcome']
ohe = OneHotEncoder()
y= ohe.fit_transform(df[['job', 'marital', 'education_qual', 'call_type', 'mon','prev_outcome']])


# In[49]:


y


# In[47]:


encoders = y.toarray()


# In[48]:


encoders


# In[50]:


encoders.shape


# In[59]:


columns_list =[]
for i in ohe_list:
    names = df[i].unique()
    columns_list.extend(names)


# In[64]:


len(columns_list)


# In[68]:


columns_list.remove("unknown")


# In[69]:


len(columns_list)


# In[73]:


columns_list.insert(34,"unknown_outcome")


# In[63]:


for j in range(len(columns_list)):
    if columns_list[j]=="unknown":
        print(j)


# In[74]:


columns_list


# In[75]:


df1 = pd.DataFrame(encoders,columns = columns_list)


# In[76]:


df1


# In[77]:


df1.reset_index(inplace=True)


# In[78]:


df1.head()


# In[80]:


df.reset_index(inplace=True)


# In[82]:


df.drop("index",axis=1,inplace=True)


# In[83]:


df.head()


# In[84]:


df["index"] = [ i for i in range(0,len(df))]


# In[85]:


df.head()


# In[86]:


df1.columns


# In[87]:


df.columns


# In[88]:


df.shape


# In[89]:


df1.shape


# # Merging the Dataframes

# In[90]:


final_df = pd.merge(df,df1,on="index")


# In[91]:


final_df


# In[92]:


final_df.columns


# In[93]:


final_df.drop(['job', 'marital', 'education_qual', 'call_type', 'mon', 'prev_outcome'],axis=1,inplace=True)


# In[94]:


final_df.drop("index",axis=1,inplace=True)


# In[95]:


final_df.shape


# In[96]:


final_df.columns


# # Determination of Dependent and Independent Features

# In[97]:


x= final_df.drop("y",axis=1)


# In[98]:


y = final_df["y"]


# # Oversampling
# 
# Have used Smute Technique for Oversampling.

# In[109]:


df["y"].value_counts()


# In[99]:


desired_sample_size = 3.5* len(final_df.loc[final_df['y']=="yes"])


# In[100]:


desired_sample_size


# In[105]:


smote = SMOTE(random_state=42)
X_resampled,y_resampled = smote.fit_resample(x,y)


# In[106]:


X_resampled.shape


# In[107]:


y_resampled.shape


# In[110]:


y_resampled.value_counts()


# # Model Development

# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


x_train,x_test,y_train,y_test=train_test_split(X_resampled,y_resampled,test_size=0.3)


# In[113]:


x_train.shape


# In[114]:


x_test.shape


# In[115]:


from sklearn.linear_model import LogisticRegression


# In[116]:


lg = LogisticRegression()
lg.fit(x_train,y_train)
y_pred = lg.predict(x_test)
prob = lg.predict_proba(x_test)


# In[117]:


y_pred


# In[118]:


prob


# In[119]:


from sklearn.metrics import recall_score,classification_report,confusion_matrix,f1_score,accuracy_score,auc


# In[133]:


print(classification_report(y_test,y_pred))


# # AUC ROC Curve
# 

# In[121]:


accuracy_score(y_test,y_pred)


# In[122]:


from sklearn.preprocessing import LabelEncoder


# In[123]:


le = LabelEncoder()
testing_data = le.fit_transform(y_test)


# In[129]:


testing_data


# In[125]:


le = LabelEncoder()
predicting_data = le.fit_transform(y_pred)


# In[130]:


predicting_data


# In[135]:


import matplotlib.pyplot as plt
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(testing_data,predicting_data)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# The derived Threshold value is 0.8. Threshold value is derived using the G-Mean.
# 
# G-Mean= √(TPR*(1-FPR)

# # SUMMARY
# 
# -Problem Statement
# 
#     Build a model which is best fit for predicting the customer conversion rate.
# 
# -Model Used
#     
#     Logistic Regression
# 
# -Why Logistic Regression
# 
#     The given dataset is binary classification problem. It is a lineraly seperable dataset hence Logistic regression can be one of the best fit model.
# 
# -Steps
# 
#     -Importing Dependecies
#     -Handling the Dataset
#     -EDA
#     -Elimination of Outliers
#     -Encoding
#     -Oversampling
#     -Model Development
#  
# -Complication
# 
#     -Due to the imbalance in the dataset the output was skewed and the model was performing with low efficiency.
# 
# -Optimization
# 
#     -To fix the imbalance and skewed output, I used the SMUTE Oversampling technique.
#     -Desired Sample Size used is 3.5 Times of "Yes" in the dataset.
#     
# -Result
# 
#     -Before Oversampling the model accuracy was 68% and precision/recall/F1-Score valued between 68-71.
#     -SMUTE increased the Accuracy to 84% and precision/recall/F1-Score to 84.
# 

# In[ ]:




