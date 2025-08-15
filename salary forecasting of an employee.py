#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn. metrics import accuracy_score


# In[72]:


data=pd.read_csv('Downloads//salary_of_employee.csv')
data.head()


# In[73]:


data.tail()


# In[74]:


data.shape


# In[75]:


data.dtypes


# In[76]:


#DUPLICATE DATA
data[data.duplicated()]


# In[77]:


#DROPPING THE DUPLICATE
sal_data=data.drop_duplicates(keep='first')
sal_data.shape


# In[78]:


sal_data.isnull().sum()


# In[79]:


#removing null data
sal_data.dropna(how = 'any',inplace=True)


# In[80]:


sal_data.shape


# In[81]:


#statistics of numerical columns
sal_data.describe()


# In[82]:


#correlation among numerical features
#Correlation summarizes the strength and direction of the linear (straight-line) association between two quantitative variables.
corr=sal_data[['Age', 'Years of Experience', 'Salary' ]].corr()
corr


# In[83]:


#label encoding


# In[84]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()


# In[85]:


sal_data['Gender_Encode']=label_encoder.fit_transform(sal_data['Gender'])


# In[86]:


sal_data['Education Level_Encode']=label_encoder.fit_transform(sal_data['Education Level'])


# In[87]:


sal_data['Job Title_Encode']=label_encoder.fit_transform(sal_data['Job Title'])


# In[88]:


sal_data.head()


# In[89]:


#feature scaler


# In[90]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()


# In[91]:


sal_data['Age_scaled'] = std_scaler.fit_transform(sal_data[['Age']])
sal_data['Years of Experience_scaled'] = std_scaler.fit_transform(sal_data[['Years of Experience']])


# In[92]:


sal_data.head()


# In[93]:


x=sal_data[['Age_scaled','Job Title_Encode','Gender_Encode','Education Level_Encode','Years of Experience_scaled']]
y=sal_data['Salary']


# In[94]:


x.head()


# In[95]:


y.head()


# In[96]:


#splitting the data in training and testing


# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)


# In[99]:


x_train.head()


# In[100]:


#Model Development


# In[101]:


from sklearn.linear_model import LinearRegression
Linear_Regression_Model=LinearRegression()


# In[102]:


#model train


# In[103]:


Linear_Regression_Model.fit(x_train,y_train)


# In[104]:


#model predictions


# In[105]:


y_pred_lr = Linear_Regression_Model.predict(x_test)
y_pred_lr


# In[106]:


df=pd.DataFrame({'y_actual':y_test,'y_predicted':y_pred_lr})
df


# In[107]:


#error


# In[108]:


df['error']=df['y_actual']-df['y_predicted']
df


# In[109]:


df['abs_error']=abs(df['error'])
df


# In[110]:


mean_abs_error=df['abs_error'].mean()
mean_abs_error


# In[111]:


from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[112]:


r2_score(y_test, y_pred_lr)


# In[113]:


round(mean_absolute_error(y_test, y_pred_lr),2)


# In[114]:


#coefficients


# In[115]:


Linear_Regression_Model.coef_


# In[116]:


#intercepts


# In[117]:


Linear_Regression_Model.intercept_


# In[118]:


#customise predictions


# In[131]:


Age1 = std_scaler.transform([[49]])
Age = -1
Gender=1
Education = 0
Job_Title = 101
Experience_years1 = std_scaler.transform([[15]])
Experience_years = 7


# In[132]:


std_scaler.transform([[15]])[0][0]


# In[133]:


Emp_Salary = Linear_Regression_Model. predict([[Age, Job_Title, Gender, Education,Experience_years]])
Emp_Salary


# In[134]:


print("Salary of that Employee with above Attributes = ", Emp_Salary[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




