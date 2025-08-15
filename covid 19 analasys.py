#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib. pyplot as plt 
import seaborn as sns 
import plotly.express as px
from plotly.subplots import make_subplots 
from datetime import datetime


# In[2]:


covid_df=pd.read_csv("Downloads//covid_19_india.csv")


# In[4]:


covid_df.head()


# In[5]:


covid_df.info()


# In[7]:


covid_df.describe()


# In[8]:


vaccine_df=pd.read_csv("Downloads//covid_vaccine_statewise.csv")


# In[9]:


vaccine_df.head()


# In[11]:


vaccine_df.describe()


# In[12]:


covid_df.drop(["Sno","Time","ConfirmedIndianNational","ConfirmedForeignNational"],inplace=True,axis=1)


# In[13]:


covid_df.head()


# In[17]:


covid_df['Date']=pd.to_datetime(covid_df['Date'], format = '%Y-%m-%d')


# In[18]:


covid_df['Active_cases']=covid_df['Confirmed']-(covid_df['Cured']+covid_df['Deaths'])


# In[19]:


covid_df.head()


# In[20]:


statewise=pd.pivot_table(covid_df,values=["Confirmed","Deaths","Cured"],index="State/UnionTerritory",aggfunc=max)


# In[21]:


statewise["Recovery_Rate"]=statewise["Cured"]*100/statewise["Confirmed"]


# In[22]:


statewise["Mortality_Rate"]=statewise["Deaths"]*100/statewise["Confirmed"]


# In[23]:


statewise=statewise.sort_values(by='Confirmed', ascending=False)


# In[24]:


statewise.style.background_gradient(cmap='cubehelix')


# In[35]:


top_10_active_cases_states=covid_df.groupby(by='State/UnionTerritory').max()[['Active_cases','Date']].sort_values(by='Active_cases',ascending=False).reset_index()


# In[28]:


fig=plt.figure(figsize=(16,9))


# In[29]:


plt.title("top 10 states with active cases",size=25)


# In[38]:


ax=sns.barplot(data=top_10_active_cases_states.iloc[:10],y='Active_cases',x='State/UnionTerritory',linewidth=1,edgecolor='red')


# In[40]:


top_10_active_cases_states=covid_df.groupby(by='State/UnionTerritory').max()[['Active_cases','Date']].sort_values(by='Active_cases',ascending=False).reset_index()
fig=plt.figure(figsize=(16,9))
plt.title("top 10 states with active cases",size=25)
ax=sns.barplot(data=top_10_active_cases_states.iloc[:10],y='Active_cases',x='State/UnionTerritory',linewidth=1,edgecolor='red')
plt.xlabel("States")
plt.ylabel("Active Cases")
plt.show()


# In[45]:


top_10_deaths=covid_df.groupby(by='State/UnionTerritory').max()[['Deaths','Date']].sort_values(by=['Deaths'],ascending=False).reset_index()
fig=plt.figure(figsize=(16,9))
plt.title("top 10 states with deaths",size=25)
ax=sns.barplot(data=top_10_deaths.iloc[:10],y='Deaths',x='State/UnionTerritory',linewidth=1,edgecolor='red')
plt.xlabel("States")
plt.ylabel("deaths")
plt.show()


# In[46]:


vaccine_df.head()


# In[48]:


vaccine_df.rename(columns={'Updated On': 'Vaccine_date'},inplace=True)


# In[49]:


vaccine_df.info()


# In[52]:


vaccine_df.isnull().sum()


# In[58]:


vaccination=vaccine_df.drop(columns=['Sputnik V (Doses Administered)','AEFI','45-60 Years (Doses Administered)','60+ Years(Individuals Vaccinated)'],axis=1)


# In[59]:


vaccination.head()


# In[64]:


male=vaccination["Male(Individuals Vaccinated)"].sum()
female=vaccination["Female(Individuals Vaccinated)"].sum()
px.pie(names=['Male','Female'],values=[male,female],title="Male and Female Vaccination")


# In[65]:


vaccine=vaccine_df[vaccine_df.State!='India']
vaccine


# In[70]:


vaccine.rename(columns={'Total Individuals Vaccinated':'Total'})


# In[73]:


max_vac=vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vac = max_vac.sort_values('Total', ascending = False)[:5]
max_vac


# In[75]:


fig=plt.figure(figsize=(16,9))
plt.title("top 10 states vaccinated",size=25)
ax=sns.barplot(data=max_vac.iloc[:10],y=max_vac.Total,x=max_vac.index,linewidth=1,edgecolor='red')
plt.xlabel("States")
plt.ylabel("Total vaccinations")
plt.show()


# In[ ]:




