#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
import statsmodels.api as sm


# In[7]:


file = "C:/Users/Asus/OneDrive/Desktop/GymIQ/realistic_synthetic_training_data.csv"
stats_data = pd.read_csv(file)


# In[10]:


stats_data['1RM'] = (stats_data['Weight'] * (1 + (stats_data['Reps'] / 30)))


# In[12]:


stats_data['Session ID'] = stats_data['Date'].factorize()[0] + 1


# In[28]:


recent_avg = stats_data['1RM'].iloc[-8:].mean()
avg_30_sessions_ago = stats_data['1RM'].iloc[-38:-30].mean()


# In[36]:


percent_change = ((recent_avg - avg_30_sessions_ago) / avg_30_sessions_ago) * 100


# In[40]:


percent_change = round(percent_change, 1)


# In[43]:


if percent_change > 5:
    print(f'You’ve improved {percent_change}% compared to nearly 30 sessions ago. Strong upward trend is observed.')
elif 0 < percent_change <= 5:
    print(f'Steady Progress Observed. You’ve improved {percent_change}% compared to nearly 30 sessions ago.')
elif -2 < percent_change <= 0:
    print(f'Performance is stable compared to nearly 30 sessions ago.')
elif -2 > percent_change:
    print(f'Your average 1RM has dropped {abs(percent_change)}% compared to nearly 30 sessions ago. Consider recovery review.')


# In[47]:


print(f'''
Current 8-session average 1RM: {round(recent_avg, 2)} kg
~30 sessions ago 1RM: {round(avg_30_sessions_ago, 2)} kg
Net improvement: +{percent_change}%
''')


# In[51]:


pr_df = stats_data.groupby('Exercise')['Weight'].max().reset_index()


# In[56]:


session_max = stats_data.groupby(['Exercise','Session ID'])['1RM'].max().reset_index()


# In[59]:


session_max['Historical Max'] = (
    session_max
    .groupby('Exercise')['1RM']
    .cummax()
    .shift(1)
)


# In[66]:


session_max['Is PR'] = (
    session_max['1RM'] > session_max['Historical Max']
)


# In[93]:


session_max['PR%'] = (
    (session_max['1RM'] - session_max['Historical Max'])
    / session_max['Historical Max']
) * 100
session_max['PR%'] = session_max['PR%'].round(1)


# In[114]:


pr_rows = session_max[session_max['Is PR'] == True]


# In[117]:


latest_pr = pr_rows.groupby('Exercise').tail(1)


# In[129]:


print(f"Your 1RM PRs as per session: {session_max['Session ID'].iloc[-1]}\n")
for _,row in latest_pr.iterrows():
    print(f"{row['Exercise']}")
    print(f'{round(row['1RM'],1)}kg (+{row['PR%']}%)\n')


# In[ ]:




