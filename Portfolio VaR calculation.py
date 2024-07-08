#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas_datareader as web
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime as dt
import scipy.stats as st
import matplotlib.pyplot as plt


# In[95]:


# Portfolio Of Stocks & the corresponding stocks in the portfolio
tickers = ['RAJESHEXPO.NS']
stocks = [1]


# In[96]:


# Download closing prices
df=pd.DataFrame()
for i in tickers:
  tick=yf.Ticker(i)
  old  = tick.history(start="2023-01-01",  end=dt.now())['Close']
  new=pd.DataFrame(old)
  df[i]=new


# In[97]:


#Calculate the initial invested value(Assuming investment is made today)
initial_investment= stocks*df.iloc[-1,:]


# In[98]:


#Calculate returns
returns=df.pct_change()


# In[99]:


#Extract the correlation matrix 
corr_matrix = returns.corr()


# In[100]:


#Get the standard deviation of returns
st_dev=returns.std()


# In[101]:


#Declare the confidence and Total Value-at-Risk variables
conf=[]
tot_vaRlist=[]


# In[102]:


#Run the simulation for calculating VaR for different confidence intervals
for i in range(0,2500):
  conf+=[75+i/100]
  vaR=st.norm.ppf(conf[i]/100)*st_dev*initial_investment*-1
  vaRlist=np.array(vaR)
  tot_vaR_sq=np.matmul( np.matmul(vaRlist,corr_matrix.to_numpy()) , np.transpose(vaRlist))
  tot_vaR=np.sqrt(tot_vaR_sq)*-1
  tot_vaRlist+=[tot_vaR]


# In[103]:


conf = np.array(conf)
tot_vaRlist = np.array(tot_vaRlist)

# Plot the VaR
plt.plot(conf, tot_vaRlist, label='VaR')

# Highlight VaR at 95% and 99% Confidence Intervals
conf_95_index = np.where(conf == 95)[0][0]
conf_99_index = np.where(conf == 99)[0][0]
vaR_95 = tot_vaRlist[conf_95_index]
vaR_99 = tot_vaRlist[conf_99_index]

plt.scatter([95], [vaR_95], color='red', label=f'VaR at 95% CI: {vaR_95:.2f}')
plt.scatter([99], [vaR_99], color='blue', label=f'VaR at 99% CI: {vaR_99:.2f}')

# Annotate the points
plt.annotate(f'{vaR_95:.2f}', (95, vaR_95), textcoords="offset points", xytext=(0, 10), ha='center')
plt.annotate(f'{vaR_99:.2f}', (99, vaR_99), textcoords="offset points", xytext=(0, 10), ha='center')

# Set labels and title
plt.xlabel('Confidence Interval (%)')
plt.ylabel('Value at Risk (VaR)')
plt.title('Value at Risk (VaR) vs. Confidence Interval')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




