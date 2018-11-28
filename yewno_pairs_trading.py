#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:29:18 2018

@author: jingjia
"""

#-----------Explore Emerging Market Pairs Trading Strategy by Using ETF-------------------------
# I believe in Emerging Market the equity market has a strong connection with 
# Sovereign debt and Commodity index. Based on the OLS regression result I can 
# build a trading strategy to capture the trading opportunities     


import pandas as pd
import numpy  as np 
import statsmodels.tsa.stattools as ts
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import statsmodels.tsa.api  as tool

#------------Getting data-----------------------------------------------------
 # DGS:EM Small Cap ETF  (I didn't use the small cap data that can be explored later)
 # EEM: MSCI EM Market ETF
 # PCY: Invesco EM Sovereign Debt ETF
 # DBC: Commodity Total return index


def loadDf1FromFile(fileName):
    return pd.read_csv(fileName,index_col='Date',na_values=['N/A'])
EM = loadDf1FromFile('EM_data.csv')# EM Market ETF data from Yahoo

# take logarithm of the raw data and plot them

log_EM = np.log(EM)
log_EM.plot()

#-------------------Get data list------------------------------------------
EEM  = log_EM['EEM'].values.tolist()
DGS = log_EM['DGS'].values.tolist()
PCY = log_EM['PCY'].values.tolist()
DBC = log_EM['DBC'].values.tolist()

#------------Unit root testing-----------------------------------------------
res_EEM=ts.adfuller(EEM,regression="c",autolag=None,maxlag=1)
print ('EEM ADF result:\n',res_EEM[0:2])
print ('Test critical value\n',res_EEM[4])
res_DGS=ts.adfuller(DGS,regression="c",autolag=None,maxlag=1)
print ('DGS ADF result:\n',res_DGS[0:2])
print ('Test critical value\n',res_DGS[4])
res_DBC=ts.adfuller(DBC,regression="c",autolag=None,maxlag=1)
print ('DBC ADF result:\n', res_DBC[0:2])
print ('Test critical value\n',res_DBC[4])

#--------------- OLS regression----------------------------------------------
y=np.array(EEM)

x1=np.array(DBC) 
x2=np.array(PCY) 
 
num_vals1=len(x1)
b=np.vstack([x1,x2,np.ones(num_vals1)]).T
test_beta1=OLS(y,b)


out=test_beta1.fit()
out.summary()
m1=out.params[0]
m2=out.params[1]

c=out.params[2]
print ('m1, m2, c:     ', m1, m2, c)


#--------------plot 3D chart ------------------------------------------------
def f(x1,x2):
    return c+m1*x1+m2*x2
m1=out.params[0]
m2=out.params[1]


c=out.params[2]

ax = plt.axes(projection='3d')


# Data for three-dimensional scattered points
ax.scatter3D(x1, x2,np.array(EEM) )
z=f(x1,x2)
ax.scatter3D(x1, x2, f(x1,x2), c=z, cmap='viridis', linewidth=0.5);

ax.set_xlabel('DBC')
ax.set_ylabel('PCY')
ax.set_zlabel('EEM')
ax.view_init(45, 35);
ax.view_init(30, 15);

#-------plain chart -------------------------------

x = np.linspace(2.8, 4, 30)
y = np.linspace(2, 3, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax = plt.axes(projection='3d')


# Data for three-dimensional in a plain
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');

ax.set_xlabel('DBC')
ax.set_ylabel('PCY')
ax.set_zlabel('EEM')
ax.view_init(60, 35);

#--------------Regression residual unit testing------------------------------

residual=y-m1*x1-m2*x2-c
test_result=ts.adfuller(residual,regression="c",autolag=None,maxlag=1)
print ('Residual ADF result:\n', test_result[0:2])
print ('Test critical value\n',test_result[4])

test_result1=ts.adfuller(residual,regression="ct",autolag=None,maxlag=1)
print ('Residual ADF result:\n', test_result1[0:2])
print ('Test critical value\n',test_result1[4])

# From the unit testing we can see the residual is stationary

plt.plot(residual, 'r', label='Residual line')
plt.legend()
plt.show()

#-------------------VAR model fitting --------------------------------------
#VAR model fitting for further analysis

df1=np.vstack([DGS,PCY,EEM]).T
model=tool.VAR(df1,freq='D')
result=model.fit(2)
print (result.summary())
