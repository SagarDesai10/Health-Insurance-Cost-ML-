#!/usr/bin/env python
# coding: utf-8

# import data set

# In[1]:


import pandas as pd

data=pd.read_csv("D:\programming\AI & ML\data set/insurance.csv")
data.head()


# In[2]:


data.tail()


# find basic info

# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isna().sum()


# find unique values

# In[7]:


data['sex'].unique()


# In[8]:


data['smoker'].unique()


# In[9]:


data['region'].unique()


# convert data to num using map

# In[10]:


data['sex']=data['sex'].map({"male":1,"female":0})
data['sex']


# In[11]:


data['smoker']=data['smoker'].map({"yes":1,"no":0})
data['smoker']


# In[12]:


data['region']=data['region'].map({"southwest":1,"southeast":2,"northwest":3,"northeast":4})
data['region']


# Store matrix in X and target in y

# In[13]:


X=data.drop(['charges'],axis=1)
X


# In[14]:


y=data['charges']
y


# Train Test Split

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train


# In[17]:


y_train


# import the model

# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# model training

# In[19]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[20]:


sv=SVR()
sv.fit(X_train,y_train)


# In[21]:


rf=RandomForestRegressor()
rf.fit(X_train,y_train)


# In[22]:


gr=GradientBoostingRegressor()
gr.fit(X_train,y_train)


# value prediction

# In[23]:


y_pred1=lr.predict(X_test)
y_pred2=sv.predict(X_test)
y_pred3=rf.predict(X_test)
y_pred4=gr.predict(X_test)


# In[24]:


data={"Actual":y_test,"lin":y_pred1,"svr":y_pred2,"rf":y_pred3,"gr":y_pred4}
df=pd.DataFrame(data)
df


# comper model

# In[25]:


import matplotlib.pyplot as plt


# In[26]:


plt.subplot(221)
plt.plot(df['Actual'].iloc[0:15],label="A")
plt.plot(df['lin'].iloc[0:15],label="lin")
plt.legend()
plt.show()

plt.subplot(222)
plt.plot(df['Actual'].iloc[0:15],label="A")
plt.plot(df['svr'].iloc[0:15],label="svr")
plt.legend()
plt.show()

plt.subplot(223)
plt.plot(df['Actual'].iloc[0:15],label="A")
plt.plot(df['rf'].iloc[0:15],label="rf")
plt.legend()
plt.show()

plt.subplot(224)
plt.plot(df['Actual'].iloc[0:15],label="A")
plt.plot(df['gr'].iloc[0:15],label="gr")
plt.tight_layout()
plt.legend()
plt.show()


# Evaluting Algoritham

# In[27]:


from sklearn import metrics


# In[28]:


score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)

print(score1,score2,score3,score4)


# In[29]:


s1=metrics.mean_absolute_error(y_test,y_pred1)
s2=metrics.mean_absolute_error(y_test,y_pred2)
s3=metrics.mean_absolute_error(y_test,y_pred3)
s4=metrics.mean_absolute_error(y_test,y_pred4)

print(s1,s2,s3,s4)


# save model using joblib

# In[30]:


gr=GradientBoostingRegressor()
gr.fit(X,y)


# In[31]:


gr.score(X_train,y_train)


# In[32]:


import joblib


# In[33]:


joblib.dump(gr,'model_joblib_gr')


# In[34]:


model=joblib.load('model_joblib_gr')


# In[35]:


d={"age":35,"sex":0,"bmi":30,"children":2,"smoker":0,"region":3}
df1=pd.DataFrame(d,index=[0])
df1


# In[36]:


model.predict(df1)


# GUI

# In[37]:


from tkinter import *
import joblib


# In[38]:


def show_result():
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    
    model=joblib.load('model_joblib_gr')
    result=model.predict([[p1,p2,p3,p4,p5,p6]])
    
    Label(master,text="Charges").grid(row=7)
    Label(master,text=result).grid(row=8)



master=Tk()

master.title("Health Insurance Cost Prediction")
laabel=Label(master,text="Health Insurance Cost Prediction",bg="black",fg="white").grid(row=0,columnspan=2)
Label(master,text="Age").grid(row=1)
Label(master,text="Sex male/female[1/0]").grid(row=2)
Label(master,text="BMI").grid(row=3)
Label(master,text="Childern").grid(row=4)
Label(master,text="Smoker yes/no[1/0]").grid(row=5)
Label(master,text="region [1-4]").grid(row=6)

e1=Entry(master)
e2=Entry(master)
e3=Entry(master)
e4=Entry(master)
e5=Entry(master)
e6=Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)

Button(master,text="prdict",command=show_result).grid()



mainloop()


# In[ ]:




