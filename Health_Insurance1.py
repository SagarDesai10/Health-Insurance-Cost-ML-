# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 16:07:41 2023

@author: Lappy store
"""

import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

data=pd.read_csv("D:\programming\AI & ML\data set/insurance.csv")
data['sex']=data['sex'].map({"male":1,"female":0})
data['smoker']=data['smoker'].map({"yes":1,"no":0})
data['region']=data['region'].map({"southwest":1,"southeast":2,"northwest":3,"northeast":4})
X=data.drop(['charges'],axis=1)
y=data['charges']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
gr=GradientBoostingRegressor()
gr.fit(X_train,y_train)
joblib.dump(gr,'model_joblib_gr')


             
def main():
    
    html_temp="""<div style="background-color:lightblue;border-radius:3px;padding:16px">
                 <h2 style="color:black";text-align:center>Health Insurance Cost Prediction</h2>
                 </div>"""
                 
    st.markdown(html_temp,unsafe_allow_html=True)
    model=joblib.load('model_joblib_gr')
    
    p1=st.slider("Enter your age",18,100)
    s=st.selectbox("sex", ("male","female"))
    
    if s=="male":
        p2=1
    else:
        p2=0
        
    p3=st.number_input("Enter your BMI value")
    
    p4=st.slider("Enter number of children ",0,4)
    
    s1=st.selectbox("Smoker", ("yes","no"))
    
    if s1=="yes":
        p5=1
    else:
        p5=0
    
    s2=st.selectbox("Region",("southwest","southeast","northwest","northeast"))
    if s2=="southwest":
        p6=1
    elif s2=="southeast":
        p6=2
    elif s2=="northwest":
        p6=3
    else:
        p6=4
   
    
    if st.button("predict"):
        result=model.predict([[p1,p2,p3,p4,p5,p6]])
        
        st.balloons()
        st.success("your insurance cost is {}".format(round(result[0], 2)))
    
    
    
#if __name__ == '__main()__':

main()
    