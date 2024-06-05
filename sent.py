import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

st.title("MOVIE REVIEW SENTIMENTAL ANALYSIS")
df1=pd.read_csv("IMDB Dataset.csv")
df=df1.head(4000)
dic={"sentiment":{"positive":1 ,"negative":0}}
df=df.replace(dic)

a= CountVectorizer()
X_train_cv=a.fit_transform(df.review.values)
X_train_ar=X_train_cv.toarray()

b = MultinomialNB()
b.fit(X_train_ar,df.sentiment)

input=st.text_input("GIVE TEXT ")
input_cv=a.transform([input])
input_ar=input_cv.toarray()
if input:
   sent=b.predict(input_ar)
   if sent[0]==1:
      st.success("POSITIVE STATEMENT")
   elif sent[0]==0:
      st.success("NEGATIVE STATEMENT")

