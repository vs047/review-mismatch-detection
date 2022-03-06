import streamlit as st
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import re
import emoji 

#function for text cleaning
def clean_text(data):
  preprocessed_text=[]
  stopwords_list=stopwords.words('english')
  lemmatizer=WordNetLemmatizer()
  for i in data:
    row_important_words=[]
    clean_string=emoji.demojize(i)
    clean_string=re.sub(r'[^\w\s]','',i)
    clean_string=re.sub('</|[a-z]*>','',i)
    clean_string=clean_string.lower()   #conversion of sentence into lowercase
    row_words=word_tokenize(clean_string,language='english')  #tokenization
    for j in row_words:
      if j not in stopwords_list:
        row_important_words.append(lemmatizer.lemmatize(j))  #removal of stopwords and lemmatization 
    preprocessed_text.append(row_important_words)
  return preprocessed_text

#function for text embedding  
def convert_text_to_vectors(data,embedding_model):
  text_vectors=[]
  for i in data:
    row_vector=np.ravel(np.zeros((1,100),dtype=float))
    for j in i:
      try:
        row_vector+=embedding_model.wv[j]
      except:
        row_vector+=np.array([0])
    text_vectors.append(np.ravel(row_vector/(len(i)+0.0001)))
  return np.array(text_vectors)

#function for showing final result
def show_table(data):
  st.header("rows with positive text but negative ratings")
  column_dict={}
  for i in data.columns.values:
    if i.lower()=="text":
      column_dict[i]=i.lower()
    elif i.lower()=="star":
      column_dict[i]="score"
  data=data[column_dict.keys()]
  data.rename(columns=column_dict,inplace=True)
  data.dropna(inplace=True)
  temp=data.copy()
  temp.text=clean_text(data.text)
  Embedding_oject=open("Embedding.pkl","rb")
  Embedder=pickle.load(Embedding_oject)
  temp.text=convert_text_to_vectors(temp.text,Embedder).tolist()
  model_oject=open("classifier.pkl","rb")
  classifier_model=pickle.load(model_oject)
  y_pred=classifier_model.predict(list(temp.text))
  data["score_pred"]=y_pred
  temp=data[data.score_pred==1]
  temp=temp[temp.score<3]
  temp.drop(columns=["score_pred"],inplace=True)
  st.write(temp[temp.score<3])
def main_screen():
  st.title("Welcome :smile:")
  with st.form("data_form"):
    st.header("Main Page")
    st.subheader("enter your csv file here")
    file_upload=st.file_uploader("Choose a csv file",type=["csv"])
    if file_upload is not None:
      data=pd.read_csv(file_upload)
      st.write("Filename:",file_upload.name)
    submitted=st.form_submit_button("Result")
    if submitted and file_upload is not None:
      show_table(data)
main_screen()
