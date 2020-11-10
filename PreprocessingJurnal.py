#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import string
# import Sastrawi
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactorydassfgfog,


# In[4]:


filedataolah = 'DataS.csv'
f = open(filedataolah,'r')
reader = csv.reader(f)


# In[5]:


def casefolding(judul, abstrak, keyword, label):
    judul = judul.lower()
    abstrak = abstrak.lower()
    keyword = keyword.lower()
    label = label.lower()
    return(judul, abstrak, keyword, label)


# In[17]:


hasil_case_folding = []
for data in reader:
    hasil = casefolding(data[1],data[2],data[3],data[4])
    hasil_case_folding.append(hasil)
print(hasil_case_folding)   


# In[16]:


print(hasil_case_folding)


# In[8]:


def remove_punctuation(judul, abstrak, keyword, label):
    judul = judul.translate(str.maketrans('','',string.punctuation)).strip()
    abstrak = abstrak.translate(str.maketrans('','',string.punctuation)).strip()
    keyword = keyword.translate(str.maketrans('','',string.punctuation)).strip()
    label = label.translate(str.maketrans('','',string.punctuation)).strip()
    return(judul, abstrak, keyword, label)


# In[9]:


hasil_remove_punctuation = []
for data in range(len(hasil_case_folding)):    
    hasil = remove_punctuation(hasil_case_folding[data][0],
                               hasil_case_folding[data][1],
                               hasil_case_folding[data][2],
                               hasil_case_folding[data][3])
    hasil_remove_punctuation.append(hasil)
print(hasil_remove_punctuation)


# In[11]:


print (hasil_remove_punctuation)


# In[12]:


def tokenize(judul,abstrak,keyword,label):
    judul = word_tokenize(judul)
    abstrak = word_tokenize(abstrak)
    keyword = word_tokenize(keyword)
    label = word_tokenize(label)
    return(judul, abstrak, keyword, label)


# In[13]:


hasil_tokenize = []
for data in range(len(hasil_remove_punctuation)):    
    hasil = tokenize(hasil_remove_punctuation[data][0],
                     hasil_remove_punctuation[data][1],
                     hasil_remove_punctuation[data][2],
                     hasil_remove_punctuation[data][3])
    hasil_tokenize.append(hasil)
print(hasil_tokenize)


# In[14]:


print(hasil_tokenize)


# In[15]:


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
def stopwords(judul,abstrak,keyword,label): 
    hasil_judul = []
    for i in judul:
        word = stopword.remove(i)
        if word !='':
            hasil_judul.append(word)
    
    hasil_abstrak = []
    for i in abstrak:
        word = stopword.remove(i)
        if word !='':
            hasil_abstrak.append(word)
            
    hasil_keyword = []
    for i in keyword:
        word = stopword.remove(i)
        if word !='':
            hasil_keyword.append(word)
            
    hasil_label = []
    for i in abstrak:
        word = stopword.remove(i)
        if word !='':
            hasil_label.append(word)
    return(hasil_judul, hasil_abstrak, hasil_keyword, hasil_label) 


# In[16]:


hasil_stopwords = []
for data in range(len(hasil_tokenize)):    
    hasil = stopwords(hasil_tokenize[data][0],
                      hasil_tokenize[data][1],  
                      hasil_tokenize[data][2], 
                      hasil_tokenize[data][3]) 
    hasil_stopwords.append(hasil)
print(hasil_stopwords)


# In[17]:


factory = StemmerFactory()
stemmer = factory.create_stemmer() 
def stemming(judul,abstrak,keyword,label): 
    hasil_judul = [] 
    for i in judul:
        word = stemmer.stem(i)
        if word !='':
            hasil_judul.append(word) 
    
    hasil_abstrak = []
    for i in abstrak:
        word = stemmer.stem(i)
        if word !='':
            hasil_abstrak.append(word)
            
    hasil_keyword = []
    for i in keyword:
        word = stemmer.stem(i)
        if word !='':
            hasil_keyword.append(word)
            
    hasil_label = []
    for i in abstrak:
        word = stemmer.stem(i)
        if word !='':
            hasil_label.append(word)
    return(hasil_judul, hasil_abstrak, hasil_keyword, hasil_label)


# In[18]:


hasil_stemming = []  
for data in range(len(hasil_stopwords)):    
    hasil = stemming (hasil_stopwords[data][0],
                      hasil_stopwords[data][1],  
                      hasil_stopwords[data][2], 
                      hasil_stopwords[data][3]) 
    hasil_stemming.append(hasil)
print(hasil_stemming)


# In[23]:


import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# In[28]:


def bagOfWords(judul): 
    hasil_judul = [] 
    for i in judul:
        bgOfWords = bagOfWords.split(i)
        if bgOfWords !='':
            hasil_judul.append(bgOfWords) 
   
    return(hasil_judul)


# In[29]:


hasil_bagOfWords = []  
for data in range(len(hasil_stemming)):    
    hasil = bagOfWords (hasil_stemming[data][0]) 
    hasil_bagOfWords.append(hasil)
print(hasil_bagOfWords)


# In[160]:


hasil_preprocessing = hasil_stemming
json_preprocessing = {
    'casefolding':hasil_case_folding,
    'remove_punctuation':hasil_remove_punctuation,
    'tokenizing':hasil_tokenize,
    'stopwords':hasil_stopwords,
    'hasil_preprocessing':hasil_preprocessing,
    'label':label
}


# In[161]:


json_preprocessing


# In[162]:


import json
with open('hasil_preprocessing.json', 'w') as json_file:
  json.dump(json_preprocessing, json_file)


# In[18]:


#cara import pandas, tidak diload dalam proses ini
import pandas as pd

judul = []
abstrak = []
keyword = []
label = []

data_dict = {'JUDUL':judul, 'ABSTRAK':abstrak, 'KEYWORD':keyword, 'LABEL':label}
df = pd.DataFrame(data_dict,columns=['JUDUL','ABSTRAK','KEYWORD','LABEL'])

