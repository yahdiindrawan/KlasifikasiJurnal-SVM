#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ex save the model 
import pickle
filename = 'TFIDF.pickle'
pickle.dump(TFIDF, open(filename, 'wb'))


# In[39]:


pip install Sastrawi


# In[40]:


import nltk
nltk.download('punkt')


# In[41]:


import string
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# In[119]:


kalimat = "PERANCANGAN SISTEM PAKAR UNTUK DIAGNOSA PENYAKIT ANAK" 
lower_case = kalimat.lower()
print (lower_case)


# In[120]:


punctuation = lower_case.translate(str.maketrans('','',string.punctuation)).strip()
print(punctuation)


# In[121]:


tokenize = word_tokenize(punctuation)
print(tokenize)


# In[122]:


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
hasil_stopword = []
for i in tokenize:
    word = stopword.remove(i)
    if word !='':
        hasil_stopword.append(word)
print(hasil_stopword)


# In[123]:


factory = StemmerFactory()
stemmer = factory.create_stemmer()  
hasil_stemming = [] 
for i in hasil_stopword:
    word = stemmer.stem(i)
    if word !='':
        hasil_stemming.append(word) 
print(hasil_stemming)


# In[124]:


text_final=[' '.join(reversed(hasil_stemming))]
print(text_final)


# In[125]:


#Load TFIDF model
import pickle
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("TFIDF.pkl", "rb")))
tfidf = transformer.fit_transform(loaded_vec.fit_transform(text_final))


# In[126]:


print (tfidf)


# In[127]:


# load SVM model
loaded_SVM_model = pickle.load(open('SVM_model.sav', 'rb'))


# In[128]:


print (loaded_SVM_model)


# In[129]:


loaded_SVM_model.predict(tfidf)


# In[130]:


# load SVM  LINIER model
loaded_SVMlinier_model = pickle.load(open('SVMLinier_model.sav', 'rb'))


# In[131]:


print (loaded_SVMlinier_model)


# In[132]:


loaded_SVMlinier_model.predict(tfidf)


# In[133]:


# load SVM  POLYNOMIAL model
loaded_SVMpolynomial_model = pickle.load(open('SVMPolynomial_model.sav', 'rb'))


# In[134]:


print (loaded_SVMpolynomial_model)


# In[135]:


loaded_SVMpolynomial_model.predict(tfidf)


# In[136]:


# load SVM RBF model
loaded_SVMrbf_model = pickle.load(open('SVMRBF_model.sav', 'rb'))


# In[137]:


print (loaded_SVMrbf_model)


# In[138]:


loaded_SVMrbf_model.predict(tfidf)


# In[ ]:




