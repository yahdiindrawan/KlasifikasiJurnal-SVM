!pip install Sastrawi
import nltk
nltk.download('punkt')
import string
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle

filename = 'TFIDF.pickle'
pickle.dump(TFIDF, open(filename, 'wb'))

kalimat = "PERANCANGAN SISTEM PAKAR UNTUK DIAGNOSA PENYAKIT ANAK" 
lower_case = kalimat.lower()
print (lower_case)


punctuation = lower_case.translate(str.maketrans('','',string.punctuation)).strip()
print(punctuation)

tokenize = word_tokenize(punctuation)
print(tokenize)


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
hasil_stopword = []
for i in tokenize:
    word = stopword.remove(i)
    if word !='':
        hasil_stopword.append(word)
print(hasil_stopword)


factory = StemmerFactory()
stemmer = factory.create_stemmer()  
hasil_stemming = [] 
for i in hasil_stopword:
    word = stemmer.stem(i)
    if word !='':
        hasil_stemming.append(word) 
print(hasil_stemming)


text_final=[' '.join(reversed(hasil_stemming))]
print(text_final)

#Load TFIDF model
import pickle
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("TFIDF.pkl", "rb")))
tfidf = transformer.fit_transform(loaded_vec.fit_transform(text_final))

print (tfidf)


# load SVM model
loaded_SVM_model = pickle.load(open('SVM_model.sav', 'rb'))
print (loaded_SVM_model)
loaded_SVM_model.predict(tfidf)

# load SVM  LINIER model
loaded_SVMlinier_model = pickle.load(open('SVMLinier_model.sav', 'rb'))
print (loaded_SVMlinier_model)
loaded_SVMlinier_model.predict(tfidf)


# load SVM  POLYNOMIAL model
loaded_SVMpolynomial_model = pickle.load(open('SVMPolynomial_model.sav', 'rb'))
print (loaded_SVMpolynomial_model)
loaded_SVMpolynomial_model.predict(tfidf)


# load SVM RBF model
loaded_SVMrbf_model = pickle.load(open('SVMRBF_model.sav', 'rb'))
print (loaded_SVMrbf_model)
loaded_SVMrbf_model.predict(tfidf)
