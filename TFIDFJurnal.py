import json

# buka file JSON-nya dulu
with open('hasil_preprocessing.json','r') as myfile:
    
# load file JSON dari file yang sudah dibuka
    data = json.loads(myfile.read())
    
# cetak isi data 
# print (data)

hasil_preprocessing = []
for i in range(len(data['hasil_preprocessing'])):
    hasil_preprocessing.append(data['hasil_preprocessing'][i][0])


text_final=[' '.join(sen) for sen in hasil_preprocessing] 
# menghubungkan menjadi 1 kalimat 


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
vectorizer = CountVectorizer(decode_error="replace")
transformer = TfidfTransformer()
TFIDF = transformer.fit_transform(vectorizer.fit_transform(text_final))


print (TFIDF)


#Save model TFIDF
import pickle
pickle.dump(vectorizer.vocabulary_,open("TFIDF.pkl","wb"))


label=[]
for i in range(len(data['casefolding'])):
    if data['casefolding'][i][3]=='grafik':
        nilai = 0
        label.append(nilai)
        
    if data['casefolding'][i][3]=='artificial intelligance da robotik':
        nilai = 1
        label.append(nilai)
        
    if data['casefolding'][i][3]=='database dan sistim retrieval informasi':
        nilai = 2
        label.append(nilai)
        
    if data['casefolding'][i][3]=='sistem operasi dan jaringan':
        nilai = 3
        label.append(nilai)
        
    if data['casefolding'][i][3]=='algoritma dan struktur data':
        nilai = 4
        label.append(nilai)

print(len(label))
print(len(text_final))
print(label)

from sklearn import svm #method untuk pross perhitungan klasifikasi
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer #method untuk menghitung vsm dan tfidf
from sklearn import metrics #method untuk pembentukan matriks 1x1, 2x2, 3x3, ...
from sklearn.metrics import accuracy_score #method perhitungan akurasi
from sklearn.model_selection import KFold #Method perhitungan K-Fold
import numpy as np #scientific computing untuk array N-dimenesi
import re #re = regular expression
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC



print(label)

#METHOD MENGHITUNG RATA2 AKURASI#
akurasi = []
akurasi1 = []
akurasi2 = []
akurasi3 = []
data_uji = []
prediksi_linear = []
prediksi_svm = []
prediksi_rbf = []
label_uji = []
def avg_akurasiLinear(): ## nama fungsi
    total = 0 ## pengosongan variabel
    for i in range(10): ## looping 10x karena ada 10 fold
        total = total + akurasi[i] ## tiap looping nilai total akan ditambahkan dengan nilai akurasi tiap fold
    print("-------------------------------------------------------------------------------------------------------") ## cetak pembatas
    print("Rata-rata akurasi keseluruhan SVM LinearSVC adalah :", total / 10) ## cetak rata-rata akurasi
def avg_akurasiSVM(): ## nama fungsi
    total = 0 ## pengosongan variabel
    for i in range(10): ## looping 10x karena ada 10 fold
        total = total + akurasi1[i] ## tiap looping nilai total akan ditambahkan dengan nilai akurasi tiap fold
    print("Rata-rata akurasi keseluruhan SVM kernel Linear adalah :", total / 10) ## cetak rata-rata akurasi
def avg_akurasiPolynomial(): ## nama fungsi
    total = 0 ## pengosongan variabel
    for i in range(10): ## looping 10x karena ada 10 fold
        total = total + akurasi2[i] ## tiap looping nilai total akan ditambahkan dengan nilai akurasi tiap fold
    print("Rata-rata akurasi keseluruhan SVM kernel Polynomial adalah :", total / 10) ## cetak rata-rata akurasi
def avg_akurasiRBF(): ## nama fungsi
    total = 0 ## pengosongan variabel
    for i in range(10): ## looping 10x karena ada 10 fold
        total = total + akurasi3[i] ## tiap looping nilai total akan ditambahkan dengan nilai akurasi tiap fold
    print("Rata-rata akurasi keseluruhan SVM kernel RBF adalah :", total / 10) ## cetak rata-rata akurasi

kFoldCrossValidation = KFold(n_splits=10)#fungsi K-Fold Cross Validation melakukan insialisasi 10x iterasi
for latih, uji in kFoldCrossValidation.split(TFIDF, label):#proses looping yg masing2 pernah jadi data latih maupun uji
    print("-----------------------------------------------------------------------")
    print("Banyak Data Latih: ", len(latih))
    print("Banyak Data Uji: ", len(uji))
    print("\nData Latih: \n", latih)
    print("\nData Uji: \n", uji)
    
    dataLatih1, dataUji1 = TFIDF[latih], TFIDF[uji]#proses inisialisasi dari masing2 data latih/uji dijadikan nilai tfidf lalu di copy ke variabel dataLatih/Uji1
    label = np.array(label)
    dataLatih2, dataUji2 = label[latih], label[uji]#proses inisialisasi dari masing2 data latih/uji dibentuk ke label untuk proses prediksi lalu di copy ke variabel dataLatih/Uji2
    
    SVMLinear = svm.LinearSVC().fit(dataLatih1, dataLatih2)#data latih melakukan proses pelatihan dengan algoritma SVM
    prediksi = SVMLinear.predict(dataUji1)#proses prediksi dari data latih yang sudah tersimpan sebagai model
    SVM = svm.SVC(kernel='linear').fit(dataLatih1, dataLatih2)  # data latih melakukan proses pelatihan dengan algoritma SVM
    prediksi1 = SVM.predict(dataUji1)  # proses prediksi dari data latih yang sudah tersimpan sebagai model
    SVMPoly = svm.SVC(kernel='poly', degree=3).fit(dataLatih1, dataLatih2)  # data latih melakukan proses pelatihan dengan algoritma SVM
    prediksi2 = SVMPoly.predict(dataUji1)  # proses prediksi dari data latih yang sudah tersimpan sebagai model
    SVMRBF = svm.SVC(kernel='rbf', gamma=0.7).fit(dataLatih1, dataLatih2)  # data latih melakukan proses pelatihan dengan algoritma SVM
    prediksi3 = SVMRBF.predict(dataUji1)  # proses prediksi dari data latih yang sudah tersimpan sebagai model
    print(uji)
    print("\nHasil Prediksi SVM Linear: \n", prediksi)
    print("\nHasil Prediksi SVM: \n", prediksi1)
    print("\nHasil Prediksi SVM Polynomial: \n", prediksi2)
    print("\nHasil Prediksi SVM RBF: \n", prediksi3)
    print(dataUji2)
    data_uji.append(list(uji))
    prediksi_linear.append(list(prediksi))
    prediksi_svm.append(list(prediksi1))
    prediksi_rbf.append(list(prediksi3))
    label_uji.append(list(dataUji2))
    dataSimpan = {
        'indeks' : list(uji),
        'linear' : list(prediksi),
        'svm' : list(prediksi1),
        'rbf' : list(prediksi3),
        'label' : list(dataUji2)
    }
    simpan.append(dataSimpan)

    print("\nConfusion Matrix: \n", metrics.confusion_matrix(dataUji2, prediksi1))#proses pembetukan metriks

    akurasi.append(accuracy_score(dataUji2, prediksi))
    akurasi1.append(accuracy_score(dataUji2, prediksi1))
    akurasi2.append(accuracy_score(dataUji2, prediksi2))
    akurasi3.append(accuracy_score(dataUji2, prediksi3))

    print("\nAkurasi SVM LinearSVC: ", accuracy_score(dataUji2, prediksi))
    print("\nAkurasi SVM kernel Linear: ", accuracy_score(dataUji2, prediksi1))
    print("\nAkurasi SVM kernel Polynomial: ", accuracy_score(dataUji2, prediksi2))
    print("\nAkurasi SVM kernel RBF: ", accuracy_score(dataUji2, prediksi3))
    print()
    label_target = ['grafik','database dan sistim retrieval informasi','artificial intelligance da robotik','sistem operasi dan jaringan','algoritma dan struktur data']
    print(metrics.classification_report(dataUji2, prediksi1, target_names=label_target))#proses pembentukan confusin matrix

avg_akurasiLinear()
avg_akurasiSVM()
avg_akurasiPolynomial()
avg_akurasiRBF()
