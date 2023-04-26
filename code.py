import sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

## FUNCTION TO FIND THE WORLD WITH THE MAXIMUM CALL
def maximum(liste):
    maxi = liste[0]
    indice=0
    for i in range(0,len(liste)):
        if liste[i] >= maxi:
            maxi = liste[i]
            indice=i

    return maxi,indice






## ---------------------------------------------------------PART 0 , a --------------------------------------------------------------------------
'''
#IMPORTATIONS
data = pd.read_csv('./IA3-train.csv')


tweets_positive=[]
tweets_negative=[]
for i in range(0,len(data)):
    if data.iloc[i][0]==0:
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_positive+=[result]
    else :
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_negative+=[result]

tweets=tweets_positive
vectorizer = CountVectorizer(lowercase=True)
X = vectorizer.fit_transform(tweets)
name_of_words=vectorizer.get_feature_names_out()
X=X.toarray()
print(name_of_words[3456])


## Counter of words in tweet ( name counter )
counter = np.zeros(len(name_of_words))
for i in range(0,len(tweets)):
    print(len(tweets)-i)
    for j in range (0,len(name_of_words)):
        counter[j]+=X[i][j]

## Find the 10 best words
top=[]
number_of_top=[]
for i in range (0,10):
    max,indice=maximum(counter)
    top+=[name_of_words[indice]]
    number_of_top+=[max]
    name_of_words = np.delete(name_of_words,indice)
    counter = np.delete(counter,indice)

print(top)
print(number_of_top)
'''
## ---------------------------------------------------------PART 0 , b --------------------------------------------------------------------------
'''
#IMPORTATIONS
data = pd.read_csv('./IA3-train.csv')


tweets_positive=[]
tweets_negative=[]
for i in range(0,len(data)):
    if data.iloc[i][0]==0:
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_positive+=[result]
    else :
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_negative+=[result]
    


#TfidVectorizer
tweets=tweets_negative

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
X = vectorizer.fit_transform(tweets)
name_of_words=vectorizer.get_feature_names_out()
X=X.toarray()
print(name_of_words[3456])


## Counter of words in tweet ( name counter )
counter = np.zeros(len(name_of_words))
for i in range(0,len(tweets)):
    print(len(tweets)-i)
    for j in range (0,len(name_of_words)):
        counter[j]+=X[i][j]

## Find the 10 best words
top=[]
number_of_top=[]
for i in range (0,10):
    max,indice=maximum(counter)
    top+=[name_of_words[indice]]
    number_of_top+=[max]
    name_of_words = np.delete(name_of_words,indice)
    counter = np.delete(counter,indice)

print(top)
print(number_of_top)
'''

#------------------------------------------------------------------Part 1 / Part 2  ---------------------------------------------------------------
'''
#IMPORTATIONS
data = pd.read_csv('./IA3-dev.csv')


tweets_positive=[]
tweets_negative=[]
tweets_general=[]
for i in range(0,len(data)):
    if data.iloc[i][0]==0:
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_positive+=[result]
    else :
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_negative+=[result]
    liste_words= data.iloc[i][1].split()
    liste_words = liste_words[1:]  # to no have the user 
    result = " ".join(liste_words)
    tweets_general+=[result]
    


#TfidVectorizer
tweets=tweets_general

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
X = vectorizer.fit_transform(tweets)
name_of_words=vectorizer.get_feature_names_out()
#X=X.toarray()
 
Y=data['sentiment']


X_train, X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.20) 

accuracy_total=[]
#C=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
C=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for C_chosen in C :
    Gamma=0.00001
    clf =svm.SVC(kernel='linear', C = C_chosen)
    #clf =svm.SVC(kernel='rbf', C = C_chosen, gamma=Gamma)
    clf.fit(X_train,Y_train)

    #Prediction
    Y_pred = clf.predict(X_test)

    #Evalutation of the model
    accuracy=metrics.accuracy_score(Y_test,Y_pred)
    class_report = classification_report(Y_test,Y_pred)
    print("########################################")
    print("for c :",C_chosen)
    print("Accuracy :", accuracy)
    print("Supports vectors for each class :", clf.n_support_)
    print ("Total number of supports vectors :", clf.n_support_[0] + clf.n_support_[1])
    print (class_report)
    accuracy_total+=[accuracy]


print("FINI")

print(C)
print(accuracy_total)
'''
#------------------------------------------------------------------Part 3  ---------------------------------------------------------------

#IMPORTATIONS
data = pd.read_csv('./IA3-train.csv')


tweets_positive=[]
tweets_negative=[]
tweets_general=[]
for i in range(0,len(data)):
    if data.iloc[i][0]==0:
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_positive+=[result]
    else :
        liste_words= data.iloc[i][1].split()
        liste_words = liste_words[1:]  # to no have the user 
        result = " ".join(liste_words)
        tweets_negative+=[result]
    liste_words= data.iloc[i][1].split()
    liste_words = liste_words[1:]  # to no have the user 
    result = " ".join(liste_words)
    tweets_general+=[result]
    


#TfidVectorizer
tweets=tweets_general

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
X = vectorizer.fit_transform(tweets)
name_of_words=vectorizer.get_feature_names_out()
#X=X.toarray()
 
Y=data['sentiment']


X_train, X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.20) 

accuracy_total=[]
C=[7,8,9]
G=[0.07,0.1,0.2]
plot_total=[]
for C_chosen in C :
    plot_inte=[]
    for Gamma in G :
        clf =svm.SVC(kernel='rbf', C = C_chosen, gamma=Gamma)
        clf.fit(X_train,Y_train)

        #Prediction
        Y_pred = clf.predict(X_test)

        #Evalutation of the model
        accuracy=metrics.accuracy_score(Y_test,Y_pred)
        class_report = classification_report(Y_test,Y_pred)
        print("########################################")
        print("for c :",C_chosen)
        print("Accuracy :", accuracy)
        print("Supports vectors for each class :", clf.n_support_)
        print ("Total number of supports vectors :", clf.n_support_[0] + clf.n_support_[1])
        print (class_report)
        accuracy_total+=[clf.n_support_[0] + clf.n_support_[1]]
        plot_inte+=[accuracy]
        print(plot_inte)
        print('1')
    plot_total+=[plot_inte]
    print(plot_total)
    


print("FINI")
'''
print(C)
print(accuracy_total)
'''
print(plot_total)



hm = sn.heatmap(data = plot_total, annot=True, cmap="crest", xticklabels=G,
                yticklabels=C)
plt.show()
