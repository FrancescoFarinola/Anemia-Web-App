import functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

scaler = StandardScaler() #scaler per standardizzare i dati
datacongiuntiva = pd.read_csv("test/conj_smote.csv")
Xc = datacongiuntiva.iloc[:, :-1].values #features della congiuntiva - train
Yc = datacongiuntiva.iloc[:, -1].values #classe della congiuntiva - train
scaler.fit(Xc)
testcongiuntiva = pd.read_csv("test/conj_smote_test.csv")
Xc_test = testcongiuntiva.iloc[:, :-1].values #features della congiuntiva - test
Yc_test = testcongiuntiva.iloc[:, -1].values #classe della congiuntiva - test
scaler.fit(Xc_test)
datapolpastrello = pd.read_csv("test/polp_smote.csv")
Xp = datapolpastrello.iloc[:, :-1].values
Yp = datapolpastrello.iloc[:, -1].values
scaler.fit(Xp)
testpolpastrello = pd.read_csv("test/polp_smote_test.csv")
Xp_test = testpolpastrello.iloc[:, :-1].values
Yp_test = testpolpastrello.iloc[:, -1].values
scaler.fit(Xp_test)
dataunghia = pd.read_csv("test/letto_smote.csv")
Xu = dataunghia.iloc[:, :-1].values
Yu = dataunghia.iloc[:, -1].values
scaler.fit(Xu)
testlettoungueale = pd.read_csv("test/letto_smote_test.csv")
Xu_test = testlettoungueale.iloc[:, :-1].values
Yu_test = testlettoungueale.iloc[:, -1].values
scaler.fit(Xu_test)

#MODELLO PER CONGIUNTIVA
classifier1 = GradientBoostingClassifier(loss='deviance',criterion='mse')  #LMT
#classifier1 = GaussianNB() #Naive Bayes
classifier1.fit(Xc, Yc)
scores1 = classifier1.predict(Xc_test)
prob1 = classifier1.predict_proba(Xc_test)
accuracy1 = []
for i in range (10):
    accuracy=cross_val_score(classifier1,Xc,Yc,scoring='accuracy',cv=10)
    accuracy1.append(accuracy.mean())
print(np.array(accuracy1).mean())

#MODELLO PER POLPASTRELLO
#classifier2 = GaussianNB() #Naive Bayes
classifier2 = KNeighborsClassifier(n_neighbors=1)       #kNN
#classifier2 = DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_leaf=5) #J48
classifier2.fit(Xp, Yp)
scores2 = classifier2.predict(Xp_test)
prob2 = classifier2.predict_proba(Xp_test)
accuracy2 = []
for i in range(10):
    accuracy=cross_val_score(classifier2,Xp,Yp,scoring='accuracy',cv=10)
    accuracy2.append(accuracy.mean())
print(np.array(accuracy2).mean())

#MODELLO PER LETTO UNGUEALE
#classifier3 = GradientBoostingClassifier(loss='deviance',criterion='mse') #LMT
classifier3 = DecisionTreeClassifier(criterion='entropy') #J48
#classifier3 = KNeighborsClassifier(n_neighbors=1)
classifier3.fit(Xu, Yu)
scores3 = classifier3.predict(Xu_test)
prob3 = classifier3.predict_proba(Xu_test)
accuracy3 = []
for i in range (10):
    accuracy = cross_val_score(classifier3, Xu, Yu, scoring='accuracy', cv=10)
    accuracy3.append(accuracy.mean())
print(np.array(accuracy3).mean())
print(scores1)
print(scores2)
print(scores3)
#effettua borda count su tre tessuti
#è stata sviluppato un'ulteriore funzione borda2 che lo effettua solo su due
scores, prob = functions.borda3(scores1, scores2, scores3)
print(scores)
print(prob)

#Modello unico che utilizza tutte le features dei 3 tessuti insieme
data = pd.read_csv("test/unico_smote.csv")
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
scaler.fit(X)
test = pd.read_csv("test/unico_smote_test.csv")
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values
scaler.fit(X_test)

#model=KNeighborsClassifier(n_neighbors=1) kNN
model = GradientBoostingClassifier(loss='deviance',criterion='mse') #LMT
#model = AdaBoostClassifier(n_estimators=100, random_state=0) #AdaBoost
model.fit(X,Y)
s=model.predict(X_test)
print(s)
print(model.predict_proba(X_test))
print(accuracy_score(s,Y_test))
dfX= np.concatenate([X,X_test])
dfY= np.concatenate([Y,Y_test])
accuracy4 = []
for i in range (10):
    accuracy = cross_val_score(classifier3, dfX, dfY, scoring='accuracy', cv=10)
    accuracy4.append(accuracy.mean())
print(np.array(accuracy4).mean())