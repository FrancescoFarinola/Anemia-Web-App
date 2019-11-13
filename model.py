import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import cv2
import pickle


def predizione():
    scaler = StandardScaler()  # scaler per normalizzare i dati

    datacongiuntiva = pd.read_csv(
        "train/congiuntiva.csv")  # lettura da csv dei dati della congiuntiva delle istanze di train
    Xc = datacongiuntiva.iloc[:, :-1].values  # assegno a Xc le features correlate della congiuntiva
    Yc = datacongiuntiva.iloc[:, -1].values  # assegno a Yc i target
    scaler.fit(Xc)  # normalizzo le features
    datapolpastrello = pd.read_csv(
        "train/polpastrello.csv")  # lettura da csv dei dati del polpastrello delle istanze di train
    Xp = datapolpastrello.iloc[:, :-1].values  # assegno a Xp le features correlate del polpastrello
    Yp = datapolpastrello.iloc[:, -1].values  # assegno a Yp i target
    scaler.fit(Xp)  # normalizzo le features
    dataunghia = pd.read_csv(
        "train/lettoungueale.csv")  # lettura da csv dei dati del letto ungueale delle istanze di train
    Xu = dataunghia.iloc[:, :-1].values  # assegno a Xu le features correlate del letto ungueale
    Yu = dataunghia.iloc[:, -1].values  # assegno a Yu i target
    scaler.fit(Xu)  # normalizzo le features

    # MODELLO PER CONGIUNTIVA
    classifier1 = GradientBoostingClassifier(loss='deviance',criterion='mse')  # corrispondente di Logistic Model Tree in Weka
    classifier1.fit(Xc, Yc)  # train del modello

    # MODELLO PER POLPASTRELLO
    classifier2 = DecisionTreeClassifier(criterion='entropy', max_depth=1,min_samples_leaf=5)  # corrispondente di J48 - usa l'algoritmo CART con massimizzazione di information gain
    classifier2.fit(Xp, Yp)  # train del modello per polpastrello

    # MODELLO PER LETTO UNGUEALE
    classifier3 = DecisionTreeClassifier(criterion='entropy')  # corrispondente di J48 - utilizza algoritmo CART con massimizzazione di Information Gain
    classifier3.fit(Xu, Yu)  # train del modello per letto ungueale

    pickle.dump(classifier1, open('model1.pkl', 'wb'))
    pickle.dump(classifier2, open('model2.pkl', 'wb'))
    pickle.dump(classifier3, open('model3.pkl', 'wb'))
    return ""


predizione()

def normalizzazioneLstar(num):
    return round(((100 - 0) / (255 - 0)) * (num - 255) + 100, 2)
def normalizzazioneAstar(num):
    return round(((128 + 128) / (255)) * (num - 255) + 128, 2)
def normalizzazioneBstar(num):
    return round(((128 + 128) / (255)) * (num - 255) + 128, 2)
def extract_conjunctiva(st1):
    image1 = cv2.imread(st1)
    b, g, r, c = 0, 0, 0, 0
    Lstar, astar, bstar = 0, 0, 0
    data = []
    lab_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if image1[i, j, 0] < 254:
                if image1[i, j, 1] < 254:
                    if image1[i, j, 2] < 254:
                        b += image1[i, j, 0]
                        g += image1[i, j, 1]
                        r += image1[i, j, 2]
                        Lstar += lab_image1[i, j, 0]
                        astar += lab_image1[i, j, 1]
                        bstar += lab_image1[i, j, 2]
                        c += 1

    data.append(round(g / c, 2))
    data.append(round(b / c, 2))
    data.append(round(Lstar / c - 100, 2))
    data.append(round(astar / c - 128, 2))
    data.append(round(bstar / c - 128, 2))
    data.append(round((r / c) - (g / c), 2))
    return data


def extract_nailbed(st2):
    image2 = cv2.imread(st2)
    g, r, c = 0, 0, 0
    astar, bstar = 0, 0
    data = []
    lab_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if image2[i, j, 0] < 254:
                if image2[i, j, 1] < 254:
                    if image2[i, j, 2] < 254:
                        g += image2[i, j, 1]
                        r += image2[i, j, 2]
                        astar += lab_image2[i, j, 1]
                        bstar += lab_image2[i, j, 2]
                        c += 1

    data.append(round(r / c, 2))
    data.append(round(astar / c - 128, 2))
    data.append(round(bstar / c - 128, 2))
    data.append(round((r / c) - (g / c), 2))
    return data


def extract_fingertip(st3):
    video1 = cv2.VideoCapture(st3)
    ret = True
    Lsum, bsum = 0, 0
    data = []
    i = 0
    while ret and i < 165:
        ret, frame = video1.read()
        if ret:
            lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            Lstar, astar, bstar = cv2.split(lab_image)
            Lsum = Lsum + Lstar.mean()
            bsum = bsum + bstar.mean()
            i = i + 1
    data.append(normalizzazioneLstar(Lsum / i))
    data.append(normalizzazioneBstar(bsum / i))
    return data


def bordaCount(scores1, scores2, scores3, prob1, prob2, prob3):
    sum1, sum2 = 0, 0
    est1, est2 = 0, 0
    for i in range(len(scores1)):
        sum1 = 0
        sum2 = 0
        if scores1[i] == 'SICURO':
            sum1 += 2
            sum2 += 1
        else:
            sum1 += 1
            sum2 += 2
        if scores2[i] == 'SICURO':
            sum1 += 2
            sum2 += 1
        else:
            sum1 += 1
            sum2 += 2
        if scores3[i] == 'SICURO':
            sum1 += 2
            sum2 += 1
        else:
            sum1 += 1
            sum2 += 2
        if sum1 > sum2:
            score = 'safe'
        else:
            score = 'at risk'

        est1 = 0
        est2 = 0
        if prob3[i, 0] == 0:
            prob3[i, 0] = 0.26
            prob3[i, 1] = 0.74
        else:
            prob3[i, 0] = 0.74
            prob3[i, 1] = 0.26

        est1 = (prob1[i, 0] + prob2[i, 0] + prob3[i, 0]) / 3 * 100
        est2 = (prob1[i, 1] + prob2[i, 1] + prob3[i, 1]) / 3 * 100
        if est1 > est2:
            prob = round(est1, 2)
        else:
            prob = round(est2, 2)

        score = score + " with " + str(prob) + "% probability"
        return score

def predict(st1,st2,st3):
    scaler = StandardScaler()
    Xc_test = np.array(extract_conjunctiva(st1)).reshape(1, -1)
    Xu_test = np.array(extract_nailbed(st2)).reshape(1, -1)
    Xp_test = np.array(extract_fingertip(st3)).reshape(1, -1)
    scaler.fit(Xc_test)
    scaler.fit(Xp_test)
    scaler.fit(Xu_test)
    classifier1 = pickle.load(open('model1.pkl', 'rb'))
    classifier2 = pickle.load(open('model2.pkl', 'rb'))
    classifier3 = pickle.load(open('model3.pkl', 'rb'))
    scores1 = classifier1.predict(Xc_test)
    prob1 = classifier1.predict_proba(Xc_test)
    scores2 = classifier2.predict(Xp_test)
    prob2 = classifier2.predict_proba(Xp_test)
    scores3 = classifier3.predict(Xu_test)
    prob3 = classifier3.predict_proba(Xu_test)
    score = bordaCount(scores1, scores2, scores3, prob1, prob2, prob3)
    return score