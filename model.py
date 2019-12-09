import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import cv2
import pickle


def predizione():
    scaler = StandardScaler()  # scaler per normalizzare i dati
    data = pd.read_csv(
        "train/unico_smote.csv")  # lettura da csv dei dati della congiuntiva delle istanze di train
    Xc = data.iloc[:, :-1].values  # assegno a Xc le features correlate della congiuntiva
    Yc = data.iloc[:, -1].values  # assegno a Yc i target
    scaler.fit(Xc)  # normalizzo le features
    # MODELLO
    classifier = GradientBoostingClassifier(loss='deviance',criterion='mse')  # LMT
    classifier.fit(Xc, Yc)  # train del modello
    pickle.dump(classifier, open('model.pkl', 'wb'))
    return ""


predizione()

def extract_conjunctiva(st1):
    image1 = cv2.imread(st1)
    g, r, astar, c = 0, 0, 0, 0
    data = []
    lab_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if image1[i, j, 0] < 254:
                if image1[i, j, 1] < 254:
                    if image1[i, j, 2] < 254:
                        g += image1[i, j, 1]
                        r += image1[i, j, 2]
                        astar += lab_image1[i, j, 1]
                        c += 1
    data.append(round(astar / c - 128, 2))
    data.append(round((r / c) - (g / c), 2))
    return data

def extract_nailbed(st2):
    image2 = cv2.imread(st2)
    b, astar, bstar, c = 0, 0, 0, 0
    data = []
    lab_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if image2[i, j, 0] < 254:
                if image2[i, j, 1] < 254:
                    if image2[i, j, 2] < 254:
                        b += image2[i, j, 0]
                        astar += lab_image2[i, j, 1]
                        bstar += lab_image2[i, j, 2]
                        c += 1

    data.append(round(b / c, 2))
    data.append(round(astar / c - 128, 2))
    data.append(round(bstar / c - 128, 2))
    return data

def extract_fingertip(st3):
    video1 = cv2.VideoCapture(st3)
    ret = True
    bsum, bstarsum = 0, 0
    data = []
    i = 0
    while ret and i < 10:
        ret, frame = video1.read()
        if ret:
            b, bstar, c = 0, 0, 0
            lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            for k in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    b = b + frame[k, j, 0]
                    bstar = bstar + lab_image[k, j, 2]
                    c = c + 1
            b = b/c
            bstar = bstar/c
            bsum = bsum + b
            bstarsum = bstarsum + bstar
            i = i + 1
    data.append(round(bsum/i, 2))
    data.append(round(bstarsum/i-128, 2))
    return data

