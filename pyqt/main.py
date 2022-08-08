import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
import functions

class Ui(QtWidgets.QMainWindow):
    def __init__(self):  # costruttore interfaccia
        super(Ui, self).__init__()
        uic.loadUi('interface.ui', self)  # carica l'interfaccia del file .ui creata con Designer
        self.setWindowIcon(QtGui.QIcon('icon.ico'))  # setta icon.ico come icona della finestra
        # determino le funzioni da chiamare quando viene cliccato un button
        self.pushButton.clicked.connect(self.BrowseImage1)
        self.pushButton_2.clicked.connect(self.BrowseImage2)
        self.pushButton_3.clicked.connect(self.BrowseVideo)
        self.pushButton_4.clicked.connect(self.Predict)

        self.show()  # mostra l'interfaccia

    # file browsing per cercare l'immagine della congiuntiva
    def BrowseImage1(self):
        filePath = QFileDialog.getOpenFileName(self, 'Choose conjunctiva image', "*.jpg")
        self.lineEdit.setText(filePath[0])

    # file browsing per cercare l'immagine del letto ungueale
    def BrowseImage2(self):
        filePath = QFileDialog.getOpenFileName(self, 'Choose nail bed image', "*.jpg")
        self.lineEdit_2.setText(filePath[0])

    # file browsing per cercare il video del polpastrello
    def BrowseVideo(self):
        filePath = QFileDialog.getOpenFileName(self, 'Choose fingertip video', "*.mp4")
        self.lineEdit_3.setText(filePath[0])

    # funzione che prende i percorsi delle immagini/video dalle lineEdit e le passe come parametri alla funzione di predizione
    # mostra un popup per controllare lo stato della predizione
    def Predict(self):
        # memorizza nella stringhe i path assoluti
        s1 = self.lineEdit.text()
        s2 = self.lineEdit_2.text()
        s3 = self.lineEdit_3.text()
        # crea la finestra di popup
        self.messageBox = QtWidgets.QMessageBox(self)
        self.messageBox.setStandardButtons(QtWidgets.QMessageBox.NoButton)  # senza bottoni
        self.messageBox.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.messageBox.setWindowTitle("Anemia Prediction")  # titolo finestra
        self.messageBox.setIcon(QtWidgets.QMessageBox.Information)  # icona informatica della finestra
        self.messageBox.setInformativeText(
            "This popup will close automatically when prediction is done")  # testo informativo
        s, prob = self.predizione(s1, s2, s3)  # chiamata alla funzione di predizione
        self.label_5.setText(s + " with " + str(prob) + "% probability")  # scrive il risultato della predizione

    def predizione(self, st1, st2, st3):
        QtGui.QGuiApplication.processEvents()  # necessario per non far freezare
        self.messageBox.setText("Please wait. Extracting conjunctiva features... (1/3)")  # messaggio di stato
        self.messageBox.show()  # mostra il popup
        QtGui.QGuiApplication.processEvents()
        scaler = StandardScaler()  # scaler per normalizzare i dati
        #estrazione dati di training
        data = pd.read_csv(
            "train/unico_smote.csv")  # lettura da csv dei dati della congiuntiva delle istanze di train
        Xc = data.iloc[:, :-1].values  # assegno a Xc le features correlate della congiuntiva
        Yc = data.iloc[:, -1].values  # assegno a Yc i target
        scaler.fit(Xc)  # normalizzo le features

        # MODELLO
        classifier = GradientBoostingClassifier(loss='deviance', criterion='mse')  # LMT
        classifier.fit(Xc, Yc)  # train del modello

        # estrazione delle features dalle immagini/video
        QtGui.QGuiApplication.processEvents() #per aggiornare interfaccia
        Xc_test = functions.extract_conjunctiva(st1) # estrazione congiuntiva
        QtGui.QGuiApplication.processEvents() #per aggiornare intefaccia
        self.messageBox.setText("Please wait. Extracting nail bed features... (2/3)")
        QtGui.QGuiApplication.processEvents()
        Xu_test = functions.extract_nailbed(st2) # estrazione letto ungueale
        QtGui.QGuiApplication.processEvents()
        self.messageBox.setText("Please wait. Extracting fingertip features... (3/3)")
        QtGui.QGuiApplication.processEvents()
        Xp_test = functions.extract_fingertip(st3) #estrazione polpastrello
        test = []
        #concateno le features estratte in un unico array
        test = np.concatenate((Xc_test, Xp_test, Xu_test)).reshape(1, -1)
        scaler.fit(test) #normalizza feature estratte
        score = classifier.predict(test) #predice classe
        prob = classifier.predict_proba(test) #predice probabilità - score
        #probabilità aggiustata moltiplicando per l'accuracy del modello
        prob = np.around(prob*100*0.919,decimals=2).reshape(-1,1)
        if score =='RISCHIO':
            score = "At risk"
            probability = prob[0].item() #estrae scalare da array 2D
        else:
            score = "Safe"
            probability = prob[1].item() #estrae scalare da array 2D
        self.messageBox.setHidden(True) #chiude il popup
        return score, probability

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()

if __name__ == '__main__':
    main()
