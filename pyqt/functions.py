import cv2

def extract_conjunctiva(st1):
    image1 = cv2.imread(st1) #salva immagine del percorso st1
    g, r, astar, c = 0, 0, 0, 0 #Green, Red per calcolare EI e a*, c contatore per media
    data = []
    lab_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB) #converte immagine da BGR a CIELAB
    for i in range(image1.shape[0]): #per ogni riga di pixel
        for j in range(image1.shape[1]): #per ogni colonna di pixel
            #Eliminazione pixel bianchi ossia con valori maggiori di [254,254,254]
            if image1[i, j, 0] < 254:
                if image1[i, j, 1] < 254:
                    if image1[i, j, 2] < 254:
                        g += image1[i, j, 1] #somma i valori di G
                        r += image1[i, j, 2] #somma i valori di R
                        astar += lab_image1[i, j, 1] #somma i valori di a*
                        c += 1 #incrementa contatore per la media
    #arrotonda a 2 cifre decimali, fa la media totale e aggiunge all'output
    data.append(round(astar / c - 128, 2)) #-128 perchè opencv normalizza
    data.append(round((r / c) - (g / c), 2)) #R-G = EI
    return data


def extract_nailbed(st2):
    image2 = cv2.imread(st2) #salva immagine del percorso st2
    b, astar, bstar, c = 0, 0, 0, 0 #blue,a*,b* e contatore per media
    data = []
    lab_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB) #converte immagine da BGR a CIELAB
    for i in range(image2.shape[0]): #per ogni riga di pixel
        for j in range(image2.shape[1]): #per ogni colonna di pixel
            #elimino i pixel bianchi cioè con valori RGB maggiori di [254,254,254]
            if image2[i, j, 0] < 254:
                if image2[i, j, 1] < 254:
                    if image2[i, j, 2] < 254:
                        b += image2[i, j, 0] #somma dei valori Blue
                        astar += lab_image2[i, j, 1] #somma dei valori a*
                        bstar += lab_image2[i, j, 2] #somma dei valori b*
                        c += 1 #incremento contatore per media
    #arrotonda a due cifre decimali, esegue la media e la aggiunge all'output
    data.append(round(b / c, 2))
    data.append(round(astar / c - 128, 2))
    data.append(round(bstar / c - 128, 2))
    return data

#estra le features dal video del polpastrello
def extract_fingertip(st3):
    video1 = cv2.VideoCapture(st3) #salva il video
    ret = True #boolean per ciclo
    bsum, bstarsum = 0, 0 #Blue e b* somme per tutti i frame
    data = [] #output
    i = 0 #iteratore per numero di frame
    while ret and i < 10:
        #ret=true se il frame è presente nel video
        ret, frame = video1.read() #salva 1 frame di video in modello BGR
        if ret: #se il frame è presente
            b, bstar, c = 0, 0, 0 #Blue, b* e contatore per n. pixel
            lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) #converte immagine in CIELAB
            for k in range(frame.shape[0]): #per ogni riga di pixel
                for j in range(frame.shape[1]): #per ogni colonna di pixel
                    b = b + frame[k, j, 0] #somma il valore Blue
                    bstar = bstar + lab_image[k, j, 2] #somma il valore b*
                    c = c + 1 #aumenta numero di pixel
            b = b/c #media dei valori di b per il frame corrente
            bstar = bstar/c #media dei valodi di b* per il frame corrente
            bsum = bsum + b #somma totale per ciascun frame Blue
            bstarsum = bstarsum + bstar #somma totale per ciascun frame b*
            i = i + 1 #incrementa contatore per frame
    # arrotonda a 2 cifre decimali e aggiunge la media all'output
    data.append(round(bsum/i, 2))
    data.append(round(bstarsum/i-128, 2))
    return data

#combina i risultati ottenuti da tre classificatori di base
def borda3(scores1, scores2, scores3):
    prob = []
    score =[]
    sum1, sum2 = 0, 0
    est1, est2 = 0, 0
    for i in range(len(scores1)):
        sum1 = 0
        sum2 = 0
        #score LMT 88.5 soglia 11.5 - score LMT 86 soglie per sesso
        if scores1[i] == 'SICURO':
            sum1 += 88.5
            sum2 += 11.5
        else:
            sum1 += 11.5
            sum2 += 88.5
        #score kNN 76.1 soglia 11.5 - score J48 88.1 soglie per sesso
        if scores2[i] == 'SICURO':
            sum1 += 76.1
            sum2 += 23.9
        else:
            sum1 += 23.9
            sum2 += 76.1
        #score J48 72.2 soglia 11.5 - score kNN 89 soglie per sesso
        if scores3[i] == 'SICURO':
            sum1 += 72.2
            sum2 += 27.8
        else:
            sum1 += 27.8
            sum2 += 72.2
        if sum1 > sum2:
            score.append("S")
            prob.append(round(sum1/3,2))
        else:
            score.append("R")
            prob.append(round(sum2/3,2))
    return score, prob

#combina i risultati di due classificatori invece che tre
def borda2(scores1, scores2, scores3):
    prob = []
    score =[]
    sum1, sum2 = 0, 0
    est1, est2 = 0, 0
    for i in range(len(scores1)):
        sum1 = 0
        sum2 = 0
        # score LMT 88.5 soglia 11.5 - score LMT 86 soglie per sesso
        if scores1[i] == 'SICURO':
            sum1 += 88.5
            sum2 += 11.5
        else:
            sum1 += 11.5
            sum2 += 88.5
        # score kNN 76.1 soglia 11.5 - score J48 88.1 soglie per sesso polpastrello
        # score J48 72.2 soglia 11.5 - score kNN 89 soglie per sesso letto ungueale
        #cambiare score3 con scores2 per combinare con polpastrello invece che letto ungueale
        #modificare anche i valori assegnati con 76.1 e 23.9
        if scores3[i] == 'SICURO':
            sum1 += 72.2
            sum2 += 27.8
        else:
            sum1 += 27.8
            sum2 += 72.2
        if sum1 > sum2:
            prob.append(round(sum1/2, 2))
            score.append('S')
        else:
            prob.append(round(sum2/2, 2))
            score.append('R')
    return score, prob