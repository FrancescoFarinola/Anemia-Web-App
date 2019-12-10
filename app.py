import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename, redirect
from sklearn.preprocessing import StandardScaler
import model as functions
import pickle
import os
from rq import Queue
from worker import conn

app = Flask(__name__)
classifier = pickle.load(open('model.pkl', 'rb')) #deserializza modello
UPLOAD_FOLDER = "static/upload" #cartella dove vengono salvati gli upload
ALLOWED_EXTENSIONS = {'jpg', 'mp4'} #estensioni dei file ammesse
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["MAX_IMAGE_FILESIZE"] = 1024 * 1024 #max filesize 1MB
q = Queue(connection=conn)

@app.route('/')
def home():
    return render_template('index.html') #apertura dell'home

#funzione per verificare se il file è dell'estensione ammessa
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#funzione per verificare se il file rispetta la dimensione fissata
def allowed_image_filesize(filesize):
    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False

#funzione chiamata quando viene premuto il tasto 'Predict'
@app.route('/predict', methods=['GET','POST'])
def predict():
    #stringhe per il percorso dei file necessari
    st1 = UPLOAD_FOLDER
    st2 = UPLOAD_FOLDER
    st3 = UPLOAD_FOLDER
    if request.method == 'POST':
        images = request.files.to_dict() #salva in un dict i file del form HTML
        for image in images: #per ogni immagine
            #se la filesize è stata presa dal cookie richiesto da JavaScript
            if "filesize" in request.cookies:
                if not allowed_image_filesize(request.cookies["filesize"]): #se supera le dim
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url) #refresh pagina
            if image and allowed_file(images[image].filename): #se estensione ammessa
                file_name = secure_filename(images[image].filename) #controlla nome file
                images[image].save(os.path.join(app.config['UPLOAD_FOLDER'], file_name)) #salva file in upload
                #scrittura dei percorsi dei file nelle rispettive stringhe
                if image == 'conjunctiva':
                    st1 = st1 + '/' + file_name
                if image == 'nailbed':
                    st2 = st2 + '/' + file_name
                if image == 'fingertip':
                    st3 = st3 + '/' + file_name
        scaler = StandardScaler() #definizione scaler
        Xc_test = functions.extract_conjunctiva(st1) #estrazione features congiuntiva
        Xu_test = functions.extract_nailbed(st2) #estrazione features letto ungueale
        Xp_test = functions.extract_fingertip(st3) #estrazione features polpastrello
        test = []
        #concatena le features estratte in un unico array
        test = np.concatenate((Xc_test, Xp_test, Xu_test)).reshape(1, -1)
        scaler.fit(test) #normalizza features
        score = classifier.predict(test) #calcola predizione
        prob = classifier.predict_proba(test) #calcola probabilità
        #aggiusta probabilità
        prob = np.around(prob*100*0.919,decimals=2).reshape(-1,1) #0.919=accuracy modello
        if score =='RISCHIO':
            score = "at risk"
            probability = prob[0].item() #salva come scalare
        else:
            score = "Safe"
            probability = prob[1].item() #salva come scalare
        score = score + " with " + str(probability) + "% probability"
        #restituisce stringa in html da sostituire
        return render_template('index.html', prediction_text='Patient is {}'.format(score))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)