import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
import model as functions
import pickle
import os



app = Flask(__name__)
classifier1 = pickle.load(open('model1.pkl', 'rb'))
classifier2 = pickle.load(open('model2.pkl', 'rb'))
classifier3 = pickle.load(open('model3.pkl', 'rb'))
UPLOAD_FOLDER = "static/upload"
ALLOWED_EXTENSIONS = {'jpg', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET','POST'])
def predict():
    st1 = UPLOAD_FOLDER
    st2 = UPLOAD_FOLDER
    st3 = UPLOAD_FOLDER
    if request.method == 'POST':
        images = request.files.to_dict()
        for image in images:
            if image and allowed_file(images[image].filename):
                file_name = secure_filename(images[image].filename)
                images[image].save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                if image == 'conjunctiva':
                    st1 = st1 + '/' + file_name
                if image == 'nailbed':
                    st2 = st2 + '/' + file_name
                if image == 'fingertip':
                    st3 = st3 + '/' + file_name
        scaler = StandardScaler()
        Xc_test = np.array(functions.extract_conjunctiva(st1)).reshape(1, -1)
        print("Extracted conjuntiva")
        Xu_test = np.array(functions.extract_nailbed(st2)).reshape(1, -1)
        print("Extracted nailbed")
        Xp_test = np.array(functions.extract_fingertip(st3)).reshape(1, -1)
        print("Extracted fingertip")
        scaler.fit(Xc_test)
        scaler.fit(Xp_test)
        scaler.fit(Xu_test)
        scores1 = classifier1.predict(Xc_test)
        prob1 = classifier1.predict_proba(Xc_test)
        scores2 = classifier2.predict(Xp_test)
        prob2 = classifier2.predict_proba(Xp_test)
        scores3 = classifier3.predict(Xu_test)
        prob3 = classifier3.predict_proba(Xu_test)
        score = functions.bordaCount(scores1, scores2, scores3, prob1, prob2, prob3)
        return render_template('index.html', prediction_text='Patient is {}'.format(score))


if __name__ == "__main__":
    app.run(debug=True)