import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
import model as functions
import pickle
import os
from rq import Queue
from worker import conn

app = Flask(__name__)
classifier1 = pickle.load(open('model1.pkl', 'rb'))
classifier2 = pickle.load(open('model2.pkl', 'rb'))
classifier3 = pickle.load(open('model3.pkl', 'rb'))
UPLOAD_FOLDER = "static/upload"
ALLOWED_EXTENSIONS = {'jpg', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
q = Queue(connection=conn)

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
        score = q.enqueue(functions.predict, st1, st2, st3)
        return render_template('index.html', prediction_text='Patient is {}'.format(score))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)