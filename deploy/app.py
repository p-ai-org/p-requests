#!/usr/bin/env python3

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from pathlib import Path
import os
import EDPipeline as edp

app = Flask(__name__)

modelCD_path = os.path.join(Path(os.getcwd()).parents[1], '311_Requests_Model', 'modelCD.pkl')
modelRT_path = os.path.join(Path(os.getcwd()).parents[1], '311_Requests_Model', 'modelRT.pkl')
modelRF_path = os.path.join(Path(os.getcwd()).parents[1], '311_Requests_Model', 'randomForest.pkl')
#encoder_path = os.path.join(Path(os.getcwd()).parents[1], '311_Requests_Model', 'encoder.pkl')

modelCD = pickle.load(open(modelCD_path, 'rb'))
modelRT = pickle.load(open(modelRT_path, 'rb'))
modelRF = pickle.load(open(modelRF_path, 'rb'))
#one_hot = pickle.load(open(encoder_path, 'rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    features = [x for x in request.form.values()]
    column_names = ['AssignTo', 'RequestType', 'RequestSource', 'Month', 'Anonymous', 'CreatedByUserOrganization','Latitude','Longitude']
    dictionary = dict(zip(column_names,features))
    df_request = pd.DataFrame(columns= column_names)
    return render_template('index.html', pred='More than 11 days')
    for key in dictionary: 
            df_request.at[0, key] = dictionary[key] 
    X,ignore, dfn = edp.preprocess(df_request,encode=False)
    if(int(modelRF.predict(X)) == 0):
        return render_template('index.html', pred='More than 11 days')
    # averaged prediction --- pulled out from lacer script
    prediction = modelCD.predict(X) + modelRT.predict(X) / 2
    return render_template('index.html', pred=str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
