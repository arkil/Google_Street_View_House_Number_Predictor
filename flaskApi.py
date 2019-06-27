# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:10:26 2018

@author: arkil
"""

import flask
from flask import Flask, render_template, request
from sklearn.externals import joblib
from scipy import misc
import numpy as np
app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('/src/index.js')   
@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':
		file = request.files['image']
		if not file: return render_template('/src/index.js', label="No file")
		
		img = misc.imread(file)
		img = img[:,:,:3]
		img = img.reshape(1, -1)

		# make prediction on new image
		prediction = model.predict(img)
	
		# squeeze value from 1D array and convert to string for clean return
		label = str(np.squeeze(prediction))

		# switch for case where label=10 and number=0
		if label=='10': label='0'

		return render_template('/src/index.js', label=label)
   
if __name__ == '__main__':
    model = joblib.load('model.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)