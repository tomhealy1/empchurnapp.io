#python app.py
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('rf_model2.pkl', 'rb'))
#cols = ['satisfaction_level', 'Work_accident',
# 'promotion_last_5years',
 #'function_RandD','salary_high']

#flask home route
@app.route('/')
def home():
    return render_template('home.html')
#flask route for predictions using post method
@app.route('/predict',methods=['POST'])
def predict():
    #takes a float from our input form and pass it to numpy array pass it to our model.
    int_feature = [x for x in request.form.values()]
    final_feature = [np.array(int_feature)]
    prediction = model1.predict(final_feature)
    # round our predict to two places
    output = round(prediction[0], 2)
    if output ==0:
        output = "This person will most likely stay"
    else:
        output="This person will most likely leave" 
    


    # Renders our home.html and our prediction text
    return render_template('home.html', prediction_text=" Our prediction is {} ".format(output))


if __name__ == "__main__":
    app.run(debug=True)