from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import math
import pickle

scale_factor = math.sqrt(442)
app = Flask(__name__)
model = pickle.load(open('./model.pkl', 'rb'))

# home endpoint
@app.route('/')
def home():
    return render_template('index.html')

# prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # each feature has to be scaled by some specific variable found here:
    # https://www4.stat.ncsu.edu/~boos/var.select/diabetes.read.tab.out.txt
    df = pd.DataFrame({ 'age': (final_features[0][0] -  48.5180995) / ( 13.1090278 * scale_factor),
                        'sex': (final_features[0][1] -  1.4683258) / (0.4995612 * scale_factor),
                        'bmi': (final_features[0][2] - 26.3757919) / (4.4181216 * scale_factor),
                        'bp': (final_features[0][3] - 94.6470136) / (13.8312834 * scale_factor),
                        's1': (final_features[0][4] - 189.1402715) / (34.6080517 * scale_factor),
                        's2': (final_features[0][5] - 115.4391403) / (30.4130810 * scale_factor),
                        's3': (final_features[0][6] - 49.7884615) / (12.9342022 * scale_factor),
                        's4': (final_features[0][7] - 4.0702489) / (1.2904499 * scale_factor),
                        's5': (final_features[0][8] - 4.6414109) / (0.5223906 * scale_factor),
                        's6': (final_features[0][9] - 152.1334842) / (77.093004 * scale_factor)},
                        index=[0])

    prediction = model.predict(df)
    return render_template('index.html', prediction_text='Regression value is {}'.format(prediction))

if __name__ == '__main__':
    app.run()

