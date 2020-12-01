from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import math
import pandas as pd
import predict
import os

app = Flask(__name__)

ordered_features = ['engine_capacity', 'type', 'registration_year', 'gearbox', 'power', 'model', 'mileage', 'fuel',
                    'brand', 'damage', 'zipcode', 'insurance_price']
string_features = ['type', 'gearbox', 'model', 'fuel', 'brand']
float_features = ['engine_capacity', 'registration_year', 'power', 'mileage', 'damage', 'zipcode', 'insurance_price']


@app.route('/')
def index():
    input_data = {i: '' for i in ordered_features}
    return render_template('index.html', input_data=input_data)


@app.route('/predict', methods=['POST'])
def predict_from_json():
    content = request.json
    data_dict = get_data_dict(content)
    x_test = pd.DataFrame(data_dict)
    x_test = x_test[ordered_features]
    price = predict.predict_result(x_test)[0]
    return jsonify(str(price))


@app.route('/result', methods=['POST'])
def predict_from_form():
    input_data = request.form
    data_dict = get_data_dict(request.form)
    x_test = pd.DataFrame(data_dict)
    x_test = x_test[ordered_features]
    price = predict.predict_result(x_test)[0]
    return render_template('index.html', prediction_text='Predicted price = {:.2f}'.format(price), input_data=input_data)


def get_data_dict(content):
    data_dict = {i: [content[i]] for i in string_features}
    data_dict.update(
        {i: [float(str(content[i]).replace(',', '.'))] if str(content[i]) != "" else math.nan for i in float_features})
    return data_dict


@app.route('/version')
def get_version():
    return 'Model: XGBRegressor; Package: v1'


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port)
