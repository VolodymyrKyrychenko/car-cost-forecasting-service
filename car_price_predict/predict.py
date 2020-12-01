import pandas as pd
import numpy as np
import pickle
import data_pre_processing
import sys
import os
from train import train_model


def predict_result(x_test: pd.DataFrame, model_name='model.pkl'):
    if not os.path.isfile(model_name):
        if model_name == 'model.pkl':
            print('Модель будет тренироваться на сохраненном датасете')
            xgb_model = train_model()
            pickle.dump(xgb_model, open('model.pkl', 'wb'))
        else:
            raise Exception('В директории проекта нет файла с таким именем')

    xgb_model = pickle.load(open(model_name, 'rb'))
    x_test = data_pre_processing.preprocess(x_test)

    # предсказанный ответ
    y_test = xgb_model.predict(x_test)

    # полученые результаты
    predicted_data = np.exp(y_test.tolist())
    return predicted_data


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Данные будут предсказываться по сохраненной модели для сохраненных данных')
        x_test = pd.read_csv("test_no_target.csv")
        x_test = x_test.drop('Unnamed: 0', axis=1)
        print(predict_result(x_test))
    elif len(sys.argv) == 2:
        print('Данные будут предсказываться для сохраненных данных')
        model_file_name = str(sys.argv[1]).split('.')[0] + '.pkl'
        x_test = pd.read_csv("test_no_target.csv")
        x_test = x_test.drop('Unnamed: 0', axis=1)
        print(predict_result(x_test, model_file_name))
    elif len(sys.argv) == 3:
        model_file_name = str(sys.argv[1]).split('.')[0] + '.pkl'
        test_file_name = str(sys.argv[2]).split('.')[0] + '.csv'
        if not os.path.isfile(test_file_name):
            raise Exception('В директории проекта нет файла с таким именем')
        x_test = pd.read_csv(test_file_name)
        x_test = x_test.drop('Unnamed: 0', axis=1)
        print(predict_result(x_test, model_file_name))
