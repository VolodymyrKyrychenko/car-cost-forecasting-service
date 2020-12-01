import pandas as pd
import numpy as np
import pickle
import data_pre_processing


def predict_result(x_test: pd.DataFrame):
    xgb_model = pickle.load(open('model.pkl', 'rb'))
    x_test = data_pre_processing.preprocess(x_test)

    # предсказанный ответ
    y_test = xgb_model.predict(x_test)

    # полученые результаты
    predicted_data = np.exp(y_test.tolist())
    return predicted_data


if __name__ == '__main__':
    x_test = pd.read_csv("test_no_target.csv")
    x_test = x_test.drop('Unnamed: 0', axis=1)
    print(predict_result(x_test))
