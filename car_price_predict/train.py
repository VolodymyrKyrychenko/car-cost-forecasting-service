import pandas as pd
import numpy as np
import pickle
import sys
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor

pd.options.mode.chained_assignment = None

target_column = 'price'
id_column = 'Unnamed: 0'
features_dict = {'mode_dict': {},
                 'frequent_labels_dict': {},
                 'ordered_labels_dict': {},
                 'categorical_variables_with_na': [],
                 'numerical_variables_with_na': [],
                 'categorical_variables': []}


def train_model(dataset_file_name='train.csv'):
    # датасет
    data = pd.read_csv(dataset_file_name)
    # Разделение датасета на тренировочный и тестовый сеты
    x_train, x_test, y_train, y_test = train_test_split(data, data[target_column], test_size=0.1, random_state=12345)
    print('Тестовый датасет составляет 10 % от всего набора данных')

    # 1) обрабатываем отсутствующие данные в категориальных признаках
    # список категориальных признаков, в которых отсутствуют данные
    categorical_variables_with_na = [var for var in x_train.columns
                                     if x_train[var].isnull().sum() > 0 and x_train[var].dtypes == 'O']

    features_dict['categorical_variables_with_na'] = categorical_variables_with_na
    # процент отсутствующих данных на признак
    # print('Процент отсутствующих данных на признак:', x_train[categorical_variables_with_na].isnull().mean(), sep='\n')

    # заменяем отсутствующие данные на метку "Missing"
    x_train[categorical_variables_with_na] = x_train[categorical_variables_with_na].fillna('Missing')
    x_test[categorical_variables_with_na] = x_test[categorical_variables_with_na].fillna('Missing')

    # 2) обрабатываем отсутствующие данные в числовых признаках
    # список числовых признаков, в которых отсутствуют данные
    numerical_variables_with_na = [var for var in x_train.columns
                                   if x_train[var].isnull().sum() > 0 and x_train[var].dtypes != 'O']

    features_dict['numerical_variables_with_na'] = numerical_variables_with_na
    # процент отсутствующих данных на признак
    # print('Процент отсутствующих данных на признак:', x_train[numerical_variables_with_na].isnull().mean(), sep='\n')

    for var in numerical_variables_with_na:
        # вычисляем моду
        mode = x_train[var].mode()[0]
        features_dict['mode_dict'][var] = mode

        # добавляем новый столбец, указывающий, отсутствовало ли значение признака или нет
        x_train[var + '_na'] = np.where(x_train[var].isnull(), 1, 0)
        x_test[var + '_na'] = np.where(x_test[var].isnull(), 1, 0)

        # заменяем отсутствующие значения на моду
        x_train[var] = x_train[var].fillna(mode)
        x_test[var] = x_test[var].fillna(mode)

    # 3) обрабатываем непрерывные числовые признаки
    # список всех числовых признаков
    numerical_variables = [var for var in x_train.columns if x_train[var].dtype != 'O']

    # список всех непрерывных числовых признаков (кроме registration_year и id)
    cont_vars = [var for var in numerical_variables
                 if len(x_train[var].unique()) >= 20 and var not in ['registration_year'] + [id_column]]

    # # распределение признаков
    # for var in cont_vars:
    #     x_train[var].hist(bins=30)
    #     plt.ylabel('Number of cars')
    #     plt.xlabel(var)
    #     plt.title(var)
    #     plt.show()

    positive_cont_vars = [i for i in cont_vars if all(x_train[i] > 0)]

    # распределения не являются нормальными
    # логарифмируем признаки
    for var in cont_vars:
        if any(x_train[var] <= 0):
            continue
        x_train[var] = np.log(x_train[var])
        if var != target_column:
            x_test[var] = np.log(x_test[var])

    # 4) обрабатываем редкие значения категориальных признаков
    # список всех категориальных признаков
    categorical_variables = [var for var in x_train.columns if x_train[var].dtype == 'O']
    features_dict['categorical_variables'] = categorical_variables

    # функция для нахождения значений, которые встречаются для более чем percent % автомобилей
    def find_frequent_labels(data, var, percent):
        data = data.copy()
        tmp = data.groupby(var)[target_column].count() / len(data)
        return tmp[tmp > percent].index

    for var in categorical_variables:
        # значения (метки), которые встречаются относительно часто
        frequent_labels = find_frequent_labels(x_train, var, 0.01)
        features_dict['frequent_labels_dict'][var] = frequent_labels.tolist()

        # замена редких меток на "Rare"
        x_train[var] = np.where(x_train[var].isin(frequent_labels), x_train[var], 'Rare')
        x_test[var] = np.where(x_test[var].isin(frequent_labels), x_test[var], 'Rare')

    # 5) заменяем значения категориальных признаков на числовые метки
    for var in categorical_variables:
        # группируем категориальные значения и упорядочиваем их по цене (ее медиане) в порядке возрастания
        ordered_labels = x_train.groupby([var])[target_column].mean().sort_values().index
        features_dict['ordered_labels_dict'][var] = ordered_labels.tolist()

        # создаем словарь типа "категориальная метка - числовое значение"
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
        # print('Словарь меток для признака \"' + var + '\":', ordinal_label, sep='\n')

        # согласно словарю заменяем категориальные метки числовыми значениями
        x_train[var] = x_train[var].map(ordinal_label)
        x_test[var] = x_test[var].map(ordinal_label)

    # 6) масштабирование признаков
    # список признаков, которые будут масшабироваться
    scale_variables = [var for var in x_train.columns if var not in [id_column, target_column]]

    # масштабирование
    scaler = MinMaxScaler()
    scaler.fit(x_train[scale_variables])
    x_train[scale_variables] = scaler.transform(x_train[scale_variables])
    x_test[scale_variables] = scaler.transform(x_test[scale_variables])

    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    # 7) выбор признаков
    # целевой признак
    y_train = x_train[target_column]
    y_test = x_test[target_column]

    x_train.drop([id_column, target_column], axis=1, inplace=True)
    x_test.drop([id_column, target_column], axis=1, inplace=True)
    x_train = x_train.drop('zipcode', axis=1)
    x_test = x_test.drop('zipcode', axis=1)

    # 8) модель
    xgb_model = XGBRegressor().fit(x_train, y_train)

    y_test_predicted = xgb_model.predict(x_test)
    result_price = [np.exp(i) for i in y_test_predicted.tolist()]
    res = mean_absolute_percentage_error(y_test.to_list(), result_price)

    print('Ошибка MAPE =', res)
    with open('features.json', 'w') as file:
        file.write(json.dumps(features_dict))

    return xgb_model


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Модель будет тренироваться на сохраненном датасете')
        xgb_model = train_model()
        pickle.dump(xgb_model, open('model.pkl', 'wb'))
    else:
        train_file_name = str(sys.argv[1]).split('.')[0]
        if not os.path.isfile(train_file_name + '.csv'):
            raise Exception('В директории проекта нет файла с таким именем')
        model_file_name = 'model_' + train_file_name + '.pkl'
        new_model = train_model(train_file_name + '.csv')
        pickle.dump(new_model, open(model_file_name, 'wb'))
    print('Тренировка окончена')
