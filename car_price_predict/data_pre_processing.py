import pandas as pd
import numpy as np
import pickle
import json


def preprocess(x_test: pd.DataFrame):
    features = {'mode_dict': {},
                'frequent_labels_dict': {},
                'ordered_labels_dict': {},
                'categorical_variables_with_na': [],
                'numerical_variables_with_na': [],
                'categorical_variables': []}

    with open('features.json', 'r') as file:
        features = json.loads(file.readline())

    # 1) обрабатываем отсутствующие данные в категориальных признаках
    # список категориальных признаков, в которых отсутствуют данные
    categorical_variables_with_na = features['categorical_variables_with_na']

    # заменяем отсутствующие данные на метку "Missing"
    x_test[categorical_variables_with_na] = x_test[categorical_variables_with_na].fillna('Missing')

    # 2) обрабатываем отсутствующие данные в числовых признаках
    # список числовых признаков, в которых отсутствуют данные
    numerical_variables_with_na = features['numerical_variables_with_na']

    for var in numerical_variables_with_na:
        # вычисляем моду
        mode = features['mode_dict'][var]

        # добавляем новый столбец, указывающий, отсутствовало ли значение признака или нет
        x_test[var + '_na'] = np.where(x_test[var].isnull(), 1, 0)

        # заменяем отсутствующие значения на моду
        x_test[var] = x_test[var].fillna(mode)

    # 3) обрабатываем непрерывные числовые признаки
    # список положительных непрерывных числовых признаков (кроме registration_year и id)
    positive_cont_vars = ['zipcode', 'insurance_price']  # в тестовом сете без 'price'

    # логарифмируем признаки
    for var in positive_cont_vars:
        x_test[var] = np.log(x_test[var])

    # 4) обрабатываем редкие значения категориальных признаков
    # список всех категориальных признаков
    categorical_variables = features['categorical_variables']

    for var in categorical_variables:
        # замена редких меток на "Rare"
        x_test[var] = np.where(x_test[var].isin(features['frequent_labels_dict'][var]), x_test[var], 'Rare')

    # 5) заменяем значения категориальных признаков на числовые метки
    for var in categorical_variables:
        # группируем категориальные значения и упорядочиваем их по цене (ее медиане) в порядке возрастания
        ordered_labels = features['ordered_labels_dict'][var]

        # создаем словарь типа "категориальная метка - числовое значение"
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

        # согласно словарю заменяем категориальные метки числовыми значениями
        x_test[var] = x_test[var].map(ordinal_label)

    # 6) масштабирование признаков
    # список признаков, которые будут масшабироваться
    scale_variables = [var for var in x_test.columns if var not in ['Unnamed: 0', 'price']]

    # масштабирование
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    x_test[scale_variables] = scaler.transform(x_test[scale_variables])

    # 7) выбор признаков
    # удаляем ненужные признаки из датасетов
    # x_test.drop(['Unnamed: 0'], axis=1, inplace=True)
    x_test = x_test.drop('zipcode', axis=1)

    return x_test
