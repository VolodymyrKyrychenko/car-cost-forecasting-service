import pandas as pd
import numpy as np
import pickle

mode_dict = {'engine_capacity': 2.0,
             'damage': 0.0,
             'insurance_price': 70.0}

frequent_labels = {
    'type': ['Missing', 'bus', 'convertible', 'coupé', 'limousine', 'small car', 'station wagon'],
    'gearbox': ['Missing', 'auto', 'manual'],
    'model': ['1er', '2_reihe', '3er', '5er', 'Missing', 'a3', 'a4', 'a6', 'a_klasse', 'andere', 'astra', 'c_klasse',
              'corsa', 'e_klasse', 'fiesta', 'focus', 'fortwo', 'golf', 'passat', 'polo', 'transporter', 'twingo',
              'vectra'],
    'fuel': ['Missing', 'diesel', 'gasoline', 'liquefied petroleum gas'],
    'brand': ['audi', 'bmw', 'citroen', 'fiat', 'ford', 'hyundai', 'mazda', 'mercedes_benz', 'mini', 'nissan', 'opel',
              'peugeot', 'renault', 'seat', 'skoda', 'smart', 'toyota', 'volkswagen']}

ordered_labels_dict = {
    'type': ['small car', 'Rare', 'Missing', 'limousine', 'station wagon', 'bus', 'coupé', 'convertible'],
    'gearbox': ['Missing', 'manual', 'auto'],
    'model': ['twingo', 'vectra', 'corsa', 'fiesta', 'polo', 'focus', 'astra', '2_reihe', 'a_klasse', 'Missing',
              'fortwo', 'golf', 'passat', 'andere', 'Rare', '3er', 'c_klasse', 'a4', 'a3', 'e_klasse', '5er', 'a6',
              'transporter', '1er'],
    'fuel': ['Missing', 'gasoline', 'liquefied petroleum gas', 'Rare', 'diesel'],
    'brand': ['renault', 'fiat', 'opel', 'peugeot', 'ford', 'citroen', 'smart', 'mazda', 'nissan', 'seat',
              'volkswagen', 'toyota', 'Rare', 'hyundai', 'skoda', 'mercedes_benz', 'bmw', 'audi', 'mini']}


def preprocess(x_test: pd.DataFrame):
    # 1) обрабатываем отсутствующие данные в категориальных признаках
    # список категориальных признаков, в которых отсутствуют данные
    categorical_variables_with_na = ['type', 'gearbox', 'model', 'fuel']

    # заменяем отсутствующие данные на метку "Missing"
    x_test[categorical_variables_with_na] = x_test[categorical_variables_with_na].fillna('Missing')

    # 2) обрабатываем отсутствующие данные в числовых признаках
    # список числовых признаков, в которых отсутствуют данные
    numerical_variables_with_na = ['engine_capacity', 'damage', 'insurance_price']

    for var in numerical_variables_with_na:
        # вычисляем моду
        mode = mode_dict[var]

        # добавляем новый столбец, указывающий, отсутствовало ли значение признака или нет
        x_test[var + '_na'] = np.where(x_test[var].isnull(), 1, 0)

        # заменяем отсутствующие значения на моду
        x_test[var] = x_test[var].fillna(mode)

    # обработка временных переменных
    # единственная временная переменная - registration_year
    # во входных данных встречались неполные значения года (например, 16)
    # попытка привести года к единой форме (к диапазону 1900-2020) привела к ухудшению точности модели

    # 3) обрабатываем непрерывные числовые признаки
    # список всех >>positive<< непрерывных числовых признаков (кроме registration_year и id)
    positive_cont_vars = ['zipcode', 'insurance_price']  # NO 'price'

    # логарифмируем признаки
    for var in positive_cont_vars:
        x_test[var] = np.log(x_test[var])

    # 4) обрабатываем редкие значения категориальных признаков
    # список всех категориальных признаков
    categorical_variables = ['type', 'gearbox', 'model', 'fuel', 'brand']

    for var in categorical_variables:
        # замена редких меток на "Rare"
        x_test[var] = np.where(x_test[var].isin(frequent_labels[var]), x_test[var], 'Rare')

    # 5) заменяем значения категориальных признаков на числовые метки
    for var in categorical_variables:
        # группируем категориальные значения и упорядочиваем их по цене (ее медиане) в порядке возрастания
        ordered_labels = ordered_labels_dict[var]

        # создаем словарь типа "категориальная метка - числовое значение"
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

        # согдлсно словарю заменяем категориальные метки числовыми значениями
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

    # при попытке выделить наиболее прогнозируемые признаки с помощью регрессии Лассо отсекались 2 признака:
    # engine_capacity и power. Но при их удалении из датасетов падала точность модели

    return x_test
