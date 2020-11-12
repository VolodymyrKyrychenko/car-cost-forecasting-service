import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn


def train_model():
    # датасет
    data = pd.read_csv("train.csv")

    # Разделение датасета на тренировочный и тестовый сеты
    x_train, x_test, y_train, y_test = train_test_split(data, data['price'], test_size=0.1, random_state=12345)

    print(x_train.shape, x_test.shape)

    # 1) обрабатываем отсутствующие данные в категориальных признаках
    # список категориальных признаков, в которых отсутствуют данные
    categorical_variables_with_na = [var for var in x_train.columns
                                     if x_train[var].isnull().sum() > 0 and x_train[var].dtypes == 'O']

    # процент отсутствующих данных на признак
    print('categorical_variables_with_na', categorical_variables_with_na)
    print('Процент отсутствующих данных на признак:', x_train[categorical_variables_with_na].isnull().mean(), sep='\n')

    # заменяем отсутствующие данные на метку "Missing"
    x_train[categorical_variables_with_na] = x_train[categorical_variables_with_na].fillna('Missing')
    x_test[categorical_variables_with_na] = x_test[categorical_variables_with_na].fillna('Missing')

    # 2) обрабатываем отсутствующие данные в числовых признаках
    # список числовых признаков, в которых отсутствуют данные
    numerical_variables_with_na = [var for var in x_train.columns
                                   if x_train[var].isnull().sum() > 0 and x_train[var].dtypes != 'O']

    # процент отсутствующих данных на признак
    print('numerical_variables_with_na', numerical_variables_with_na)
    print('Процент отсутствующих данных на признак:', x_train[numerical_variables_with_na].isnull().mean(), sep='\n')

    print('variable - mode')
    for var in numerical_variables_with_na:
        # вычисляем моду
        mode = x_train[var].mode()[0]
        print(var, mode)

        # добавляем новый столбец, указывающий, отсутствовало ли значение признака или нет
        x_train[var + '_na'] = np.where(x_train[var].isnull(), 1, 0)
        x_test[var + '_na'] = np.where(x_test[var].isnull(), 1, 0)

        # заменяем отсутствующие значения на моду
        x_train[var] = x_train[var].fillna(mode)
        x_test[var] = x_test[var].fillna(mode)

    # обработка временных переменных
    # единственная временная переменная - registration_year
    # во входных данных встречались неполные значения года (например, 16)
    # попытка привести года к единой форме (к диапазону 1900-2020) привела к ухудшению точности модели

    # 3) обрабатываем непрерывные числовые признаки
    # список всех числовых признаков
    numerical_variables = [var for var in x_train.columns if x_train[var].dtype != 'O']
    print('numerical_variables', numerical_variables)

    # список всех непрерывных числовых признаков (кроме registration_year и id)
    cont_vars = [var for var in numerical_variables
                 if len(x_train[var].unique()) >= 20 and var not in ['registration_year'] + ['Unnamed: 0']]
    print('Непрерывные числовые признаки: ', cont_vars)

    # # распределение признаков
    # for var in cont_vars:
    #     x_train[var].hist(bins=30)
    #     plt.ylabel('Number of cars')
    #     plt.xlabel(var)
    #     plt.title(var)
    #     plt.show()

    positive_cont_vars = [i for i in cont_vars if all(x_train[i] > 0)]
    print('positive_cont_vars', positive_cont_vars)

    # распределения не являются нормальными
    # логарифмируем признаки
    for var in cont_vars:
        if any(x_train[var] <= 0):
            continue
        x_train[var] = np.log(x_train[var])
        if var != 'price':
            x_test[var] = np.log(x_test[var])

    # 4) обрабатываем редкие значения категориальных признаков
    # список всех категориальных признаков
    categorical_variables = [var for var in x_train.columns if x_train[var].dtype == 'O']
    print('categorical_variables', categorical_variables)

    # функция для нахождения значений, которые встречаются для более чем percent % автомобилей
    def find_frequent_labels(data, var, percent):
        data = data.copy()
        tmp = data.groupby(var)['price'].count() / len(data)
        return tmp[tmp > percent].index

    print('frequent_labels')
    for var in categorical_variables:
        # значения (метки), которые встречаются относительно часто
        frequent_labels = find_frequent_labels(x_train, var, 0.01)
        print(var, 'frequent_labels', frequent_labels)

        # замена редких меток на "Rare"
        x_train[var] = np.where(x_train[var].isin(frequent_labels), x_train[var], 'Rare')
        x_test[var] = np.where(x_test[var].isin(frequent_labels), x_test[var], 'Rare')

    print('ordered labels')
    # 5) заменяем значения категориальных признаков на числовые метки
    for var in categorical_variables:
        # группируем категориальные значения и упорядочиваем их по цене (ее медиане) в порядке возрастания
        ordered_labels = x_train.groupby([var])['price'].mean().sort_values().index
        print(var, 'ordered_labels', ordered_labels)

        # создаем словарь типа "категориальная метка - числовое значение"
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
        print('Словарь меток для признака \"' + var + '\":', ordinal_label, sep='\n')

        # согдлсно словарю заменяем категориальные метки числовыми значениями
        x_train[var] = x_train[var].map(ordinal_label)
        x_test[var] = x_test[var].map(ordinal_label)

    # 6) масштабирование признаков
    # список признаков, которые будут масшабироваться
    scale_variables = [var for var in x_train.columns if var not in ['Unnamed: 0', 'price']]

    # масштабирование
    scaler = MinMaxScaler()
    scaler.fit(x_train[scale_variables])
    x_train[scale_variables] = scaler.transform(x_train[scale_variables])
    x_test[scale_variables] = scaler.transform(x_test[scale_variables])

    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    # 7) выбор признаков
    # целевой признак
    y_train = x_train['price']
    y_test = x_test['price']

    x_train.drop(['Unnamed: 0', 'price'], axis=1, inplace=True)
    x_test.drop(['Unnamed: 0', 'price'], axis=1, inplace=True)
    x_train = x_train.drop('zipcode', axis=1)
    x_test = x_test.drop('zipcode', axis=1)

    # 8) модель
    # при использовании XGBRegressor точность была выше, чем при использовании регрессии Лассо
    xgb_model = XGBRegressor().fit(x_train, y_train)

    y_test_predicted = xgb_model.predict(x_test)
    result_price = [np.exp(i) for i in y_test_predicted.tolist()]
    res = mean_absolute_percentage_error(y_test.to_list(), result_price)

    print(res)
    pickle.dump(xgb_model, open('model.pkl', 'wb'))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    train_model()
