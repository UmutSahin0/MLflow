# In this code I loaded and used the champion model from MLflow, printed the performance metrics and compared them
# with the results in the MLflow ui.


import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# mlflow.set_experiment(experiment_name='startups')
# with mlflow.start_run(run_name='startups_run'):
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data


def label_encoder(data, columns_to_encode):
    le = LabelEncoder()

    for col in columns_to_encode:
        data[col] = le.fit_transform(data[col])

    return data, le


def train_test(data):
    y = data.iloc[:, -1]
    X = data.iloc[:, 0:-1]

    train_test_split_params = {'test_size': 0.33, 'random_state': 42}
    X_train, X_test, y_train, y_test = train_test_split(X, y, **train_test_split_params)

    return X_train, X_test, y_train, y_test, train_test_split_params


def calculate_metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('r2_score: ', r2)
    print('mean_squared_error: ', mse)
    print('mean_absolute_error: ', mae)

    return r2, mse, mae


def plot_model_performance(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title('y_test vs. y_pred')
    plt.grid(True)
    plt.savefig("test_vs_pred.png")
    plt.show()


columns_to_encode = ['sex', 'smoker', 'region']

if __name__ == '__main__':
    mlflow.set_experiment(experiment_name='startups')

    # data load
    data = load_data('medical_insurance.csv')

    # encoding
    data, le = label_encoder(data, columns_to_encode)

    # split data
    X_train, X_test, y_train, y_test, train_test_split_params = train_test(data)

    # Regression model
    # model = LinearRegression().fit(X_train, y_train)
    logged_model = 'runs:/3df4fa2805344bb5b832f02212f60c1e/model'

    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(logged_model)

    # predict with regression model which is created
    y_pred = model.predict(X_test)

    # calculate metrics
    r2, mse, mae = calculate_metrics(y_test, y_pred)

    plot_model_performance(y_test, y_pred)
