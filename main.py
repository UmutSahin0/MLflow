import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# mlflow.set_experiment(experiment_name='startups')
with mlflow.start_run(run_name='startups_run'):

    data=pd.read_csv('medical_insurance.csv')
    le = LabelEncoder()

    columns_to_encode = ['sex', 'smoker', 'region']
    for col in columns_to_encode:
        data[col] = le.fit_transform(data[col])

    y = data.iloc[:,-1]
    X = data.iloc[:,0:-1]

    train_test_split_params = {'test_size': 0.33, 'random_state': 42}
    X_train, X_test, y_train, y_test = train_test_split(X, y, **train_test_split_params)
    mlflow.log_params({'train_test_split_params': train_test_split_params})

    num_columns = len(data.columns)
    mlflow.log_param('num_columns', num_columns)
    mlflow.log_param('encoded_columns', columns_to_encode)
    mlflow.log_param('encoding_type', 'LabelEncoder')

    reg = LinearRegression().fit(X_train, y_train)
    mlflow.sklearn.log_model(reg, "linear_regression_model")

    y_pred = reg.predict(X_test)

    score = r2_score(y_test, y_pred)
    print(score)
    mlflow.log_metric("r2_score", score)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title('y_test vs. y_pred')
    plt.grid(True)

    plt.savefig("test_vs_pred.png")
    mlflow.log_artifact("test_vs_pred.png")

    plt.show()

    mlflow.end_run()











