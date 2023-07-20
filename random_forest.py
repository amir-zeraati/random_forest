
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def read_data():
    df = pd.read_excel('data/MP_Revised_900_DataSet.xlsx')
    return df


def split_data(df):
    X = df.drop('Tmk', axis=1)
    y = df['Tmk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def create_model():
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    return model


def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def calculate_error(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def plot_results(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.show()


def main():
    df = read_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = create_model()
    model = fit_model(model, X_train, y_train)
    y_pred = predict(model, X_test)
    mse, r2 = calculate_error(y_test, y_pred)
    plot_results(y_test, y_pred)
    print('MSE: ', mse)
    print('R2: ', r2)
    assert mse > 0
    assert r2 > 0
    print('Tests passed')


if __name__ == '__main__':
    main()
