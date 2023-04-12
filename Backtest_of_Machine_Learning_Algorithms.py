# Import necessary libraries and modules for data manipulation, visualization, and machine learning
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import requests
import bs4 as bs
import yfinance as yf
import ta
import quantstats as qs
from empyrical import max_drawdown, alpha_beta
from sklearn import svm, preprocessing 
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn import preprocessing, model_selection

# Set up warnings to ignore
import warnings
warnings.filterwarnings('ignore')

# Load a list of tickers from a CSV file and scrape the S&P 500 company data from Wikipedia using Beautiful Soup
tickers = pd.read_csv('tickers.csv')
resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]

# Set the start and end dates for the historical data
start_date = datetime.datetime(2010, 1, 3)
end_date = datetime.datetime(2020, 6, 3)

# Download the historical data for each ticker from Yahoo Finance for the specified time period
data = yf.download(tickers, start=start_date, end=end_date)
print(data)

# Define a function to preprocess the data for a given ticker and return a DataFrame of features
def preprocess_data(ticker, data):
    # Create a DataFrame for the ticker's historical data
    df = pd.DataFrame()
    adj = pd.DataFrame()
    df['Open ' + ticker] = data['Open'][ticker].fillna(0)
    df['High ' + ticker] = data['High'][ticker].fillna(0)
    df['Low ' + ticker] = data['Low'][ticker].fillna(0)
    df['Close ' + ticker] = data['Close'][ticker].fillna(0)
    df['Adj Close ' + ticker] = data['Adj Close'][ticker].fillna(0)
    df['Volume ' + ticker] = data['Volume'][ticker].fillna(0)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    adj['Adj Close'] = data['Adj Close'][ticker].fillna(0)
    
    # Calculate technical analysis indicators using the ta library and add them as features
    df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1, inplace=True)

    # Convert the DataFrame of targets (returns) to a binary classification problem
    returns = adj.pct_change().fillna(0)
    returns.drop(returns.head(1).index,inplace=True)  
    targets = np.where(returns > 0, 1, -1)
    
    # Split the data into training and validation sets
    index = 2000
    features = df.fillna(0)
    training_features = features.iloc[:index, :]
    validation_features = features.iloc[index:, :]
    training_targets = targets[:index]
    validation_targets = targets[index:]
    
    return training_features, validation_features, training_targets, validation_targets

# Define a list of machine learning models to train and evaluate
models = [
    ('LR', LogisticRegression(C=1e09)),
    ('LR_L2', LogisticRegression(penalty='l2')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('GB', GradientBoostingClassifier()),
    ('ABC', AdaBoostClassifier()),
    ('RF', RandomForestClassifier()),
    ('SVM', SVC(probability=True)),
    ('SVMP', SGDClassifier(loss='perceptron')),
    ('SVMH', SGDClassifier(loss='hinge')),
    ('MLP', MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500))
]

# Define a function to train and evaluate the models for a given ticker
def evaluate_models(ticker, data):
    # Preprocess the data for the given ticker
    training_features, validation_features, training_targets, validation_targets = preprocess_data(ticker, data)
    
    # Train each model and evaluate its performance on the validation set
    for name, model in models:
        # Scale the features to improve model performance
        X = preprocessing.scale(training_features)
        Y = training_targets
        clf = model
        try1 = clf.fit(X, Y)
        yhat = try1.predict(validation_features)
        
        # Calculate performance metrics for the model
        correct_count = sum(yhat == validation_targets)
        total_invests = len(yhat)
        invest_return = -np.sum((validation_targets * adj['Adj Close'][ticker].iloc[index:]) * (yhat - 1))
        final_value = invest_return + 10000
        final_return = final_value / 10000
        accuracy = (correct_count / total_invests) * 100
        sharpe_ratio_algo = qs.stats.sharpe(df_tracking['log returns'])
        sharpe_ratio_bh = np.sqrt(623) * adj['Adj Close'][ticker].pct_change().fillna(0).mean() / adj['Adj Close'][ticker].pct_change().fillna(0).std()
        drawdown = abs(max_drawdown(df_tracking['cummulative_sum']))
        bh_drawdown = 10000 * ((adj['Adj Close'][ticker].max() - adj['Adj Close'][ticker].min()) / adj['Adj Close'][ticker].max())
        
        # Print performance metrics for the model
        print(f"{ticker}: {name}: {accuracy:.2f}: {total_invests}: {final_value:.2f}: {final_return:.2f}: {adj['Adj Close'][ticker].iloc[-1]:.2f}: {sharpe_ratio_algo:.2f}: {sharpe_ratio_bh:.2f}: {drawdown:.2f}: {bh_drawdown:.2f}")

        # Print performance metrics for the model
        print(f"{ticker}: {name}: {accuracy:.2f}: {total_invests}: {final_value:.2f}: {final_return:.2f}: {adj['Adj Close'][ticker].iloc[-1]:.2f}: {sharpe_ratio_algo:.2f}: {sharpe_ratio_bh:.2f}: {drawdown:.2f}: {bh_drawdown:.2f}")
        
        # Plot the cumulative returns for the model and the buy-and-hold strategy
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_tracking['cummulative_sum'], label=f"{name}")
        ax.plot(10000 * (1 + adj['Adj Close'][ticker].iloc[index:].pct_change().cumsum()), label="Buy and Hold")
        ax.set_title(f"{ticker} - {name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        plt.show()

# Evaluate the models for each ticker in the tickers list
for ticker in tickers:
    evaluate_models(ticker, data1)

   
