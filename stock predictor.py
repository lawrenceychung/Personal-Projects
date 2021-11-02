#import the libraries that we need
from datetime import date
import yfinance as yf
import pandas as pd

#ability to choose which stock you want to predict
stock_ticker = "MSFT"

#using the yahoo finance API, scape the data of the historical prices for the specified stock
stock_data = yf.Ticker(stock_ticker)
info = stock_data.info


#categories of data that we want to clean from the results of the API
categories = [  "regularMarketPrice",
                "recommendationKey",
                "currentPrice",
                "earningsGrowth",
                "currentRatio",
                "debtToEquity",
                "revenuePerShare",
                "quickRatio",
                "sharesOutstanding",
                "shortRatio",
                "pegRatio",
                "previousClose",
                "regularMarketDayHigh",
                "dayHigh"
            ]


#new dictionary to hold ONLY the values that we want
new_dict = {"date": date.today()}
for item in categories:
    if item in info:
        new_dict[item] = (info[item])

# #turn the new dictionary into a dataframe for easy use
df = pd.DataFrame.from_dict(new_dict, orient="index")
df = df.T
df = df.set_index("date")
print (df)

df.to_csv("/home/lawrencechung/stock_data.csv", mode = "a", header = False)

#______________________________________________________________________________________
#read in data
data = pd.read_csv("/home/lawrencechung/stock_data.csv")

#clean data by dropping the duplicated rows which are the weekends
data.drop_duplicates(subset="regularMarketPrice", keep="first", inplace=True)
data = data.drop(['date'], axis=1)

#Find the Change in close price & turn into a binary class
data["change"] = (data["currentPrice"] - data["previousClose"])

shifted_close = data["change"].shift(-1)
data["change"] = shifted_close

data.change[data["change"] >= 0] = 1
data.change[data["change"] < 0] = 0

data.recommendationKey[data["recommendationKey"] == "buy"] = 1
data.recommendationKey[data["recommendationKey"] == "sell"] = 0

predictor = data.tail(1)
data.drop(data.tail(1).index, inplace=True)

print(data)
#_______________________________________________________________________________________
#importing ML packages

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


#Creating X and Y values for ML Model
X = data.drop("change", axis=1)
y = data.change

#splitting data set into 70% training data and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#Training using Linear Discriminant Model
LDA = LinearDiscriminantAnalysis()

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'roc_auc']
scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=5)

sorted(scores.keys())
LDA_fit_time = scores['fit_time'].mean()
LDA_score_time = scores['score_time'].mean()
LDA_accuracy = scores['test_accuracy'].mean()
LDA_precision = scores['test_precision_macro'].mean()
LDA_recall = scores['test_recall_macro'].mean()
LDA_roc = scores['test_roc_auc'].mean()

models_results = pd.DataFrame({
    'Model'       : ['Linear Discriminant Analysis'],
    'Fitting time': [LDA_fit_time],
    'Scoring time': [LDA_score_time],
    'Accuracy'    : [LDA_accuracy],
    'Precision'   : [LDA_precision],
    'Recall'      : [LDA_recall],
    'AUC_ROC'     : [LDA_roc],
    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'AUC_ROC'])

models_results.sort_values(by='Accuracy', ascending=False)

print(models_results)
LDA.fit(X, y)

predictor = predictor.drop("change", axis=1)
prediction = (LDA.predict(predictor))

#_____________________________________________________________________________________________
#sending email notification every day of the prediction
import smtplib

gmail_user = 'lawslemon@gmail.com'
gmail_password = 'McLaren675LT'

sent_from = gmail_user
to = ['lawrencecchung@gmail.com']
subject = 'MFST Prediction'
body = 'prediction: %s' %(prediction,)

email_text = """\
From: %s
To: %s
Subject: %s

%s
""" % (sent_from, ", ".join(to), subject, body)

try:
    smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtp_server.ehlo()
    smtp_server.login(gmail_user, gmail_password)
    smtp_server.sendmail(sent_from, to, email_text)
    smtp_server.close()
    print ("Email sent successfully!")
except Exception as ex:
    print ("Something went wrong….",ex)


