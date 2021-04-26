import re
import numpy as np
import pandas as pd

from twitter import *
# Tweepy - Python library for accessing the Twitter API.
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import time

#nltk.download('stopwords')
#all_stopwords = stopwords.words('english')
    
#loading data=============================================================================

dataset_columns = ["sentiment" , "id" , "time" , "flag" , "user" , "text" ]
dataset_coding = "ISO-8859-1"
dataset = pd.read_csv("twitter_dataset.csv", names = dataset_columns , encoding = dataset_coding )

#preprocessing function=============================================================================

def preprocess(tweet , stem = True):
    processedText = []
    
    urlPattern =  r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphanumericPattern = r'\w*\d\w*'
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    punc = r'[^\w\s]'
    
    #tweet = tweet.lower()
    tweet = re.sub(urlPattern , '' , tweet)
    tweet = re.sub(userPattern , '' , tweet)
    tweet = re.sub(alphanumericPattern , '' , tweet)
    tweet = re.sub(sequencePattern , seqReplacePattern , tweet)
    tweet = re.sub(punc,"",tweet)
    
    tweet = tweet.split()   
    ps = PorterStemmer()
    
    all_stopwords = stopwords.words('english')
  
    
    for word in tweet:
            if word not in (all_stopwords):
                if stem:
                    processedText.append(ps.stem(word))
                else:
                    processedText.append(word)
                
    return " ".join(processedText)
    
        
#droping 600000 data------------------------------------------------------------------

cleared_dataset = dataset.sample(frac=1).reset_index(drop=True)
cleared_dataset = cleared_dataset.iloc[0:1000000]

#appling preprocessing=============================================================================

cleared_dataset = cleared_dataset.drop(cleared_dataset.index[0]).reset_index()
cleared_dataset = cleared_dataset.drop(['time','flag','user','id','index'],axis=1)
cleared_dataset["text"] = cleared_dataset["text"].apply(preprocess)
X = cleared_dataset['text']


#splitting data---------------------------------------------------------------------------------

y = cleared_dataset['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.30, random_state = 10)

#vectorizing data

vectorizer = TfidfVectorizer(analyzer='word',max_df=0.90, min_df=2, max_features = 500000,ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train)
tfidf_tokens = vectorizer.get_feature_names()
print("Number of feature_words = ", len(tfidf_tokens))
#print(tfidf_tokens[1:2000])

X_test  = vectorizer.transform(X_test)  #transforming x_test on X_train's transformation

#scaling

x_max = X_train.max()
x_min = X_train.min()

X_train = (X_train - x_min)/x_max
X_test = (X_test - x_min)/x_max

#model evaluation=============================================================================

def model(model):
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.values 
    y_test_ = pd.Series(y_test) 
    y_test_ = y_test_.values 
    #print("Comparison:")
    #result = np.concatenate((y_pred.reshape(len(y_pred),1), y_test_.reshape(len(y_test_),1)),1)
    #print(result[1:500])
    
    c_matrix = confusion_matrix(y_test_, y_pred)
    print("Confusion Matrix = \n",c_matrix)
    print("Accuracy Score = ",accuracy_score(y_test_, y_pred))
    print("Train Score = ",model.score(X_train, y_train))
    print("Test Score = ",model.score(X_test, y_test))
    
    
    
    categories = ['Negative','Positive']
    prediction = ['True Negative','False Positive', 'False Negative','True Positive']
    percentage = ['{0:.2%}'.format(value) for value in c_matrix.flatten() / np.sum(c_matrix)]

    labels = [f'{m}\n{n}' for m, n in zip(prediction,percentage)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(c_matrix,cmap = 'Blues' , fmt = '',annot = labels, xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.title ("Confusion Matrix")
    
def logisticRegression():
    global logistic_reg
    start_time = time.time()
    logistic_reg = LogisticRegression(solver = 'sag',C = 2, max_iter = 1500)
    logistic_reg.fit(X_train, y_train) 
    model(logistic_reg)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
def naiveBayes():    
    global naive_bayes 
    start_time = time.time()
    naive_bayes = BernoulliNB()
    naive_bayes.fit(X_train, y_train) 
    model(naive_bayes)
    print("--- %s seconds ---" % (time.time() - start_time))
    
def svm():
    global svm_model
    svm = LinearSVC()
    svm_model = CalibratedClassifierCV(svm) 
    svm_model.fit(X_train, y_train)
    model(svm_model)
    
svm()
logisticRegression()        
naiveBayes()    

config = pd.read_csv("config.csv")
    

twitterApiKey = str(config['twitterApiKey'][0])
twitterApiSecret = str(config['twitterApiSecret'][0])
twitterApiAccessToken = str(config['twitterApiAccessToken'][0])
twitterApiAccessTokenSecret = str(config['twitterApiAccessTokenSecret'][0])

def predict_text(text,model):
    start_time = time.time()
    sentiment=0
    sentiment_prob=[[]]
    textdata = vectorizer.transform([preprocess(text)])
    sentiment = model.predict(textdata)
    sentiment_prob = model.predict_proba(textdata)   
    prob = ('%.2f'%max(max(sentiment_prob*100)))
    timer = "%.2f seconds" % (time.time() - start_time)
    return sentiment[0],timer,prob      

def tweets_of_twitter_user(user_name,no_of_tweets):
    

    # Authenticate
    auth = tweepy.OAuthHandler(twitterApiKey, twitterApiSecret)
    auth.set_access_token(twitterApiAccessToken, twitterApiAccessTokenSecret)
    twetterApi = tweepy.API(auth, wait_on_rate_limit = True)


    twitterAccount = user_name

    tweets = tweepy.Cursor(twetterApi.user_timeline, 
                            screen_name=twitterAccount, 
                            count=None,
                            since_id=None,
                            max_id=None,
                            trim_user=True,
                            exclude_replies=True,
                            contributor_details=False,
                            include_entities=False
                            ).items(no_of_tweets);
    print(tweets)

    tweet_DataBase = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweet'])
    
    ### DataFrame for the sentiment of the user
    sentiment_score=[]
    for i in range(0,no_of_tweets):
        sentiment_score.append(0)
    
    for model in [logistic_reg,naive_bayes,svm_model]:
            prediction=[]
            i=0
            for tweets in tweet_DataBase["Tweet"]:
                sen,timer,prob=predict_text(tweets,model)
                
                prediction.append(float(prob))
                sentiment_score[i]=sentiment_score[i]+sen
                
                i+=1
            tweet_DataBase[model]=prediction
            
    sentiment_score=["Positive" if i>4 else "Negative" for i in sentiment_score]
    tweet_DataBase.columns=["Tweets","log_reg","naive","svm"]
    tweet_DataBase.insert(1,"Sentiment",sentiment_score,True)
    return tweet_DataBase

    
def predict(model,text):
    textdata = vectorizer.transform([preprocess(text)])
    sentiment = model.predict(textdata)
    return sentiment[0]

def Twitter_account_name(user_name):
    account_name=""
    twitter = Twitter(auth = OAuth(twitterApiAccessToken,
                                   twitterApiAccessTokenSecret,
                                   twitterApiKey,
                                   twitterApiSecret))
    results = twitter.users.search(q = user_name)
    
    for user in results:
        if user["screen_name"]==user_name or user["name"]==user_name :
            account_name=user["name"]
    return account_name





