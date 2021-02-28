from flask import Flask,render_template,request
import model as sm
#import pickle
    
app=Flask(__name__)

@app.route("/")
def home():
    
   # return tf_idf,log_reg,naive_bayes
    return render_template("home.html")

@app.route("/result")
def result():
    return render_template("result.html")


def predict_text(text):
    textdata = sm.vectorizer.transform([sm.preprocess(text)])
    
    sentiment = sm.naive_bayes.predict(textdata)
    if sentiment==0:
        return "Negative"
    else:
        return "Positive"
    
@app.route("/" ,methods=["POST"])
def basic():
    if request.method == 'POST':
        twee=request.form["tweet"]
        
    tweet_prediction_from_LR=sm.predict_text(twee)
    
        
    return tweet_prediction_from_LR
#render_template('home.html')

if __name__ == '__main__':
    app.run()
