from flask import Flask,render_template,request
import model as sm
import time

app=Flask(__name__)

@app.route("/")
def home():
    
   # return tf_idf,log_reg,naive_bayes
    return render_template("home.html")

"""@app.route("/result")
def result():
    return render_template("result.html")

"""


@app.route("/analyser")
def analyser():
    
    
    return render_template("analyser.html",s_value=10,choosed_model="none")

def predict_for_all(text):
    model="logistic_reg"
    sentiment_by_log,log_timer,log_prob=sm.predict_text(text,model)
    
    model="naive_bayes"        
    sentiment_by_bayes,bayes_timer,naive_prob=sm.predict_text(text,model)
    
    model="svm_model"
    sentiment_by_svm,svm_timer,svm_prob=sm.predict_text(text,model)
    
    return sentiment_by_log,log_timer,sentiment_by_bayes,bayes_timer,sentiment_by_svm,svm_timer,log_prob,naive_prob,svm_prob
    


    
@app.route("/analyser_result" ,methods=["POST" ,"GET"])
def result():
    if request.method == 'POST':
        
        twee=request.form["tweet_box"]
        model_request=request.form["radio"]
        
        
        if model_request=="all":
            sentiment_by_log,log_timer,sentiment_by_bayes,bayes_timer,sentiment_by_svm,svm_timer,log_prob,naive_prob,svm_prob=predict_for_all(twee)
            
            return render_template("analyser.html",choosed_model=model_request,tweet=twee,log_score=sentiment_by_log,log_time=log_timer,naive_score=sentiment_by_bayes,naive_time=bayes_timer,svm_score=sentiment_by_svm,svm_time=svm_timer,log_acc=log_prob,naive_acc=naive_prob,svm_acc=svm_prob)
        else:
            score,timer,prob=sm.predict_text(twee, model_request)
            print(prob)
            return render_template("analyser.html",s_value=score ,tweet=twee,choosed_model=model_request,log_time=timer,naive_time=timer,svm_time=timer,log_acc=prob,naive_acc=prob,svm_acc=prob)
   
if __name__ == '__main__':
    app.run()
