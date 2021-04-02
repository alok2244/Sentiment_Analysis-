from flask import Flask,render_template,request
import model as sm
import time

app=Flask(__name__)

@app.route("/")
def home():
    
   # return tf_idf,log_reg,naive_bayes
    return render_template("home.html")

@app.route("/user")
def user():
    return render_template("user.html",table=False)

@app.route("/user_result" ,methods=["POST" ,"GET"])
def user_result():
    if request.method == 'POST':
        user_name=request.form["user"]
        total_no_of_tweets=int(request.form["tweet_no."])
        
        dataset=sm.tweets_of_twitter_user(user_name,total_no_of_tweets)
        return render_template("user.html", dset=dataset,table=True)
    
    

@app.route("/project")
def project():
    return render_template("project.html")

'''@app.route("/about")
def about():
    return render_template("Enter the fiel name")'''




@app.route("/analyser")
def analyser():
    
    
    return render_template("analyser.html",s_value=10,choosed_model="none")

def predict_for_all(text):
    
    sentiment_by_log,log_timer,log_prob=sm.predict_text(text,sm.logistic_reg)
    
         
    sentiment_by_bayes,bayes_timer,naive_prob=sm.predict_text(text,sm.naive_bayes)
    
   
    sentiment_by_svm,svm_timer,svm_prob=sm.predict_text(text,sm.svm_model)
    
    return sentiment_by_log,log_timer,sentiment_by_bayes,bayes_timer,sentiment_by_svm,svm_timer,log_prob,naive_prob,svm_prob
    


    
@app.route("/analyser_result" ,methods=["POST" ,"GET"])
def result():
    if request.method == 'POST':
        
        twee=request.form["tweet_box"]
        model_request=request.form["radio"]
        
        if model_request=="logistic_reg":
            model=sm.logistic_reg
        if model_request=="naive_bayes":
            model=sm.naive_bayes
        if model_request=="svm_model":
            model=sm.svm_model
        
        if model_request=="all":
            sentiment_by_log,log_timer,sentiment_by_bayes,bayes_timer,sentiment_by_svm,svm_timer,log_prob,naive_prob,svm_prob=predict_for_all(twee)
            
            return render_template("analyser.html",choosed_model=model_request,tweet=twee,log_score=sentiment_by_log,log_time=log_timer,naive_score=sentiment_by_bayes,naive_time=bayes_timer,svm_score=sentiment_by_svm,svm_time=svm_timer,log_acc=log_prob,naive_acc=naive_prob,svm_acc=svm_prob)
        else:
            score,timer,prob=sm.predict_text(twee, model)
            print(prob)
            return render_template("analyser.html",s_value=score ,tweet=twee,choosed_model=model_request,log_time=timer,naive_time=timer,svm_time=timer,log_acc=prob,naive_acc=prob,svm_acc=prob)
   
if __name__ == '__main__':
    app.run()
