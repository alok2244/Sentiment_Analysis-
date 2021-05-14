from flask import Flask,render_template,request
import model as sm

app=Flask(__name__)

@app.route("/")
def home():
    
   # return tf_idf,log_reg,naive_bayes}
    return render_template("home.html")

#@app.errorhandler(404) 
def invalid_route(e): 
 return render_template('hello')
 

@app.route("/user")
def user():
    return render_template("user.html",table=False)

@app.route("/user_result" ,methods=["POST"])
def user_result():
    if request.method == 'POST':
        user_name=request.form["user"]
        print("user name="+user_name)
        acc_name=""
        
        total_no_of_tweets=int(request.form["tweet_no"])
        print(total_no_of_tweets)
        
        dataset=sm.tweets_of_twitter_user(user_name,total_no_of_tweets)
        print("here")
        '''avg_log=dataset['log_reg'].sum(axis=0)/len(dataset)
        avg_naive=dataset['naive'].sum(axis=0)/len(dataset)
        avg_svm=dataset['svm'].sum(axis=0)/len(dataset)
        log=avg_log,naive=avg_naive,s=avg_svm
        
        ''' 
        if 'Positive' in dataset.values :
            pos=dataset["Sentiment"].value_counts()['Positive']
        else:
            pos=0
        
        if 'Negative' in dataset.values :
            neg=dataset["Sentiment"].value_counts()['Negative']
        else:
            neg=0
       
        print(dataset)
        acc_name=sm.Twitter_account_name(user_name)
        print("Accouct_name="+acc_name)
      
        
        return render_template("user.html", dset=dataset,table=True,positive=pos,negative=neg,length=len(dataset),user_name_for_user_tab=user_name,account_name=acc_name)
    
    

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
