"""
Before running this, we need to set an environment variable
>>>export FLASK_APP=application.py

This will set the environment variable FLASK_APP to application.py
temporarily (until user ends the session)

if you want a permanent variable, need to set bash profile

if you want to check what the environment variable is set as
>>>echo $FLASK_APP
"""

from flask import Flask, render_template, request
import sys
sys.path.append("../../utils/")
import serve_model_results as smr

#This command says that turn this file is my web application
app = Flask(__name__)

#i want to build an app that has a route listening for slash 
@app.route("/")
#whenever you see a request from some user, call this function (index)
def index():

    return render_template("index.html")

#build a route called classify that listens for tweets to be classified
@app.route("/classify", methods=["POST"])
def classify():
    #for POST requests, its request.form.get() - normally it is request.args.get
    user_tweet = request.form.get("user_inputed_tweet")
    classified_party = smr.serve_model_prediction(user_tweet)

    if not user_tweet:
        return render_template("failure.html")
    if not isinstance(user_tweet, str):
        return render_template("failure.html")
    if len(user_tweet) > 140:
        return render_template("character_limit.html")

    if classified_party == 'Democrat':
        return render_template("democrat.html")
    else:
        return render_template("republican.html")