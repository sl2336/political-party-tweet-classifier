import ml_functions as mlf
import tensorflow as tf
import json

def serve_model_prediction(tweet):
    #Load model
    model = tf.keras.models.load_model('../demsvsreps_embedding/')
    
    #load in config paramaters after training
    f = open('../../config/config.json')
 
    # returns JSON object as
    # a dictionary
    data = json.load(f)

    optimal_threshold = data['optimal_threshold']

    # Closing file
    f.close()
    
    #preprocess tweet
    x = mlf.preprocess_tweet(tweet)
    
    #Get prediction
    score = model.predict(tf.convert_to_tensor(x))
    
    return 'Democrat' if (score > optimal_threshold)[0] else 'Republican'