# political-party-tweet-classifier
Classifier to determine whether a tweet leans more democratic or republican
The underlying model is a neural network containing an embedding layer to vectorize tweets in feature space based on the political party affiliated by the twitter user

## Setup environment variables and load conda environment
Make sure you are in [src/](./src)
```console
source setup.sh
```

## Launch Tweet Classifier

First make sure you are in [src/flask_code/](./src/flask_cde)
```console
flask run
```
