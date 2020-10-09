import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:////home/workspace/data/DisasterResponse.db')
print(engine.table_names())
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    ## 1. how many messages are there in genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    ## 2. how many messages are there in each category
    df_categories = df.drop(["id", "message", "original", "genre"], axis=1)
    output_categories_percentage =(df_categories.sum()/df_categories.shape[0]).round(decimals=3)
    output_categories = list(output_categories_percentage.index)
    category_frequency = output_categories_percentage.tolist()    
    ## 3. message length distribution
    message_len = df.message.str.split().str.len()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=output_categories,
                    y=category_frequency
                )
            ],

            'layout': {
                'title': 'Frequency of output categories',
                'yaxis': {
                    'title': "Proportion of total"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=message_len
                )
            ],

            'layout': {
                'title': 'Histogram of Message Length',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Message Lens"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()