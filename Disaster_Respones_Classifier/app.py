import os
import sys
sys.path.append(r"model") 
sys.path.append(r"templates")
from ada_classifier import *
import pickle
import plotly
from plotly.graph_objs import Bar
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import plotly
import json
import uvicorn

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)

# load model
with open('./model/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

def return_figures():
    """Creates two plotly visualizations
    Args:
        None
    Returns:
        list (dict): list containing the two plotly visualizations
    """
    graph_one = []
    category_names = df.iloc[:,4:].columns.values.tolist()
    value_1 = [df[df[category]==1][category].value_counts()[1] for category in category_names]
    value_0 = [df[df[category]==0][category].value_counts()[0] for category in category_names]
    graph_one.append(
        Bar(
        name = '1',
        x = category_names,
        y =  value_1,        
      )
    )
    graph_one.append(
        Bar(
        name = '0',
        x = category_names,
        y =  value_0,
       
      )
    )
    layout_one = dict(title = 'Distribution of Message Categories',
                xaxis = dict(title = "Category",),
                yaxis = dict(title = "Count"),
                )
    graph_two = [] 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_two.append(
      Bar(
        x = genre_names,
        y = genre_counts,
      )
    )
    layout_two = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = "Genre",),
                yaxis = dict(title = "Count"),
                )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures


# instantiate FastAPI object
app = FastAPI()
templates = Jinja2Templates(directory="templates/") 

# index webpage displays visuals and receives user input text for model
@app.get("/")
@app.get("/index")
def index(request: Request):
    figures = return_figures()
    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return templates.TemplateResponse("index.html", {"request": request, "ids": ids, "graphJSON": graphJSON})


# web page that handles user query and displays model results
@app.get("/go")
def go(request: Request):
    # save user input in query
    query = request.query_params.get("query", "") 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return templates.TemplateResponse(
        'go.html',
        {"request": request, "query": query, "classification_result": classification_results}
    )


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)