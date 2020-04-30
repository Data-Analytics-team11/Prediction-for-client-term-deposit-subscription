import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd 
from dash.dependencies import Input,Output
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics 
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,accuracy_score ,f1_score,r2_score,roc_curve,roc_auc_score,balanced_accuracy_score
import pickle
from imblearn.over_sampling import SMOTE
import dash_table

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

colors = {
    'background': '#111111',
    'background2': '#FF0',
    'text': 'yellow'
    }
model_list = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVM','KNN', 'Logistic Regression','Neural Network']     
data_list = ['Imbalanced','SMOTE', 'Over-Sampled']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children = [html.H1(children = 'Prediction of term deposit subscription',
                               style={
                                      'textAlign': 'center',
                                      "background": "yellow"}),
             html.Div(children = [html.Label(children = 'Select the Alogrithm:',style={'display': 'inline-block','vertical-align': 'left',
                                                                 'margin': 0,'padding': '8px',}),
                       dcc.Dropdown(id='model',
                            options=[{'label': i, 'value': i} for i in model_list],
                            value='Decision Tree'),
                       html.Label(children = 'Select the data:',style={'width': '50%', 'display': 'inline-block','vertical-align': 'middle','position': 'middle'}),
                       dcc.Dropdown(id='data',
                            options=[{'label': i, 'value': i} for i in data_list],
                            value='SMOTE'),             
                            ]),
             html.Div(id='Output1')                        
                            ])        


@app.callback(Output('Output1','children'),
              [Input('model','value'),
               Input('data', 'value')]
              )
                     
def output(model,data):
    if model =='Decision Tree' and data == 'SMOTE':
        pkl1_filename = "KNN_model1.pkl"
    with open(pkl1_filename, 'rb') as file1:
        DT_model = pickle.load(file1)
    y_pred=DT_model.predict(X_test)
    return 
    
if __name__ == '__main__':
 app.run_server(debug=True,use_reloader=False)