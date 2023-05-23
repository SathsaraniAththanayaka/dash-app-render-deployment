# For data analysis
import pandas as pd
import numpy as np
# For model creation and performance evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
# For visualizations and interactive dashboard creation
import dash
import dash_html_components as html
from dash import dcc, html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
data = pd.read_csv("data/winequality-red.csv")
# check for missing values
data.isnull().sum()
# drop rows with missing values
data.dropna(inplace=True)
# Drop duplicate rows
data.drop_duplicates(keep='first')
# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)
# Calculate the correlation matrix
corr_matrix = data.corr()
# Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']
# Split the data into training and testing sets (20% testing and 80% training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Create an object of the logistic regression model
logreg_model = LogisticRegression()
# Fit the model to the training data
logreg_model.fit(X_train, y_train)
# Predict the labels of the test set
y_pred = logreg_model.predict(X_test)


# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
app.layout = html.Div(children=[html.H1('CO544-2023 Lab 3: Wine Quality Prediction'),
# Layout for exploratory data analysis: correlation between two selected features
html.Div([
html.H3('Exploratory Data Analysis'),
html.Label('Feature 1 (X-axis)'),
dcc.Dropdown(
id='x_feature',
options=[{'label': col, 'value': col} for col in data.columns],
value=data.columns[0]
)
], style={'width': '30%', 'display': 'inline-block'}),
html.Div([
html.Label('Feature 2 (Y-axis)'),
dcc.Dropdown(
id='y_feature',
options=[{'label': col, 'value': col} for col in data.columns],
value=data.columns[1]
)
], style={'width': '30%', 'display': 'inline-block'}),
dcc.Graph(id='correlation_plot'),
# Layout for wine quality prediction based on input feature values
html.H3("Wine Quality Prediction"),
html.Div([
html.Label("Fixed Acidity:", style={'display': 'block','margin-bottom': '10px'}),
dcc.Input(id='fixed_acidity', type='number', required=True),
    
html.Label("Volatile Acidity:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='volatile_acidity', type='number', required=True),
    
html.Label("Citric Acid:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='citric_acid', type='number', required=True),
    
html.Label("Residual Sugar:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='residual_sugar', type='number', required=True),

html.Label("Chlorides:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='chlorides', type='number', required=True),
    
html.Label("Free Sulfur Dioxide:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='free_sulfur_dioxide', type='number', required=True),
    
html.Label("Total Sulfur Dioxide:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='total_sulfur_dioxide', type='number', required=True),
    
html.Label("Density:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='density', type='number', required=True),
    
html.Label("pH:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='ph', type='number', required=True),

html.Label("Sulphates:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='sulphates', type='number', required=True),
    
html.Label("Alcohol:", style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
dcc.Input(id='alcohol', type='number', required=True),
html.Br(),
]),
html.Div([
html.Button('Predict', id='predict-button', n_clicks=0, style={'display': 'block','margin-bottom': '10px', 'margin-top': '10px'}),
]),
html.Div([
html.H4("Predicted Quality"),
html.Div(id='prediction-output')
])
]
,style={'backgroundColor': 'lightgray'})


# Define the callback to update the correlation plot
@app.callback(
  dash.dependencies.Output('correlation_plot', 'figure'),
  [dash.dependencies.Input('x_feature', 'value'),
   dash.dependencies.Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
  fig = px.scatter(data, x=x_feature, y=y_feature, color='quality')
  fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
  return fig
# Define the callback function to predict wine quality
@app.callback(
  Output(component_id='prediction-output', component_property='children'),
  [Input('predict-button', 'n_clicks')],
  [State('fixed_acidity', 'value'),
  State('volatile_acidity', 'value'),
  State('citric_acid', 'value'),
  State('residual_sugar', 'value'),
  State('chlorides', 'value'),
  State('free_sulfur_dioxide', 'value'),
  State('total_sulfur_dioxide', 'value'),
  State('density', 'value'),
  State('ph', 'value'),
  State('sulphates', 'value'),
  State('alcohol', 'value')]
)
def predict_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid,
  residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
  density, ph, sulphates, alcohol):
# Create input features array for prediction
  input_features = np.array([fixed_acidity, volatile_acidity, citric_acid,
    residual_sugar, chlorides, free_sulfur_dioxide,
    total_sulfur_dioxide, density, ph, sulphates, alcohol]).reshape(1, -1)
# Predict the wine quality (0 = bad, 1 = good)
  prediction = logreg_model.predict(input_features)[0]
# Return the prediction
  if prediction == 1:
    return 'This wine is predicted to be good quality.'
  else:
    return 'This wine is predicted to be bad quality.'

if __name__ == '__main__':
  app.run_server(debug=False)