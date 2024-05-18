#importing libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import tensorflow as tf
import os 
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout
from keras import backend as k
import warnings 
warnings.simplefilter('ignore')
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
##Interactive visuals
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
#pip install -r requirements.txt
# Load data
df = pd.read_csv('Meat_market.csv')
df['Item'].unique()

# Set up page
st.set_page_config(page_title='Meat Market', page_icon=":flag-eu:", layout="wide")

# Configure the sidebar for user input
st.sidebar.header('Please Filter Here:')
year_range = st.sidebar.slider(
    'Select the year range',
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(int(df['Year'].min()), int(df['Year'].max()))
)

selected_area = st.sidebar.selectbox('Select Area', options=df['Area'].unique())

# Filter the DataFrame based on user input

df_selection = df.query('Year >= @year_range[0] and Year <= @year_range[1]  and Area == @selected_area')

#Main page
st.title('European Meat Market ')
Country = df_selection['Area'].iloc[0]
Population = int(df_selection['Population'].mean())*1000 # original unit is *1000 inhabitants
Land = int(df_selection['Country area'].mean()*1000) # same as land area multiplied by 1000 hectars
Pastures = int(df_selection['Permanent meadows and pastures'].mean()*1000)
Total_export = int(df_selection['Export Quantity'].sum()*1000)
Total_Production = int(df_selection['Production'].sum()*1000)
Total_SupplyQuantity = int(df_selection['Domestic supply quantity'].sum()*1000)
#Calling the most recent GDP 
df_selection_sorted = df_selection.sort_values(by='Year', ascending=False)
most_recent_gdp_per_capita = int(df_selection_sorted['GDP per capita in USD'].iloc[0])
GDP = most_recent_gdp_per_capita
          
#Splitting the header into 3 columns
left_column, middle_column, right_column = st.columns(3)
with left_column:
    
    st.header(f'{Country}')
    st.subheader(f'Average Population: {Population:,} inh')
    st.subheader(f'Average Land Area: {Land:,}Ha')
                       
with middle_column:
    st.subheader(f'Total Production: {Total_Production:,}t') 
    st.subheader(f'Total Export Quantity: {Total_export:,}t')
    st.subheader(f'Total Supply Quantity: {Total_SupplyQuantity:,}t')
    st.subheader(f'GDP: {GDP:,} in Million USD') 
    
# Placing a markdown   
st.markdown("""---""")

#creating 2 columns                       
col1, col2, col3 = st.columns(3)   

# Grouping by item and the sum of production 
production_by_item = df_selection.groupby('Item')['Production'].sum().sort_values(ascending=False)*1000 #converting the unit 

# Plotting using Plotly Express
#Production in the first column
with col1: 
    Prod_ = px.bar(
        production_by_item,
        x=production_by_item.values,
        y=production_by_item.index,
        orientation='h',
        title="<b>Production by Item</b>",
        color_discrete_sequence=["#0083B8"],  
        template="plotly_white"
    )

    Prod_.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title='Total Production in t' )
    )
# plotting
    st.plotly_chart(Prod_)
                       
#Supply Quantity in the second column
supplyquantity_by_item = df_selection.groupby('Item')['Domestic supply quantity'].sum().sort_values(ascending=False)*1000 #converting the unit   
                 
with col2: 
    Prod_ = px.bar(
        production_by_item,
        x=supplyquantity_by_item.values,
        y=supplyquantity_by_item.index,
        orientation='h',
        title="<b>Supply Quantity by Item</b>",
        color_discrete_sequence=["#0083B8"],  
        template="plotly_white"
    )

    Prod_.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title='Total Domestic Supply Quantity in t' )
    )
# plotting
    st.plotly_chart(Prod_)     
                 
#Export Quantity in the second column                       
export_by_item = df_selection.groupby('Item')['Export Quantity'].sum().sort_values(ascending=False)*1000

with col3:                       
    exp_ = px.bar(
        export_by_item,
        x=export_by_item.values,
        y=export_by_item.index,
        orientation='h',
        labels={'x': 'Total Quantity Exported in Tonnes', 'index': 'Item'},
        title="<b>Export by Item</b>",
        color_discrete_sequence=["#0083B8"], 
        template="plotly_white"
    )

    Prod_.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title='Export Quantity in t' )
    )
# plotting
    st.plotly_chart(exp_)   
    
# placing another markdown    
st.markdown("""---""")

# Group by Year and Item
yearly_data = df_selection.groupby(['Year', 'Item']).agg({
    'Production': 'sum',
    'Export Quantity': 'sum',
    'Domestic supply quantity': 'sum'
}).reset_index()


yearly_data['Production'] *= 1000
yearly_data['Export Quantity'] *= 1000
yearly_data['Domestic supply quantity'] *= 1000


# Sorting by Year
yearly_data.sort_values('Year', inplace=True)

# Create initial traces for Production
fig = go.Figure()


# Production
for item in yearly_data['Item'].unique():
    item_data = yearly_data[yearly_data['Item'] == item]
    fig.add_trace(go.Scatter(
        x=item_data['Year'],
        y=item_data['Production'],
        mode='lines',
        name=f'{item} - Production'
    ))

# Export Quantity
for item in yearly_data['Item'].unique():
    item_data = yearly_data[yearly_data['Item'] == item]
    fig.add_trace(go.Scatter(
        x=item_data['Year'],
        y=item_data['Export Quantity'],
        mode='lines',
        name=f'{item} - Export Quantity',
        visible=False 
    ))

# Domestic supply quantity
for item in yearly_data['Item'].unique():
    item_data = yearly_data[yearly_data['Item'] == item]
    fig.add_trace(go.Scatter(
        x=item_data['Year'],
        y=item_data['Domestic supply quantity'],
        mode='lines',
        name=f'{item} - Domestic supply quantity',
        visible=False  
    ))

# Update layout
fig.update_layout(
    title='Annual Sum of Production, Export Quantities, and Domestic Supply Quantity by Meat Type',
    xaxis_title='Year',
    yaxis_title='Total Quantity',
    legend_title='Metrics',
    plot_bgcolor='rgba(0,0,0,0)',
    updatemenus=[{
        'buttons': [
            {
                'method': 'update',
                'label': 'Production',
                'args': [
                    {'visible': [i < len(yearly_data['Item'].unique()) for i in range(len(fig.data))]},
                    {'title': 'Production Quantities by Meat Type'}
                ]
            },
            {
                'method': 'update',
                'label': 'Export Quantity',
                'args': [
                    {'visible': [len(yearly_data['Item'].unique()) <= i < 2 * len(yearly_data['Item'].unique()) for i in range(len(fig.data))]},
                    {'title': 'Export Quantities by Meat Type'}
                ]
            },
            {
                'method': 'update',
                'label': 'Domestic supply quantity',
                'args': [
                    {'visible': [2 * len(yearly_data['Item'].unique()) <= i < 3 * len(yearly_data['Item'].unique()) for i in range(len(fig.data))]},
                    {'title': 'Domestic Supply Quantities by Meat Type'}
                ]
            }
        ],
        'direction': 'down',
        'showactive': True,
    }]
)

# Plotting
st.plotly_chart(fig, use_container_width=True)


# Define paths using raw string literals to avoid escape sequence issues
base_path = r'C:\Users\lucas\OneDrive\Área de Trabalho\CCT\Github\InteractiveApp'

model_path = os.path.join(base_path, 'NNmodel.h5')
scaler_path = os.path.join(base_path, 'scaler.pkl')
item_encoder_path = os.path.join(base_path, 'Item_encoder.pkl')
area_encoder_path = os.path.join(base_path, 'Area_encoder.pkl')

# Function to check if file exists and is readable
def check_file(path):
    if os.path.exists(path):
        if os.access(path, os.R_OK):
            return True
        else:
            st.error(f"File is not readable: {path}")
            print(f"File is not readable: {path}")
            return False
    else:
        st.error(f"File does not exist: {path}")
        print(f"File does not exist: {path}")
        return False

# Load the trained model
if check_file(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully")
        print("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")

# Load the scaler
if check_file(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        st.success("Scaler loaded successfully")
        print("Scaler loaded successfully")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        print(f"Error loading scaler: {e}")

# Load the item encoder
if check_file(item_encoder_path):
    try:
        with open(item_encoder_path, 'rb') as f:
            item_encoder = pickle.load(f)
        st.success("Item encoder loaded successfully")
        print("Item encoder loaded successfully")
    except Exception as e:
        st.error(f"Error loading item encoder: {e}")
        print(f"Error loading item encoder: {e}")

# Load the area encoder
if check_file(area_encoder_path):
    try:
        with open(area_encoder_path, 'rb') as f:
            area_encoder = pickle.load(f)
        st.success("Area encoder loaded successfully")
        print("Area encoder loaded successfully")
    except Exception as e:
        st.error(f"Error loading area encoder: {e}")
        print(f"Error loading area encoder: {e}")

# Streamlit UI
st.title("Meat Market Prediction")
st.markdown("""---""")
st.header("Predict Future Values")

# Create input fields for user to enter prediction data
if 'item_encoder' in locals() and hasattr(item_encoder, 'classes_'):
    item_input = st.selectbox('Item', options=item_encoder.classes_)
else:
    item_input = st.selectbox('Item', options=[])

if 'area_encoder' in locals() and hasattr(area_encoder, 'classes_'):
    area_input = st.selectbox('Area', options=area_encoder.classes_)
else:
    area_input = st.selectbox('Area', options=[])

population_input = st.number_input('Population (in thousands)', min_value=0)
land_input = st.number_input('Land Area (in hectares)', min_value=0)
pastures_input = st.number_input('Permanent Meadows and Pastures (in hectares)', min_value=0)
export_input = st.number_input('Export Quantity (in tonnes)', min_value=0)
production_input = st.number_input('Production (in tonnes)', min_value=0)
supply_input = st.number_input('Domestic Supply Quantity (in tonnes)', min_value=0)
gdp_input = st.number_input('GDP (in Million USD)', min_value=0)

# Function to preprocess inputs similar to training data
def preprocess_inputs(inputs):
    num_inputs = inputs[2:]  # Extract numerical inputs
    scaled_num_inputs = scaler.transform([num_inputs])  # Scale numerical inputs
    preprocessed_inputs = [inputs[0]] + [inputs[1]] + scaled_num_inputs[0].tolist()  # Combine inputs
    return preprocessed_inputs

# Make predictions based on user inputs
if st.button('Predict'):
    item_encoded = item_encoder.transform([item_input])[0]
    area_encoded = area_encoder.transform([area_input])[0]
    inputs = [item_encoded, area_encoded, population_input, land_input, pastures_input, export_input, production_input, supply_input, gdp_input]
    processed_inputs = preprocess_inputs(inputs)
    prediction = model.predict([processed_inputs])
    st.subheader(f'Predicted Value: {prediction[0][0]:.2f}')



