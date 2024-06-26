#importing libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import tensorflow as tf
import os 
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import warnings 
warnings.simplefilter('ignore')
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import pickle
##Interactive visuals
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
#important libraries in the txt file
#pip install -r requirements.txt

# Load data
df = pd.read_csv('Meat_market.csv',encoding='utf-8')
df['Item'].unique()


# Set up the Streamlit page
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

metric = st.sidebar.selectbox(
    'Select Metric',
    options=['Production', 'Export Quantity', 'Domestic supply quantity']
)

# Filter the DataFrame based on user input
df_selection = df.query('Year >= @year_range[0] and Year <= @year_range[1]  and Area == @selected_area')

# Main page
st.title('European Meat Market')
df_selection_sorted = df_selection.sort_values(by='Year', ascending=False)
Country = df_selection['Area'].iloc[0]
Population = int(df_selection_sorted['Population'].iloc[0]) * 1000  # original unit is *1000 inhabitants
Land = int(df_selection_sorted['Country area'].iloc[0] * 1000)  # same as land area multiplied by 1000 hectares
Pastures = int(df_selection_sorted['Permanent meadows and pastures'].iloc[0] * 1000)
Total_export = int(df_selection['Export Quantity'].sum() * 1000)
Total_Production = int(df_selection['Production'].sum() * 1000)
Total_SupplyQuantity = int(df_selection['Domestic supply quantity'].sum() * 1000)
# Calling the most recent GDP
most_recent_gdp_per_capita = int(df_selection_sorted['GDP per capita in USD'].iloc[0])
GDP = most_recent_gdp_per_capita

# Placing a markdown
st.markdown("""---""")
# Splitting the header into 3 columns
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.header(f'{Country}')
    st.subheader(f'Population: {Population:,}inh')
    st.subheader(f'Land Area: {Land:,}Ha')

with middle_column:
    st.header(f'')
    st.subheader(f'Total Production: {Total_Production:,}t')
    st.subheader(f'Total Export Quantity: {Total_export:,}t')

with right_column:
    st.header(f'')
    st.subheader(f'Total Supply Quantity: {Total_SupplyQuantity:,}t')
    st.subheader(f'GDP: {GDP:,} in Million USD')

# Placing a markdown
st.markdown("""---""")

# Creating 3 columns
col1, col2, col3 = st.columns(3)

# Grouping by item and the sum of production
production_by_item = df_selection.groupby('Item')['Production'].sum().sort_values(ascending=False) * 1000  # converting the unit

# Plotting using Plotly Express
# Production in the first column
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
        xaxis=dict(showgrid=False, title='Total Production in t')
    )
    st.plotly_chart(Prod_)

# Supply Quantity in the second column
supplyquantity_by_item = df_selection.groupby('Item')['Domestic supply quantity'].sum().sort_values(ascending=False) * 1000  # converting the unit

with col2:
    Supply_ = px.bar(
        supplyquantity_by_item,
        x=supplyquantity_by_item.values,
        y=supplyquantity_by_item.index,
        orientation='h',
        title="<b>Supply Quantity by Item</b>",
        color_discrete_sequence=["#0083B8"],
        template="plotly_white"
    )

    Supply_.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title='Total Domestic Supply Quantity in t')
    )
    st.plotly_chart(Supply_)

# Export Quantity in the second column
export_by_item = df_selection.groupby('Item')['Export Quantity'].sum().sort_values(ascending=False) * 1000

with col3:
    Exp_ = px.bar(
        export_by_item,
        x=export_by_item.values,
        y=export_by_item.index,
        orientation='h',
        labels={'x': 'Total Quantity Exported in Tonnes', 'index': 'Item'},
        title="<b>Export by Item</b>",
        color_discrete_sequence=["#0083B8"],
        template="plotly_white"
    )

    Exp_.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title='Export Quantity in t')
    )
    st.plotly_chart(Exp_)

# Placing another markdown
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

# Create initial traces for the selected metric
fig = go.Figure()

# Add traces for the selected metric
for item in yearly_data['Item'].unique():
    item_data = yearly_data[yearly_data['Item'] == item]
    fig.add_trace(go.Scatter(
        x=item_data['Year'],
        y=item_data[metric],
        mode='lines',
        name=f'{item} - {metric}'
    ))

# Update layout
fig.update_layout(
    title=f' {metric} by Meat Type over the years',
    xaxis_title='Year',
    yaxis_title='Total Quantity',
    legend_title='Metrics',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Plotting
st.plotly_chart(fig, use_container_width=True)


# Paths
model_path = 'ANNmodel.h5'
scaler_path = 'scaler.pkl'
item_encoder_path = 'Item_encoder.pkl'
area_encoder_path = 'Area_encoder.pkl'
#loading the ANN model
def load_nn_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"File not found: {model_path}.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
nn_model = load_nn_model(model_path)

# load a pickle file
def load_pickle_file(path):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except FileNotFoundError:
        st.error(f"File not found: {path}.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        st.stop()

# Load the scaler
scaler = load_pickle_file(scaler_path)

# Load the label encoders
item_encoder = load_pickle_file(item_encoder_path)
area_encoder = load_pickle_file(area_encoder_path)

# Streamlit for prediction
st.title("Meat Market Prediction")
st.markdown("""---""")
st.header("Predict Future Values")

# Function to preprocess inputs similar to training data
def preprocess_inputs(inputs):
    try:
        # Normalize 
        inputs = scaler.transform([inputs])[0]
        return inputs
    except Exception as e:
        st.error(f"Error in preprocessing inputs: {e}")
        st.stop()

# Create input fields for user to enter prediction data
if item_encoder and hasattr(item_encoder, 'classes_'):
    item_input = st.selectbox('Item', options=item_encoder.classes_)
else:
    st.error("Error loading item encoder.")
    st.stop()

if area_encoder and hasattr(area_encoder, 'classes_'):
    area_input = st.selectbox('Area', options=area_encoder.classes_)
else:
    st.error("Error loading area encoder.")
    st.stop()

# Retrieve the latest values for the selected country and item
df_filtered = df[(df['Area'] == area_input) & (df['Item'] == item_input)].sort_values(by='Year', ascending=False)

if not df_filtered.empty:
    latest_record = df_filtered.iloc[0]
    last_production = latest_record['Production']
    last_supply = latest_record['Domestic supply quantity']
    last_time = latest_record['Time']
    population_input = latest_record['Population']  # original unit
    land_input = latest_record['Country area']  # original unit
    pastures_input = latest_record['Permanent meadows and pastures']  # original unit
    gdp_input = latest_record['GDP per capita in USD']  # original unit
else:
    st.error("No data available for the selected area and item.")
    st.stop()

# Allow the production and supply values up to 25% higher than the latest values
if last_production > 0:
    production_input = st.slider(
        'Production (in tonnes)',
        min_value=0,
        max_value=int(1.25 * last_production),  #up to 25% more
        value=int(last_production)
    )
else:
    st.warning("No production data available for the selected area and item.")
    production_input = 0

if last_supply > 0:
    supply_input = st.slider(
        'Domestic Supply Quantity (in tonnes)',
        min_value=0,
        max_value=int(1.25 * last_supply),  #  up to 25% more
        value=int(last_supply)
    )
else:
    st.warning("No supply data available for the selected area and item.")
    supply_input = 0

# Use the last time + 1 for predictions
time_input = st.slider(
    'Time (+1 for each future year):',
    min_value=int(last_time + 1),
    max_value=int(last_time + 10),
    value=int(last_time + 1)
)

# Prediction
if st.button('Predict'):
    try:
        if nn_model and scaler and item_encoder and area_encoder:
            item_encoded = int(item_encoder.transform([item_input])[0])#encoding input Item 
            area_encoded = int(area_encoder.transform([area_input])[0])#encoding input Area 
            # list of inputs (import to not consider year)
            inputs = [population_input, land_input, pastures_input, gdp_input, production_input, supply_input, time_input, area_encoded, item_encoded]
            
            # Scale the inputs
            processed_inputs = preprocess_inputs(inputs)
            # Ensure the input is a 2D array
            processed_inputs = np.array([processed_inputs])
            #applying the model 
            prediction = nn_model.predict(processed_inputs)
            
            st.subheader(f'Predicted Export Quantity: {prediction[0][0]:.2f}t')
        else:
            st.error("please check it")
    except Exception as e:
        st.error(f"Error in : {e}")
