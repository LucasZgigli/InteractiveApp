#importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.simplefilter('ignore')
!pip install requirements.txt
# Load data
df = pd.read_csv('Meat_market.csv', encoding='utf-8')

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
df_selection = df.query('Year >= @year_range[0] and Year <= @year_range[1] and Area == @selected_area')
df_selection_sorted = df_selection.sort_values(by='Year', ascending=False)

# Main page
st.title('European Meat Market')
Country = df_selection['Area'].iloc[0]
Population = int(df_selection_sorted['Population'].iloc[0]) * 1000
Land = int(df_selection_sorted['Country area'].iloc[0] * 1000)
Pastures = int(df_selection_sorted['Permanent meadows and pastures'].iloc[0] * 1000)
Total_export = int(df_selection['Export Quantity'].sum() * 1000)
Total_Production = int(df_selection['Production'].sum() * 1000)
Total_SupplyQuantity = int(df_selection['Domestic supply quantity'].sum() * 1000)
most_recent_gdp_per_capita = int(df_selection_sorted['GDP per capita in USD'].iloc[0])
GDP = most_recent_gdp_per_capita

# Display statistics
st.markdown("""---""")
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.header(f'{Country}')
    st.subheader(f'Population: {Population:,} inh')
    st.subheader(f'Land Area: {Land:,} Ha')

with middle_column:
    st.header(f'')
    st.subheader(f'Total Production: {Total_Production:,} t')
    st.subheader(f'Total Export Quantity: {Total_export:,} t')

with right_column:
    st.header(f'')
    st.subheader(f'Total Supply Quantity: {Total_SupplyQuantity:,} t')
    st.subheader(f'GDP: {GDP:,} USD')

st.markdown("""---""")

# Plotting
col1, col2, col3 = st.columns(3)
production_by_item = df_selection.groupby('Item')['Production'].sum().sort_values(ascending=False) * 1000
supplyquantity_by_item = df_selection.groupby('Item')['Domestic supply quantity'].sum().sort_values(ascending=False) * 1000
export_by_item = df_selection.groupby('Item')['Export Quantity'].sum().sort_values(ascending=False) * 1000

with col1:
    Prod_ = px.bar(production_by_item, x=production_by_item.values, y=production_by_item.index, orientation='h',
                   title="<b>Production by Item</b>", color_discrete_sequence=["#0083B8"], template="plotly_white")
    Prod_.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, title='Total Production in t'))
    st.plotly_chart(Prod_)

with col2:
    Supply_ = px.bar(supplyquantity_by_item, x=supplyquantity_by_item.values, y=supplyquantity_by_item.index,
                     orientation='h', title="<b>Supply Quantity by Item</b>", color_discrete_sequence=["#0083B8"],
                     template="plotly_white")
    Supply_.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, title='Total Domestic Supply Quantity in t'))
    st.plotly_chart(Supply_)

with col3:
    Exp_ = px.bar(export_by_item, x=export_by_item.values, y=export_by_item.index, orientation='h',
                  labels={'x': 'Total Quantity Exported in Tonnes', 'index': 'Item'}, title="<b>Export by Item</b>",
                  color_discrete_sequence=["#0083B8"], template="plotly_white")
    Exp_.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, title='Export Quantity in t'))
    st.plotly_chart(Exp_)

st.markdown("""---""")

# Line plot by year and item
yearly_data = df_selection.groupby(['Year', 'Item']).agg({
    'Production': 'sum',
    'Export Quantity': 'sum',
    'Domestic supply quantity': 'sum'
}).reset_index()
yearly_data[['Production', 'Export Quantity', 'Domestic supply quantity']] *= 1000
yearly_data.sort_values('Year', inplace=True)

fig = go.Figure()
for item in yearly_data['Item'].unique():
    item_data = yearly_data[yearly_data['Item'] == item]
    fig.add_trace(go.Scatter(x=item_data['Year'], y=item_data[metric], mode='lines', name=f'{item} - {metric}'))

fig.update_layout(
    title=f'{metric} by Meat Type over the years',
    xaxis_title='Year',
    yaxis_title='Total Quantity',
    legend_title='Metrics',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig, use_container_width=True)

# Load models and scalers
model_path = 'ANNmodel.h5'
scaler_path = 'scaler.pkl'
item_encoder_path = 'Item_encoder.pkl'
area_encoder_path = 'Area_encoder.pkl'

def load_nn_model(model_path):
    try:
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Recompile the model
        return model
    except FileNotFoundError:
        st.error(f"File not found: {model_path}.")
    except Exception as e:
        st.error