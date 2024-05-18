#importing libraries
import pandas as pd 
import numpy as np
import seaborn as sns 
import os 
import numpy as np
import warnings 
warnings.simplefilter('ignore')
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
Land = int(df_selection['Land area'].mean()*1000) # same as land area multiplied by 1000 hectars
Pastures = int(df_selection['Permanent meadows and pastures'].mean()*1000)
Total_export = int(df_selection['Export Quantity'].sum()*1000)
Total_Production = int(df_selection['Production'].sum()*1000)
Total_SupplyQuantity = int(df_selection['Domestic supply quantity'].sum()*1000)
GDP = int(df_selection['GDP in Million USD'].mean())
          
#Splitting the header into 3 columns
left_column, middle_column, right_column = st.columns(3)
with left_column:
    
    st.header(f'{Country}')
    st.subheader(f'Average Population: {Population:,} inh')
    st.subheader(f'Average Land Area: {Land:,} Ha')
                       
with middle_column:
    st.subheader(f'Total Production: {Total_Production:,} t') 
    st.subheader(f'Total Export Quantity: {Total_export:,} t')
    st.subheader(f'Total Supply Quantity: {Total_SupplyQuantity:,} t')
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


st.markdown("""---""")
st.header("Predict Future Values")
population_input = st.number_input('Population (in thousands)', min_value=0)
land_input = st.number_input('Land Area (in hectares)', min_value=0)
pastures_input = st.number_input('Permanent Meadows and Pastures (in hectares)', min_value=0)
export_input = st.number_input('Export Quantity (in tonnes)', min_value=0)
production_input = st.number_input('Production (in tonnes)', min_value=0)
supply_input = st.number_input('Domestic Supply Quantity (in tonnes)', min_value=0)
gdp_input = st.number_input