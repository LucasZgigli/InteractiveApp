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

#line plot by the years selected
yearly_data = df_selection.groupby(['Year', 'Item']).agg({
    'Production': 'sum',
    'Export Quantity': 'sum',
    'Domestic supply quantity': 'sum'
}).reset_index()

yearly_data['Production'] *= 1000
yearly_data['Export Quantity'] *= 1000
yearly_data['Domestic supply quantity'] *= 1000
yearly_data.sort_values('Year', inplace=True)
                       
fig = px.line(
    yearly_data,
    x='Year',
    y=['Production', 'Export Quantity', 'Domestic supply quantity'],  # Plotting both metrics
    labels={'value': 'Quantity', 'variable': 'Metric'},
    title='Annual Sum of Production, Export Quantities, and Domestic Supply Quantity'
)


# Updating layout
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Total Quantity',
    legend_title='Metrics',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Plotting
st.plotly_chart(fig, use_container_width=True)                
                       


