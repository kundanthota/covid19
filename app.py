import numpy as np
import json 
import requests
import pandas as pd
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from scipy.integrate import odeint
from scipy.optimize import minimize,curve_fit

def data_from_github():
    confirmed_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    confirmed_cases = confirmed_cases.drop(["Province/State", "Lat", "Long"], axis =1)
    confirmed_cases = confirmed_cases.groupby(["Country/Region"]).sum()
    recovered_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
    recovered_cases=recovered_cases.drop(["Province/State","Lat", "Long"],axis=1)
    recovered_cases = recovered_cases.groupby(["Country/Region"]).sum()
    deaths=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    deaths=deaths.drop(["Province/State","Lat", "Long"],axis=1)
    deaths = deaths.groupby(["Country/Region"]).sum()

    present_confirmed = confirmed_cases[confirmed_cases.columns[-1]]
    present_recovered = recovered_cases[recovered_cases.columns[-1]]
    present_deaths = deaths[deaths.columns[-1]]
    present_active = present_confirmed - present_recovered - present_deaths
    recovery_Index = np.round(np.divide(present_recovered, present_confirmed),2)
    death_Index = np.round(np.divide(present_deaths, present_confirmed),2)

    data_table_df = pd.DataFrame([present_confirmed.index.values, present_confirmed.values, present_recovered.values,present_deaths.values, present_active.values, recovery_Index, death_Index])
    data_table_df =data_table_df.T
    data_table_df.columns = ['Country', 'Confirmed(C)', 'Recovered(R)', 'Deaths(D)', 'Active(A)', 'RecoveryIndex(R/C)', 'DeathIndex(D/C)']
    data_table_df = data_table_df.sort_values(by = 'Confirmed(C)', ascending = False)
    
    return data_table_df, present_confirmed, present_recovered, present_deaths, present_active, confirmed_cases, recovered_cases, deaths


def get_data_for_map():
    data_table = []
    url="https://corona.lmao.ninja/v2/countries?yesterday&sort"
    data= requests.get(url)
    data=json.loads(data.text)
    for item in data:
        data_table.append([item['countryInfo']['iso3'],item['country'],item['cases'],item['recovered'],item['active'],item['deaths']])
    data = pd.DataFrame(data_table,columns = ['Code','Country', 'Confirmed', 'Recovered', 'Active', 'Deaths'])
    data = data.sort_values(by = 'Confirmed', ascending=False)
    return data

def world_map():
    df = get_data_for_map()
    fig = go.Figure(data=go.Choropleth(
                locations = df['Code'],
                z = df['Confirmed'],
                text = df.Recovered,
                colorscale = 'Blues',
                autocolorscale=False,
                marker_line_color='darkgray',
                marker_line_width=1.5,
                colorbar_title = 'Affected',
                hovertext = df.Deaths,
                hovertemplate =df.Country + "<extra>Confirmed : %{z}<br>Recovered : %{text} <br>Deaths : %{hovertext}</extra>",
    ))

    fig.update_layout(
        height = 1000,
        width = 1200
    )
    return fig

def text_format(data) :
    
    data['Confirmed(C)'] = list(map("{:,}".format, data['Confirmed(C)'].astype(int).values))
    data['Recovered(R)'] = list(map("{:,}".format, data['Recovered(R)'].astype(int).values))
    data['Deaths(D)'] = list(map("{:,}".format, data['Deaths(D)'].astype(int).values))
    data['Active(A)'] = list(map("{:,}".format, data['Active(A)'].astype(int).values))

    return data

def day_wise_data(country_confirmed, country_recovered, country_deaths):
    day_wise_infections = country_confirmed.T
    day_wise_infections = day_wise_infections - day_wise_infections.shift(1)
    day_wise_infections = day_wise_infections.fillna(0)
    day_wise_infections = day_wise_infections.mask(day_wise_infections < 0, 0)
    
    day_wise_recoveries = country_recovered.T
    day_wise_recoveries = day_wise_recoveries - day_wise_recoveries.shift(1)
    day_wise_recoveries = day_wise_recoveries.fillna(0)
    day_wise_recoveries = day_wise_recoveries.mask(day_wise_recoveries < 0, 0)

    day_wise_deaths = country_deaths.T
    day_wise_deaths = day_wise_deaths - day_wise_deaths.shift(1)
    day_wise_deaths = day_wise_deaths.fillna(0)
    day_wise_deaths = day_wise_deaths.mask(day_wise_deaths < 0, 0)

    return  day_wise_infections, day_wise_recoveries, day_wise_deaths

def sir_simulations(country):

    N = 1000000
    betas = []
    gammas = []
    simulations = []

    infections, recoveries = country_confirmed.T, country_recovered.T
    def SIR(y, t, beta, gamma):    
        S = y[0]
        I = y[1]
        R = y[2]
        return -beta*S*I/N, (beta*S*I)/N-(gamma*I), gamma*I

    def fit_odeint(t,beta, gamma):
        return odeint(SIR,(s_0,i_0,r_0), t, args = (beta,gamma))[:,1]
    
    def loss(point, data, s_0, i_0, r_0):
        predict = fit_odeint(t, *point)
        l1 = np.sqrt(np.mean((predict - data)**2))
        return l1
  
    for index in range(len(infections)):
        if index % 7== 0:
            if index+1 <= len(data)-7:
                train = infections[f'{country}'].values[index:index+7]
            else:
                train =  infections[f'{country}'].values[index:]
            i_0 = train[0]
            r_0 = recoveries[f'{country}'].values[index]
            s_0 = N - i_0 - r_0

            if len(train) > 2:
                t = np.arange(len(train))
                params, cerr = curve_fit(fit_odeint,t, train)
                optimal = minimize(loss, params, args=(train, s_0, i_0, r_0))
                beta,gamma = optimal.x
                betas.append(beta)
                gammas.append(gamma)
                
            predict = list(fit_odeint(np.arange(7),beta,gamma))
            simulations.extend(predict)
    i_0 =  infections[f'{country}'].values[-1]
    r_0 =  recoveries[f'{country}'].values[-1]
    s_0 = N - i_0 - r_0 
    future_simulations = list(fit_odeint(np.arange(7), beta, gamma ))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dates, y=simulations,
                        mode='lines+markers',
                        name='Simulated'))
    fig.add_bar(x = dates, y= infections[f'{country}'].values, name = "Actual")
    fig.add_bar(x = future_dates, y= future_simulations, name = "Expected")
    fig.update_layout(height = 800,  xaxis_title="Date",
    yaxis_title="Infections",  hovermode='x unified' )
    return fig, [betas,gammas]

data, present_confirmed, present_recovered, present_deaths, present_active, country_confirmed, country_recovered, country_deaths = data_from_github()
day_wise_infections, day_wise_recoveries, day_wise_deaths = day_wise_data(country_confirmed, country_recovered, country_deaths)
table_data = text_format(data)
start_date = np.array('2020-01-22', dtype=np.datetime64)
dates = start_date + np.arange(len(day_wise_infections.index.values))
future_dates = dates[-1] + np.arange(7)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([

    html.Div( [html.H1("COVID-19 TRACKER", style = {'color' : 'white', 'paddingTop' :'10px', 'fontFamily' : 'arial'})]
    ,style = {'width' : '100%', 'height' : '50px', 'backgroundColor' : '#008080', 'textAlign' : 'center'}),

    html.Div([html.Table([
        html.Tr([
            html.Td([html.P("CONFIRMED",style = {'fontSize' : '20px', 'fontFamily' : 'arial'}), html.P(f"{present_confirmed.sum():,}", style = {'fontSize' : '50px', 'color' : '#999900', 'marginTop' : '-15PX'})]),
            html.Td([html.P("RECOVERED",style = {'fontSize' : '20px', 'fontFamily' : 'arial'}), html.P(f"{present_recovered.sum():,}", style = {'fontSize' : '50px', 'color' : 'green', 'marginTop' : '-15PX' })]),
            html.Td([html.P("DEATHS",style = {'fontSize' : '20px', 'fontFamily' : 'arial'}), html.P(f"{present_deaths.sum():,}", style = {'fontSize' : '50px', 'color' : 'red', 'marginTop' : '-15PX'})]),
            html.Td([html.P('ACTIVE',style = {'fontSize' : '20px', 'fontFamily' : 'arial'}), html.P(f"{present_active.sum():,}", style = {'fontSize' : '50px', 'color' : 'grey', 'marginTop' : '-15PX' })])
        ])
    ],style =  {
        'textAlign' :'center',
        'width' : '80%',
        'margin' : 'auto',
        'backgroundColour' : '#C1C1C1'
    }
    )]),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div([html.Button('GLOBAL DATA' , id = "table_button", n_clicks_timestamp = 1,
        style = {'width' : '300px',
        'backgroundColor': '#008080',
        'border': 'none',
        'color': 'white',
        'padding': '15px 32px',
        'textAlign': 'center',
        'textDecoration': 'none',
        'display': 'inline-block',
        'font-size': '16px',
        'borderRadius' : '15px 15px 0px 0px' 
        })],style = {'float' : 'left', 'width' : '300px'}),

       html.Div([html.Button('GLOBAL HEAT MAP' , id = "map_button", n_clicks_timestamp = 0,
        style = {'width' : '300px',
        'backgroundColor': '#008080',
        'border': 'none',
        'color': 'white',
        'padding': '15px 32px',
        'textAlign': 'center',
        'textDecoration': 'none',
        'display': 'inline-block',
        'font-size': '16px',
        'borderRadius' : '15px 15px 0px 0px' })],
         style = {'marginLeft': '300px', 'width' : '300px'}),
    
    ], style = {'width': '600px', 'margin': '0 auto'}),
    html.Div(id = 'output_initial'),
    html.Br(),
    html.Br(),
    
    html.P(['COUNTRY STATS'], style = {'fontFamily' : 'arial', 'fontSize': '25px', 'textAlign' : 'center'}),

    html.Div([
        
    html.Div(
            dcc.Dropdown(id = 'country_list',
        options=[{'label': each , 'value': each} for each in data['Country'].values ],
        value="India",
        style = {'width' : '400px', 'float' : 'right', 'marginRight' : '100px', 'paddingDown' : '50px' ,'fontSize' : '21px',}
    )),
    html.Div([

        html.Div(dcc.Graph(id = 'infections'), style = {'width' :'48%','height':'600px', 'float': 'left', }),
        html.Div(dcc.Graph(id = 'daily_infections'), style = {'width' :'48%', 'height':'600px', 'float': 'right',})
    ]),
    html.Div([
        html.Div(dcc.Graph(id ='recoveries'), style = {'width' :'48%','height':'600px', 'float': 'left', }),
        html.Div(dcc.Graph(id = 'daily_recoveries'), style = {'width' :'48%', 'height':'600px', 'float': 'right', })
    ]),
    html.Div([
        html.Div(dcc.Graph(id = 'deaths'), style = {'width' :'48%','height':'600px', 'float': 'left', }),
        html.Div(dcc.Graph(id = 'daily_deaths'), style = {'width' :'48%', 'height':'600px', 'float': 'right', })
    ]),

    html.Div([
    html.Div(
            dcc.Dropdown(id = 'sir_list',
        options=[{'label': each , 'value': each} for each in data['Country'].values ],
        value="India",
        style = {'width' : '400px', 'marginRight' : '100px', 'paddingDown' : '50px' ,'fontSize' : '21px',}
    ))], style = {'width' : '100%'}),
    html.Br(),
    html.Br(),
    html.P(['SIR PREDICTIONS'], style = {'fontFamily' : 'arial', 'fontSize': '25px', 'textAlign' : 'center'}),
    html.Div([
        html.Div([dcc.Graph(id = 'sir_simulations')],style = {'width' : '80%', 'float' : 'center', 'margin' : 'auto'}),
        html.Div([dcc.Graph(id = 'sir_parameters')],style = {'width' : '80%', 'float' : 'center', 'margin' : 'auto'}),

    ], style = {'width' : '100%'}),

    html.P(['DIFFERENT COUNTRY STATS COMPARISION'],  style = {'fontFamily' : 'arial', 'fontSize': '25px', 'textAlign' : 'center'}),

    html.Table([
            html.Tr([
                html.Td(dcc.Dropdown(id = 'comparision_countries_dd',
        options=[{'label': each, 'value': each} for each in data['Country'].values  ],
        value=["US", "India"],
        multi = True
    )),

                html.Td(dcc.RadioItems(
               id = 'comparision_countries_radio',
               options = [ {'label': 'Confirmed', 'value': 'Confirmed'},
                {'label': 'Recovered', 'value': 'Recovered'},
                {'label': 'Deaths', 'value': 'Deaths'}],
                value='Confirmed',
    ))
            ])
        ],style = {'width': '100%','textAlign':'center'}),
    html.Div(dcc.Graph(id = 'comparision_output'),style = {'width' : '80%', 'float' : 'center', 'margin' : 'auto'}),
    
    

    ], style = {'margin': '0 auto', 'width' : '100%'})

])

table_layout = html.Div([
    
    
    dash_table.DataTable(
    id = 'global_data',
    data=table_data.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in table_data.columns],
    page_action='none',
    style_table={'height': '800px','width' : '1300px', 'overflowY': 'auto', 'fontSize' : '21px', 'marginRight': 'auto','marginLeft': 'auto', 'fontFamily' : 'arial'},
    style_cell={
         'width': '180px', 'maxWidth': '400px','textAlign': 'left'
    },
    style_header={
        'backgroundColor': 'Black',
        'fontWeight': 'bold',
        'color' :  'white'
    },
    )])

heat_map_layout = html.Div([
    html.Div(dcc.Graph( figure = world_map(),  id = 'output_map'), style = {'width' : '1200px', "margin" : 'auto'})
])

@app.callback(
    [Output('output_initial', 'children')],
    [Input('map_button', 'n_clicks_timestamp'),
    Input('table_button', 'n_clicks_timestamp')])

def return_layout(map_button, table_button) :
    if table_button > map_button :
        return [table_layout]
    else:
        return [heat_map_layout]

@app.callback(
    [Output('infections', 'figure'),
    Output('daily_infections', 'figure'),
    Output('recoveries', 'figure'),
    Output('daily_recoveries', 'figure'),
    Output('deaths','figure'),
    Output('daily_deaths','figure')],
    
    [Input('country_list', 'value')])

def return_figures(country) : 

    confirmed = px.line(x = dates ,y = country_confirmed.loc[f'{country}'] , labels = { "x" : "Date", "y" : "Confirmed cases"}, height = 600)
    confirmed.update_layout(title_text = " Confirmed cases" ,title_x=0.5 ,hovermode='x unified')
    confirmed.update_traces(line = dict(color='#999900'))

    recovered = px.line(x = dates ,y = country_recovered.loc[f'{country}'] , labels = { "x" : "Date", "y" : "Recovered cases"}, height = 600)
    recovered.update_layout(title_text = " Recovered cases" ,title_x=0.5, hovermode='x unified')
    recovered.update_traces(line = dict(color='green'))

    deaths = px.line(x = dates ,y = country_deaths.loc[f'{country}'] , labels = { "x" : "Date", "y" : "Deaths"} ,height = 600)
    deaths.update_layout(title_text = " Deaths" ,title_x=0.5, hovermode='x unified')
    deaths.update_traces(line = dict(color='red'))

    day_wise_confirmed = px.bar(x = dates ,y = day_wise_infections[f'{country}'] , labels = { "x" : "Date", "y" : "Infections"}, height = 600)
    day_wise_confirmed.add_trace(go.Scatter(x = dates ,y = day_wise_infections[f'{country}'], mode = 'lines',hoverinfo = 'skip'))
    day_wise_confirmed.update_layout(title_text = " Daily Infections" ,title_x=0.5, hovermode='x unified', showlegend = False)
    day_wise_confirmed.update_traces(marker_color = '#999900')

  
    day_wise_recovered = px.bar(x = dates ,y = day_wise_recoveries[f'{country}'] , labels = { "x" : "Date", "y" : "Recoveries"}, height = 600)
    day_wise_recovered.add_trace(go.Scatter(x = dates, y =  day_wise_recoveries[f'{country}'], mode = 'lines', hoverinfo = 'skip'))
    day_wise_recovered.update_layout(title_text = " Daily Recoveries" ,title_x=0.5, hovermode='x unified', showlegend = False )
    day_wise_recovered.update_traces(marker_color = 'green')

    day_wise_death_cases = px.bar(x = dates ,y = day_wise_deaths[f'{country}'] , labels = { "x" : "Date", "y" : "Deaths"}, height = 600)
    day_wise_death_cases.add_trace(go.Scatter(x = dates, y =  day_wise_deaths[f'{country}'], mode = 'lines', hoverinfo = 'skip'))
    day_wise_death_cases.update_layout(title_text = " Daily Deaths" ,title_x=0.5, hovermode='x unified', showlegend = False)
    day_wise_death_cases.update_traces(marker_color = 'red')

    return confirmed, day_wise_confirmed, recovered, day_wise_recovered, deaths, day_wise_death_cases

@app.callback([Output('sir_simulations', 'figure'),
Output('sir_parameters', 'figure')],
[Input('sir_list', 'value')])

def sir_figure(country):

    sir, parameters = sir_simulations(country)

    parameters_figure = go.Figure()
    parameters_figure.add_trace(go.Scatter(x = np.arange(len(parameters[0])), y = parameters[0],
                        mode='lines+markers',
                        name='Beta'))
    parameters_figure.add_trace(go.Scatter(x = np.arange(len(parameters[1])), y = parameters[1],
                        mode='lines+markers',
                        name='Gamma'))
    parameters_figure.update_layout(height = 800,  xaxis_title="Week",
    yaxis_title="parameters",  hovermode='x unified', title = 'Parameters change w.r.t Time', title_x = 0.5 )

    return sir, parameters_figure

@app.callback(
    Output('comparision_output','figure'),
    [Input('comparision_countries_dd', 'value'),
    Input('comparision_countries_radio','value')]
)
def countries_comparision_charts(comparision_countries_dd, comparision_countries_radio) :

    confirmed, recovered, deaths = country_confirmed.T, country_recovered.T, country_deaths.T

    if comparision_countries_radio == 'Confirmed':
        fig = go.Figure()
        for each in comparision_countries_dd:
            fig.add_traces( go.Scatter(x= dates, y = confirmed[f'{each}'], mode='lines+markers', name = each))
        fig.update_layout(
         height = 900,
        hovermode='x unified',
        xaxis_title="Date",
    yaxis_title="Infections"
        )
        return fig        

    elif comparision_countries_radio == 'Recovered' :
        fig = go.Figure()
        for each in comparision_countries_dd:
            fig.add_traces( go.Scatter(x= dates, y = recovered[f'{each}'], mode='lines+markers', name = each))
        fig.update_layout(
         height = 800,
          hovermode='x unified',
          xaxis_title="Date",
    yaxis_title="Recoveries"
        )
        return fig  

    else:
        fig = go.Figure()
        for each in comparision_countries_dd:
            fig.add_traces( go.Scatter(x= dates, y = deaths[f'{each}'], mode='lines+markers',name = each))
        fig.update_layout(
         height = 800,
          hovermode='x unified',
          xaxis_title="Date",
    yaxis_title="Deaths"
        )
        return fig 


#application tab title 
app.title = 'COVID-19 Dashboard'  


#application favicion 
app._favicon = "favicon.ico"

if __name__ == '__main__':
    app.run_server(debug=True)
