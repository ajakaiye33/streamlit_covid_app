#!/usr/bin/env python
# coding: utf-8


# import modules and packages
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import sys
import json
from bs4 import BeautifulSoup
from scrape_html_table import get_data

st.title('Predicting Confirmed Cases of Covid19')

df = get_data()
global_data = 'https://pomber.github.io/covid19/timeseries.json'

time_series_data = 'https://raw.githubusercontent.com/ajakaiye33/covid19Naija/master/covid19Naija/data/Records_covid19.csv'
non_time_series_data = df


def clean_col(name):
    # print('pretifying the column names')
    pretify_name = name.strip().lower().replace(" ", "_").replace('/', '_')
    return pretify_name


affected_column = ['no._of_cases_(lab_confirmed)', 'no._of_cases_(on_admission)', 'no._discharged']


def polish_data(df):
    clean_columns = df.rename(columns=clean_col)
    for i in clean_columns.columns:
        if i in affected_column:
            clean_columns[i] = clean_columns[i].str.replace(',', '').astype('int64')
    return clean_columns
# data from json


def naija_cases(url=global_data):
    time_series = str(url)
    response = requests.get(time_series)

    # function to check status of webpage
    def status_check(r):
        if r.status_code == 200:
            return 1
        else:
            return -1

    def encoding_check(r):
        return (r.encoding)

    def decode_content(r, encoding):
        return (r.content.decode(encoding))
    status = status_check(response)
    if status == 1:
        contents = decode_content(response, encoding_check(response))
    else:
        print('Sorry could not reach the web page!')
        return -1
    # load into pandas
    str_data = json.loads(contents)
    isolate_nig = str_data['Nigeria']
    pand_data = pd.DataFrame(isolate_nig)
    return pand_data


naija_data = naija_cases()
naija_data['date'] = pd.to_datetime(naija_data['date'])


# Loading Data
@st.cache()
def load_tm_data():
        # print('f Loading data from {filename} ...')
    df = pd.read_csv(time_series_data, parse_dates=['Dates'], index_col='Dates')
    return df


# load data
@st.cache()
def non_tm_data():
    df = get_data()
    return df

# cached data


@st.cache()
def naija_json():
    df = naija_cases()
    return df


loaded_tm = non_tm_data()
cleany = polish_data(loaded_tm)


if st.checkbox('Display States Data'):
    '', cleany.head()

if st.checkbox('Display Timeseries Data'):
    '', naija_data.tail()


# clean_data
second_data = cleany


def states_stat(df, st_col, st_death):
    death_by_state = df[[st_col, st_death]].sort_values(st_death, axis=0, ascending=False)
    return death_by_state


death_by_states = states_stat(second_data, 'states_affected', 'no._of_deaths')

# Visualize Deaths By States
ax = px.bar(death_by_states,
            x='states_affected',
            y='no._of_deaths',
            hover_name='states_affected',
            title='Deaths By States')
#
#
if st.checkbox('Show Deaths By States'):
    st.plotly_chart(ax)


# wrangle Data
def recov_ratio(df, st_col, recov_col, conf_cases, new_col):
    df[new_col] = df[recov_col] / df[conf_cases]
    recovey_ratio = df[[st_col, new_col]].sort_values(by=new_col, ascending=False)
    return recovey_ratio


state_recov_ratio = recov_ratio(second_data, 'states_affected', 'no._discharged',
                                'no._of_cases_(lab_confirmed)', 'recov_ratio')


# Visualize Rate pf Recoveries
rcov = px.bar(state_recov_ratio,
              x='states_affected',
              y='recov_ratio',
              hover_name='states_affected',
              title='Recovery Rate By States')

if st.checkbox('See Recoveries/Discharged Rates By States'):
    st.plotly_chart(rcov)
    st.markdown(
        'Wow, you have a high chance of recovery from the disease if you are in any of those states from the left')


# Wrangle Data
discharged_by_states = states_stat(second_data, 'states_affected', 'no._discharged')


# Visualize Discharge By States
discharge = px.bar(discharged_by_states,
                   x='states_affected',
                   y='no._discharged',
                   hover_name='states_affected',
                   title='Discharged By States')

if st.checkbox(' Show Patients Dischage By States'):
    st.plotly_chart(discharge)

# Wrangle Data
confirmed_cases_states = states_stat(second_data, 'states_affected', 'no._of_cases_(lab_confirmed)')


# Visualize Confirmed Cases By States
conf_cases = px.bar(confirmed_cases_states,
                    x='states_affected',
                    y='no._of_cases_(lab_confirmed)',
                    hover_name='states_affected',
                    title='Confirmed Cases By States')

if st.checkbox(' Show Confirmed Cases By States'):
    st.plotly_chart(conf_cases)

second_data = second_data.pivot(columns='states_affected',
                                values='no._of_cases_(lab_confirmed)').fillna(0)


# Extract geopolitical zone


def extract_zones(df):
    df['south_west'] = df['Lagos'] + df['Ondo'] + df['Osun'] + df['Oyo'] + \
        df['Ekiti'] + df['Ogun']
    df['north_west'] = df['Jigawa'] + df['Kaduna'] + df['Kano'] + \
        df['Katsina'] + df['Kebbi'] + df['Zamfara'] + df['Sokoto']
    df['south_south'] = df['Edo'] + df['Rivers'] + df['Delta'] + \
        df['Cross River'] + df['Bayelsa'] + df['Akwa Ibom']
    df['south_east'] = df['Anambra'] + df['Imo'] + df['Enugu'] + df['Abia'] + \
        df['Ebonyi']
    df['north_central'] = df['Benue'] + df['Kogi'] + df['Nasarawa'] + \
        df['Niger'] + df['Plateau'] + df['Kwara'] + df['FCT']
    df['north_east'] = df['Adamawa'] + df['Bauchi'] + df['Borno'] + df['Gombe'] + \
        df['Taraba'] + df['Yobe']

    df = df[['south_south', 'south_west', 'south_east', 'north_central', 'north_west', 'north_east']]
    df = df.melt(value_vars=['south_west', 'south_south', 'south_east', 'north_central', 'north_east',
                             'north_west'], var_name='geopolitical_zones', value_name='confirmed_zone_cases')
    df = df.groupby('geopolitical_zones').agg({'confirmed_zone_cases': 'sum'}).reset_index()
    df = df.sort_values('confirmed_zone_cases', ascending=False)
    return df


geopolitical_zone = extract_zones(second_data)
if st.checkbox('Show Geopolitical Zones Data'):
    '', geopolitical_zone


# ## Visualize Cases By Geopolitical Zones

geo = px.bar(geopolitical_zone,
             x='confirmed_zone_cases',
             y='geopolitical_zones',
             hover_name='geopolitical_zones',
             title='Total Confirmed Cases By Geopolitical Zones')

if st.checkbox('Show Confirmed Cases By Geopolitical Zones'):
    fig = px.sunburst(geopolitical_zone,
                      path=['geopolitical_zones'],
                      values='confirmed_zone_cases')
    st.plotly_chart(geo)
    st.plotly_chart(fig)

# Calculate Case Fatality Rate


def case_fatality(df):
    df = df.set_index('date')
    df = df.diff()
    total_death = df['deaths'].sum()
    total_confirm_cases = df['confirmed'].sum()
    cfr = total_death/total_confirm_cases * 100
    return cfr


cfr = case_fatality(naija_data)

st.markdown(f'### The Case Fatality rate In Nigeria is: {round(cfr,2)}%')


# wrangle Data
def monthly_stats(df):
    df = df.set_index('date')
    df = df.diff()
    #make_date_index = df.set_index(data_column)
    monthly_data = df.resample('M').agg(
        {'deaths': 'sum', 'confirmed': 'sum', 'recovered': 'sum'})
    monthly_data['date'] = monthly_data.index
    monthly_data['month'] = monthly_data['date'].dt.date
    return monthly_data.drop('date', axis=1)


df_month = monthly_stats(naija_data)


# Visualize Confirmed Cases By Month
month = px.bar(df_month,
               x='month',
               y='confirmed',
               hover_name='month',
               title='Monthly Confirm Cases')


if st.checkbox('See Confirmed Cases By Month'):
    st.plotly_chart(month)


# Preapre data
def clean_model_data(df):
    # cumulate_data = df
    # clean_index = cumulate_data.reset_index()
    return df


def smooth(df, window=5, repeat=10):
    """
    Smooth data using repeated moving average

    Parameters
    ----------
    df : DataFrame

    window : integer window size

    repeat : integer number of repeats

    Returns
    -------
    DataFrame
    """
    df = df.set_index('date')
    df = df.diff()
    for _ in range(repeat):
        df = df.rolling(window, min_periods=1, center=True).mean()
    return df.cumsum().reset_index()


smooth_data = smooth(naija_data)
# st.write(smooth_data)


log_model_data = clean_model_data(naija_data)


# Line graph of confirm cases over time
def line_graph():
    ax = px.line(smooth_data,
                 x='date',
                 y='confirmed',
                 title='Line graph of Confirmed Daily Cases Over Time')
    st.plotly_chart(ax)


if st.checkbox('See Forecast of Confirmed Cases, 30 days from today(For better result, check all preceeding check boxes above)'):
    line_graph()

    # Build Logistic Model
    def logistic_model(x, a, b, c, d):
        return a / (1 + np.exp(-c * (x - d))) + b

    def build_data(df):
        df['time_stamp'] = df.index
        return df

    build_model = build_data(log_model_data)
    # st.write(build_model.head())

    # extract x(days) & y(cases) from dataframe

    x = list(build_model.iloc[:, 4])
    y = list(build_model.iloc[:, 1])
    # randomly initialize a,b,c,d
    p0 = np.random.exponential(size=4)

    # set upper and lower bounds a,b,class
    bounds = (0, [10000000., 2., 100000000., 100000000.])
    (a_, b_, c_, d_), cov = curve_fit(logistic_model, x, y, bounds=bounds, p0=p0)

    t_fastest = np.log(a_)/b_

    check_fastest = logistic_model(t_fastest, a_, b_, c_, d_)

    # wrangle dataframe to fit prophet requirement
    def forecast_data(df):
        df['ds'] = df['date']
        df['y'] = df['confirmed']
        df['cap'] = check_fastest
        prof_df = df[['ds', 'y', 'cap']]
        return prof_df

    prophet_data = forecast_data(build_model)
    # st.write(prophet_data.tail())

# Build Prophet Model

    m = Prophet(growth='logistic')
    m.fit(prophet_data)
    future = m.make_future_dataframe(periods=30)

    future['cap'] = prophet_data['cap'].iloc[0]
    forecast = m.predict(future)

    # Visualize Prophet Model
    # st.write(forecast.tail())

    lowyhat = forecast.iloc[-1, 3]
    upperyhat = forecast.iloc[-1, 4]
    st.markdown(
        f'### The confirmed cases in Nigeria will be in the range of {round(lowyhat,2)} and {round(upperyhat,2)} 30 days from today ')

    fig = m.plot(forecast)
    st.write(fig)

    fig2 = m.plot_components(forecast)
    st.write(fig2)
