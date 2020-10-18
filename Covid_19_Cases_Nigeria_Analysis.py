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
from bs4 import BeautifulSoup
from scrape_html_table import get_data

st.title('Predicting Confirmed Cases of Covid19')

df = get_data()

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


cleany = polish_data(non_time_series_data)


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


if st.checkbox('Display States Data'):
    '', cleany.head()


# def clean_col(name):
#     # print('pretifying the column names')
#     pretify_name = name.strip().lower().replace(" ", "_").replace('/', '_')
#     return pretify_name


# clean_data
second_data = cleany
#affected_column = ['no._of_cases_(lab_confirmed)', 'no._of_cases_(on_admission)', 'no._discharged']


# def polish_data(df):
#     clean_columns = df.rename(columns=clean_col)
#     # for i in clean_columns.columns:
#     #     if i in affected_column:
#     #         clean_columns[i] = clean_columns[i].str.replace(',', '').astype('int64')
#     return clean_columns
#
#
# cleany = polish_data(second_data)


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


covid_ng = load_tm_data()


# format header columns
# clean columns
clean_covid_ng = covid_ng.rename(columns=clean_col)
columns = {}
for col in clean_covid_ng.columns:
    if col == 'abuja(fct)':
        columns['abuja(fct)'] = 'abuja'
    elif col == 'dealth':
        columns['dealth'] = 'deaths'
    elif col == 'dischared_revovered':
        columns['dischared_revovered'] = 'discharged_recovered'
clean_covid_ng.rename(columns=columns, inplace=True)


# Extract geopolitical zone
def extract_features(df):
    df['south_west'] = df['lagos'] + df['ondo'] + df['osun'] + df['oyo'] + df['ekiti'] + df['ogun']
    df['south_south'] = df['edo'] + df['rivers'] + df['delta'] + \
        df['cross_river'] + df['bayelsa'] + df['akwa_ibom']
    df['south_east'] = df['anambra'] + df['imo'] + df['enugu'] + df['abia'] + df['ebonyi']
    df['north_central'] = df['benue'] + df['kogi'] + df['nasarawa'] + \
        df['niger'] + df['plateau'] + df['kwara'] + df['abuja']
    df['north_east'] = df['adamawa'] + df['bauchi'] + \
        df['borno'] + df['gombe'] + df['taraba'] + df['yobe']
    df['north_west'] = df['jigawa'] + df['kaduna'] + df['kano'] + \
        df['katsina'] + df['kebbi'] + df['sokoto'] + df['zamfara']
    return df.head(2)


extract_features(clean_covid_ng)


# extract model data
def m_data(df):
    m_data = df[['total_daily_cases', 'deaths', 'discharged_recovered']]
    return m_data


model_data = m_data(clean_covid_ng)

# extract Geopolitcal zone


def zones(df):
    zone_data = df[['south_west', 'south_south', 'south_east',
                    'north_central', 'north_east', 'north_west']]
    melt_zone = zone_data.melt(value_vars=['south_west', 'south_south', 'south_east', 'north_central', 'north_east', 'north_west'],
                               var_name='geopolitical_zones', value_name='daily_zone_cases')
    group_by_zone = melt_zone.groupby('geopolitical_zones').agg(
        {'daily_zone_cases': 'sum'}).reset_index()
    sort_by_zonal_cases = group_by_zone.sort_values('daily_zone_cases', ascending=True)
    return sort_by_zonal_cases


geopolitical_zone = zones(clean_covid_ng)
if st.checkbox('Show Geopolitical Zones Data'):
    '', geopolitical_zone


# ## Visualize Cases By Geopolitical Zones

geo = px.bar(geopolitical_zone,
             x='daily_zone_cases',
             y='geopolitical_zones',
             hover_name='geopolitical_zones',
             title='Total Confirmed Cases By Geopolitical Zones')

if st.checkbox('Show Confirmed Cases By Geopolitical Zones'):
    fig = px.sunburst(geopolitical_zone,
                      path=['geopolitical_zones'],
                      values='daily_zone_cases')
    st.plotly_chart(geo)
    st.plotly_chart(fig)

# Calculate Case Fatality Rate


def case_fatality(df):
    total_death = df['deaths'].sum()
    total_confirm_cases = df['total_daily_cases'].sum()
    cfr = total_death/total_confirm_cases * 100
    return cfr


cfr = case_fatality(model_data)

st.markdown(f'### The Case Fatality rate In Nigeria is: {round(cfr,2)}%')


# Calculate Mortality Rate/Deaths Per One Million
# def mortality_rate(df):
#     estimated_population = 200000000
#     covid_deaths = df['deaths'].sum()
#     mr = covid_deaths/estimated_population
#     return mr
#
#
# mr = mortality_rate(model_data)
# death_per1million = model_data['deaths'].sum()/1000000
# print(f'The Mortality Rate of Covid19: {mr}, whereas deaths per one million is:{death_per1million}')


# wrangle Data
def monthly_stats(df):
    #make_date_index = df.set_index(data_column)
    monthly_data = df.resample('M').agg(
        {'deaths': 'sum', 'total_daily_cases': 'sum', 'discharged_recovered': 'sum'})
    monthly_data['dates'] = monthly_data.index
    monthly_data['month'] = monthly_data['dates'].dt.month
    return monthly_data.drop('dates', axis=1)


df_month = monthly_stats(model_data)


# Visualize Confirmed Cases By Month
month = px.bar(df_month,
               x='month',
               y='total_daily_cases',
               hover_name='month',
               title='Monthly Confirm Cases')


if st.checkbox('See Confirmed Cases By Month'):
    st.plotly_chart(month)


# get data into shape
nt_needed = ['discharged_recovered', 'deaths', 'total_daily_cases', 'south_west',
             'south_south', 'south_east', 'north_central', 'north_east', 'north_west']


def tidyrc_data(df):
    race_chart_data = df.drop(nt_needed, axis=1)
    clean_rc_data = race_chart_data.cumsum(axis=0)
    return clean_rc_data


clean_rb = tidyrc_data(clean_covid_ng)


# Preapre data
def clean_model_data(df):
    cumulate_data = df.cumsum(axis=0)
    clean_index = cumulate_data.reset_index()
    return clean_index


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
    df = df.diff()
    for _ in range(repeat):
        df = df.rolling(window, min_periods=1, center=True).mean()
    return df.cumsum().reset_index()


smooth_data = smooth(model_data)
# st.write(smooth_data)


log_model_data = clean_model_data(model_data)


# Line graph of confirm cases over time
def line_graph():
    ax = px.line(smooth_data,
                 x='Dates',
                 y='total_daily_cases',
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


#conex = np.array(y)

#
# # Calculate Time of Plateau
# def plateau(confirmed, logistic_params, diff=200):
#     a_, b_, c_, d_ = logistic_params
#     confirm_now = confirmed[-1]
#     confirmed_then = confirmed[-2]
#     days = 0
#     now = x[-1]
#     while confirm_now - confirmed_then > diff:
#         days += 1
#         confirmed_then = confirm_now
#         confirm_now = logistic_model(now + days, a_, b_, c_, d_)
#     return days, confirm_now

#
# days, confirmy = plateau(y, (a_, b_, c_, d_))
#
# print(f"last day's case:{conex[-1] - conex[-2]}")


# carrying cappacity from above logistic growth model

    t_fastest = np.log(a_)/b_

    check_fastest = logistic_model(t_fastest, a_, b_, c_, d_)

    # wrangle dataframe to fit prophet requirement
    def forecast_data(df):
        df['ds'] = df['Dates']
        df['y'] = df['total_daily_cases']
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
