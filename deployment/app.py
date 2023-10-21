import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from model import PredictionModel

data_path = './data/data_daily.csv'
damped_model_path = './models/damped_hw_model.pkl'
model_path = './models/hw_model.pkl'

months = {1: [31, "2022-01-31"], 2: [59, "2022-02-28"], 3: [90, "2022-03-31"], 4: [120, "2022-04-30"], 5: [151, "2022-05-31"], 6: [181, "2022-06-30"], 7: [212, "2022-07-31"], 8: [243, "2022-08-31"], 9: [273, "2022-09-30"], 10: [304, "2022-010-31"], 11: [334, "2022-11-30"], 12: [365, "2022-12-31"]}

damped_model = PredictionModel(damped_model_path, data_path)
model = PredictionModel(model_path, data_path)

def plot_daily_chart(result, month):
    fig, axes = plt.subplots(2,2, figsize=(15,15))

    axes[0][0].plot(result.ds, result.y, label='Train')
    axes[0][0].plot(result.ds, result.yhat, label='yhat')
    axes[0][0].plot(result.ds, result.upper_bound, label='Upper Bound')
    axes[0][0].plot(result.ds, result.lower_bound, label='Lower Bound')
    axes[0][0].set_xticklabels(result.ds.dt.date, rotation=45)
    axes[0][0].legend(loc='best')
    axes[0][0].set_title("Daily Forecast")
    axes[0][0].set_xlabel("Date")
    axes[0][0].set_ylabel("Receipt Count")
    axes[0][0].grid(True)

    monthly_result = result.drop('ds', axis=1).groupby(['year', 'month']).sum().reset_index()
    monthly_result['monthly'] = monthly_result['year'].astype(str) + '-' + monthly_result['month'].astype(str)
    monthly_result['y'].iloc[-int(month):] = np.nan
    monthly_result['yhat'].iloc[:12] = np.nan
    monthly_result['upper_bound'].iloc[:12] = np.nan
    monthly_result['lower_bound'].iloc[:12] = np.nan

    axes[1][0].plot(monthly_result.monthly, monthly_result.y, label='Train')
    axes[1][0].plot(monthly_result.monthly, monthly_result.yhat, label='yhat')
    axes[1][0].plot(monthly_result.monthly, monthly_result.upper_bound, label='Upper Bound')
    axes[1][0].plot(monthly_result.monthly, monthly_result.lower_bound, label='Lower Bound')
    axes[1][0].set_xticklabels(monthly_result.monthly, rotation=45)
    axes[1][0].legend(loc='best')
    axes[1][0].set_title("Monthly Forecast")
    axes[1][0].set_xlabel("Date")
    axes[1][0].set_ylabel("Receipt Count")
    axes[1][0].grid(True)
    
    axes[0][1].plot(result.ds, result.y, label='Train')
    axes[0][1].plot(result.ds, result.damped_yhat, label='yhat')
    axes[0][1].plot(result.ds, result.damped_upper_bound, label='Upper Bound')
    axes[0][1].plot(result.ds, result.damped_lower_bound, label='Lower Bound')
    axes[0][1].set_xticklabels(result.ds.dt.date, rotation=45)
    axes[0][1].legend(loc='best')
    axes[0][1].set_title("Damped Model - Daily Forecast")
    axes[0][1].set_xlabel("Date")
    axes[0][1].set_ylabel("Receipt Count")
    axes[0][1].grid(True)

    monthly_result['damped_yhat'].iloc[:12] = np.nan
    monthly_result['damped_upper_bound'].iloc[:12] = np.nan
    monthly_result['damped_lower_bound'].iloc[:12] = np.nan

    axes[1][1].plot(monthly_result.monthly, monthly_result.y, label='Train')
    axes[1][1].plot(monthly_result.monthly, monthly_result.damped_yhat, label='yhat')
    axes[1][1].plot(monthly_result.monthly, monthly_result.damped_upper_bound, label='Upper Bound')
    axes[1][1].plot(monthly_result.monthly, monthly_result.damped_lower_bound, label='Lower Bound')
    axes[1][1].set_xticklabels(monthly_result.monthly, rotation=45)
    axes[1][1].legend(loc='best')
    axes[1][1].set_title("Damped Model - Monthly Forecast")
    axes[1][1].set_xlabel("Date")
    axes[1][1].set_ylabel("Receipt Count")
    axes[1][1].grid(True)

    fig.tight_layout()
    
    return fig


def cal_result(alpha, month):
    alpha = float(alpha)
    period = months[int(month)][0]
    train_data_len = model.df.shape[0]
    yhat, intervals = model.predict(period, alpha)
    damped_yhat, damped_intervals = damped_model.predict(period, alpha)

    START = "2021-01-01"
    END = months[int(month)][1]
    
    result = pd.DataFrame({
        'ds': pd.date_range(START, END, freq='D'),
        'y': list(model.df.receipt_count.values) + [np.nan] * period, 
        'yhat': [np.nan] * train_data_len + list(yhat),
        'upper_bound': [np.nan] * train_data_len + list(intervals[1]),
        'lower_bound': [np.nan] * train_data_len + list(intervals[0]),
        'damped_yhat': [np.nan] * train_data_len + list(damped_yhat),
        'damped_upper_bound': [np.nan] * train_data_len + list(damped_intervals[1]),
        'damped_lower_bound': [np.nan] * train_data_len + list(damped_intervals[0])
    })

    result['year'] = result.ds.dt.year
    result['month'] = result.ds.dt.month

    daily_chart = plot_daily_chart(result, month)
    
    return daily_chart


with gr.Blocks() as demo:
    with gr.Row():
        alpha = gr.Dropdown(choices=["0.1", "0.05", "0.01"], label="Significance level", info="The significance level for the confidence intervals.", value="0.05")
        month = gr.Dropdown(choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], label="Forecast period", info="The number of months to forecast.", value="12")
    
    submit = gr.Button("Submit")
    
    daily = gr.Plot()

    submit.click(fn=cal_result, inputs=[alpha, month], outputs=daily)


demo.launch(server_name="0.0.0.0", server_port=7860)