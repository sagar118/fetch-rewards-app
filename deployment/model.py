import pickle
import pandas as pd
import numpy as np

class PredictionModel:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.load_model()
        self.load_data()
    
    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df.columns = ['date', 'receipt_count']

    def predict(self, period, alpha=0.05):
        yhat = self.model.forecast(period)
        intervals = self.compute_confidence_intervals(alpha, yhat)
        return yhat, intervals

    def compute_confidence_intervals(self, alpha, yhat):
        residuals = self.df.receipt_count - self.model.fittedvalues
        z = abs(np.percentile(np.random.standard_normal(10000), [100 * alpha/2, 100 * (1 - alpha/2)]))
        std_residual = residuals.std()
        interval_upper = yhat + z[1] * std_residual * np.sqrt(1 + 1/len(self.df))
        interval_lower = yhat - z[1] * std_residual * np.sqrt(1 + 1/len(self.df))
        return [interval_lower, interval_upper]
