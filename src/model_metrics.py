from numba import njit
import numpy as np
import torch as tc
import torch.nn as tcn
from sklearn.model_selection import KFold

class CustomMetrics:    
    '''@staticmethod
    def calculate_all_metrics(y_true, y_pred):
        metrics = {}
        
        # Convert inputs to tensors if they aren't already
        y_true = tc.tensor(y_true) if not isinstance(y_true, tc.Tensor) else y_true
        y_pred = tc.tensor(y_pred) if not isinstance(y_pred, tc.Tensor) else y_pred
        
        # Ensure same shape
        y_true = y_true.float().view(-1)
        y_pred = y_pred.float().view(-1)
        
        # Basic metrics
        metrics['mse'] = tcn.functional.mse_loss(y_pred, y_true).item()
        metrics['rmse'] = tc.sqrt(tc.tensor(metrics['mse'])).item()
        metrics['mae'] = tcn.functional.l1_loss(y_pred, y_true).item()
        
        # Improved Gaussian NLL calculation
        variance = tc.var(y_pred - y_true)
        if variance > 0:  # Avoid log(0)
            metrics['gaussian_nll'] = (0.5 * tc.log(2 * tc.pi * variance) + 
                                    (y_pred - y_true) ** 2 / (2 * variance)).mean().item()
        else:
            metrics['gaussian_nll'] = float('inf')
        
        # Improved Poisson NLL calculation
        eps = 1e-8  # Small constant to avoid log(0)
        y_pred_positive = tc.clamp(y_pred, min=eps)  # Ensure positive values
        metrics['poisson_nll'] = (y_pred_positive - y_true * tc.log(y_pred_positive)).mean().item()
        
        # R-squared calculation
        try:
            ss_tot = tc.sum((y_true - tc.mean(y_true)) ** 2)
            ss_res = tc.sum((y_true - y_pred) ** 2)
            metrics['r2'] = (1 - ss_res / (ss_tot + 1e-8)).item()
        except Exception as e:
            print(f"Error calculating R2: {e}")
            metrics['r2'] = 0.0
        
        return metrics'''

    @staticmethod
    def calculate_all_metrics(y_true, y_pred):
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        metrics = {}

        # Basic metrics
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(mse)

        # Gaussian NLL
        residual = y_pred - y_true
        variance = np.var(residual)
        if variance > 0:
            gaussian_nll = np.mean(0.5 * np.log(2 * np.pi * variance) + (residual ** 2) / (2 * variance))
        else:
            gaussian_nll = float('inf')

        # Poisson NLL
        eps = 1e-8
        y_pred_positive = np.clip(y_pred, eps, None)
        poisson_nll = np.mean(y_pred_positive - y_true * np.log(y_pred_positive))

        # R-squared
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        r2 = min(r2, 1.0)  # Ensure R2 is not greater than 1

        metrics.update({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'gaussian_nll': gaussian_nll,
            'poisson_nll': poisson_nll,
            'r2': r2
        })

        return metrics

    
    @staticmethod
    def cross_validate_metrics(X, y, model, num_folds=5):
        X = X
        y = y
        model = model
        num_folds = num_folds
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = CustomMetrics.calculate_all_metrics(y_test, y_pred)
            fold_metrics.append(metrics)

        # Average metrics across folds
        avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}

        return avg_metrics
'''
    @njit
    def compute_metrics_np(self, y_true, y_pred):
        metrics = {}
        
        self.y_true = y_true
        self.y_pred = y_pred

        # Ensure input shape
        self.y_true = self.y_true.ravel()
        self.y_pred = self.y_pred.ravel()

        # Basic metrics
        mse = np.mean((self.y_pred - self.y_true) ** 2)
        mae = np.mean(np.abs(self.y_pred - self.y_true))
        rmse = np.sqrt(mse)

        # Gaussian NLL
        residuals = self.y_pred - self.y_true
        variance = np.var(residuals)
        if variance > 0:
            gaussian_nll = np.mean(0.5 * np.log(2 * np.pi * variance) + (residuals ** 2) / (2 * variance))
        else:
            gaussian_nll = np.inf

        # Poisson NLL
        eps = 1e-8
        self.y_pred_positive = np.clip(self.y_pred, a_min=eps, a_max=None)
        poisson_nll = np.mean(self.y_pred_positive - self.y_true * np.log(self.y_pred_positive))

        # R2
        try:
            ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
            ss_res = np.sum((self.y_true - self.y_pred) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
        except:
            r2 = 0.0

        metrics['mse'] = mse
        metrics['rmse'] = rmse
        metrics['mae'] = mae
        metrics['gaussian_nll'] = gaussian_nll
        metrics['poisson_nll'] = poisson_nll
        metrics['r2'] = r2

        return metrics
    
    def calculate_all_metrics(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        
        # Ensure tensors and flatten
        self.y_true = tc.tensor(self.y_true).float().view(-1) if not isinstance(y_true, tc.Tensor) else y_true.float().view(-1)
        self.y_pred = tc.tensor(self.y_pred).float().view(-1) if not isinstance(y_pred, tc.Tensor) else y_pred.float().view(-1)

        # Convert to NumPy
        self.y_true_np = self.y_true.cpu().numpy()
        self.y_pred_np = self.y_pred.cpu().numpy()

        # Use the Numba-accelerated function
        return self.compute_metrics_np(self.y_true_np, self.y_pred_np)'''
