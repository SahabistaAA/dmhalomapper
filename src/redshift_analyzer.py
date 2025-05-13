import os
import numpy as np
import pandas as pd
from model_metrics import CustomMetrics
import plotly.graph_objects as go
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()

class RedshiftAnalyzer:
    def __init__(self, data=None, selected_features=None):
        self.data = data
        self.selected_features = selected_features

    def process_redshift_data(self):
        """
        Process redshift data from the EAGLE simulation dataset.
        
        Parameters:
        data : DataFrame
            The input EAGLE simulation data
        selected_features : list
            List of selected features including 'Redshift'
            
        Returns:
        numpy.ndarray
            Processed redshift values
        """
        if 'Redshift' not in self.selected_features:
            logger.warning("'Redshift' must be included in selected_features")
            raise ValueError("'Redshift' must be included in selected_features")
        
        # Extract redshift values
        redshift_values = self.data['Redshift'].values
        cosmological_time_values = self.data['UniverseAge'].values
        
        
        # Sort unique redshift values in descending order
        unique_redshifts = np.sort(np.unique(redshift_values))[::-1]
        
        print(f"Redshift range: {unique_redshifts.min():.5E} to {unique_redshifts.max():.5E}")
        logger.info(f"Redshift range: {unique_redshifts.min():.5E} to {unique_redshifts.max():.5E}")
        print(f"Number of unique redshift values: {len(unique_redshifts)}")
        logger.info(f"Number of unique redshift values: {len(unique_redshifts)}")
        
        return redshift_values, unique_redshifts, cosmological_time_values
    
    def analyze_redshift_evolution(self, predictions, true_values, redshifts, save_dir, model_name):
        """
        Analyze how predictions vary with redshift.
        
        Parameters:
        predictions : array-like
            Model predictions (can be (N, 1) or (N,))
        true_values : array-like
            True values (can be (N, 1) or (N,))
        redshifts : array-like
            Corresponding redshift values (should be (N,))
        save_dir : str
            Directory to save analysis results
        model_name : str
            Name of the model being analyzed
        """
        try:
            # --- 1. Squeeze to ensure shapes are (N,) ---
            predictions = np.squeeze(predictions)
            true_values = np.squeeze(true_values)
            redshifts = np.squeeze(redshifts)

            logger.info(f"Shapes after squeezing - predictions: {predictions.shape}, true_values: {true_values.shape}, redshifts: {redshifts.shape}")

            # --- 2. Safety check ---
            if not (predictions.shape == true_values.shape == redshifts.shape):
                logger.error("Shape mismatch: predictions, true_values, and redshifts must have the same length after squeezing.")
                raise ValueError("Shape mismatch between predictions, true_values, and redshifts.")

            # --- 3. Unique redshift bins ---
            unique_z = np.sort(np.unique(redshifts))
            metrics_by_z = []
        
            # --- 4. Loop through redshifts and calculate metrics ---
            for z in unique_z:
                mask = np.squeeze(redshifts == z)  # Ensure mask is 1D

                preds_z = predictions[mask]
                trues_z = true_values[mask]

                if len(preds_z) == 0 or len(trues_z) == 0:
                    logger.warning(f"No samples found for redshift {z:.3f}. Skipping...")
                    continue

                z_metrics = CustomMetrics.calculate_all_metrics(
                    trues_z,
                    preds_z
                )
                z_metrics['redshift'] = z
                metrics_by_z.append(z_metrics)

            # --- 5. Save metrics to CSV ---
            metrics_df = pd.DataFrame(metrics_by_z)
            metrics_path = os.path.join(save_dir, f'{model_name}_redshift_evolution.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved redshift evolution metrics to {metrics_path}")

            # --- 6. Plot evolution ---
            fig = go.Figure()
            for metric in ['mse', 'rmse', 'mae', 'r2']:
                if metric in metrics_df.columns:
                    fig.add_trace(go.Scatter(
                        x=metrics_df['redshift'],
                        y=metrics_df[metric],
                        mode='lines+markers',
                        name=fr"$\text{{{metric.upper()}}}$"
                    ))

            fig.update_layout(
                title=fr"$\text{{{model_name} Metrics Evolution with Redshift}}$",
                xaxis_title=r"$\text{Redshift}$",
                yaxis_title=r"$\text{Metric Value}$",
                showlegend=True,
                template="plotly_white",
                font=dict(size=16),
                legend=dict(
                    title=r"$\text{Metric}$",
                    font=dict(size=14),
                    bordercolor="gray",
                    borderwidth=1
                )
            )

            html_path = os.path.join(save_dir, f'{model_name}_redshift_evolution.html')
            fig.write_html(html_path, include_mathjax='cdn')
            logger.info(f"Saved redshift evolution plot to {html_path}")

        except Exception as e:
            logger.error(f"Error during redshift evolution analysis: {e}", exc_info=True)
