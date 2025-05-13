import os
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from collections import Counter
from scipy.stats import skew, kurtosis, sigmaclip
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import pandas as pd
import torch as tc
import torch.utils.data as tcud
from astropy.cosmology import FlatLambdaCDM
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()


class DataProcessor:
    def __init__(self, data_path, scaler_type=None, coverage_percentage=None):
        """
        Initialize the DataProcessor with a given dataset path and optional settings.
        
        Args:
            data_path (str): Path to the dataset file.
            scaler_type (str, optional): Type of scaler to use ('standard', 'robust', 'minmax').
            coverage_percentage (float, optional): Coverage threshold for processing.
        """
        
        self.data_path = data_path
        self.data = None
        self.train_data = None
        self.val_data = None
        self.scaler_type = scaler_type
        self.coverage_percentage= coverage_percentage
        
        # Impor PathManger
        self.path_manager = PathManager()
        
        #self.logger = logger.getLogger(__name__)
    
    def __add_cosmological_time(self):
        """
        Add lookback time and age of the universe (in Gyr) based on the Redshift column.
        """
        try:
            if self.data is None:
                logger.warning("Data is not loaded.")
                return
            
            if "Redshift" not in self.data.columns:
                logger.error("Redshift column is missing.")
                return
            
            # Filter invalid redshift
            invalid_redshift = self.data[self.data["Redshift"] < 0]
            if not invalid_redshift.empty:
                logger.warning(f"Found {len(invalid_redshift)} rows with invalid redshift (< 0). Setting LookbackTime and UniverseAge to NaN for these.")
            
            # Define cosmolical parameters (Planck 2018)
            cosmo = FlatLambdaCDM(name='Planck18', H0=67.66, Om0=0.30966, Tcmb0=2.7255, Neff=3.046, m_nu=[0.  , 0.  , 0.06], Ob0=0.04897)
            
            # Initialize columns
            self.data["LookbackTime"] = np.nan
            self.data["UniverseAge"] = np.nan

            # Only compute for valid redshift (>= 0)
            valid_idx = self.data[self.data["Redshift"] >= 0].index
            self.data.loc[valid_idx, "LookbackTime"] = cosmo.lookback_time(self.data.loc[valid_idx, "Redshift"]).value
            self.data.loc[valid_idx, "UniverseAge"] = cosmo.age(self.data.loc[valid_idx, "Redshift"]).value
            
            '''# Add columns
            self.data["LookbackTime"] = cosmo.lookback_time(self.data["Redshift"]).value
            self.data["UniverseAge"] = cosmo.age(self.data["Redshift"]).value
            '''
            logger.info("Cosmological time features added successfully.")
        except Exception as e:
            logger.error(f"Failed to add cosmological time: {e}")
    
    def load_data(self):
        """
        Load dataset from a CSV or Feather file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported or the data is empty.
        """        
        try:
            if not os.path.exists(self.data_path):
                logger.error(f"File not found: {self.data_path}")
                raise FileNotFoundError(f"File not found: {self.data_path}")
            if self.data_path is None:
                logger.error("Data path is not provided.")
                raise ValueError("Data path is not provided.")
            if self.data_path.endswith(".csv"):
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.endswith(".feather"):
                self.data = pd.read_feather(self.data_path)
            else:
                logger.error("Unsupported file format. Only CSV and Feather are supported.")
                raise ValueError("Unsupported file format. Only CSV and Feather are supported.")
            
            if self.data.empty:
                logger.error("Loaded data is empty.")
                raise ValueError("Loaded data is empty.")
            
            #self.__add_cosmological_time()
            
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
        
    def data_info(self):
        """
        Provide information about the loaded dataset.
        
        Returns:
            None: Prints the dataset information.
        """
        if self.data is not None:
            return self.data.info()
        else:
            logger.info("Data loaded successfully.")
            return None
    
    def basic_statistics(self):
        """
        Compute and save basic statistics for numerical columns.
        
        Returns:
            pd.DataFrame: Dataframe containing statistics like mean, std, skewness, and kurtosis.
        """
        try:
            if self.data is not None:
                # Compute statistics
                stats = self.data.describe().T
                stats["Skewness"] = self.data.skew()
                stats["Kurtosis"] = self.data.kurt()
            
                # Path to save CSV
                stats_file_path = os.path.join(self.path_manager.csv_file_path, 'basic_statistics_before.csv')

                # Format with scientific notation
                formatted_stats = stats.applymap(lambda x: f"{x:.5E}" if isinstance(x, (float, int)) else x)

                # Save formatted stats
                formatted_stats.to_csv(stats_file_path, index=True)
                logger.info(f"Basic statistics file saved to {stats_file_path}")

                return stats
            else:
                logger.warning("Data is not loaded.")
                return None
        except Exception as e:
            logger.error(f"Error computing or saving basic statistics: {e}")
            return None
    
    def __non_category_features(self):
        """
        Identify non-categorical features in the dataset.
        
        Returns:
            pd.Index: Index of non-categorical features.
        """
        if self.data is not None:
            self.non_categorical_features = self.data.select_dtypes(exclude=['object']).columns
            return self
        else:
            logger.warning("Data is not loaded.")
            return None
    
    def missing_values(self):
        """
        Check for missing values in the dataset.
        
        Returns:
            pd.Series: A series containing the count of missing values per column.
        """
        if self.data is not None:
            return self.data.isnull().sum()
        else:
            logger.warning("Data is not loaded.")
            return None
    
    def duplicated_values(self):
        """
        Check for duplicated values in the dataset.
        
        Returns:
            int: Number of duplicated rows.
        """
        if self.data is None:
            logger.warning("Data is not loaded.")
            return None
        
        return self.data.duplicated().sum()
    
    def __data_columns(self):
        """
        Get the columns of the dataset.
        
        Returns:
            pd.Index: Index of dataset columns.
        """
        if self.data is not None:
            return self.data.columns
        else:
            logger.warning("Data is not loaded.")
            return None
    
    def __outlier_analysis(self):
        """
        Perform outlier analysis using the Interquartile Range (IQR) method.
        
        Returns:
            tuple: Number of outliers, percentage of outliers, Q1, Q3, IQR, lower bound, and upper bound.
        """
        self.Q1 = self.data[self.__data_columns()].quantile(0.25)
        self.Q3 = self.data[self.__data_columns()].quantile(0.75)
        self.IQR = self.Q3 - self.Q1

        # define the boundary
        self.lower_bound = self.Q1 - 1.5 * self.IQR
        self.upper_bound = self.Q3 + 1.5 * self.IQR

        # checking outliers
        self.outliers = self.data[(self.data[self.__data_columns()] < self.lower_bound) | (self.data[self.__data_columns()] > self.upper_bound)]

        # number of outliers
        self.n_outliers = self.outliers.shape[0]

        # percentage outlier
        self.pct_outliers = (self.n_outliers / self.data.shape[0]) * 100

        return self.n_outliers, self.pct_outliers, self.Q1, self.Q3, self.IQR, self.lower_bound, self.upper_bound

    def analyze_outliers(self):
        """
        Analyze and summarize outliers in the dataset.
        
        Returns:
            pd.DataFrame: Dataframe containing outlier summary for each feature.
        """
        outliers_file_path = os.path.join(self.path_manager.csv_file_path, 'analyze_outliers.csv')

        outlier_summary = []
        for column in self.__non_category_features():
            n_outliers, pct_outliers, Q1, Q3, IQR, lower, upper_bound = self.__outlier_analysis()
            if n_outliers > 0:
                outlier_summary.append({
                    'Feature': column,
                    'Number of Outliers': n_outliers,
                    'Percentage (%)': pct_outliers,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'Lower Bound': lower,
                    'Upper Bound': upper_bound,
                })
        outlier_summary.to_csv(outliers_file_path)
        logger.info(f"Analyze outliers file saved to {outliers_file_path}")
        return pd.DataFrame(outlier_summary)

    def __plot_distribution(self, features, output_dir, title_prefix=""):
        """
        Plot and save distribution histograms for the given features.
        Args:
            features (list): List of feature names to plot.
            output_dir (str): Directory to save the plots.
            title_prefix (str): Prefix for the plot titles (optional).
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for feature in features:
            try:
                data = self.data[feature].dropna()
                hist_data = [data]
                group_labels = [fr"$\text{{{feature.replace('_', ' ')}}}$"]  # Format feature name in LaTeX

                fig = ff.create_distplot(
                    hist_data, group_labels, 
                    show_hist=True, show_curve=True, 
                    colors=["skyblue"]
                )

                fig.update_layout(
                    title=fr"$\text{{{title_prefix} Histogram of {feature.replace('_', ' ')}}}$",
                    xaxis_title=fr"$\text{{{feature.replace('_', ' ')}}}$",
                    yaxis_title=r"$\text{Density}$",
                    template="plotly_white",
                    font=dict(size=16),
                    width=800,
                    height=600,
                    legend=dict(
                        title=r"$\text{Legend}$",
                        font=dict(size=14),
                        bordercolor="gray",
                        borderwidth=1
                    )
                )

                # Save plot
                file_path = os.path.join(output_dir, f"{feature}.png")
                fig.write_image(
                    file_path, 
                    scale=1
                )  # High resolution
                logger.info(f"Saved histogram: {file_path}")

            except Exception as e:
                logger.error(f"Error plotting feature {feature}: {e}")
    
    def data_distribution(self):
        """
        Generate and save histograms for the non-categorical features.
        """
        if self.data is None:
            logger.warning("Data is not loaded.")
            return None
        
        output_dir = self.path_manager.distribution_plot_path
        non_categorical_features = self.data.select_dtypes(exclude=["object"]).columns.tolist()
        non_categorical_features.remove("GroupID")  # Exclude Image_ID if present
        self.__plot_distribution(non_categorical_features, output_dir, title_prefix="Original Data: ")
        logger.info("Histograms for original data saved successfully.")
    
    def correlation_heatmap(self):
        """
        Create a correlation matrix heatmap.
        
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        try:
            corr_path = self.path_manager.correlation_plot_path
            os.makedirs(corr_path, exist_ok=True)

            corr_matrix = self.data.drop(columns=['Image_ID'], errors='ignore').corr(numeric_only=True)
            corr_file_path = os.path.join(self.path_manager.csv_file_path, 'correlation_matrix.csv')
            corr_matrix.to_csv(corr_file_path)

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=[fr"$\text{{{col.replace('_', ' ')}}}$" for col in corr_matrix.columns],
                    y=[fr"$\text{{{col.replace('_', ' ')}}}$" for col in corr_matrix.columns],
                    colorscale="inferno",
                    colorbar=dict(
                        title=r"$\text{Correlation}$",
                        tickfont=dict(size=12)
                    )
                )
            )

            fig.update_layout(
                title=r"$\text{Correlation Matrix}$",
                width=1000,
                height=800,
                template="plotly_white",
                font=dict(size=16),
                legend=dict(
                    title=r"$\text{Legend}$",
                    font=dict(size=14),
                    bordercolor="gray",
                    borderwidth=1
                )
            )

            # Save the plot
            heatmap_path = os.path.join(corr_path, "corr_matrix.png")
            fig.write_image(
                heatmap_path,
                scale=1
            )  # High resolution
            logger.info(f"Correlation matrix saved: {heatmap_path}")

            return corr_matrix

        except Exception as e:
            logger.error(f"Error generating correlation heatmap: {e}", exc_info=True)
            return None
        
    
    def split_data(self, train_size=0.8, random_state=42):
        """
        Split the dataset into training and validation sets.
        
        Args:
            train_size (float): Proportion of data to be used for training (default is 0.8).
            random_state (int): Seed for reproducibility.
        
        Returns:
            tuple: Training and validation datasets.
        """
        try:
            self.train_set, self.val_set = train_test_split(self.data, train_size=train_size, random_state=random_state)
            logger.info("Data split into training and validation sets.")
            return self.train_set, self.val_set
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
    
    def handle_duplicated_values(self):
        """
        Handle duplicated values by removing them from the dataset.
        
        Returns:
            int: Number of duplicated rows removed.
        """
        if self.data is None:
            logger.warning("Data is not loaded.")
            return None
        
        # Get the number of duplicated rows before removal
        num_duplicates = self.duplicated_values()
        
        if num_duplicates > 0:
            # Remove duplicated rows
            self.data = self.data.drop_duplicates()
            logger.info(f"Removed {num_duplicates} duplicated rows.")
        else:
            logger.info("No duplicated values found.")
        
        return num_duplicates
    
    def handle_missing_values(self, strategy="mean"):
        """
        Handle missing values using the specified strategy.

        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent').
        """
        if self.data is not None:
            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = self.data.select_dtypes(include=["float64", "int64"]).columns

            # Only keep columns that have at least one non-missing value
            valid_cols = [col for col in numeric_cols if self.data[col].notna().any()]

            if valid_cols:
                self.data[valid_cols] = imputer.fit_transform(self.data[valid_cols])
                logger.info("Missing values handled.")
            else:
                logger.warning("No valid numeric columns to impute.")
        else:
            logger.warning("Data is not loaded.")


    def __detect_outliers(self, columns=None, iqr_factor=1.5):
        """
        Detect outliers using the IQR method.
        
        Args:
            columns (list, optional): Columns to analyze for outliers.
            iqr_factor (float, optional): Factor to multiply the IQR by to determine bounds.
        
        Returns:
            dict: Dictionary containing outlier information for each column.
        """
        if columns is None:
            columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        
        outliers = {}
        Q1 = self.data[columns].quantile(0.25)
        Q3 = self.data[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        
        for column in columns:
            outliers[column] = {
                "lower_bound": lower_bound[column],
                "upper_bound": upper_bound[column],
                "outliers": self.data[column][(self.data[column] < lower_bound[column]) | (self.data[column] > upper_bound[column])].count()
            }
        
        return outliers

    def __detect_skewed_cols(self, columns=None, skew_threshold=0.5):
        """
        Detect skewed columns based on the skew threshold.
        
        Args:
            columns (list, optional): Columns to analyze for skewness.
            skew_threshold (float, optional): Threshold to determine if a column is skewed.
        
        Returns:
            pd.Series: Series containing skewness values for skewed columns.
        """
        if columns is None:
            columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        skewness = self.data[columns].skew()
        return skewness[skewness > skew_threshold]

    def __detect_clip_cols(self, columns, lower_percentile=0.01, upper_percentile=0.99):
        """
        Detect columns needing clipping based on percentiles.
        
        Args:
            columns (list): Columns to analyze for clipping.
            lower_percentile (float, optional): Lower percentile for clipping.
            upper_percentile (float, optional): Upper percentile for clipping.
        
        Returns:
            dict: Dictionary containing clipping information for each column.
        """
        lower_limits = self.data[columns].quantile(lower_percentile)
        upper_limits = self.data[columns].quantile(upper_percentile)

        clip_cols = {}
        for col in columns:
            outliers = self.data[col][(self.data[col] < lower_limits[col]) | (self.data[col] > upper_limits[col])].count()
            clip_cols[col] = {
                "lower_limit": lower_limits[col],
                "upper_limit": upper_limits[col],
                "outliers": outliers
            }
        return clip_cols

    def auto_handle_outliers(self, numerical_columns):
        """
        Automatically handle outliers in the dataset.
        
        Args:
            numerical_columns (list): List of numerical columns to handle outliers for.
        
        Returns:
            tuple: Lists of columns to impute, transform, and clip.
        """
        outliers = self.__detect_outliers(columns=numerical_columns)
        cols_to_impute = [col for col, stats in outliers.items() if stats["outliers"] > 0]

        skewed_cols = self.__detect_skewed_cols(columns=numerical_columns)
        cols_to_transform = skewed_cols.index.tolist()

        clip_cols = self.__detect_clip_cols(numerical_columns)
        cols_to_clip = [col for col, stats in clip_cols.items() if stats["outliers"] > 0]

        imputer = SimpleImputer(strategy="median")
        self.data[cols_to_impute] = imputer.fit_transform(self.data[cols_to_impute])

        for col in cols_to_transform:
            self.data[col] = np.log1p(self.data[col])

        for col in cols_to_clip:
            lower_limit = self.data[col].quantile(0.01)
            upper_limit = self.data[col].quantile(0.99)
            self.data[col] = self.data[col].clip(lower=lower_limit, upper=upper_limit)

        return cols_to_impute, cols_to_transform, cols_to_clip

    def __separate_obs_data(self, data):
        """
        Separate observation and feature data.
        
        Args:
            data (pd.DataFrame): The dataset to separate.
        
        Returns:
            tuple: Observation data and feature data.
        """        
        self.obs_data_columns = [
            "GroupID", "SnapNum", "Redshift", "RandomNumber", "GalaxyID", "DescendantID",
            "LastProgID", "TopLeafID", "GroupNumber", "SubGroupNumber", "nodeIndex",
            "PosInFile", "FileNum", "Image_ID", "Redshift", "LookbackTime", "UniverseAge"
        ]
        self.obs_data = data[self.obs_data_columns]
        self.features = data.drop(columns=self.obs_data_columns)
        return self.obs_data, self.features

    def scale_data(self, scaler_type="standard"):
        """
        Scale numerical features using a specified scaler.

        Args:
            scaler_type (str): Type of scaler ('standard', 'robust', 'minmax').

        Returns:
            tuple: Scaled training and validation data.
        """
        scaler = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler()
        }.get(scaler_type)
        if scaler is None:
            raise ValueError(f"Invalid scaler type: {scaler_type}. Choose from 'standard', 'robust', or 'minmax'.")

        self.train_data_sc, self.train_features = self.__separate_obs_data(self.train_set)
        self.val_data_sc, self.val_features = self.__separate_obs_data(self.val_set)

        # Drop columns where all values are NaN
        self.train_features = self.train_features.dropna(axis=1, how="all")
        self.val_features = self.val_features.dropna(axis=1, how="all")

        # Fill remaining NaNs with column mean
        self.train_features = self.train_features.fillna(self.train_features.mean())
        self.val_features = self.val_features.fillna(self.val_features.mean())

        self.train_data_sc = pd.DataFrame(scaler.fit_transform(self.train_features), columns=self.train_features.columns)
        self.val_data_sc = pd.DataFrame(scaler.transform(self.val_features), columns=self.val_features.columns)

        logger.info(f"Data scaled using {scaler_type} scaler.")
        return self.train_data_sc, self.val_data_sc


    def normalize_data(self):
        """
        Normalize the scaled data (except the target column).
        
        Returns:
            tuple: Normalized training and validation data.
        """
        normalizer = Normalizer()
        TARGET_COL = "Group_M_Crit200"

        # Identify which columns to normalize (exclude the target)
        features_to_normalize = [col for col in self.train_features.columns if col != TARGET_COL]

        # Apply normalization **only on the features**
        train_features_normalized = pd.DataFrame(
            normalizer.fit_transform(self.train_data_sc[features_to_normalize]),
            columns=features_to_normalize
        )
        val_features_normalized = pd.DataFrame(
            normalizer.transform(self.val_data_sc[features_to_normalize]),
            columns=features_to_normalize
        )

        # Add the target column back **without modifying it**
        train_features_normalized[TARGET_COL] = self.train_data_sc[TARGET_COL].values
        val_features_normalized[TARGET_COL] = self.val_data_sc[TARGET_COL].values

        # Save
        self.train_data_sc_dn = train_features_normalized
        self.val_data_sc_dn = val_features_normalized

        logger.info("Data have been successfully normalized (excluding target).")
        return self.train_data_sc_dn, self.val_data_sc_dn

    
    def integrate_data(self):
        """
        Integrate observation data and processed features.
        
        Returns:
            tuple: Integrated training and validation datasets.
        """
        
        # Extract observation data for training
        try:
            train_obs_data = self.train_set[self.obs_data_columns]
        except KeyError as e:
            logger.error(f"Missing columns in train_data: {e}")
            raise KeyError(f"Missing columns in train_data: {e}")

        # Combine observation data with scaled and normalized features for training
        self.train_data_integrate = pd.concat(
            [
                train_obs_data.reset_index(drop=True),
                pd.DataFrame(self.train_data_sc_dn[self.train_features.columns].values, columns=self.train_features.columns)
            ],
            axis=1
        )

        # Extract observation data for validation
        try:
            val_obs_data = self.val_set[self.obs_data_columns]
        except KeyError as e:
            logger.error(f"Missing columns in val_data: {e}")
            raise KeyError(f"Missing columns in val_data: {e}")

        # Combine observation data with scaled and normalized features for validation
        self.val_data_integrate = pd.concat(
            [
                val_obs_data.reset_index(drop=True),
                pd.DataFrame(self.val_data_sc_dn[self.val_features.columns].values, columns=self.val_features.columns)
            ],
            axis=1
        )

        return self.train_data_integrate, self.val_data_integrate

    def final_clip(self, coverage_percentage=95.0):
        """
        Apply sigma clipping to the integrated datasets.
        
        Args:
            coverage_percentage (float): The desired coverage percentage (e.g., 95 for 95%).
        
        Returns:
            tuple: Clipped training and validation datasets.
        """
        
        if coverage_percentage is not None:
            self.coverage_percentage = coverage_percentage

        
        # Ensure coverage_percentage is valid
        if not (1 < coverage_percentage < 99):
            raise ValueError("Coverage percentage must be between 1 and 99 (exclusive).")

        
        # Calculate the z-score threshold based on the coverage percentage
        alpha = (100 - self.coverage_percentage) / 100
        rng = np.random.default_rng()

        z_threshold = abs(np.percentile(rng.standard_normal(1000000), [alpha * 50, 100 - alpha * 50]))[1]
        
        print("\nBefore final_clip, train_data:", type(self.train_data_integrate) , self.train_data_integrate.shape)
        
        # Calculate mean and standard 
        for column in self.train_data_integrate.columns:
            mean = self.train_data_integrate[column].mean()
            std = self.train_data_integrate[column].std()

            # Apply sigma clipping
            self.train_data_integrate = self.train_data_integrate[(self.train_data_integrate[column] >= mean - z_threshold * std) & 
                        (self.train_data_integrate[column] <= mean + z_threshold * std)]

        for column in self.val_data_integrate.columns:
            mean = self.val_data_integrate[column].mean()
            std = self.val_data_integrate[column].std()
            
            # Apply sigma clipping
            self.val_data_integrate = self.val_data_integrate[(self.val_data_integrate[column] >= mean - z_threshold * std) & (self.val_data_integrate[column] <= mean + z_threshold * std)]
        
        print("\nAfter final_clip, train_data:", type(self.train_data_integrate), self.train_data_integrate.shape)
        
        return self.train_data_integrate, self.val_data_integrate

    def preprocess_data(self, scaler_type="standard", coverage_percentage=99.7):
        """
        Run the full preprocessing pipeline.
        
        Args:
            scaler_type (str): Type of scaler to use ('standard', 'robust', 'minmax').
            coverage_percentage (float): Coverage threshold for processing.
        
        Returns:
            tuple: Preprocessed training and validation datasets.
        """
        try:
            self.handle_duplicated_values()
            self.handle_missing_values()
            non_categorical_features = self.data.select_dtypes(exclude=["object"]).columns.tolist()
            self.auto_handle_outliers(non_categorical_features)
            self.split_data()
            self.scale_data(scaler_type)
            self.normalize_data()
            self.integrate_data()
            #self.final_clip(coverage_percentage)
            logger.info("Preprocessed data have been completed.")
        except KeyError as e:
            logger.error(f"Error while preprocessing data: {e}")
        return self.train_data_integrate, self.val_data_integrate
    
    def preprocessed_distribution(self):
        """
        Generate and save histograms for preprocessed training data.
        """
        if self.train_data_integrate is None:
            logger.warning("Training data is not available.")
            return None
        
        output_dir = self.path_manager.preprocessed_distribution_plot_path
        train_dir = os.path.join(output_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(output_dir, "val")
        os.makedirs(val_dir, exist_ok=True)
        train_numeric_features = self.train_data_integrate.select_dtypes(include=["float64", "int64"]).columns.tolist()
        train_numeric_features.remove("GroupID")
        val_numeric_features = self.val_data_integrate.select_dtypes(include=["float64", "int64"]).columns.tolist()
        val_numeric_features.remove("GroupID")
        self.__plot_distribution(train_numeric_features, train_dir, title_prefix="Preprocessed Train Data: ")
        self.__plot_distribution(val_numeric_features, val_dir, title_prefix="Preprocessed Validation Data: ")
        logger.info("Histograms for preprocessed data saved successfully.")
