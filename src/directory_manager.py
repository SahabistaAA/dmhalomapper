import os
import logging
from datetime import datetime

class PathManager:
    """
    A utility class to manage directory structures and logging configurations for output files.

    This class creates a hierarchical directory structure based on a timestamped folder and sets up
    logging for both general and error logs. It ensures all necessary directories are created and
    provides easy access to commonly used paths.

    Attributes:
        base_path (str): The root directory for all outputs (default is './Output').
        running_path (str): The main timestamped directory for the current run.
        saved_path (str): Directory for saved files (e.g., models, plots, CSV files).
        log_path (str): Directory for log files.
        model_path (str): Directory for saving model files.
        plot_path (str): Directory for saving plots.
        distribution_plot_path (str): Directory for saving distribution plots.
        correlation_plot_path (str): Directory for saving correlation plots.
        preprocessed_distribution_plot_path (str): Directory for saving preprocessed data distribution plots.
        csv_file_path (str): Directory for saving CSV files.
        error_log_path (str): Path to the error log file.
    """

    def __init__(self, base_path='./out'):
        """
        Initialize the PathManager with a base directory and create the directory structure.

        Args:
            base_path (str, optional): The root directory for all outputs. Defaults to './Output'.
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        
        # Generate timestamp for unique folder name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
        self.running_path = os.path.join(self.base_path, timestamp)  # Fixed typo: base_pah -> base_path
        
        # Define directory structure
        self.saved_path = os.path.join(self.running_path, 'saved')
        self.log_path = os.path.join(self.running_path, 'log')
        self.model_path = os.path.join(self.saved_path, 'model')
        self.visualization_path = os.path.join(self.model_path, 'visualization')
        self.plot_path = os.path.join(self.saved_path, 'plots')
        self.distribution_plot_path = os.path.join(self.plot_path, 'distribution')
        self.correlation_plot_path = os.path.join(self.plot_path, 'correlation')
        self.preprocessed_distribution_plot_path = os.path.join(self.plot_path, 'preprocessed_distribution')
        self.csv_file_path = os.path.join(self.saved_path, 'csv_file')
        #self.error_log_path = os.path.join(self.log_path, 'errors.log')  # Define error log path
    
        self.__create_directories()
        self.logger = self.__setup_logging()
    
    def __create_directories(self):
        """
        Create all directories in the defined structure.

        This method ensures that all required directories (e.g., for logs, models, plots, etc.) exist.
        If they do not exist, they are created.
        """
        paths = [
            self.running_path,
            self.saved_path,  # Replaced output_path with saved_path
            self.log_path,
            self.model_path,
            self.visualization_path,
            self.plot_path,
            self.distribution_plot_path,
            self.correlation_plot_path,
            self.preprocessed_distribution_plot_path,
            self.csv_file_path
            #self.error_log_path
        ]
        
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    def __setup_logging(self):
        """
        Configure the logging settings.

        This method sets up three log outputs:
        1. General log file ('rnn_dmhalomapper_process.log') for debug/info.
        2. Warning log file ('warning_process.log') for warnings.
        3. Console output for real-time monitoring.

        Ensures no duplicate handlers are attached.
        """
        try:
            logger = logging.getLogger('dmmapper')
            logger.setLevel(logging.DEBUG)

            # Prevent adding handlers multiple times
            if logger.handlers:
                return logger

            os.makedirs(self.log_path, exist_ok=True)

            log_file = os.path.join(self.log_path, 'rnn_dmhalomapper_process.log')
            warning_log_file = os.path.join(self.log_path, 'warning_process.log')

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # General log handler
            general_handler = logging.FileHandler(log_file)
            general_handler.setLevel(logging.DEBUG)
            general_handler.setFormatter(formatter)

            # Warning log handler
            warning_handler = logging.FileHandler(warning_log_file)
            warning_handler.setLevel(logging.WARNING)
            warning_handler.setFormatter(formatter)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            # Attach handlers
            logger.addHandler(general_handler)
            logger.addHandler(warning_handler)
            logger.addHandler(console_handler)

            return logger

        except Exception as e:
            print(f"[Logger Setup Error] Failed to set up logging: {e}")
            raise  # Ensure the calling code knows logging failed


    def get_logger(self):
        """Return the configured logger."""
        return self.logger