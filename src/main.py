from data_processing import DataProcessor,  DMDataset
from model_train import ModelTrainer
from model_hyperparameter_optimizer import RNNHyperParameterOptimizer
from rnn_model import RNNDMHaloMapper, LSTMDMHaloMapper, GRUDMHaloMapper
from redshift_analyzer import RedshiftAnalyzer
from model_visualization import ModelVisualizer
import torch
import torch.optim as tco
import torch.utils.data as tcud
import torch.nn as tcn


class DMHaloMapper:
    def __init__(self, data=None, device=None, model=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dp = DataProcessor(data_path=data)
        self.mt = ModelTrainer()
        self.dm = DMDataset()
        self.hp = None
        self.mv = ModelVisualizer()
        self.rnn = RNNDMHaloMapper()
        self.lstm = LSTMDMHaloMapper()
        self.gru = GRUDMHaloMapper()
        self.train_data = None
        self.val_data = None
        self.X = None
        self.y = None
        self.features = None
        self.targets = None
        self.redshift_values = None
        self.unique_redshifts = None
        self.best_hyperparameter = None
        self.model=model

    def data_process(self):
        self.dp.load_data()
        self.dp.missing_values()
        self.dp.basic_statistics()
        self.dp.data_distribution()
        self.dp.correlation_heatmap()
        self.train_data, self.val_data = self.dp.preprocess_data(scaler_type="robust", coverage=0.85)
        self.dp.preprocessed_distribution()
    
    def data_to_tensor(self):
        features = [
            'Redshift',
            'NumOfSubhalos',
            'Velocity_x',
            'Velocity_y',
            'Velocity_z',
            'MassType_DM',
            'CentreOfPotential_x',
            'CentreOfPotential_y',
            'CentreOfPotential_z',
            'GroupCentreOfPotential_x',
            'GroupCentreOfPotential_y',
            'GroupCentreOfPotential_z',
            'Vmax',
            'VmaxRadius',
            'Group_R_Crit200'
        ]
        
        targets = ['Group_M_Crit200']
        
        self.X = self.train_data[features]
        self.y = self.train_data[targets].values.reshape(-1, 1)
        
        self.features, self.targets = self.dm(self.X, self.y)
    
    def process_redshift(self):
        self.ra = RedshiftAnalyzer(self.train_data, self.features)
        self.redshift_values, self.unique_redshifts = self.ra.process_redshift_data()
    
    def discover_hyperparameter(self):
        print(f"Starting discover_hyperparameter for {len(self.models_to_optimize)} models...")
        self.models_to_optimize = [self.rnn, self.lstm, self.gru]
        
        optimization_results = []
        
        self.hp = RNNHyperParameterOptimizer(
            X=self.X,
            y=self.y,
            base_model_class=RNNDMHaloMapper,  # You can change this to LSTMDMHaloMapper or GRUDMHaloMapper
            device=self.device,
            num_features=self.X.shape[1],
            pop_size=20,
            maxiter=100,
            cross_val_folds=13
        )
        
        # Run optimization
        optimization_result = self.hp.optimize()
        
        if optimization_result:
            self.best_hyperparameters = optimization_result['best_params']
            print("Best hyperparameters found:")
            for param, value in self.best_hyperparameters.items():
                print(f"{param}: {value}")
        else:
            print("Hyperparameter optimization failed. Using default hyperparameters.")
            self.best_hyperparameters = {
                'hidden_size': 256,
                'num_layers': 2,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'num_epochs': 10
            }
        
    def train_rnn_model(self):
        """
        Train the RNN, LSTM, and GRU models using the best hyperparameters.
        """
        if not self.best_hyperparameters:
            print("No hyperparameters found. Running discover_hyperparameter first.")
            self.discover_hyperparameter()

        input_size = self.X.shape[1]
        output_size = 1

        # Define models with the best hyperparameters
        models = {
            'RNN': RNNDMHaloMapper(
                input_size=input_size,
                hidden_size=self.best_hyperparameters['hidden_size'],
                output_size=output_size,
                num_layers=self.best_hyperparameters['num_layers'],
                dropout_rate=self.best_hyperparameters['dropout_rate'],
                learning_rate=self.best_hyperparameters['learning_rate'],
                batch_size=self.best_hyperparameters['batch_size'],
                weight_decay=self.best_hyperparameters['weight_decay'],
                num_epochs=self.best_hyperparameters['num_epochs']
            ),
            'LSTM': LSTMDMHaloMapper(
                input_size=input_size,
                hidden_size=self.best_hyperparameters['hidden_size'],
                output_size=output_size,
                num_layers=self.best_hyperparameters['num_layers'],
                dropout_rate=self.best_hyperparameters['dropout_rate'],
                learning_rate=self.best_hyperparameters['learning_rate'],
                batch_size=self.best_hyperparameters['batch_size'],
                weight_decay=self.best_hyperparameters['weight_decay'],
                num_epochs=self.best_hyperparameters['num_epochs']
            ),
            'GRU': GRUDMHaloMapper(
                input_size=input_size,
                hidden_size=self.best_hyperparameters['hidden_size'],
                output_size=output_size,
                num_layers=self.best_hyperparameters['num_layers'],
                dropout_rate=self.best_hyperparameters['dropout_rate'],
                learning_rate=self.best_hyperparameters['learning_rate'],
                batch_size=self.best_hyperparameters['batch_size'],
                weight_decay=self.best_hyperparameters['weight_decay'],
                num_epochs=self.best_hyperparameters['num_epochs']
            )
        }

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=42)

        self.X_val = X_val
        self.y_val = y_val
        
        train_dataset = DMDataset(X_train, y_train)
        val_dataset = DMDataset(X_val, y_val)

        train_loader = DMDataset.data_loader(train_dataset, batch_size=self.best_hyperparameters['batch_size'])
        val_loader = DMDataset.data_loader(val_dataset, batch_size=self.best_hyperparameters['batch_size'])
        '''train_loader = tcud.DataLoader(
            train_dataset,
            batch_size=self.best_hyperparameters['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = tcud.DataLoader(
            val_dataset,
            batch_size=self.best_hyperparameters['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )'''

        model_results = {}

        for name, model in models.items():
            print(f"\nTraining {name} model with optimized hyperparameters...")
            optimizer = tco.Adam(model.parameters(), lr=self.best_hyperparameters['learning_rate'], weight_decay=self.best_hyperparameters['weight_decay'])
            criterion = tcn.MSELoss()

            history, model_dir = self.mt.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=self.best_hyperparameters['num_epochs'],
                device=self.device,
                model_name=name,
                save_dir='./best_model'
            )

            model_results[name] = {
                'history': history,
                'model_dir': model_dir
            }

        return model_results
    
    def process_model(self):
        """
        Process the validation dataset, make predictions, and store results for visualization.
        """
        if not hasattr(self, 'models'):
            print("No trained models found. Run train_rnn_model first.")
            return

        # Load validation data
        X_val_tensor = tc.tensor(self.X_val.values, dtype=tc.float32)
        self.y_val = self.y[self.X_val.index]

        # Make predictions on the validation set
        self.val_predictions = {}
        for name, model in self.models.items():
            model.eval()
            with tc.no_grad():
                outputs = model(X_val_tensor.to(self.device))
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                self.val_predictions[name] = outputs.cpu().numpy()

        print("Validation dataset processed and predictions stored.")
    
    def model_visualization(self):
        """
        Visualize the model's predictions, including dark matter distribution, triaxial model, and redshift evolution.
        """
        if not hasattr(self, 'val_predictions'):
            print("No validation predictions found. Run process_model first.")
            return

        # Extract spatial coordinates for visualization
        X_val_tensor = tc.tensor(self.X_val.values, dtype=tc.float32)
        column_indices = [self.X.columns.get_loc(f) for f in 
            [
                'GroupCentreOfPotential_x',
                'GroupCentreOfPotential_y',
                'GroupCentreOfPotential_z'
            ]
        ]
        spatial_coords = X_val_tensor[:, column_indices].numpy()

        # Initialize ModelVisualizer
        self.mv = ModelVisualizer(
            predictions=self.val_predictions['RNN'],  # Use RNN predictions as an example
            true_values=self.y_val,
            spatial_coords=spatial_coords,
            redshifts=self.redshift_values[self.X_val.index],
            save_dir='./visualizations'
        )

        # Generate and save visualizations
        self.mv.visualize_dm_distribution(a=1.0, b=0.8, c=0.6)

    
    def run_process(self):
        try:
            self.data_process()
            self.data_to_tensor()
            self.process_redshift()
            self.discover_hyperparameter()
            self.train_rnn_model()
            self.model_visualization()
            self.process_model()
        except Exception as e:
            print(f"An error occurred: {e}")
    
def __main__():
    dmhalo_mapper = DMHaloMapper()
    dmhalo_mapper.run_process(model='GRU')

if __name__ == "__main__":
    __main__()