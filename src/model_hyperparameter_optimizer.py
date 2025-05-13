import os
import gc
import json
import optuna
import torch as tc
import numpy as np
import pandas as pd
import torch.utils.data as tcud
import torch.optim as tco
import torch.nn as tcn
from tqdm import tqdm
from custom_dataset import DMDataset
from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()

class RNNDifferentialEvolutionHyperParameterOptimizer:
    """
    Hyperparameter optimization for RNN model using differential evolution algorithm.
    Args:
    X (numpy.ndarray): Input features
    y (numpy.ndarray): Target values
    base_model_class (class): Base model class to be optimized, should inherit from tcn.RNNBase
    device (str): Default Cuda device
    num_features (int): Number of input features
    pop_size (int): Population size for differential evolution
    maxiter (int): Maximum number of iterations for differential evolution
    cross_val_folds (int): Number of cross-validation folds
    """
    def __init__(self, X, y, base_model_class, device, num_features, 
                    pop_size=5, maxiter=13, cross_val_folds=3, logger=None, path_manager=None):
        self.X = pd.DataFrame(X)
        self.y = np.array(y)
        self.base_model_class = base_model_class
        self.device = device
        self.num_features = num_features
        self.pop_size = pop_size
        self.maxiter = maxiter
        self.cross_val_folds = cross_val_folds
        
        #self.activation_functions = ['relu', 'tanh', 'gelu', 'elu', 'sigmoid', 'mish', 'silu']
        self.activation_functions = ['gelu', 'mish', 'silu']
        #self.activation_functions = ['relu', 'tanh', 'gelu', 'elu', 'sigmoid', 'mish', 'silu']
        #self.optimizers = ['Adam', 'SGD', 'RMSprop']
        self.optimizers = ['Adam', 'SGD']
        self.logger= logger or PathManager().get_logger()
        self.path_manager = path_manager or PathManager()
        
        # Define parameter bounds
        '''self.bounds = [
            (32, 512),      # hidden_size
            (1, 4),         # num_layers
            (0.0, 0.5),     # dropout_rate
            (0.0001, 0.1),  # learning_rate
            (16, 256),      # batch_size
            (0.0, 0.1),     # weight_decay
            (10, 100),      # number of epochs
            (0, len(self.activation_functions) - 1),  # activation index
            (0, len(self.optimizers) - 1),            # optimizer index
        ]'''
        self.bounds = [
            (64, 256),      # hidden_size
            (1, 4),         # num_layers
            (0.0, 0.5),     # dropout_rate
            (0.001, 0.01),  # learning_rate
            (32, 128),      # batch_size
            (0.0, 0.1),     # weight_decay
            (20, 50),      # number of epochs
            (0, len(self.activation_functions) - 1),  # activation index
            (0, len(self.optimizers) - 1),            # optimizer index
        ]
    
    def __get_activation_layer(self):
        """
        Get activation layer based on the specified activation function.
            
        Returns:
            torch.nn.Module: Activation layer.
        """
        activation_name = self.activation.lower()
    
        activation_map = {
            'relu': tcn.ReLU(),
            'leaky_relu': tcn.LeakyReLU(),
            'elu': tcn.ELU(),
            'tanh': tcn.Tanh(),
            'sigmoid': tcn.Sigmoid(),
            'gelu': tcn.GELU(),
            'selu': tcn.SELU(),
            'mish': tcn.Mish(),
            'hardshrink': tcn.Hardshrink(),
            'celu': tcn.CELU(),
            'swish': tcn.SiLU(),  # PyTorch calls Swish as SiLU
            'glu': tcn.GLU()
        }
        
        if activation_name not in activation_map:
            logger.warning(f"Unknown activation function '{self.activation}', defaulting to GELU.")
        
        return activation_map.get(activation_name, tcn.GELU())
    
    def __create_model(self, params):
        """Create model with given hyperparameters"""
        hidden_size = int(params[0])
        num_layers = int(params[1])
        dropout_rate = params[2]
        learning_rate = params[3]
        batch_size = int(params[4])
        weight_decay = params[5]
        num_epochs = int(params[6])
        activation_fn = self.activation_functions[int(round(params[7]))]
        optimizer_name = self.optimizers[int(round(params[8]))]
        
        model = self.base_model_class(
            input_size=self.num_features,
            hidden_size=hidden_size,
            output_size=1,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            #learning_rate=learning_rate,
            #batch_size=batch_size,
            #weight_decay=weight_decay,
            #num_epochs=num_epochs,
            #activation=activation_fn,  # <-- string name passed to model
            bidirectional=True,
            use_layer_norm=True,
            return_hidden=True
        )
        
        return model, num_epochs, optimizer_name, learning_rate, weight_decay, batch_size

    def __get_optimizer(self, optimizer_name, model_params, lr, wd):
            if optimizer_name == 'Adam':
                return tco.Adam(model_params, lr=lr, weight_decay=wd)
            elif optimizer_name == 'SGD':
                return tco.SGD(model_params, lr=lr, weight_decay=wd)
            elif optimizer_name == 'RMSprop':
                return tco.RMSprop(model_params, lr=lr, weight_decay=wd)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _objective_function(self, params):
        try:
            model, _, optimizer_name, lr, wd, batch_size, = self.__create_model(params)
            kf = KFold(n_splits=self.cross_val_folds, shuffle=True, random_state=42)
            cv_scores = []
            num_epochs = 10
            
            for train_idx, val_idx in kf.split(self.X):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]

                train_loader = tcud.DataLoader(DMDataset(X_train, y_train),
                                               batch_size=batch_size, shuffle=True, num_workers=1)
                val_loader = tcud.DataLoader(DMDataset(X_val, y_val),
                                             batch_size=batch_size, shuffle=False, num_workers=1)

                model = model.to(self.device)
                optimizer = self.__get_optimizer(optimizer_name, model.parameters(), lr, wd)
                criterion = tcn.MSELoss()

                best_val_loss = float('inf')
                no_improve = 0
                patience = 3

                model.train()
                for epoch in range(num_epochs):
                    for batch_features, batch_targets in train_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        optimizer.zero_grad()
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_targets)
                        loss.backward()
                        optimizer.step()

                    # Evaluate validation loss
                    model.eval()
                    val_loss = 0
                    with tc.no_grad():
                        for batch_features, batch_targets in val_loader:
                            batch_features = batch_features.to(self.device)
                            batch_targets = batch_targets.to(self.device)
                            outputs = model(batch_features)
                            val_loss += criterion(outputs, batch_targets).item()
                    val_loss /= len(val_loader)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break

                cv_scores.append(best_val_loss)

            mean_score = np.mean(cv_scores)
            print(f"Score: {mean_score:.5E} | Activation: {self.activation_functions[int(round(params[7]))]} | Optimizer: {self.optimizers[int(round(params[8]))]}")
            self.logger.info(f"Score: {mean_score:.5E} | Activation: {self.activation_functions[int(round(params[7]))]} | Optimizer: {self.optimizers[int(round(params[8]))]}")
            return mean_score

        except Exception as e:
            self.logger.error(f"Error: {e}")
            return float('inf')
    
    def optimize(self):
        """Run differential evolution"""
        
        def callback(self, xk, convergence):
            self.pbar.update(1)
        
        try:
            pbar = tqdm(total=self.maxiter, desc="Optimization Progress")
            
            result = differential_evolution(
                func=self._objective_function,
                bounds=self.bounds,
                maxiter=self.maxiter,
                popsize=self.pop_size,
                callback=callback,
                mutation=(0.7, 1.0),
                recombination=0.5,
                tol=0.005,
                atol=0.001,
                seed=42,
                disp=True,
                workers=-1
            )

            best_params = {
                'hidden_size': int(result.x[0]),
                'num_layers': int(result.x[1]),
                'dropout_rate': f"{result.x[2]:.5E}",
                'learning_rate': f"{result.x[3]:.5E}",
                'batch_size': int(result.x[4]),
                'weight_decay': f"{result.x[5]:.5E}",
                'num_epochs': int(result.x[6]),
                'activation_fn': self.activation_functions[int(round(result.x[7]))],
                'optimizer': self.optimizers[int(round(result.x[8]))]
            }
            
            json_path = os.path.join(self.path_manager.csv_file_path, 'best_hyperparameters.json')
            with open(json_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            self.logger.info(f"Best hyperparameters saved to {json_path}")


            return {
                'best_params': best_params,
                'best_score': result.fun,
                'convergence': result.success,
                'iterations': result.nit,
                'optimization_result': result
            }

        except Exception as e:
            print(f"Error in optimization: {e}")
            self.logger.error(f"Error in optimization: {e}")
            return None

class RNNOptunaHyperParameterOptimizer:
    """
    Hyperparameter optimization for RNN model using Optuna.
    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target values
        base_model_class (class): Base model class to be optimized, should inherit from tcn.RNNBase
        device (str): Default Cuda device
        num_features (int): Number of input features
        pop_size (int): Population size for differential evolution
        maxiter (int): Maximum number of iterations for differential evolution
        cross_val_folds (int): Number of cross-validation folds
    """
    def __init__(self, X, y, base_model_class,
                 device, num_features,
                 n_trials=50, cross_val_folds=3, timeout=None):
        self.X = pd.DataFrame(X)
        self.y = np.array(y)
        self.base_model_class = base_model_class
        self.device = device
        self.num_features = num_features
        self.n_trials = n_trials
        self.cross_val_folds = cross_val_folds
        self.timeout = timeout
        
        self.activation_functions = ['gelu', 'mish', 'silu']
        self.optimizers = ['Adam']
        
        self.logger = logger
        self.path_manager = path_manager
        
    def __objective_functions(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.
        Args:
            trial (optuna.Trial): Optuna trial object
        Returns:
            float: Mean cross-validation score
        """
        try:
            # Suggest hyperparameters
            hidden_size = trial.suggest_int('hidden_size', 64, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 128, 256)
            weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
            num_epochs = trial.suggest_int('num_epochs', 20, 50)
            activation_fn = trial.suggest_categorical("activation_fn", self.activation_functions)
            optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)

            # Create model
            model = self.base_model_class(
                input_size=self.num_features,
                hidden_size=hidden_size,
                output_size=1,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                bidirectional=True,
                use_layer_norm=True,
                return_hidden=True
                #activation=activation_fn  # Pass activation function name to model
            ).to(self.device)
            
            criterion = tcn.MSELoss()
            optimizer = self.__get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)
            activation_fn = self.__get_activation(activation_fn)
            
            # Cross-validation
            kf = KFold(n_splits=self.cross_val_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
                # Split data manually
                X_train, y_train = self.X[train_idx], self.y[train_idx]
                X_val, y_val = self.X[val_idx], self.y[val_idx]

                train_loader = tcud.DataLoader(DMDataset(X_train, y_train),
                                            batch_size=batch_size, shuffle=True, num_workers=1)
                val_loader = tcud.DataLoader(DMDataset(X_val, y_val),
                                            batch_size=batch_size, shuffle=False, num_workers=1)

                # --- Create model fresh for each fold
                model = self.base_model_class(
                    input_size=self.num_features,
                    hidden_size=hidden_size,
                    output_size=1,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate,
                    bidirectional=True,
                    use_layer_norm=True,
                    return_hidden=True
                    #activation=activation_fn
                ).to(self.device)

                criterion = tcn.MSELoss()
                optimizer = self.__get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)
                activation_fn = self.__get_activation(activation_fn)

                best_val_loss = float('inf')
                patience = 10
                no_improve = 0

                for epoch in range(num_epochs):
                    model.train()
                    for batch_features, batch_targets in train_loader:
                        batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(batch_features)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = criterion(outputs, batch_targets)
                        loss.backward()
                        optimizer.step()

                    # Validation
                    model.eval()
                    val_loss = 0
                    with tc.no_grad():
                        for batch_features, batch_targets in val_loader:
                            batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                            outputs = model(batch_features)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            val_loss += criterion(outputs, batch_targets).item()
                    val_loss /= len(val_loader)

                    trial.report(val_loss, epoch)

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break

                # Save best val_loss for this fold
                cv_scores.append(best_val_loss)

                # Cleanup
                del model
                gc.collect()
                tc.cuda.empty_cache()

            # --- After all folds
            mean_score = np.mean(cv_scores)
            return mean_score

        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            return float('inf')
    
    def __get_optimizer(self, optimizer_name, model_params, lr, wd):
        if optimizer_name == 'Adam':
            return tco.Adam(model_params, lr=lr, weight_decay=wd)
        elif optimizer_name == 'SGD':
            return tco.SGD(model_params, lr=lr, weight_decay=wd)
        elif optimizer_name == 'RMSprop':
            return tco.RMSprop(model_params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    def __get_activation(self, activation_fn):
        """
        Get activation layer based on the specified activation function.
            
        Returns:
            torch.nn.Module: Activation layer.
        """
        activation_name = activation_fn.lower()
    
        activation_map = {
            'relu': tcn.ReLU(),
            'leaky_relu': tcn.LeakyReLU(),
            'elu': tcn.ELU(),
            'tanh': tcn.Tanh(),
            'sigmoid': tcn.Sigmoid(),
            'gelu': tcn.GELU(),
            'selu': tcn.SELU(),
            'mish': tcn.Mish(),
            'hardshrink': tcn.Hardshrink(),
            'celu': tcn.CELU(),
            'swish': tcn.SiLU(),  # PyTorch calls Swish as SiLU
            'glu': tcn.GLU(),
            'silu': tcn.SiLU(),
        }
        
        if activation_name not in activation_map:
            logger.warning(f"Unknown activation function '{self.activation}', defaulting to GELU.")
        
        return activation_map.get(activation_name, tcn.GELU())

    def optimize(self):
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study.optimize(
            self.__objective_functions,
            n_trials=self.n_trials,
            n_jobs=8,  # Max parallelism
            timeout=self.timeout
        )

        best_params = study.best_trial.params
        best_score = study.best_trial.value

        # Save results
        save_path = os.path.join(self.path_manager.csv_file_path, "optuna_best_hyperparameters.json")
        with open(save_path, 'w') as f:
            json.dump(best_params, f, indent=4)

        logger.info(f"Best hyperparameters saved to {save_path}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "study": study
        }        