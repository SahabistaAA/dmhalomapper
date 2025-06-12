import os
import csv
import datetime
import time
import numpy as np
import torch as tc
import torch.nn as tcn
import torch.optim as tco
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
#from torch.cuda.amp import autocast
from torch.amp import autocast, GradScaler
from model_metrics import CustomMetrics
from custom_dataset import DMDataset
from rnn_model import RNNDMHaloMapper
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()

class ModelTrainer:
    
    def __init__(self, model=None, criterion=None, optimizer=None, patience=10, device=None, save_dir=None):
        # Model setup
        self.model = model if model is not None else RNNDMHaloMapper()
        self.device = device if device else ('cuda' if tc.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Criterion and optimizer
        self.criterion = criterion or tc.nn.MSELoss()
        self.optimizer = optimizer or tc.optim.Adam(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.0001,
            eps=1e-8,
        )

        # Save and tracking
        self.save_dir = save_dir
        self.metrics_calculator = CustomMetrics()
        self.best_val_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        # Data loaders
        self.train_loader = None
        self.val_loader = None

        # AMP scaler only for CUDA
        if self.device == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None  # Skip AMP on CPU

        # Misc
        self.log_interval = 50
        self.accumulation_steps = 1
        
    def set_loaders(self, X, y):
        """Set up the data loaders for training and validation."""
        self.train_loader, self.val_loader = self.__set_loader__(X, y)
        
    def __set_loader__(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = self.__split__(X, y)
        self.train_dataset, self.val_dataset = self.__dataset__(self.X_train, self.X_val, self.y_train, self.y_val)
        self.train_loader, self.val_loader = self.__loader__(self.train_dataset, self.val_dataset)
        return self.train_loader, self.val_loader
        
    def __split__(self, X, y):
        return train_test_split(X, y, test_size=0.25, random_state=42)
    
    def __dataset__(self, X_train, X_val, y_train, y_val):
        try: 
            train_dataset = DMDataset(X_train, y_train)
            val_dataset = DMDataset(X_val, y_val)
            return train_dataset, val_dataset
        except Exception as e:
            print(e)
            logger.error(e)
    
    def __loader__(self, train_dataset, val_dataset):
        try:
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True if self.device == 'cuda' else False)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True if self.device == 'cuda' else False)
            return train_loader, val_loader
        except Exception as e:
            print(e)
            logger.error(e)
    
    def __create_save_directory(self, model_name):
        model_dir = os.path.join(self.save_dir, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(model_dir, exist_ok=True)
        return {
            'model_dir': model_dir,
            'best_model_path': os.path.join(model_dir, "best_model.pth"),
            'train_history_path': os.path.join(model_dir, "train_history.npy"),
            'train_history_file': os.path.join(model_dir, "train_history.csv"),
            'model_comparison_path': os.path.join(model_dir, "model_comparison_results.npy"),
            'plot_path': os.path.join(model_dir, "dmhalo_mapper.html")
        }

    def __train_epoch(self):
        self.model.train()
        train_loss = 0
        epoch_metrics = {
            'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0,
            'gaussian_nll': 0, 'poisson_nll': 0
        }
        grad_norms = []
        
        # Increase log interval on CPU for better performance
        effective_log_interval = self.log_interval * (5 if self.device == 'cpu' else 1)
        # Reduce accumulation steps on CPU
        effective_accumulation = 1 if self.device == 'cpu' else self.accumulation_steps
        
        pbar = tqdm(self.train_loader, desc="Training")

        # Determine if we're using CUDA for mixed precision
        use_amp = self.device == 'cuda'

        for batch_idx, (batch_features, batch_targets) in enumerate(pbar):
            try:
                # Move data to device
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                # Mixed precision training
                if use_amp:
                    with autocast(device_type=self.device):
                        outputs = self.model(batch_features)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.criterion(outputs, batch_targets)
                        # Scale loss for gradient accumulation
                        loss = loss / effective_accumulation
                    
                    # Scale gradients for mixed precision
                    self.scaler.scale(loss).backward()
                else:
                    # Standard forward pass and loss calculation for CPU
                    outputs = self.model(batch_features)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, batch_targets)
                    # Scale loss for gradient accumulation
                    loss = loss / effective_accumulation
                    
                    # Standard backward for CPU
                    loss.backward()
                
                # Only step optimizer and update metrics at accumulation boundary
                if (batch_idx + 1) % effective_accumulation == 0:
                    # Skip gradient clipping on CPU for speed when possible
                    if self.device == 'cuda' or batch_idx % effective_log_interval == 0:
                        grad_norm = tc.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0, norm_type=2
                        )
                        if batch_idx % effective_log_interval == 0:
                            grad_norms.append(grad_norm.item())
                    
                    # Update weights based on device type
                    if use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                
                # Accumulate unscaled loss for reporting
                train_loss += loss.item() * effective_accumulation
                
                # Update metrics less frequently - much less on CPU
                if batch_idx % effective_log_interval == 0:
                    with tc.no_grad():
                        batch_metrics = self.metrics_calculator.calculate_all_metrics(
                            batch_targets.detach(), outputs.detach()
                        )
                        for metric, value in batch_metrics.items():
                            epoch_metrics[metric] += value
                    
                    # Update progress bar with minimal info
                    if batch_idx > 0:
                        pbar.set_postfix({'loss': f"{loss.item()*effective_accumulation:.5f}"})
            
            except RuntimeError as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Adjust for actual number of evaluation steps
        num_eval_steps = max(1, len(self.train_loader) // effective_log_interval)
        avg_loss = train_loss / len(self.train_loader)
        avg_metrics = {metric: value / num_eval_steps for metric, value in epoch_metrics.items()}
        
        return avg_loss, avg_metrics, grad_norms    

    def __validate_epoch(self):
        self.model.eval()
        val_loss = 0
        epoch_metrics = {
            'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0,
            'gaussian_nll': 0, 'poisson_nll': 0
        }
        
        val_batches = 0  # Count actual batches processed
        
        # Increase log interval on CPU for better performance
        effective_log_interval = self.log_interval * (5 if self.device == 'cpu' else 1)
        
        # Determine if we're using CUDA for mixed precision
        use_amp = self.device == 'cuda'
        
        with tc.no_grad():
            # Use tqdm with minimal description
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, (batch_features, batch_targets) in enumerate(pbar):
                try:
                    # Move data to device with non_blocking=True
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    
                    # Only use autocast for CUDA devices
                    if use_amp:
                        with autocast(device_type=self.device):
                            outputs = self.model(batch_features)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            loss = self.criterion(outputs, batch_targets)
                    else:
                        # Standard forward pass for CPU
                        outputs = self.model(batch_features)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.criterion(outputs, batch_targets)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Calculate metrics less frequently to improve speed
                    if batch_idx % effective_log_interval == 0:
                        batch_metrics = self.metrics_calculator.calculate_all_metrics(
                            batch_targets, outputs
                        )
                        for metric, value in batch_metrics.items():
                            epoch_metrics[metric] += value
                        
                        if batch_idx > 0:  # Skip first update
                            pbar.set_postfix({'loss': f"{loss.item():.5f}"})
                
                except RuntimeError as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Adjust for actual number of evaluation steps
        num_eval_steps = max(1, len(self.val_loader) // effective_log_interval)
        avg_loss = val_loss / max(1, val_batches)
        avg_metrics = {metric: value / num_eval_steps for metric, value in epoch_metrics.items()}
        
        return avg_loss, avg_metrics

    '''def __train_epoch(self):
        self.model.train()
        total_loss = 0
        metrics = {k: 0 for k in ['mse', 'rmse', 'mae', 'r2', 'gaussian_nll', 'poisson_nll']}

        for i, (x, y) in enumerate(tqdm(self.train_loader, desc="Training Batches")):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            with tc.no_grad():
                ytrue = y.detach().cpu().numpy()
                ypred = output.detach().cpu().numpy()
                batch_metrics = self.metrics_calculator.calculate_all_metrics(ytrue, ypred)
                for k, v in batch_metrics.items():
                    metrics[k] += v

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in metrics.items()}

    def __validate_epoch(self):
        self.model.eval()
        total_loss = 0
        metrics = {k: 0 for k in ['mse', 'rmse', 'mae', 'r2', 'gaussian_nll', 'poisson_nll']}

        with tc.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation Batches"):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y)
                total_loss += loss.item()

                batch_metrics = self.metrics_calculator.calculate_all_metrics(y.cpu(), output.cpu())
                for k, v in batch_metrics.items():
                    metrics[k] += v

        n = len(self.val_loader)
        return total_loss / n, {k: v / n for k, v in metrics.items()}
'''
    def __save_training_progress(self, epoch, train_loss, val_loss, file_path):
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            writer.writerow([epoch + 1, train_loss, val_loss])

    def train(self, num_epochs, model_name, train_loader=None, val_loader=None, X=None, y=None):
        if train_loader and val_loader:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            # Set up data loaders from X and y
            self.set_loaders(X, y)
        
        # Create save directory
        save_paths = self.__create_save_directory(model_name)

        # Initialize learning rate scheduler and early stopping
        scheduler = tco.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
        
        # Training variables
        grad_norms = []
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.__optimize_batch_parameters()
        
                # Use torch.backends.cudnn optimizations
        if self.device != 'cpu':
            tc.backends.cudnn.benchmark = True  # Find best algorithms
        
        # Track time for benchmarking
        start_time = time.time()
        samples_processed = 0


        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"\nEpoch [{epoch + 1}/{num_epochs}] | LR: {current_lr:.2e}")
            logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}] | LR: {current_lr:.2e}")
            
            # Training phase with gradient clipping
            train_loss, train_metrics, epoch_grad_norsm = self.__train_epoch()
            grad_norms.extend(epoch_grad_norsm)
            
            # Validation phase
            val_loss, val_metrics = self.__validate_epoch()
            
            # Update learning rate
            scheduler.step(val_loss)
            print(f"Current learning rate: {scheduler.get_last_lr()}")
            
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"Learning rate updated: {current_lr:.2e} → {new_lr:.2e}")
                logger.info(f"Learning rate updated: {current_lr:.2e} → {new_lr:.2e}")
                current_lr = new_lr
            
            self.__save_training_progress(epoch, train_loss, val_loss, save_paths['train_history_file'])

            # Track gradient statistics
            avg_grad_norm = sum(grad_norms[-len(self.train_loader):])/len(self.train_loader)
            self.history.setdefault('grad_norms', []).append(avg_grad_norm)

            # Update samples_processed for benchmarking
            samples_processed += len(self.train_loader) * self.train_loader.batch_size

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                #tc.save(self.model.state_dict(), save_paths['best_model_path'])
                self.__save_model_checkpoint(save_paths['best_model_path'])

                print(f"New best model saved (Val Loss: {val_loss:.5E})")
                logger.info(f"New best model saved (Val Loss: {val_loss:.5E})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)

            # Calculate and log throughput
            epoch_time = time.time() - epoch_start
            samples_per_second = len(self.train_loader)

            # Print epoch summary
            epoch_summary = (
                f"Train Loss: {train_loss:.5E}, Val Loss: {val_loss:.5E}, "
                f"Grad Norm: {avg_grad_norm:.2f}"
                f"Throughput: {samples_per_second:.1f} samples/sec, "
                f"Time: {epoch_time:.2f}s"
            )
         
            print(epoch_summary)
            logger.info(epoch_summary)
            
            # Print metrics every 5 epochs
            if (epoch + 1) % 5 == 0:
                            
                print("\nTraining Metrics:")
                logger.info("Training Metrics:")

                train_metrics_line = "  ".join([f"{metric}: {value:.5E}" for metric, value in train_metrics.items()])
                print(train_metrics_line)
                logger.info(train_metrics_line)

                print("Validation Metrics:")
                logger.info("Validation Metrics:")

                val_metrics_line = "  ".join([f"{metric}: {value:.5E}" for metric, value in val_metrics.items()])
                print(val_metrics_line)
                logger.info(val_metrics_line)

                print()
                
                self.__log_metrics(train_metrics, val_metrics)

        
        # Final performance summary
        total_time = time.time() - start_time
        avg_throughput = samples_processed / total_time
        logger.info(f"Training complete. Total time: {total_time:.2f}s, Avg throughput: {avg_throughput:.1f} samples/sec")

        # Save training history
        np.save(save_paths['train_history_path'], self.history)
        print(f"Training complete. Best model saved to {save_paths['best_model_path']}")
        logger.info(f"Training complete. Best model saved to {save_paths['best_model_path']}")

        return self.history, save_paths['model_dir']

    def __optimize_batch_parameters(self):
        """Determine optimal batch size and gradient accumulation steps"""
        # This is a simplified version - could be expanded with actual benchmarking
        if self.device != 'cpu':
            # Get available GPU memory and model size
            total_mem = tc.cuda.get_device_properties(0).total_memory
            allocated_mem = tc.cuda.memory_allocated(0)
            free_mem = total_mem - allocated_mem
            
            # Rough estimate of memory per sample
            with tc.no_grad():
                # Try to estimate memory per sample based on model size
                dummy_input = next(iter(self.train_loader))[0][:1].to(self.device)
                
                # Measure memory usage for a forward pass
                before_forward = tc.cuda.memory_allocated(0)
                _ = self.model(dummy_input)
                after_forward = tc.cuda.memory_allocated(0)
                
                mem_per_sample = (after_forward - before_forward) * 3  # Forward + backward + optimizer
                
                # Set accumulation steps if needed
                if mem_per_sample * self.train_loader.batch_size > free_mem * 0.7:
                    # Calculate accumulation steps needed
                    ideal_batch = int(free_mem * 0.7 / mem_per_sample)
                    self.accumulation_steps = max(1, self.train_loader.batch_size // ideal_batch)
                    logger.info(f"Using gradient accumulation with {self.accumulation_steps} steps")
    
    def __save_model_checkpoint(self, path):
        """More efficient model saving"""
        # Only save model state dict instead of whole model
        tc.save(self.model.state_dict(), path)
    
    def __log_metrics(self, train_metrics, val_metrics):
        """Log metrics in a more organized way"""
        logger.info("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric}: {value:.5E}")
        
        logger.info("Validation Metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.5E}")

# Example usage
#if __name__ == "__main__":
#    from torch.utils.data import DataLoader
#    # Replace with actual implementations of model, dataset, and metrics calculator
#    model = MyModel()
#    criterion = tc.nn.MSELoss()
#    optimizer = tc.optim.Adam(model.parameters(), lr=0.001)
#    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
#    save_dir = "./training_results"

#    # Mock datasets and dataloaders
#    train_dataset = MyTrainDataset()
#    val_dataset = MyValDataset()
#    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#    trainer = ModelTrainer(model, criterion, optimizer, device, save_dir)
#    history, model_dir = trainer.train(num_epochs=50, model_name="my_model")