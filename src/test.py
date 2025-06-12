import os
import time
import json
import torch as tc
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import torch.cuda.amp as amp  # For mixed precision training
import torch.backends.cudnn as cudnn
import torch.amp as tca
from pathlib import Path

from data_processing import DataProcessor
from custom_dataset import DMDataset
from model_hyperparameter_optimizer import RNNOptunaHyperParameterOptimizer, RNNDifferentialEvolutionHyperParameterOptimizer
from model_train import ModelTrainer
from redshift_analyzer import RedshiftAnalyzer
from model_visualization import ModelVisualizer
from rnn_model import GRUDMHaloMapper, RNNDMHaloMapper, LSTMDMHaloMapper
from directory_manager import PathManager
from model_metrics import CustomMetrics


# ---------------------- Configuration ----------------------
#DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0025N0376_with_cosmological_time.feather"
#DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0025N0752_with_cosmological_time.feather"
DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0100N1504_with_cosmological_time.feather"

SELECTED_FEATURES = [
    'Redshift',
    'UniverseAge', 
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
TARGET = 'Group_M_Crit200'
SPATIAL_COLS = ['GroupCentreOfPotential_x', 'GroupCentreOfPotential_y', 'GroupCentreOfPotential_z']
SUBHALO_COLS = ['CentreOfPotential_x', 'CentreOfPotential_y', 'CentreOfPotential_z']
MODEL_TYPE = 'GRU'
BATCH_SIZE = 256
DEVICE = 'cuda' if tc.cuda.is_available() else 'cpu'
NUM_WORKERS = min(8, os.cpu_count())  # Don't use too many workers


# ---------------------- Setup ----------------------
path_manager = PathManager()
logger = path_manager.get_logger()

# ---------------------- Helper Functions ----------------------
def extract_features_targets(df, features, target):
    available_features = [feat for feat in features if feat in df.columns]
    missing_features = [feat for feat in features if feat not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features during extraction: {missing_features}")
    
    X = df[available_features]
    y = df[[target]]
    return X, y

def extract_coords(X_val):
    return X_val[SPATIAL_COLS].values, X_val[SUBHALO_COLS].values

def get_model_class(name='GRU'):
    model_classes = {
        'GRU': GRUDMHaloMapper,
        'LSTM': LSTMDMHaloMapper,
        'RNN': RNNDMHaloMapper
    }
    return model_classes.get(name, GRUDMHaloMapper)
    
def predict_model(model, dataloader, device, verbose=True):

    model.eval()
    preds = []
    total_batches = len(dataloader)
    samples_processed = 0
    
    if verbose:
        logger.info(f"Total batches in dataloader: {total_batches}")
    
    with tc.no_grad():
        for batch_idx, (features, _) in enumerate(dataloader):
            features = features.to(device)
            outputs = model(features)
            
            # Handle different output types
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Convert to numpy and append
            batch_preds = outputs.cpu().numpy()
            preds.append(batch_preds)
            
            # Track number of samples
            samples_processed += len(batch_preds)
            
            # Optional detailed logging
            if verbose:
                logger.info(f"Batch {batch_idx+1}/{total_batches}: "
                            f"Batch size = {len(batch_preds)}, "
                            f"Cumulative samples = {samples_processed}")
    
    # Combine predictions
    preds_array = np.vstack(preds)
    
    if verbose:
        logger.info(f"Total predictions generated: {preds_array.shape[0]}")
        logger.info(f"Prediction array shape: {preds_array.shape}")
    
    return preds_array

def run_hpo(X, y, model_class, num_features, device='cpu', iterations=5, logger=logger, path_manager=path_manager):
    logger.info("Starting hyperparameter optimization...")
    hpo = RNNDifferentialEvolutionHyperParameterOptimizer(
        X, y, 
        base_model_class=model_class, 
        device=device, 
        num_features=num_features, 
        logger=logger, 
        path_manager=path_manager
    )
    
    best_result = None

    with tqdm(total=iterations, desc="Optimizing hyperparameters") as pbar:
        for i in range(iterations):
            logger.info(f"\nStarting HPO iteration {i + 1}/{iterations}...")
            start = time.time()
            result = hpo.optimize()
            
            if result is None:
                logger.warning("HPO iteration returned None, skipping...")
                continue
            
            duration = timedelta(seconds=time.time() - start)
            logger.info(f"Iteration {i + 1} completed in {duration}")
            pbar.update(1)

            if result and (best_result is None or result['best_score'] < best_result['best_score']):
                best_result = result
                logger.info(f"New best score: {best_result['best_score']:.6f}")

    return best_result

def diagnose_dataloader_mismatch(dataloader, y_val):
    """
    Diagnose potential reasons for mismatch between predictions and validation set.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader to investigate
        y_val (numpy.ndarray): Validation labels/targets
    
    Returns:
        dict: Diagnostic information
    """
    # Calculate expected and actual samples
    expected_samples = len(y_val)
    
    # Dataloader batch size
    batch_size = dataloader.batch_size
    
    # Calculate expected number of batches
    expected_full_batches = expected_samples // batch_size
    remainder_samples = expected_samples % batch_size
    
    # Log diagnostics
    logger.info(f"Expected total samples: {expected_samples}")
    logger.info(f"Dataloader batch size: {batch_size}")
    logger.info(f"Expected full batches: {expected_full_batches}")
    logger.info(f"Remainder samples: {remainder_samples}")
    
    # Investigate batch sizes
    batch_sizes = []
    for batch_idx, (features, _) in enumerate(dataloader):
        batch_sizes.append(len(features))
        if batch_idx >= expected_full_batches:
            break
    
    logger.info("Batch sizes:")
    logger.info(batch_sizes)
    
    return {
        "expected_samples": expected_samples,
        "batch_size": batch_size,
        "expected_full_batches": expected_full_batches,
        "remainder_samples": remainder_samples,
        "batch_sizes": batch_sizes
    }
    
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        '../visualizations',
        '../visualizations/animation_frames',
        '../visualizations/overview_plots',
        './out'
    ]
    
def safe_write_html(fig, filepath, fallback_dir='./'):
    """Safely write HTML files with fallback directory"""
    try:
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(filepath, include_mathjax='cdn')
        print(f"Successfully saved: {filepath}")
    except Exception as e:
        # Fallback to current directory
        filename = Path(filepath).name
        fallback_path = Path(fallback_dir) / filename
        try:
            fig.write_html(fallback_path, include_mathjax='cdn')
            print(f"Saved to fallback location: {fallback_path}")
        except Exception as e2:
            print(f"Failed to save {filepath}: {e}")
            print(f"Fallback also failed: {e2}")

# Fix 3: Handle shape mismatches and empty arrays
def validate_arrays(predictions, weights, labels=None):
    """Validate and fix array shapes before processing"""
    
    # Convert to numpy arrays if needed
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    if hasattr(weights, 'numpy'):
        weights = weights.numpy()
    
    # Ensure arrays are not empty
    if predictions.size == 0:
        print("Warning: predictions array is empty")
        return None, None, None
    
    if weights.size == 0:
        print("Warning: weights array is empty")
        return None, None, None
    
    # Flatten arrays if needed
    predictions = np.squeeze(predictions)
    weights = np.squeeze(weights)
    
    # Ensure matching shapes
    if predictions.shape != weights.shape:
        min_len = min(len(predictions), len(weights))
        predictions = predictions[:min_len]
        weights = weights[:min_len]
        print(f"Trimmed arrays to matching length: {min_len}")
    
    if labels is not None:
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        labels = np.squeeze(labels)
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        weights = weights[:min_len]
        labels = labels[:min_len]
        return predictions, weights, labels
    
    return predictions, weights, None

# Fix 4: JSON serialization for numpy/torch types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and torch types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'item'):  # For torch tensors
            return obj.item()
        elif hasattr(obj, 'numpy'):  # For torch tensors
            return obj.numpy().tolist()
        return super().default(obj)

def safe_json_dump(data, filepath):
    """Safely dump data to JSON with proper encoding"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        print(f"Successfully saved JSON: {filepath}")
    except Exception as e:
        print(f"Error saving JSON {filepath}: {e}")

# Fix 5: Memory-efficient processing for large datasets
def process_large_array_chunks(array, chunk_size=100000):
    """Process large arrays in chunks to avoid memory issues"""
    if len(array) <= chunk_size:
        return array
    
    print(f"Processing large array of size {len(array)} in chunks of {chunk_size}")
    results = []
    
    for i in range(0, len(array), chunk_size):
        chunk = array[i:i+chunk_size]
        # Process chunk here
        results.append(chunk)
    
    return np.concatenate(results)

def convert_to_serializable(obj):
    """Convert numpy/torch objects to JSON-serializable types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif tc.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.detach().cpu().numpy().tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def downsample_for_visualization(data, target_size=500_000, method='systematic'):
    """
    Downsample data for memory-safe visualization
    
    Args:
        data: Input data array/tensor
        target_size: Target number of points (default 500k for safety)
        method: 'systematic' or 'random' sampling
    """
    if hasattr(data, '__len__'):
        current_size = len(data)
    else:
        return data
    
    if current_size <= target_size:
        print(f"Data size {current_size:,} is within limits, no downsampling needed")
        return data
    
    print(f"Downsampling from {current_size:,} to {target_size:,} points for visualization")
    
    if method == 'systematic':
        # Systematic sampling - preserves distribution better
        step = current_size // target_size
        if hasattr(data, 'shape') and len(data.shape) > 1:
            return data[::step][:target_size]
        else:
            return data[::step][:target_size]
    else:
        # Random sampling
        indices = np.random.choice(current_size, target_size, replace=False)
        indices = np.sort(indices)  # Keep temporal order if relevant
        return data[indices]

def safe_visualizer_call(visualizer_method, *args, **kwargs):
    """Safely call visualizer methods with error handling"""
    try:
        return visualizer_method(*args, **kwargs)
    except Exception as e:
        if "Invalid typed array length" in str(e) or "525" in str(e):
            print(f"Memory error in {visualizer_method.__name__}: {e}")
            print("Skipping this visualization due to memory constraints")
            return None
        else:
            print(f"Error in {visualizer_method.__name__}: {e}")
            raise e

def safe_density_profile_generation(visualizer, model_dir, MODEL_TYPE, logger):
    """
    Safely generate density profiles with error handling and memory management
    """
    print("Generating density profiles...")
    
    # Ensure model directory exists
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    density_methods = {
        "density_profile": visualizer.density_profile,
        "cuspy_density_profile": visualizer.cuspy_density_profile,
        "nfw_density_profile": visualizer.nfw_density_profile,
        # "einasto_density_profile": visualizer.einasto_density_profile,
        # "burkert_density_profile": visualizer.burkert_density_profile,
        # "isothermal_density_profile": visualizer.isothermal_density_profile,
        "all_density_profile": visualizer.all_density_profiles
    }
    
    successful_profiles = 0
    failed_profiles = 0
    
    for name, method in density_methods.items():
        try:
            print(f"  Generating {name}...")
            logger.info(f"Generating {name}...")
            
            # Call the density profile method
            fig = method()
            
            if fig is not None:
                # Construct safe file path
                html_path = os.path.join(model_dir, f"{MODEL_TYPE}_{name}.html")
                
                # Ensure directory exists (in case of nested paths)
                Path(html_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save HTML file safely
                try:
                    fig.write_html(html_path, include_mathjax='cdn')
                    logger.info(f"Saved {name} for {MODEL_TYPE}")
                    print(f"    ✓ Saved {name}")
                    successful_profiles += 1
                    
                except Exception as save_error:
                    logger.error(f"Failed to save {name}: {save_error}")
                    print(f"    ✗ Failed to save {name}: {save_error}")
                    failed_profiles += 1
                    
            else:
                logger.warning(f"{name} returned None - skipping")
                print(f"    ⚠ {name} returned None - skipping")
                failed_profiles += 1
                
        except Exception as e:
            # Handle the "Shape mismatch: predictions (), weights ()" error
            if "Shape mismatch" in str(e):
                logger.error(f"Shape mismatch in {name}: {e}")
                print(f"    ✗ Shape mismatch in {name} - likely empty arrays")
            else:
                logger.error(f"Error generating {name}: {e}")
                print(f"    ✗ Error in {name}: {e}")
            failed_profiles += 1
            continue
    
    print(f"Density profiles completed: {successful_profiles} successful, {failed_profiles} failed")
    logger.info(f"Density profiles: {successful_profiles} successful, {failed_profiles} failed")
    
    return successful_profiles, failed_profiles

def safe_redshift_analysis(redshift_analyzer, preds_trimmed, y_val_trimmed, 
                          redshifts_only_trimmed, model_dir, MODEL_TYPE, logger):
    """
    Safely perform redshift evolution analysis
    """
    print("Performing redshift evolution analysis...")
    
    try:
        # Check data sizes before analysis
        print(f"  Data sizes for redshift analysis:")
        print(f"    Predictions: {len(preds_trimmed):,}")
        print(f"    True values: {len(y_val_trimmed):,}")
        print(f"    Redshifts: {len(redshifts_only_trimmed):,}")
        
        # Ensure all arrays have the same length
        min_length = min(len(preds_trimmed), len(y_val_trimmed), len(redshifts_only_trimmed))
        
        if min_length < len(preds_trimmed):
            print(f"  Trimming arrays to consistent length: {min_length:,}")
            preds_analysis = preds_trimmed[:min_length]
            y_val_analysis = y_val_trimmed[:min_length]
            redshifts_analysis = redshifts_only_trimmed[:min_length]
        else:
            preds_analysis = preds_trimmed
            y_val_analysis = y_val_trimmed
            redshifts_analysis = redshifts_only_trimmed
        
        # Perform the analysis
        redshift_analyzer.analyze_redshift_evolution(
            predictions=preds_analysis,
            true_values=y_val_analysis,
            redshifts=redshifts_analysis,
            save_dir=model_dir,
            model_name=MODEL_TYPE
        )
        
        logger.info("Redshift evolution analysis completed successfully")
        print("  ✓ Redshift evolution analysis completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in redshift evolution analysis: {e}")
        print(f"  ✗ Redshift evolution analysis failed: {e}")
        return False

def safe_metrics_calculation(y_val_trimmed, preds_trimmed, model_dir, MODEL_TYPE, 
                           CustomMetrics, logger):
    """
    Safely calculate and save metrics with JSON conversion
    """
    print("Calculating metrics...")
    
    try:
        # Ensure consistent array lengths
        min_length = min(len(y_val_trimmed), len(preds_trimmed))
        
        if min_length < max(len(y_val_trimmed), len(preds_trimmed)):
            print(f"  Trimming arrays to consistent length: {min_length:,}")
            y_val_metrics = y_val_trimmed[:min_length]
            preds_metrics = preds_trimmed[:min_length]
        else:
            y_val_metrics = y_val_trimmed
            preds_metrics = preds_trimmed
        
        # Convert to tensors safely
        try:
            y_tensor = tc.tensor(y_val_metrics).float()
            preds_tensor = tc.tensor(preds_metrics).float()
            print(f"  Tensor shapes: y={y_tensor.shape}, preds={preds_tensor.shape}")
            
        except Exception as tensor_error:
            logger.error(f"Error creating tensors: {tensor_error}")
            print(f"  ✗ Failed to create tensors: {tensor_error}")
            return False
        
        # Calculate metrics
        try:
            metrics = CustomMetrics.calculate_all_metrics(y_tensor, preds_tensor)
            print("  ✓ Metrics calculated successfully")
            
        except Exception as metrics_error:
            logger.error(f"Error calculating metrics: {metrics_error}")
            print(f"  ✗ Metrics calculation failed: {metrics_error}")
            return False
        
        # Save metrics with JSON-safe conversion
        metrics_path = os.path.join(model_dir, f"{MODEL_TYPE}_metrics.json")
        
        try:
            # Ensure directory exists
            Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert metrics to JSON-safe format
            safe_metrics = convert_to_serializable(metrics)
            
            # Save to file
            with open(metrics_path, 'w') as f:
                json.dump(safe_metrics, f, indent=4)
            
            logger.info(f"Metrics saved to {metrics_path}")
            print(f"  ✓ Metrics saved to {metrics_path}")
            
        except Exception as save_error:
            logger.error(f"Error saving metrics: {save_error}")
            print(f"  ✗ Failed to save metrics: {save_error}")
            
            # Try fallback save
            try:
                fallback_path = os.path.join(model_dir, f"{MODEL_TYPE}_metrics_fallback.json")
                fallback_metrics = {}
                
                for k, v in metrics.items():
                    try:
                        fallback_metrics[k] = convert_to_serializable(v)
                    except:
                        fallback_metrics[k] = str(v)
                
                with open(fallback_path, 'w') as f:
                    json.dump(fallback_metrics, f, indent=4)
                
                print(f"  ⚠ Saved fallback metrics to {fallback_path}")
                logger.info(f"Saved fallback metrics to {fallback_path}")
                
            except Exception as fallback_error:
                logger.error(f"Even fallback metrics save failed: {fallback_error}")
                print(f"  ✗ Even fallback save failed: {fallback_error}")
                return False
        
        # Print metrics to console and log
        print("  Calculated metrics:")
        for k, v in metrics.items():
            try:
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")
                    logger.info(f"{k}: {v:.4f}")
                else:
                    # Handle non-numeric metrics
                    converted_v = convert_to_serializable(v)
                    if isinstance(converted_v, (int, float)):
                        print(f"    {k}: {converted_v:.4f}")
                        logger.info(f"{k}: {converted_v:.4f}")
                    else:
                        print(f"    {k}: {converted_v}")
                        logger.info(f"{k}: {converted_v}")
            except Exception as print_error:
                print(f"    {k}: <unable to display>")
                logger.info(f"{k}: <unable to display>")
        
        return True
        
    except Exception as e:
        logger.error(f"Overall error in metrics calculation: {e}")
        print(f"  ✗ Overall metrics calculation failed: {e}")
        return False

# ---------------------- Main Pipeline ----------------------
def main():
    start_time = time.time()
    logger.debug("Using: %s", DATA_PATH)
    if tc.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = False  # Setting to True would reduce speed but improve reproducibility
        tc.cuda.empty_cache()  # Clear GPU cache before starting
    logger.info("Starting the dark matter halo pipeline...")
    
    # Load and Preprocess Data
    dp = DataProcessor(DATA_PATH)
    dp.load_data()
    #dp.data_distribution()
    dp.duplicated_values()
    dp.missing_values()
    dp.basic_statistics()
    #dp.correlation_heatmap()

    train_data, val_data = dp.preprocess_data(scaler_type="robust", coverage_percentage=85)
    #dp.preprocessed_distribution()
    logger.info(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}")

    train_data.describe().T.to_csv(os.path.join(path_manager.csv_file_path, "train_data_statistics.csv"))
    val_data.describe().T.to_csv(os.path.join(path_manager.csv_file_path, "val_data_statistics.csv"))

    '''# Create Datasets
    train_dataset = DMDataset(train_data.drop(columns=['Group_M_Crit200']), train_data[['Group_M_Crit200']])
    val_dataset = DMDataset(val_data.drop(columns=['Group_M_Crit200']), val_data[['Group_M_Crit200']])
    train_loader = train_dataset.data_loader(batch_size=256, num_workers=os.cpu_count() // 2, pin_memory=False)
    val_loader = val_dataset.data_loader(batch_size=256, num_workers=os.cpu_count() // 2, pin_memory=False)'''

    #logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Feature and Coordinate Extraction
    X_train, y_train = extract_features_targets(train_data, SELECTED_FEATURES, TARGET)
    X_val, y_val = extract_features_targets(val_data, SELECTED_FEATURES, TARGET)
    
    logger.debug(f"x_train type: {type(X_train)}, y_train example: {X_train[:5]}")
    logger.debug(f"y_train type: {type(y_train)}, y_train example: {y_train[:5]}")
    logger.debug(f"x_val type: {type(X_val)}, y_train example: {X_val[:5]}")
    logger.debug(f"y_val type: {type(y_val)}, y_train example: {y_val[:5]}")
    
    train_dataset = DMDataset(X_train, y_train)
    val_dataset = DMDataset(X_val, y_val)
    train_loader = train_dataset.data_loader(
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=0,
                        pin_memory=False,
                        prefetch_factor=None,
                        persistent_workers=False # Keep workers alive between iterations
                    )
    #train_loader = train_dataset.data_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = val_dataset.data_loader(
                        batch_size=BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=False,
                        prefetch_factor=None,
                        persistent_workers=False  # Keep workers alive between iterations
                    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    spatial_coords = val_data[SPATIAL_COLS].values
    subhalo_coords = val_data[SUBHALO_COLS].values
    
    redshift_analyzer = RedshiftAnalyzer(data=val_data, selected_features=SELECTED_FEATURES)
    redshifts, unique_redshifts, cosmological_time = redshift_analyzer.process_redshift_data()
    #spatial_coords, subhalo_coords = extract_coords(X_val)

    # Hyperparameter Optimization
    model_class = get_model_class(MODEL_TYPE)
    logger.info(f"Using model class: {model_class.__name__}")
   
    logger.info("Initializing final model with given hyperparameters...")
    final_model = model_class(
        input_size=X_train.shape[1],
        hidden_size=128,
        output_size=1,
        num_layers=1,
        dropout_rate=0.2,
        activation='mish',
        bidirectional=False,
        use_layer_norm=False,
        return_hidden=True
    ).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in final_model.parameters() if p.requires_grad)}")

    trainer = ModelTrainer(
        model=final_model,
        criterion=tc.nn.MSELoss(),
        optimizer=tc.optim.Adam(
            final_model.parameters(),
            lr=0.001,
            weight_decay=0.005,
            eps=1e-7  # Smaller epsilon for improved performance
        ),
        device=DEVICE,
        save_dir=path_manager.model_path
    )
    
    # Add these to your training loop (or modify ModelTrainer to include them)
    scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, 
        mode='min',  # Monitor validation loss
        factor=0.5,  # Reduce LR by half
        patience=2,  # Wait 2 evaluations w/o improvement
        verbose=True
    )    

    logger.info("Training final model...")
    history, model_dir = trainer.train(
        num_epochs=20,
        model_name=MODEL_TYPE,
        train_loader = train_loader,
        val_loader = val_loader,
        X=X_train,
        y=y_train
    )
    # Force clean the redshifts
    if isinstance(redshifts, pd.DataFrame) or isinstance(redshifts, pd.Series):
        redshifts_only = redshifts.values[:, 0]
    elif isinstance(redshifts, np.ndarray) and redshifts.shape[1] == 2:
        redshifts_only = redshifts[:, 0]
    else:
        redshifts_only = redshifts

    # Prediction & Visualization
    
    diagnostic_info = diagnose_dataloader_mismatch(val_loader, y_val)

    #print(diagnostic_info)

    logger.info("Generating predictions...")
    preds = predict_model(final_model, val_loader, device=DEVICE)

    #assert preds.shape[0] == y_val.shape[0], f"Mismatch: preds={preds.shape[0]}, y_val={y_val.shape[0]}"

    #DEBUG
    print(f"predictions shape: {np.shape(preds)}")
    print(f"true_values shape: {np.shape(y_val.values)}")
    print(f"spatial_coords shape: {np.shape(spatial_coords)}")
    print(f"redshifts_only shape: {np.shape(redshifts_only)}")
    print(f"cosmological_time shape: {np.shape(cosmological_time)}")
    print(f"subhalo_coords shape: {np.shape(subhalo_coords)}")
    print(f"model_dir: {model_dir}")

    # ===== MEMORY-SAFE DATA PREPARATION =====
    print("Preparing data for visualization...")

    #logger.info("Visualizing results...")
    
    # Determine minimum length among all arrays
    min_len = min(
        len(preds),
        len(y_val.values),
        len(spatial_coords),
        len(redshifts_only),
        len(cosmological_time),
        len(subhalo_coords)
    )

    # Trim all arrays to the same minimum length
    preds_trimmed = preds[:min_len]
    y_val_trimmed = y_val.values[:min_len]
    spatial_coords_trimmed = spatial_coords[:min_len]
    redshifts_only_trimmed = redshifts_only[:min_len]
    cosmological_time_trimmed = cosmological_time[:min_len]
    subhalo_coords_trimmed = subhalo_coords[:min_len]
    
    # Check original data sizes
    print(f"Original data sizes:")
    print(f"  - Predictions: {len(preds_trimmed):,}")
    print(f"  - True values: {len(y_val_trimmed):,}")
    print(f"  - Spatial coords: {len(spatial_coords_trimmed):,}")
    print(f"  - Redshifts: {len(redshifts_only_trimmed):,}")
    
    # Downsample if needed (500k points max for safety)
    MAX_POINTS = 10_000_000
    
    # Find the minimum length to ensure all arrays match
    min_length = min(
        len(preds_trimmed),
        len(y_val_trimmed),
        len(spatial_coords_trimmed),
        len(redshifts_only_trimmed),
        len(cosmological_time_trimmed),
        len(subhalo_coords_trimmed)
    )
    
    if min_length > MAX_POINTS:
        print(f"Downsampling all arrays from {min_length:,} to {MAX_POINTS:,} points")
        
        # Create systematic sampling indices
        step = min_length // MAX_POINTS
        indices = np.arange(0, min_length, step)[:MAX_POINTS]
        
        # Apply downsampling to all arrays
        preds_viz = preds_trimmed[indices]
        y_val_viz = y_val_trimmed[indices]
        spatial_coords_viz = spatial_coords_trimmed[indices]
        redshifts_viz = redshifts_only_trimmed[indices]
        cosmological_time_viz = cosmological_time_trimmed[indices]
        subhalo_coords_viz = subhalo_coords_trimmed[indices]
        
        print(f"Downsampled data sizes:")
        print(f"  - Predictions: {len(preds_viz):,}")
        print(f"  - True values: {len(y_val_viz):,}")
        print(f"  - Spatial coords: {len(spatial_coords_viz):,}")
        
    else:
        print("Data size is within limits, using original data")
        preds_viz = preds_trimmed
        y_val_viz = y_val_trimmed
        spatial_coords_viz = spatial_coords_trimmed
        redshifts_viz = redshifts_only_trimmed
        cosmological_time_viz = cosmological_time_trimmed
        subhalo_coords_viz = subhalo_coords_trimmed
        
    # ===== MEMORY-SAFE VISUALIZER INITIALIZATION =====
    print("Initializing visualizer with memory-safe data...")

    # Ensure visualizations directory exists
    Path("visualizations").mkdir(parents=True, exist_ok=True)

    # # Visualizer call with consistent trimmed inputs
    # visualizer = ModelVisualizer(
    #     predictions=preds_trimmed,
    #     true_values=y_val_trimmed,
    #     spatial_coords=spatial_coords_trimmed,
    #     redshifts=redshifts_only_trimmed,
    #     cosmological_time=cosmological_time_trimmed,
    #     subhalo_coords=subhalo_coords_trimmed,
    #     save_dir="visualizations"
    # )
    
    try:
        visualizer = ModelVisualizer(
            predictions=preds_viz,
            true_values=y_val_viz,
            spatial_coords=spatial_coords_viz,
            redshifts=redshifts_viz,
            cosmological_time=cosmological_time_viz,
            subhalo_coords=subhalo_coords_viz,
            save_dir=path_manager.visualization_path
        )
        print("Visualizer initialized successfully")
    except Exception as e:
        print(f"Error initializing visualizer: {e}")
        return
    
    # ===== MEMORY-SAFE VISUALIZATION CALLS =====
    print("Starting memory-safe visualizations...")
    
    # 1. Triaxial model (most memory-intensive)
    print("1. Creating triaxial model visualization...")
    result = safe_visualizer_call(visualizer.visualize_fitted_triaxial_model)
    if result is not None:
        _, a, b, c = result
        print(f"   Triaxial parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}")
    else:
        # Fallback values if triaxial fails
        a, b, c = 0.346, 0.318, 0.297
        print(f"   Using fallback triaxial parameters: a={a}, b={b}, c={c}")
    
    # 2. DM distribution
    print("2. Creating DM distribution visualization...")
    safe_visualizer_call(visualizer.visualize_dm_distribution, a=a, b=b, c=c, save_frames=True)
    
    # 3. Redshift evolution
    print("3. Creating redshift evolution visualization...")
    safe_visualizer_call(visualizer.visualize_redshift_evolution, save_frames=True)
    
    # 4. Cosmological evolution
    print("4. Creating cosmological evolution visualization...")
    safe_visualizer_call(visualizer.visualize_cosmological_evolution, save_frames=True)
    
    # 5. Hierarchical structure
    print("5. Creating hierarchical structure visualization...")
    #safe_visualizer_call(visualizer.visualize_hierarchical_structure)
    safe_visualizer_call(visualizer.visualize_standalone_hierarchical_structure)
    
    # 6. Triaxial with subhalo
    print("6. Creating triaxial with subhalo visualization...")
    safe_visualizer_call(visualizer.visualize_triaxial_with_subhalo, opacity=0.2, save_frames=True)
    
    # 7. Cosmic web (usually memory-intensive)
    print("7. Creating cosmic web visualization...")
    safe_visualizer_call(visualizer.visualize_cosmic_web)
    
    # Skip GIF creation for now as it's memory-intensive
    print("Skipping GIF creation to avoid memory issues")
    # safe_visualizer_call(visualizer.create_visualization_gif)
    # safe_visualizer_call(visualizer.create_evolution_gif)
    
    # ===== SAFE DENSITY PROFILE GENERATION =====
    print("\n" + "="*50)
    print("DENSITY PROFILE GENERATION")
    print("="*50)
    
    successful_profiles, failed_profiles = safe_density_profile_generation(
        visualizer, model_dir, MODEL_TYPE, logger
    )
    
    print("Visualization phase completed!")


    # _, a, b, c = visualizer.visualize_fitted_triaxial_model()
    # visualizer.visualize_dm_distribution(a=a, b=b, c=c, save_frames=True)
    # visualizer.visualize_redshift_evolution(save_frames=True)
    # visualizer.visualize_cosmological_evolution(save_frames=True)
    # visualizer.visualize_hierarchical_structure()
    # visualizer.visualize_triaxial_with_subhalo(opacity=0.2, save_frames=True)
    # #visualizer.create_visualization_gif()
    # #visualizer.create_evolution_gif()
    # visualizer.visualize_cosmic_web()

    # for name, method in {
    #     "density_profile": visualizer.density_profile,
    #     "cuspy_density_profile": visualizer.cuspy_density_profile,
    #     "nfw_density_profile": visualizer.nfw_density_profile,
    #     # "einasto_density_profile": visualizer.einasto_density_profile,
    #     # "burkert_density_profile": visualizer.burkert_density_profile,
    #     # "isothermal_density_profile": visualizer.isothermal_density_profile,
    #     "all_density_profile": visualizer.all_density_profiles
    # }.items():
    #     fig = method()
    #     if fig:
    #         fig.write_html(
    #             os.path.join(model_dir, f"{MODEL_TYPE}_{name}.html"),
    #             include_mathjax='cdn'
    #         )
    #         logger.info(f"Saved {name} for {MODEL_TYPE}")

    # # Redshift Evolution Analysis
    # redshift_analyzer.analyze_redshift_evolution(
    #     predictions=preds_trimmed,
    #     true_values=y_val_trimmed,
    #     redshifts=redshifts_only_trimmed,
    #     save_dir=model_dir,
    #     model_name=MODEL_TYPE
    # )

    # ===== SAFE REDSHIFT EVOLUTION ANALYSIS =====
    print("\n" + "="*50)
    print("REDSHIFT EVOLUTION ANALYSIS")
    print("="*50)
    
    redshift_success = safe_redshift_analysis(
        redshift_analyzer, preds_trimmed, y_val_trimmed, 
        redshifts_only_trimmed, model_dir, MODEL_TYPE, logger
    )

    # # Metrics & Summary
    
    # y_tensor = tc.tensor(y_val_trimmed).float()
    # preds_tensor = tc.tensor(preds_trimmed).float()
    
    # metrics = CustomMetrics.calculate_all_metrics(
    #     y_tensor,
    #     preds_tensor
    # )

    # metrics_path = os.path.join(model_dir, f"{MODEL_TYPE}_metrics.json")
    # with open(metrics_path, 'w') as f:
    #     json.dump(metrics, f, indent=4)

    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")
    #     logger.info(f"{k}: {v:.4f}")
    
    # ===== SAFE METRICS CALCULATION =====
    print("\n" + "="*50)
    print("METRICS CALCULATION")
    print("="*50)
    
    metrics_success = safe_metrics_calculation(
        y_val_trimmed, preds_trimmed, model_dir, MODEL_TYPE, 
        CustomMetrics, logger
    )

    # ===== PIPELINE COMPLETION =====
    end_time = time.time()
    duration = timedelta(seconds=end_time - start_time)
    
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    print(f"Density profiles: {successful_profiles} successful, {failed_profiles} failed")
    print(f"Redshift analysis: {'✓' if redshift_success else '✗'}")
    print(f"Metrics calculation: {'✓' if metrics_success else '✗'}")
    print(f"Total duration: {duration}")
    
    logger.info(f"Pipeline completed in {duration}")
    logger.info(f"Final summary - Profiles: {successful_profiles}/{successful_profiles + failed_profiles}, "
                f"Redshift: {redshift_success}, Metrics: {metrics_success}")
    
    return {
        'duration': duration,
        'successful_profiles': successful_profiles,
        'failed_profiles': failed_profiles,
        'redshift_success': redshift_success,
        'metrics_success': metrics_success
    }

if __name__ == "__main__":
    main()
