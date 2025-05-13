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
DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0025N0752_with_cosmological_time.feather"
#DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0100N1504_with_cosmological_time.feather"

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
        activation='gelu',
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
        num_epochs=30,
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

    print(diagnostic_info)

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


    logger.info("Visualizing results...")
    visualizer = ModelVisualizer(
        predictions=preds,
        true_values=y_val.values,
        spatial_coords=spatial_coords,
        redshifts=redshifts_only,  # <-- important
        cosmological_time=cosmological_time,
        subhalo_coords=subhalo_coords,
        save_dir=model_dir
    )
    _, a, b, c = visualizer.visualize_fitted_triaxial_model()
    visualizer.visualize_dm_distribution(a=a, b=b, c=c)
    visualizer.visualize_cosmological_evolution()
    visualizer.visualize_hierarchical_structure()
    visualizer.visualize_triaxial_with_subhalo(opacity=0.2)
    #visualizer.create_visualization_gif()
    #visualizer.create_evolution_gif()
    visualizer.visualize_cosmic_web()

    for name, method in {
        "density_profile": visualizer.density_profile,
        "cuspy_density_profile": visualizer.cuspy_density_profile,
        "nfw_density_profile": visualizer.nfw_density_profile,
        "einasto_density_profile": visualizer.einasto_density_profile,
        "burkert_density_profile": visualizer.burkert_density_profile,
        "isothermal_density_profile": visualizer.isothermal_density_profile,
        "all_density_profile": visualizer.all_density_profiles
    }.items():
        fig = method()
        if fig:
            fig.write_html(
                os.path.join(model_dir, f"{MODEL_TYPE}_{name}.html"),
                include_mathjax='cdn'
            )
            logger.info(f"Saved {name} for {MODEL_TYPE}")


    # Redshift Evolution Analysis
    redshift_analyzer.analyze_redshift_evolution(
        predictions=preds,
        true_values=y_val.values,
        redshifts=redshifts_only,
        save_dir=model_dir,
        model_name=MODEL_TYPE
    )

    # Metrics & Summary
    metrics = CustomMetrics.calculate_all_metrics(
        tc.tensor(y_val.values).float(),
        tc.tensor(preds).float()
    )

    metrics_path = os.path.join(model_dir, f"{MODEL_TYPE}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        logger.info(f"{k}: {v:.4f}")

    end_time = time.time()
    duration = timedelta(seconds=end_time - start_time)
    logger.info(f"Pipeline completed in {duration}")
    print(f"Pipeline completed in {duration}")

if __name__ == "__main__":
    main()
