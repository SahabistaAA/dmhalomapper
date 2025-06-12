import os
import time
import json
import torch as tc
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

from data_processing import DataProcessor
from custom_dataset import DMDataset
from model_train import ModelTrainer
from model_visualization import ModelVisualizer
from redshift_analyzer import RedshiftAnalyzer
from rnn_model import GRUDMHaloMapper, RNNDMHaloMapper, LSTMDMHaloMapper
from directory_manager import PathManager
from model_metrics import CustomMetrics
from model_hyperparameter_optimizer import RNNOptunaHyperParameterOptimizer  # <<< NEW OPTUNA VERSION

# ---------------------- Configuration ----------------------
#DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0025N0376.feather"
DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0025N0752.feather"
#DATA_PATH = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0100N1504_with_cosmological_time.feather"
SELECTED_FEATURES = [
    'Redshift', 'NumOfSubhalos', 'Velocity_x', 'Velocity_y', 'Velocity_z',
    'MassType_DM', 'CentreOfPotential_x', 'CentreOfPotential_y',
    'CentreOfPotential_z', 'GroupCentreOfPotential_x', 'GroupCentreOfPotential_y',
    'GroupCentreOfPotential_z', 'Vmax', 'VmaxRadius', 'Group_R_Crit200'
]
TARGET = 'Group_M_Crit200'
SPATIAL_COLS = ['GroupCentreOfPotential_x', 'GroupCentreOfPotential_y', 'GroupCentreOfPotential_z']
SUBHALO_COLS = ['CentreOfPotential_x', 'CentreOfPotential_y', 'CentreOfPotential_z']
MODEL_TYPE = 'GRU'
DEVICE = 'cuda' if tc.cuda.is_available() else 'cpu'
HPO_TRIALS = 50  # <<< Number of Optuna trials

path_manager = PathManager()
logger = path_manager.get_logger()

# ---------------------- Helper Functions ----------------------
def extract_features_targets(df, features, target):
    available_features = [feat for feat in features if feat in df.columns]
    X = df[available_features]
    y = df[[target]]
    return X, y

def get_model_class(name='GRU'):
    model_classes = {
        'GRU': GRUDMHaloMapper,
        'LSTM': LSTMDMHaloMapper,
        'RNN': RNNDMHaloMapper
    }
    return model_classes.get(name, GRUDMHaloMapper)

def predict_model(model, dataloader, device):
    model.eval()
    preds = []
    with tc.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            outputs = model(features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds.append(outputs.cpu().numpy())
    return np.vstack(preds)

# ---------------------- Main Pipeline ----------------------
def main():
    start_time = time.time()
    logger.debug("Using: %s", DATA_PATH)
    logger.info("Starting the dark matter halo pipeline with Optuna optimization...")

    dp = DataProcessor(DATA_PATH)
    dp.load_data()
    dp.data_distribution()
    dp.visualize_data_distribution()
    dp.duplicated_values()
    dp.missing_values()
    dp.basic_statistics()
    dp.correlation_heatmap()

    train_data, val_data = dp.preprocess_data(scaler_type="robust", coverage_percentage=85)

    X_train, y_train = extract_features_targets(train_data, SELECTED_FEATURES, TARGET)
    X_val, y_val = extract_features_targets(val_data, SELECTED_FEATURES, TARGET)

    train_dataset = DMDataset(X_train, y_train)
    val_dataset = DMDataset(X_val, y_val)
    train_loader = train_dataset.data_loader(batch_size=128, shuffle=True, num_workers=12)
    val_loader = val_dataset.data_loader(batch_size=128, shuffle=False, num_workers=12)

    spatial_coords = val_data[SPATIAL_COLS].values
    subhalo_coords = val_data[SUBHALO_COLS].values

    redshift_analyzer = RedshiftAnalyzer(data=val_data, selected_features=SELECTED_FEATURES)
    redshifts, unique_redshifts, cosmological_time = redshift_analyzer.process_redshift_data()

    model_class = get_model_class(MODEL_TYPE)

    logger.info("Running Optuna Hyperparameter Optimization...")
    optimizer = RNNOptunaHyperParameterOptimizer(
        X=X_train,
        y=y_train,
        base_model_class=model_class,
        device=DEVICE,
        num_features=X_train.shape[1],
        n_trials=HPO_TRIALS,
        cross_val_folds=3
    )

    optuna_result = optimizer.optimize()
    best_params = optuna_result['best_params']

    logger.info(f"Best Hyperparameters found: {json.dumps(best_params, indent=2)}")

    logger.info("Initializing final model with best Optuna hyperparameters...")
    final_model = model_class(
        input_size=X_train.shape[1],
        hidden_size=best_params['hidden_size'],
        output_size=1,
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        bidirectional=True,
        use_layer_norm=True,
        return_hidden=True
        #activation=best_params['activation']  # 🔥 here too
    ).to(DEVICE)

    trainer = ModelTrainer(
        model=final_model,
        criterion=tc.nn.MSELoss(),
        optimizer=tc.optim.Adam(final_model.parameters(),
                                lr=best_params['learning_rate'],
                                weight_decay=best_params['weight_decay']),
        device=DEVICE,
        save_dir=path_manager.model_path
    )

    logger.info("Training final model...")
    history, model_dir = trainer.train(
        num_epochs=best_params['num_epochs'],
        model_name=MODEL_TYPE,
        train_loader=train_loader,
        val_loader=val_loader,
        X=X_train,
        y=y_train
    )

    if isinstance(redshifts, (pd.DataFrame, pd.Series)):
        redshifts_only = redshifts.values[:, 0]
    else:
        redshifts_only = redshifts[:, 0]

    logger.info("Generating predictions...")
    preds = predict_model(final_model, val_loader, device=DEVICE)

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
        redshifts=redshifts_only,
        cosmological_time=cosmological_time,
        subhalo_coords=subhalo_coords,
        save_dir=model_dir
    )
    _, a, b, c = visualizer.visualize_fitted_triaxial_model()
    visualizer.visualize_dm_distribution(a=a, b=b, c=c)
    visualizer.visualize_cosmological_evolution()
    visualizer.visualize_hierarchical_structure()
    visualizer.visualize_triaxial_with_subhalo(opacity=0.2)
    visualizer.create_evolution_gif()
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
            fig.write_html(os.path.join(model_dir, f"{MODEL_TYPE}_{name}.html"), include_mathjax='cdn')

    logger.info("Analyzing redshift evolution...")
    redshift_analyzer.analyze_redshift_evolution(
        predictions=preds,
        true_values=y_val.values,
        redshifts=redshifts_only,
        save_dir=model_dir,
        model_name=MODEL_TYPE
    )

    metrics = CustomMetrics.calculate_all_metrics(tc.tensor(y_val.values).float(), tc.tensor(preds).float())
    with open(os.path.join(model_dir, f"{MODEL_TYPE}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        logger.info(f"{k}: {v:.4f}")

    end_time = time.time()
    logger.info(f"Pipeline completed in {timedelta(seconds=end_time - start_time)}")

if __name__ == "__main__":
    main()
