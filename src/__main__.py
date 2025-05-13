def __main__():
    start_time = time.time()  # Start timer
    print("Starting the pipeline...")
    logger.info("Starting the pipeline...")

    #path_manager = PathManager(base_path='/mgpfs/home/sarmantara/dmmapper/out')

    # Load Data
    data_path = "/mgpfs/home/sarmantara/dmmapper/data/merge_L0100N1504.feather"
    dp = DataProcessor(data_path=data_path)
    print("Loading data...")
    logger.info("Loading data...")
    dp.load_data()
    dp.duplicated_values()
    dp.missing_values()
    
    print("Computing basic statistics...")
    logger.info("Computing basic statistics...")
    dp.basic_statistics()
    
    #print("Generating data visualizations...")
    #logger.info("Generating data visualizations...")
    #dp.data_distribution()
    
    print("Generating Correlation Matrix")
    logger.info("Generating Correlation Matrix")
    dp.correlation_heatmap()
    
    print("Preprocessing data...")
    logger.info("Preprocessing data...")
    train_data, val_data = dp.preprocess_data(scaler_type="robust", coverage_percentage=85)
    print(f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
    logger.info(f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
    
    stats_train = train_data.describe().T
    stats_val = val_data.describe().T
    stats_train.to_csv(os.path.join(path_manager.csv_file_path, "train_data_statistics.csv"))
    stats_val.to_csv(os.path.join(path_manager.csv_file_path, "val_data_statistics.csv"))
    print("Train data statistics saved.")
    logger.info("Train data statistics saved.")
    print("Validation data statistics saved.")
    logger.info("Validation data statistics saved.")
    
    #print("Generating preprocessed data visualization...")
    #logger.info("Generating preprocessed data visualization...")
    #dp.preprocessed_distribution()
    
    #print("Creating dataset loader...")
    logger.info("Creating dataset loader...")
    train_dataset = DMDataset(train_data.drop(columns=['Group_M_Crit200']), train_data[['Group_M_Crit200']])
    val_dataset = DMDataset(val_data.drop(columns=['Group_M_Crit200']), val_data[['Group_M_Crit200']])
    train_loader = train_dataset.data_loader(batch_size=256, num_workers=os.cpu_count() // 2, pin_memory=False)
    val_loader = val_dataset.data_loader(batch_size=256, num_workers=os.cpu_count() // 2, pin_memory=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Validation loader batches: {len(val_loader)}")
    
    selected_features = [
        'Redshift', 'NumOfSubhalos', 'Velocity_x', 'Velocity_y', 'Velocity_z', 
        'MassType_DM', 'Mass', 'CentreOfPotential_x', 'CentreOfPotential_y', 
        'CentreOfPotential_z', 'GroupCentreOfPotential_x', 'GroupCentreOfPotential_y', 
        'GroupCentreOfPotential_z', 'Vmax', 'VmaxRadius', 'Group_R_Crit200'
    ]
    
    X_train = train_data[selected_features]
    y_train = train_data['Group_M_Crit200'].values.reshape(-1, 1)
    
    X_val = val_data[selected_features]
    y_val = val_data['Group_M_Crit200'].values.reshape(-1, 1)
    
    redshift_analyzer = RedshiftAnalyzer(data=val_data, selected_features=selected_features)
    redshift_values, _, cosmological_time_values = redshift_analyzer.process_redshift_data()
    
    # Get spatial coordinates for visualization
    spatial_columns = ['GroupCentreOfPotential_x', 'GroupCentreOfPotential_y', 'GroupCentreOfPotential_z']
    spatial_coords = X_val[spatial_columns].values
    
    subhalo_columns = ['CentreOfPotential_x', 'CentreOfPotential_y', 'CentreOfPotential_z']
    subhalo_coords = X_val[subhalo_columns].values

   
    model_classes = {
        'RNN': RNNDMHaloMapper,
        'LSTM': LSTMDMHaloMapper,
        'GRU': GRUDMHaloMapper
    }

    model_classes = {
        'GRU': GRUDMHaloMapper
        #'LSTM': LSTMDMHaloMapper
        #'RNN': RNNDMHaloMapper
    }
    
    def get_model_class(name='GRU'):
        model_classes = {
            'GRU': GRUDMHaloMapper,
            'LSTM': LSTMDMHaloMapper,
            'RNN': RNNDMHaloMapper
        }
        return model_classes.get(name, GRUDMHaloMapper)

    '''
    print("Starting hyperparameter optimization...")
    logger.info("Starting hyperparameter optimization...")
    
    def run_hpo(X_train, y_train, model_class, num_features, device='cpu', total_iterations=5):
        logger.info("Initializing hyperparameter optimizer...")
        hpo = RNNHyperParameterOptimizer(
            X=X_train,
            y=y_train,
            base_model_class=model_class(),
            device=device,
            num_features=num_features
        )

        best_result = None

        with tqdm(total=total_iterations, desc="Optimizing hyperparameters") as pbar:
            for i in range(total_iterations):
                logger.info(f"Starting HPO iteration {i + 1}/{total_iterations}...")
                start = time.time()
                result = hpo.optimize()
                duration = timedelta(seconds=time.time() - start)

                logger.info(f"Iteration {i + 1} completed in {duration}")
                pbar.update(1)

                # Keep the best result so far
                if best_result is None or result['best_score'] < best_result['best_score']:
                    best_result = result
                    logger.info(f"New best score: {best_result['best_score']:.6f}")

        return best_result

    # Initialize model with best hyperparameters
    model_class = get_model_class('GRU')  # change to 'RNN' or 'LSTM' if needed

    # HPO
    print("Starting hyperparameter optimization...")
    logger.info("Starting hyperparameter optimization...")'''

    '''best_result = run_hpo(X_train, y_train, model_class, X_train.shape[1], device='cpu', total_iterations=5)

    print("Extracting best hyperparameters...")
    logger.info("Extracting best hyperparameters...")

    best_params = best_result['best_params']
    print(f"Best parameters: {best_params}")
    logger.info(f"Best parameters: {best_params}")
    
    #hpo = RNNHyperParameterOptimizer(X_train, y_train, base_model_class=RNNDMHaloMapper(), device='cpu', num_features=X_train.shape[1])
    
    total_iterations = 5
    with tqdm(total=total_iterations, desc="Optimizing hyperparameters") as pbar:
        for _ in range(total_iterations):
            iter_start_time = time.time()
            params = hpo.optimize()
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            formatted_iter_time = time.strftime("%H:%M:%S", time.gmtime(iter_time))
            logger.info(f"Iteration completed in {formatted_iter_time} ({iter_time:.2f} seconds)")
            pbar.update(1)
    
    #print("Extracting best hyperparameters...")
    #logger.info("Extracting best hyperparameters...")
    #best_params = hpo.optimize()
    #print(f"Best parameters: {best_params}")
    #logger.info(f"Best parameters: {best_params}")
    
    
    # Initialize model with best hyperparameters
    #print("Initializing model with best hyperparameters...")
    #logger.info("Initializing model with best hyperparameters...")

    model = GRUDMHaloMapper(
        input_size=X.shape[1],
        hidden_size=best_params['best_params']['hidden_size'],
        output_size=1,
        num_layers=best_params['best_params']['num_layers'],
        dropout_rate=best_params['best_params']['dropout_rate'],
        nonlinearity='gelu'
    )
    '''

    #print("Initializing model with given hyperparameters...")
    logger.info("Initializing model with given hyperparameters...")
    
    model_results = {}
    
    for name, ModelClass in model_classes.items():
        logger.info(f"Training {name} model...")
        model = ModelClass(
            input_size=X_train.shape[1],
            hidden_size=256,
            output_size=1,
            num_layers=2,
            dropout_rate=0.25,
            activation = 'gelu',
            
            bidirectional = True
        )
    
        # Initialize ModelTrainer
        print(f"Initializing ModelTrainer with {name} model...")
        logger.info(f"Initializing ModelTrainer with {name} model...")
        trainer = ModelTrainer(
            model=model,
            criterion=tc.nn.MSELoss(),
            optimizer=tc.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01),
            device='cuda' if tc.cuda.is_available() else 'cpu',
            save_dir=path_manager.model_path  # Use model_path from PathManager
        )
    
        # Train the model
        print(f"Starting model training with {name} model...")
        logger.info(f"Starting model training with {name} model...")
        history, model_dir = trainer.train(
            num_epochs=50,
            model_name=name,
            X= X_train,
            y= y_train
        )
        
        # Predict on validation data
        predictions = []
        with tc.no_grad():
            for batch_features in np.array_split(X_val.values, len(y_val) // 128):
                batch_tensor = tc.tensor(batch_features, dtype=tc.float32).to(trainer.device)
                
                output = model(batch_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                
                batch_preds = output.cpu().numpy()
                predictions.append(batch_preds)

        predictions = np.vstack(predictions).flatten()
        
        # Save model
        y_true = y_val.values.flatten()
        
        # Visualize predictions
        visualizer = ModelVisualizer(predictions, y_val, spatial_coords, redshifts=redshift_values, cosmological_time=cosmological_time_values, subhalo_coords=subhalo_coords ,save_dir=model_dir)
    
        _, a, b, c = visualizer.visualize_fitted_triaxial_model()
    
        visualizer.visualize_dm_distribution(a=a, b=b, c=c)
        
        visualizer.visualize_cosmological_evolution()
        
        visualizer.visualize_hierarchical_structure()
        
        visualizer.create_visualization_gif()
        visualizer.create_evolution_gif()
        
        visualizer.visualize_cosmic_web()
                    
        # Density Profile Visualizations
        density_fig = visualizer.density_profile()
        if density_fig:
            density_fig.write_html(
                os.path.join(model_dir, f"{name}_density_profile.html"),
                include_mathjax='cdn'
            )
            #print(f"Density profile saved for {name}.")
            logger.info(f"Density profile saved for {name}.")

        cuspy_fig = visualizer.cuspy_density_profile()
        if cuspy_fig:
            cuspy_fig.write_html(
                os.path.join(model_dir, f"{name}_cuspy_density_profile.html"),
                include_mathjax='cdn'
            )
            #print(f"Cuspy density profile saved for {name}.")
            logger.info(f"Cuspy density profile saved for {name}.")

        nfw_fig = visualizer.nfw_density_profile()
        if nfw_fig:
            nfw_fig.write_html(
                os.path.join(model_dir, f"{name}_NFW_density_profile.html"),
                include_mathjax='cdn'
            )
            #print(f"NFW density profile saved for {name}.")
            logger.info(f"NFW density profile saved for {name}.")

        all_fig = visualizer.all_density_profiles()
        if all_fig:
            all_fig.write_html(
                os.path.join(model_dir, f"{name}_all_density_profile.html"),
                include_mathjax='cdn'    
            )
            #print(f"Comparison of density profiles saved for {name}")
            logger.info(f"Comparison of density profiles saved for {name}")

        # Redshift evolution analysis
        redshift_analyzer.analyze_redshift_evolution(
            predictions=predictions,
            true_values=y_val,
            redshifts=redshift_values,
            save_dir=model_dir,
            model_name=name
        )
        
        save_path = os.path.join(model_dir)
        #visualizer.create_evolution_gif(save_path)

        # Metrics
        metrics = CustomMetrics.calculate_all_metrics(tc.tensor(y_val), tc.tensor(predictions))
        model_results[name] = {
            'metrics': metrics,
            'history': history
        }

        print(f"\nFinal metrics for {name} model:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        logger.info(f"\nFinal metrics for {name} model:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    # Save model comparison results
    results_df = pd.DataFrame({
        name: {
            'MSE': results['metrics']['mse'],
            'RMSE': results['metrics']['rmse'],
            'MAE': results['metrics']['mae'],
            'R2': results['metrics']['r2'],
            'GaussianNLL': results['metrics']['gaussian_nll'],
            'PoissonNLL': results['metrics']['poisson_nll']
        } for name, results in model_results.items()
    }).T  # Transpose to get models as rows

    comparison_path = os.path.join(path_manager.model_path, "model_comparison_results.csv")
    results_df.to_csv(comparison_path)
    print(f"Model comparison results saved to {comparison_path}")
    logger.info(f"Model comparison results saved to {comparison_path}")

    end_time = time.time()
    total_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"Total execution time: {formatted_time} ({total_time:.2f} seconds)")
    logger.info(f"Total execution time: {formatted_time} ({total_time:.2f} seconds)")

 
if __name__ == "__main__":
    __main__()
