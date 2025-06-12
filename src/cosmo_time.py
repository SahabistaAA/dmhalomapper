import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import os


def add_cosmological_time(data):
    """
    Add lookback time and age of the universe (in Gyr) based on the Redshift column.
    """
    try:
        if data is None:
            print("Warning: Input data is None")
            return None
            
        if "Redshift" not in data.columns:
            print("Warning: 'Redshift' column not found in the data")
            return data
            
        # Create a copy to avoid modifying the original dataframe
        result = data.copy()
            
        # Filter invalid redshift
        invalid_redshift = result[result["Redshift"] < 0]
        if not invalid_redshift.empty:
            print(f"Invalid redshift values found: {invalid_redshift['Redshift'].values}")
            result = result[result["Redshift"] >= 0]   
                     
        # Define cosmological parameters (Planck 2018)
        cosmo = FlatLambdaCDM(name='Planck18', H0=67.66, Om0=0.30966, Tcmb0=2.7255, 
                             Neff=3.046, m_nu=[0.0, 0.0, 0.06], Ob0=0.04897)
            
        # Initialize columns
        result["LookbackTime"] = np.nan
        result["UniverseAge"] = np.nan

        # Only compute for valid redshift (>= 0)
        valid_idx = result[result["Redshift"] >= 0].index
        result.loc[valid_idx, "LookbackTime"] = cosmo.lookback_time(result.loc[valid_idx, "Redshift"]).value
        result.loc[valid_idx, "UniverseAge"] = cosmo.age(result.loc[valid_idx, "Redshift"]).value
        
        return result
    except Exception as e:
        print(f"Error in add_cosmological_time: {e}")
        return None
        
def main():
    # Input/output directory paths
    input_dir = "/mgpfs/home/sarmantara/dmmapper/data"
    output_dir = "/mgpfs/home/sarmantara/dmmapper/data"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List of input files
    l_data = ['merge_L0100N1504.feather', 'merge_L0025N0376.feather', 'merge_L0025N0752.feather']
    
    for file in l_data:
        try:
            input_path = os.path.join(input_dir, file)
            base_name = file.split('.')[0]
            
            print(f"Processing {file}...")
            
            # Read input file
            df = pd.read_feather(input_path)
            print(f"  Loaded data with shape: {df.shape}")
            
            # Add cosmological time data
            result = add_cosmological_time(df)
            if result is None:
                print(f"  Failed to process {file}")
                continue
                
            print(f"  Added cosmological time columns")
            
            # Output feather file
            feather_output = os.path.join(output_dir, f"{base_name}_with_cosmological_time.feather")
            result.to_feather(feather_output)
            print(f"  Saved feather file: {feather_output}")
            
            # Output CSV file
            csv_output = os.path.join(output_dir, f"{base_name}_with_cosmological_time.csv")
            result.to_csv(csv_output, index=False)
            print(f"  Saved CSV file: {csv_output}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()