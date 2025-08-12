import os
import zipfile
import requests
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def setup_kaggle_api():
    """
    Set up Kaggle API credentials.
    You need to download kaggle.json from your Kaggle account settings.
    """
    print("Setting up Kaggle API...")
    
    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        print("✓ Kaggle CLI is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing Kaggle CLI...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
            print("Kaggle CLI installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install Kaggle CLI: {e}")
            return False
    
    # Check for kaggle.json credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    current_dir_json = Path.cwd() / "kaggle.json"
    
    if current_dir_json.exists():
        print("✓ Kaggle credentials found in current directory")
        # Set environment variable to use current directory
        os.environ['KAGGLE_CONFIG_DIR'] = str(Path.cwd())
        return True
    elif kaggle_json.exists():
        print("✓ Kaggle credentials found in home directory")
        return True
    else:
        print("✗ Kaggle credentials not found")
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place kaggle.json in one of these locations:")
        print(f"   - {kaggle_dir}")
        print(f"   - {Path.cwd()}")
        print("   - Or set KAGGLE_CONFIG_DIR environment variable")
        return False

def download_credit_card_dataset():
    """
    Download credit card fraud detection dataset from Kaggle.
    Using the popular 'Credit Card Fraud Detection' dataset.
    """
    print("\nDownloading Credit Card Fraud Detection dataset...")
    
    try:
        # Download the dataset
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "mlg-ulb/creditcardfraud",
            "-p", "data"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Dataset downloaded successfully")
        
        # Extract the zip file
        zip_path = Path("data") / "creditcardfraud.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data")
            
            # Remove the zip file
            zip_path.unlink()
            print("✓ Dataset extracted successfully")
            
            # List downloaded files
            data_files = list(Path("data").glob("*.csv"))
            if data_files:
                print(f"\nDownloaded files:")
                for file in data_files:
                    print(f"  - {file.name}")
                    # Show file size
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"    Size: {size_mb:.2f} MB")
            else:
                print("No CSV files found in extracted dataset")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def download_alternative_dataset():
    """
    Alternative method: Download using direct URL if Kaggle API fails
    """
    print("\nTrying alternative download method...")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Alternative dataset URL (you can replace this with other credit card datasets)
    url = "https://raw.githubusercontent.com/curiousily/Credit-Card-Fraud-Detection-using-A-TensorFlow-2.0-GAN/master/creditcard.csv"
    
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_path = Path("data") / "creditcard.csv"
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {percent:.1f}%", end="")
        
        print(f"\n✓ Alternative dataset downloaded: {file_path}")
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"✗ Alternative download failed: {e}")
        return False

def create_enhanced_features(data_path):
    """
    Create additional engineered features for the credit card fraud dataset.
    This function adds new features that can improve fraud detection performance.
    """
    print("\nCreating enhanced features...")
    
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        print(f"✓ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Store original features
        original_features = df.columns.tolist()
        
        # 1. STATISTICAL FEATURES
        print("Adding statistical features...")
        
        # Calculate rolling statistics for V1-V28 features
        v_features = [col for col in df.columns if col.startswith('V')]
        
        # Rolling mean and std (window=3)
        for col in v_features:
            df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_rolling_std_3'] = df[col].rolling(window=3, min_periods=1).std()
        
        # Rolling mean and std (window=5)
        for col in v_features:
            df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
            df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
        
        # 2. INTERACTION FEATURES
        print("Adding interaction features...")
        
        # Amount interactions with V features
        for col in v_features[:10]:  # Limit to first 10 to avoid too many features
            df[f'amount_{col}_interaction'] = df['Amount'] * df[col]
        
        # V feature interactions (pairwise)
        for i in range(0, len(v_features), 2):
            if i + 1 < len(v_features):
                col1, col2 = v_features[i], v_features[i + 1]
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        # 3. RATIO FEATURES
        print("Adding ratio features...")
        
        # Amount ratios
        df['amount_v1_ratio'] = df['Amount'] / (df['V1'] + 1e-8)
        df['amount_v2_ratio'] = df['Amount'] / (df['V2'] + 1e-8)
        df['amount_v3_ratio'] = df['Amount'] / (df['V3'] + 1e-8)
        
        # V feature ratios
        df['v1_v2_ratio'] = df['V1'] / (df['V2'] + 1e-8)
        df['v3_v4_ratio'] = df['V3'] / (df['V4'] + 1e-8)
        df['v5_v6_ratio'] = df['V5'] / (df['V6'] + 1e-8)
        
        # 4. POLYNOMIAL FEATURES
        print("Adding polynomial features...")
        
        # Square and cube of important features
        df['amount_squared'] = df['Amount'] ** 2
        df['amount_cubed'] = df['Amount'] ** 3
        
        # Square of key V features
        for col in v_features[:5]:
            df[f'{col}_squared'] = df[col] ** 2
        
        # 5. BINNING FEATURES
        print("Adding binning features...")
        
        # Amount bins
        df['amount_bins'] = pd.cut(df['Amount'], bins=10, labels=False, include_lowest=True)
        
        # V feature bins (for first few features)
        for col in v_features[:3]:
            df[f'{col}_bins'] = pd.cut(df[col], bins=5, labels=False, include_lowest=True)
        
        # 6. AGGREGATE FEATURES
        print("Adding aggregate features...")
        
        # Sum of V features
        df['v_sum'] = df[v_features].sum(axis=1)
        df['v_mean'] = df[v_features].mean(axis=1)
        df['v_std'] = df[v_features].std(axis=1)
        df['v_max'] = df[v_features].max(axis=1)
        df['v_min'] = df[v_features].min(axis=1)
        
        # 7. DIFFERENCE FEATURES
        print("Adding difference features...")
        
        # Differences between consecutive V features
        for i in range(len(v_features) - 1):
            col1, col2 = v_features[i], v_features[i + 1]
            df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
        
        # 8. CROSSING FEATURES
        print("Adding crossing features...")
        
        # Cross features between amount and key V features
        df['amount_v1_cross'] = (df['Amount'] > df['Amount'].median()) & (df['V1'] > df['V1'].median())
        df['amount_v2_cross'] = (df['Amount'] > df['Amount'].median()) & (df['V2'] > df['V2'].median())
        
        # Convert boolean to int
        df['amount_v1_cross'] = df['amount_v1_cross'].astype(int)
        df['amount_v2_cross'] = df['amount_v2_cross'].astype(int)
        
        # 9. NORMALIZED FEATURES
        print("Adding normalized features...")
        
        # Z-score normalization for V features
        for col in v_features:
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        # Min-max normalization for amount
        df['amount_normalized'] = (df['Amount'] - df['Amount'].min()) / (df['Amount'].max() - df['Amount'].min() + 1e-8)
        
        # 10. CATEGORICAL FEATURES
        print("Adding categorical features...")
        
        # High amount flag
        df['high_amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
        
        # Extreme V values flag
        for col in v_features[:5]:
            q95 = df[col].quantile(0.95)
            q05 = df[col].quantile(0.05)
            df[f'{col}_extreme'] = ((df[col] > q95) | (df[col] < q05)).astype(int)
        
        # Clean up any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Save enhanced dataset
        enhanced_path = data_path.replace('.csv', '_enhanced.csv')
        df.to_csv(enhanced_path, index=False)
        
        print(f"✓ Enhanced dataset saved to: {enhanced_path}")
        print(f"✓ Original features: {len(original_features)}")
        print(f"✓ New features: {len(df.columns) - len(original_features)}")
        print(f"✓ Total features: {len(df.columns)}")
        
        # Show feature summary
        print("\nFeature Summary:")
        print(f"  - Original features: {len(original_features)}")
        print(f"  - Statistical features: {len([col for col in df.columns if 'rolling' in col or 'mean' in col or 'std' in col])}")
        print(f"  - Interaction features: {len([col for col in df.columns if 'interaction' in col])}")
        print(f"  - Ratio features: {len([col for col in df.columns if 'ratio' in col])}")
        print(f"  - Polynomial features: {len([col for col in df.columns if 'squared' in col or 'cubed' in col])}")
        print(f"  - Binning features: {len([col for col in df.columns if 'bins' in col])}")
        print(f"  - Aggregate features: {len([col for col in df.columns if col in ['v_sum', 'v_mean', 'v_std', 'v_max', 'v_min']])}")
        print(f"  - Difference features: {len([col for col in df.columns if 'diff' in col])}")
        print(f"  - Crossing features: {len([col for col in df.columns if 'cross' in col])}")
        print(f"  - Normalized features: {len([col for col in df.columns if 'zscore' in col or 'normalized' in col])}")
        print(f"  - Categorical features: {len([col for col in df.columns if col in ['high_amount'] or 'extreme' in col])}")
        
        return enhanced_path
        
    except Exception as e:
        print(f"✗ Error creating enhanced features: {e}")
        return None

def main():
    """
    Main function to orchestrate the dataset acquisition process
    """
    print("=" * 60)
    print("Credit Card Fraud Dataset Acquisition")
    print("=" * 60)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Try Kaggle API first
    if setup_kaggle_api():
        if download_credit_card_dataset():
            print("\n🎉 Dataset acquisition completed successfully!")
            
            # Find the downloaded CSV file
            data_files = list(Path("data").glob("*.csv"))
            if data_files:
                csv_file = data_files[0]
                print(f"\nFound dataset: {csv_file}")
                
                # Create enhanced features
                enhanced_file = create_enhanced_features(str(csv_file))
                if enhanced_file:
                    print(f"\n🎉 Feature enhancement completed!")
                    print(f"Enhanced dataset saved to: {enhanced_file}")
                
            print("You can now use the dataset for your fraud detection analysis.")
            return
    
    # Fallback to alternative method
    print("\nKaggle API method failed. Trying alternative download...")
    if download_alternative_dataset():
        print("\n🎉 Alternative dataset acquisition completed successfully!")
        
        # Create enhanced features for alternative dataset
        csv_file = Path("data") / "creditcard.csv"
        if csv_file.exists():
            enhanced_file = create_enhanced_features(str(csv_file))
            if enhanced_file:
                print(f"\n🎉 Feature enhancement completed!")
                print(f"Enhanced dataset saved to: {enhanced_file}")
        
        print("You can now use the dataset for your fraud detection analysis.")
    else:
        print("\n❌ All download methods failed.")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
