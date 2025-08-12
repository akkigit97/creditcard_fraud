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
        print("Kaggle credentials found in current directory")
        # Set environment variable to use current directory
        os.environ['KAGGLE_CONFIG_DIR'] = str(Path.cwd())
        return True
    elif kaggle_json.exists():
        print("Kaggle credentials found in home directory")
        return True
    else:
        print("!!Kaggle credentials not found!!")
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
        print("Dataset downloaded successfully")
        
        # Extract the zip file
        zip_path = Path("data") / "creditcardfraud.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data")
            
            # Remove the zip file
            zip_path.unlink()
            print("Dataset extracted successfully")
            
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
        print(f"Failed to download dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Use this step only if the API doesn't work for some reason
'''def download_alternative_dataset():
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
        
        print(f"\n Alternative dataset downloaded: {file_path}")
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f" Alternative download failed: {e}")
        return False '''

def main():
    """
    Main function to automate the dataset acquisition process
    """
    print("=" * 60)
    print("Credit Card Fraud Dataset Acquisition")
    print("=" * 60)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Try Kaggle API first
    if setup_kaggle_api():
        if download_credit_card_dataset():
            print("\n Dataset acquisition completed successfully!")
            
            # Find the downloaded CSV file
            data_files = list(Path("data").glob("*.csv"))
            if data_files:
                csv_file = data_files[0]
                print(f"\nFound dataset: {csv_file}")
            return
    
    # Fallback to alternative method
    print("\nKaggle API method failed. Trying alternative download...")
    if download_alternative_dataset():
        print("\n Alternative dataset acquisition completed successfully!")
    else:
        print("\n All download methods failed.")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()

