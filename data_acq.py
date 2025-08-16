#!/usr/bin/env python3
"""
Credit Card Fraud Dataset Acquisition & MongoDB Storage
This script handles downloading the dataset from Kaggle and storing it in MongoDB cloud storage.
"""

import os
import zipfile
import requests
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import warnings
warnings.filterwarnings('ignore')

def setup_kaggle_api():
    """
    Set up Kaggle API credentials.
    You need to download kaggle.json from your Kaggle account settings.
    """
    print("Setting up Kaggle API")
    
    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        print("Kaggle CLI is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing Kaggle CLI")
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
        print("Kaggle credentials not found")
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place kaggle.json in one of these locations:")
        print(f"   - {kaggle_dir}")
        print(f"   - {Path.cwd()}")
        print("   - Or set KAGGLE_CONFIG_DIR environment variable")
        return False

def setup_mongodb_connection():
    """
    Set up MongoDB connection to cloud database.
    """
    print("\nSetting up MongoDB connection")
    
    # MongoDB connection string
    mongo_uri = "mongodb+srv://akhilamohan24:LmEVJjsGeYFcBGKY@datastore.13xdrir.mongodb.net/"
    
    try:
        # Test connection
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB connection successful")
        
        # Get database and collection
        db = client['creditcard_fraud']
        collection = db['transactions']
        
        print(f"Connected to database: {db.name}")
        print(f"Using collection: {collection.name}")
        
        return client, db, collection
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection failed: {e}")
        print("Please check your internet connection and MongoDB credentials.")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error connecting to MongoDB: {e}")
        return None, None, None

def download_credit_card_dataset():
    """
    Download credit card fraud detection dataset from Kaggle.
    Using the popular 'Credit Card Fraud Detection' dataset.
    """
    print("\nDownloading Credit Card Fraud Detection dataset")
    
    try:
        # Create temp directory
        Path("temp_data").mkdir(exist_ok=True)
        
        # Download the dataset
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "joebeachcapital/credit-card-fraud",
            "-p", "temp_data"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Dataset downloaded successfully")
        
        # Extract the zip file
        zip_path = Path("temp_data") / "creditcardfraud.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("temp_data")
            
            # Remove the zip file
            zip_path.unlink()
            print("Dataset extracted successfully")
            
            # List downloaded files
            data_files = list(Path("temp_data").glob("*.csv"))
            if data_files:
                print(f"\nDownloaded files:")
                for file in data_files:
                    print(f"  - {file.name}")
                    # Show file size
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"    Size: {size_mb:.2f} MB")
                return data_files[0]  # Return the CSV file path
            else:
                print("No CSV files found in extracted dataset")
                return None
        
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def download_alternative_dataset():
    """
    Alternative method: Download using direct URL if Kaggle API fails
    """
    print("\nTrying alternative download method")
    
    # Create temp directory
    Path("temp_data").mkdir(exist_ok=True)
    
    # Alternative dataset URL
    url = "https://raw.githubusercontent.com/curiousily/Credit-Card-Fraud-Detection-using-A-TensorFlow-2.0-GAN/master/creditcard.csv"
    
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_path = Path("temp_data") / "creditcard.csv"
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
        return file_path
        
    except Exception as e:
        print(f" Alternative download failed: {e}")
        return None

def upload_to_mongodb(csv_file_path, collection):
    """
    Upload the dataset to MongoDB collection.
    """
    print(f"\nUploading dataset to MongoDB")
    
    try:
        # Read CSV file in chunks to handle large datasets
        chunk_size = 10000  # Process 10k rows at a time
        
        # Get total rows for progress tracking
        total_rows = sum(1 for _ in open(csv_file_path)) - 1  # Subtract header
        print(f"Total rows to upload: {total_rows:,}")
        
        # Clear existing data
        collection.delete_many({})
        print(" Cleared existing data from collection")
        
        # Upload data in chunks
        uploaded_rows = 0
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
            # Convert chunk to list of dictionaries
            records = chunk.to_dict('records')
            
            # Upload chunk to MongoDB
            result = collection.insert_many(records)
            uploaded_rows += len(records)
            
            # Show progress
            progress = (uploaded_rows / total_rows) * 100
            print(f"\rUpload progress: {progress:.1f}% ({uploaded_rows:,}/{total_rows:,} rows)", end="")
        
        print(f"\n Dataset uploaded successfully to MongoDB!")
        print(f"  - Total rows uploaded: {uploaded_rows:,}")
        print(f"  - Collection: {collection.name}")
        print(f"  - Database: {collection.database.name}")
        
        # Get collection statistics
        collection_stats = collection.database.command("collstats", collection.name)
        print(f"  - Storage size: {collection_stats['size'] / (1024*1024):.2f} MB")
        print(f"  - Index size: {collection_stats['totalIndexSize'] / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n Failed to upload to MongoDB: {e}")
        return False

def create_mongodb_indexes(collection):
    """
    Create useful indexes for the credit card fraud dataset.
    """
    print("\nCreating MongoDB indexes for optimal performance")
    
    try:
        # Create indexes for common query patterns
        indexes_created = []
        
        # Index on Class (target variable) for fraud detection queries
        collection.create_index("Class")
        indexes_created.append("Class")
        
        # Index on Amount for amount-based queries
        collection.create_index("Amount")
        indexes_created.append("Amount")
        
        # Index on Time for temporal queries
        collection.create_index("Time")
        indexes_created.append("Time")
        
        # Compound index on Class and Amount (common fraud detection pattern)
        collection.create_index([("Class", 1), ("Amount", 1)])
        indexes_created.append("Class_Amount")
        
        # Index on V-features for pattern analysis (first few V-features)
        for i in range(1, 6):  # V1 to V5
            feature_name = f"V{i}"
            if feature_name in collection.find_one().keys():
                collection.create_index(feature_name)
                indexes_created.append(feature_name)
        
        print(f" Created {len(indexes_created)} indexes: {', '.join(indexes_created)}")
        
        # Show index information
        index_info = collection.list_indexes()
        print(f"\nIndex details:")
        for index in index_info:
            print(f"  - {index['name']}: {index['key']}")
        
        return True
        
    except Exception as e:
        print(f" Failed to create indexes: {e}")
        return False

def verify_mongodb_data(collection):
    """
    Verify the uploaded data in MongoDB.
    """
    print("\nVerifying uploaded data")
    
    try:
        # Get total count
        total_count = collection.count_documents({})
        print(f" Total documents in collection: {total_count:,}")
        
        # Check class distribution
        class_counts = collection.aggregate([
            {"$group": {"_id": "$Class", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ])
        
        print(f"\nClass distribution:")
        for class_info in class_counts:
            class_name = "Legitimate" if class_info["_id"] == 0 else "Fraudulent"
            percentage = (class_info["count"] / total_count) * 100
            print(f"  - {class_name} (Class {class_info['_id']}): {class_info['count']:,} ({percentage:.2f}%)")
        
        # Check amount statistics
        amount_stats = collection.aggregate([
            {"$group": {
                "_id": None,
                "min_amount": {"$min": "$Amount"},
                "max_amount": {"$max": "$Amount"},
                "avg_amount": {"$avg": "$Amount"},
                "total_amount": {"$sum": "$Amount"}
            }}
        ]).next()
        
        print(f"\nAmount statistics:")
        print(f"  - Min: ${amount_stats['min_amount']:.2f}")
        print(f"  - Max: ${amount_stats['max_amount']:.2f}")
        print(f"  - Average: ${amount_stats['avg_amount']:.2f}")
        print(f"  - Total: ${amount_stats['total_amount']:,.2f}")
        
        # Check V-features
        sample_doc = collection.find_one()
        v_features = [key for key in sample_doc.keys() if key.startswith('V')]
        print(f"\nV-features found: {len(v_features)}")
        print(f"  - Range: {v_features[0]} to {v_features[-1]}")
        
        return True
        
    except Exception as e:
        print(f" Failed to verify data: {e}")
        return False

def cleanup_temp_files():
    """
    Clean up temporary files after successful upload.
    """
    print("\nCleaning up temporary files")
    
    try:
        temp_dir = Path("temp_data")
        if temp_dir.exists():
            # Remove all files in temp directory
            for file in temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    import shutil
                    shutil.rmtree(file)
            
            # Remove temp directory
            temp_dir.rmdir()
            print(" Temporary files cleaned up")
        else:
            print(" No temporary files to clean")
            
    except Exception as e:
        print(f" Warning: Could not clean up temporary files: {e}")

def main():
    """
    Main function to orchestrate the dataset acquisition and MongoDB upload process.
    """
    print("=" * 80)
    print("Credit Card Fraud Dataset Acquisition & MongoDB Storage")
    print("=" * 80)
    
    # Setup MongoDB connection
    client, db, collection = setup_mongodb_connection()
    if not client:
        print(" MongoDB setup failed. Exiting.")
        return
    
    try:
        # Try Kaggle API first
        if setup_kaggle_api():
            csv_file = download_credit_card_dataset()
            if csv_file:
                print(f"\n Dataset downloaded successfully from Kaggle!")
                
                # Upload to MongoDB
                if upload_to_mongodb(csv_file, collection):
                    # Create indexes
                    create_mongodb_indexes(collection)
                    
                    # Verify data
                    verify_mongodb_data(collection)
                    
                    # Cleanup
                    cleanup_temp_files()
                    
                    print("\n Complete! Dataset is now stored in MongoDB cloud storage.")
                    print(f"\nMongoDB Connection Details:")
                    print(f"  - Database: {db.name}")
                    print(f"  - Collection: {collection.name}")
                
                    return
                else:
                    print(" Failed to upload dataset to MongoDB")
            else:
                print(" Failed to download dataset from Kaggle")
        
        # Fallback to alternative method
        print("\nKaggle API method failed. Trying alternative download")
        csv_file = download_alternative_dataset()
        if csv_file:
            print(f"\n Alternative dataset acquisition completed successfully!")
            
            # Upload to MongoDB
            if upload_to_mongodb(csv_file, collection):
                # Create indexes
                create_mongodb_indexes(collection)
                
                # Verify data
                verify_mongodb_data(collection)
                
                # Cleanup
                cleanup_temp_files()
                
                print("\n Complete! Dataset is now stored in MongoDB cloud storage.")
                print(f"\nMongoDB Connection Details:")
                print(f"  - Database: {db.name}")
                print(f"  - Collection: {collection.name}")
                
            else:
                print(" Failed to upload alternative dataset to MongoDB")
        else:
            print(" All download methods failed.")
            print("Please check your internet connection and try again.")
    
    finally:
        # Always close MongoDB connection
        if client:
            client.close()
            print("\n✓ MongoDB connection closed")

if __name__ == "__main__":
    main()
