#!/usr/bin/env python3
"""
MongoDB Data Access Script for Credit Card Fraud Dataset
This script provides easy access to the dataset stored in MongoDB cloud storage.
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import warnings
warnings.filterwarnings('ignore')

class MongoDBDataAccess:
    """
    Class to access and work with credit card fraud data stored in MongoDB.
    """
    
    def __init__(self):
        """
        Initialize MongoDB connection.
        """
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
    
    def connect(self):
        """
        Connect to MongoDB cloud database.
        """
        print("Connecting to MongoDB cloud storage...")
        
        # MongoDB connection string
        mongo_uri = "mongodb+srv://akhilamohan24:LmEVJjsGeYFcBGKY@datastore.13xdrir.mongodb.net/"
        
        try:
            # Test connection
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            print("MongoDB connection successful")
            
            # Get database and collection
            self.db = self.client['creditcard_fraud']
            self.collection = self.db['transactions']
            
            print(f" Connected to database: {self.db.name}")
            print(f" Using collection: {self.collection.name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f" MongoDB connection failed: {e}")
            print("Please check your internet connection and MongoDB credentials.")
        except Exception as e:
            print(f" Unexpected error connecting to MongoDB: {e}")
    
    def get_dataset_info(self):
        """
        Get basic information about the stored dataset.
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return
        
        try:
            # Get total count
            total_count = self.collection.count_documents({})
            print(f"\nDataset Information:")
            print(f"  - Total transactions: {total_count:,}")
            
            # Check class distribution
            class_counts = self.collection.aggregate([
                {"$group": {"_id": "$Class", "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ])
            
            print(f"\nClass Distribution:")
            for class_info in class_counts:
                class_name = "Legitimate" if class_info["_id"] == 0 else "Fraudulent"
                percentage = (class_info["count"] / total_count) * 100
                print(f"  - {class_name} (Class {class_info['_id']}): {class_info['count']:,} ({percentage:.2f}%)")
            
            # Check amount statistics
            amount_stats = self.collection.aggregate([
                {"$group": {
                    "_id": None,
                    "min_amount": {"$min": "$Amount"},
                    "max_amount": {"$max": "$Amount"},
                    "avg_amount": {"$avg": "$Amount"},
                    "total_amount": {"$sum": "$Amount"}
                }}
            ]).next()
            
            print(f"\nAmount Statistics:")
            print(f"  - Min: ${amount_stats['min_amount']:.2f}")
            print(f"  - Max: ${amount_stats['max_amount']:.2f}")
            print(f"  - Average: ${amount_stats['avg_amount']:.2f}")
            print(f"  - Total: ${amount_stats['total_amount']:,.2f}")
            
            # Check V-features
            sample_doc = self.collection.find_one()
            v_features = [key for key in sample_doc.keys() if key.startswith('V')]
            print(f"\nFeatures:")
            print(f"  - V-features: {len(v_features)} (V1 to V{len(v_features)})")
            print(f"  - Other features: Time, Amount, Class")
            print(f"  - Total features: {len(sample_doc.keys())}")
            
        except Exception as e:
            print(f" Error getting dataset info: {e}")
    
    def load_to_dataframe(self, limit=None, sample_fraction=None):
        """
        Load data from MongoDB to pandas DataFrame.
        
        Args:
            limit (int): Maximum number of documents to load
            sample_fraction (float): Fraction of data to sample (0.0 to 1.0)
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return None
        
        try:
            print("Loading data from MongoDB")
            
            # Build query
            query = {}
            
            # Apply sampling if requested
            if sample_fraction and 0 < sample_fraction < 1:
                pipeline = [
                    {"$sample": {"size": int(self.collection.count_documents({}) * sample_fraction)}}
                ]
                if limit:
                    pipeline.append({"$limit": limit})
                cursor = self.collection.aggregate(pipeline)
            else:
                # Apply limit if requested
                if limit:
                    cursor = self.collection.find(query).limit(limit)
                else:
                    cursor = self.collection.find(query)
            
            # Convert to DataFrame
            df = pd.DataFrame(list(cursor))
            
            print(f" Loaded {len(df):,} records from MongoDB")
            print(f"  - Shape: {df.shape}")
            print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except Exception as e:
            print(f" Error loading data: {e}")
            return None
    
    def get_fraud_transactions(self, limit=None):
        """
        Get only fraudulent transactions.
        
        Args:
            limit (int): Maximum number of fraud transactions to return
        
        Returns:
            pd.DataFrame: Fraudulent transactions
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return None
        
        try:
            print("Loading fraudulent transactions")
            
            query = {"Class": 1}
            if limit:
                cursor = self.collection.find(query).limit(limit)
            else:
                cursor = self.collection.find(query)
            
            df = pd.DataFrame(list(cursor))
            print(f" Loaded {len(df):,} fraudulent transactions")
            
            return df
            
        except Exception as e:
            print(f" Error loading fraud data: {e}")
            return None
    
    def get_legitimate_transactions(self, limit=None):
        """
        Get only legitimate transactions.
        
        Args:
            limit (int): Maximum number of legitimate transactions to return
        
        Returns:
            pd.DataFrame: Legitimate transactions
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return None
        
        try:
            print("Loading legitimate transactions")
            
            query = {"Class": 0}
            if limit:
                cursor = self.collection.find(query).limit(limit)
            else:
                cursor = self.collection.find(query)
            
            df = pd.DataFrame(list(cursor))
            print(f"Loaded {len(df):,} legitimate transactions")
            
            return df
            
        except Exception as e:
            print(f" Error loading legitimate data: {e}")
            return None
    
    def get_transactions_by_amount_range(self, min_amount, max_amount, limit=None):
        """
        Get transactions within a specific amount range.
        
        Args:
            min_amount (float): Minimum transaction amount
            max_amount (float): Maximum transaction amount
            limit (int): Maximum number of transactions to return
        
        Returns:
            pd.DataFrame: Transactions in amount range
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return None
        
        try:
            print(f"Loading transactions with amount between ${min_amount:.2f} and ${max_amount:.2f}...")
            
            query = {"Amount": {"$gte": min_amount, "$lte": max_amount}}
            if limit:
                cursor = self.collection.find(query).limit(limit)
            else:
                cursor = self.collection.find(query)
            
            df = pd.DataFrame(list(cursor))
            print(f" Loaded {len(df):,} transactions in amount range")
            
            return df
            
        except Exception as e:
            print(f" Error loading data by amount: {e}")
            return None
    
    def get_transactions_by_time_range(self, start_time, end_time, limit=None):
        """
        Get transactions within a specific time range.
        
        Args:
            start_time (int): Start time in seconds
            end_time (int): End time in seconds
            limit (int): Maximum number of transactions to return
        
        Returns:
            pd.DataFrame: Transactions in time range
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return None
        
        try:
            print(f"Loading transactions between time {start_time} and {end_time}...")
            
            query = {"Time": {"$gte": start_time, "$lte": end_time}}
            if limit:
                cursor = self.collection.find(query).limit(limit)
            else:
                cursor = self.collection.find(query)
            
            df = pd.DataFrame(list(cursor))
            print(f" Loaded {len(df):,} transactions in time range")
            
            return df
            
        except Exception as e:
            print(f" Error loading data by time: {e}")
            return None
    
    def run_aggregation_query(self, pipeline, limit=None):
        """
        Run custom MongoDB aggregation pipeline.
        
        Args:
            pipeline (list): MongoDB aggregation pipeline
            limit (int): Maximum number of results to return
        
        Returns:
            list: Aggregation results
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return None
        
        try:
            print("Running aggregation query...")
            
            if limit:
                pipeline.append({"$limit": limit})
            
            results = list(self.collection.aggregate(pipeline))
            print(f" Aggregation returned {len(results)} results")
            
            return results
            
        except Exception as e:
            print(f" Error running aggregation: {e}")
            return None
    
    def export_to_csv(self, filename="creditcard_fraud_data.csv", sample_fraction=None):
        """
        Export data from MongoDB to CSV file.
        
        Args:
            filename (str): Output CSV filename
            sample_fraction (float): Fraction of data to export (0.0 to 1.0)
        """
        if not self.collection:
            print(" Not connected to MongoDB")
            return
        
        try:
            print(f"Exporting data to {filename}...")
            
            # Load data
            df = self.load_to_dataframe(sample_fraction=sample_fraction)
            if df is None:
                return
            
            # Export to CSV
            df.to_csv(filename, index=False)
            print(f" Data exported to {filename}")
            print(f"  - File size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f" Error exporting data: {e}")
    
    def close_connection(self):
        """
        Close MongoDB connection.
        """
        if self.client:
            self.client.close()
            print(" MongoDB connection closed")

def main():
    """
    Main function to demonstrate MongoDB data access.
    """
    print("=" * 80)
    print("MongoDB Data Access for Credit Card Fraud Dataset")
    print("=" * 80)
    
    # Initialize data access
    data_access = MongoDBDataAccess()
    
    if not data_access.collection:
        print(" Could not connect to MongoDB. Exiting.")
        return
    
    try:
        # Show dataset information
        data_access.get_dataset_info()
        
        # Example: Load a sample of data
        print(f"\n" + "=" * 60)
        print("Loading Sample Data")
        print("=" * 60)
        
        # Load 10% sample for analysis
        df_sample = data_access.load_to_dataframe(sample_fraction=0.1)
        
        if df_sample is not None:
            print(f"\nSample data loaded successfully!")
            print(f"Shape: {df_sample.shape}")
            print(f"Columns: {list(df_sample.columns)}")
            
            # Show first few rows
            print(f"\nFirst 5 rows:")
            print(df_sample.head())
            
            # Show basic statistics
            print(f"\nBasic statistics:")
            print(df_sample.describe())
        
        # Example: Get fraud transactions
        print(f"\n" + "=" * 60)
        print("Loading Fraudulent Transactions")
        print("=" * 60)
        
        fraud_df = data_access.get_fraud_transactions(limit=1000)
        if fraud_df is not None:
            print(f"Loaded {len(fraud_df)} fraud transactions")
            print(f"Average fraud amount: ${fraud_df['Amount'].mean():.2f}")
        
        # Example: Export sample to CSV
        print(f"\n" + "=" * 60)
        print("Exporting Sample Data")
        print("=" * 60)
        
        data_access.export_to_csv("sample_creditcard_data.csv", sample_fraction=0.05)
        
        print(f"\n" + "=" * 60)
        print("Usage Examples")
        print("=" * 60)
        print("""
# Load all data (be careful with large datasets)
df = data_access.load_to_dataframe()

# Load 20% sample
df_sample = data_access.load_to_dataframe(sample_fraction=0.2)

# Load only fraud transactions
fraud_df = data_access.get_fraud_transactions()

# Load transactions in amount range
amount_df = data_access.get_transactions_by_amount_range(100, 1000)

# Export to CSV
data_access.export_to_csv("my_data.csv", sample_fraction=0.1)

# Custom aggregation
pipeline = [
    {"$group": {"_id": "$Class", "avg_amount": {"$avg": "$Amount"}}}
]
results = data_access.run_aggregation_query(pipeline)
        """)
        
    except KeyboardInterrupt:
        print("\n\n Operation interrupted by user")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
    finally:
        # Always close connection
        data_access.close_connection()

if __name__ == "__main__":
    main()
