#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Credit Card Fraud Dataset
This script fetches data from MongoDB and performs comprehensive EDA and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import warnings
warnings.filterwarnings('ignore')

class CreditCardEDA:
    """
    Class to perform EDA on credit card fraud dataset from MongoDB.
    """
    
    def __init__(self, mongo_uri=None):
        """
        Initialize EDA with MongoDB connection.
        
        Args:
            mongo_uri (str): MongoDB connection string. If None, uses default.
        """
        self.client = None
        self.db = None
        self.collection = None
        self.df = None
        
        # Default MongoDB connection string
        if mongo_uri is None:
            mongo_uri = "mongodb+srv://akhilamohan24:LmEVJjsGeYFcBGKY@datastore.13xdrir.mongodb.net/"
        
        self.mongo_uri = mongo_uri
        self.connect()
        
        # Create plots directory
        Path("plots").mkdir(exist_ok=True)
        
        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def connect(self):
        """
        Connect to MongoDB cloud database.
        """
        print("Connecting to MongoDB for EDA")
        
        try:
            # Test connection
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            print(" MongoDB connection successful")
            
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
    
    def load_data(self, sample_fraction=0.2):
        """
        Load data from MongoDB to pandas DataFrame.
        
        Args:
            sample_fraction (float): Fraction of data to sample (0.0 to 1.0)
        """
        if self.collection is None:
            print(" Not connected to MongoDB")
            return False
        
        try:
            print("Loading data from MongoDB for EDA")
            
            # Get total count
            total_count = self.collection.count_documents({})
            print(f"Total documents in collection: {total_count:,}")
            
            # Apply sampling if requested
            if sample_fraction and 0 < sample_fraction < 1:
                sample_size = int(total_count * sample_fraction)
                print(f"Sampling {sample_size:,} records ({sample_fraction*100:.1f}%)")
                
                try:
                    # Try aggregation with allowDiskUse first
                    pipeline = [
                        {"$sample": {"size": sample_size}}
                    ]
                    cursor = self.collection.aggregate(pipeline, allowDiskUse=True)
                    print(" Using aggregation pipeline with disk usage allowed")
                except Exception as agg_error:
                    print(f" Aggregation failed: {agg_error}")
                    print("Falling back to find() with limit")
                    
                    # Fallback: use find() with limit (less random but more reliable)
                    cursor = self.collection.find({}).limit(sample_size)
                    print(" Using find() with limit as fallback")
            else:
                print("Loading all data (this may take a while for large datasets)")
                cursor = self.collection.find({})
            
            # Convert to DataFrame
            self.df = pd.DataFrame(list(cursor))
            
            print(f" Loaded {len(self.df):,} records from MongoDB")
            print(f"  - Shape: {self.df.shape}")
            print(f"  - Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f" Error loading data: {e}")
            return False
    
    def basic_info(self):
        """
        Display basic information about the dataset.
        """
        if self.df is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumns:")
        for i, col in enumerate(self.df.columns):
            dtype = str(self.df[col].dtype)
            non_null = self.df[col].count()
            null_count = self.df[col].isnull().sum()
            print(f"  {i+1:2d}. {col:15s} | {dtype:10s} | {non_null:8,} non-null | {null_count:5,} null")
        
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
    
    def feature_statistics(self):
        """
        Display statistical information about features.
        """
        if self.df is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        
        # Numerical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"Numerical Features ({len(numerical_cols)}):")
        print(numerical_cols.tolist())
        
        # Categorical features
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical Features ({len(categorical_cols)}):")
            print(categorical_cols.tolist())
        
        # Display statistics for numerical features
        print(f"\nDescriptive Statistics:")
        print(self.df[numerical_cols].describe())
        
        # Class distribution
        if 'Class' in self.df.columns:
            print(f"\nClass Distribution:")
            class_counts = self.df['Class'].value_counts().sort_index()
            for class_val, count in class_counts.items():
                percentage = (count / len(self.df)) * 100
                class_name = "Legitimate" if class_val == 0 else "Fraudulent"
                print(f"  Class {class_val} ({class_name}): {count:,} ({percentage:.2f}%)")
    
    def create_distribution_plots(self):
        """
        Create distribution plots for key features.
        """
        if self.df is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\nCreating distribution plots")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        
        # Amount distribution
        if 'Amount' in self.df.columns:
            axes[0, 0].hist(self.df['Amount'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Transaction Amount Distribution')
            axes[0, 0].set_xlabel('Amount ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Time distribution
        if 'Time' in self.df.columns:
            axes[0, 1].hist(self.df['Time'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Transaction Time Distribution')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Class distribution
        if 'Class' in self.df.columns:
            class_counts = self.df['Class'].value_counts().sort_index()
            colors = ['lightblue', 'lightcoral']
            axes[1, 0].bar(['Legitimate', 'Fraudulent'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Class Distribution')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(class_counts.values):
                axes[1, 0].text(i, v + max(class_counts.values) * 0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        # V1 distribution (example V-feature)
        v_features = [col for col in self.df.columns if col.startswith('V')]
        if v_features:
            axes[1, 1].hist(self.df[v_features[0]], bins=50, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].set_title(f'{v_features[0]} Distribution')
            axes[1, 1].set_xlabel(v_features[0])
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Distribution plots saved to plots/feature_distributions.png")
    
    def create_correlation_analysis(self):
        """
        Create correlation analysis and heatmap.
        """
        if self.df is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\nCreating correlation analysis")
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find top correlations with Class
        if 'Class' in correlation_matrix.columns:
            class_correlations = correlation_matrix['Class'].abs().sort_values(ascending=False)
            print(f"\nTop correlations with Class:")
            for feature, corr in class_correlations.head(10).items():
                if feature != 'Class':
                    print(f"  {feature}: {corr:.4f}")
        
        print(" Correlation heatmap saved to plots/correlation_heatmap.png")
    
    def create_feature_analysis(self):
        """
        Create detailed feature analysis plots.
        """
        if self.df is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\nCreating feature analysis plots")
        
        # Create subplots for different analyses
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Analysis', fontsize=16, fontweight='bold')
        
        # Amount vs Class boxplot
        if 'Amount' in self.df.columns and 'Class' in self.df.columns:
            # Filter out extreme outliers for better visualization
            q99 = self.df['Amount'].quantile(0.99)
            filtered_df = self.df[self.df['Amount'] <= q99]
            
            sns.boxplot(data=filtered_df, x='Class', y='Amount', ax=axes[0, 0])
            axes[0, 0].set_title('Amount Distribution by Class')
            axes[0, 0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
            axes[0, 0].set_ylabel('Amount ($)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Time vs Class boxplot
        if 'Time' in self.df.columns and 'Class' in self.df.columns:
            sns.boxplot(data=self.df, x='Class', y='Time', ax=axes[0, 1])
            axes[0, 1].set_title('Time Distribution by Class')
            axes[0, 1].set_xlabel('Class (0=Legitimate, 1=Fraud)')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # V-features analysis (first few V-features)
        v_features = [col for col in self.df.columns if col.startswith('V')][:3]
        if v_features and 'Class' in self.df.columns:
            # Create violin plot for V1
            sns.violinplot(data=self.df, x='Class', y=v_features[0], ax=axes[1, 0])
            axes[1, 0].set_title(f'{v_features[0]} Distribution by Class')
            axes[1, 0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
            axes[1, 0].set_ylabel(v_features[0])
            axes[1, 0].grid(True, alpha=0.3)
        
        # Amount distribution by class (histogram)
        if 'Amount' in self.df.columns and 'Class' in self.df.columns:
            for class_val in [0, 1]:
                class_name = "Legitimate" if class_val == 0 else "Fraudulent"
                class_data = self.df[self.df['Class'] == class_val]['Amount']
                axes[1, 1].hist(class_data, bins=30, alpha=0.6, label=class_name, density=True)
            
            axes[1, 1].set_title('Amount Distribution by Class (Normalized)')
            axes[1, 1].set_xlabel('Amount ($)')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Feature analysis plots saved to plots/feature_analysis.png")
    
    def create_summary_report(self):
        """
        Create a comprehensive summary report.
        """
        if self.df is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\nCreating summary report")
        
        # Generate report
        report = []
        report.append("="*80)
        report.append("CREDIT CARD FRAUD DATASET - EDA SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        
        # Basic info
        report.append("DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Records: {len(self.df):,}")
        report.append(f"Total Features: {len(self.df.columns)}")
        report.append(f"Dataset Size: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append("")
        
        # Feature breakdown
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        report.append("FEATURE BREAKDOWN")
        report.append("-" * 40)
        report.append(f"Numerical Features: {len(numerical_cols)}")
        report.append(f"Categorical Features: {len(categorical_cols)}")
        report.append("")
        
        # Class distribution
        if 'Class' in self.df.columns:
            report.append("CLASS DISTRIBUTION")
            report.append("-" * 40)
            class_counts = self.df['Class'].value_counts().sort_index()
            for class_val, count in class_counts.items():
                percentage = (count / len(self.df)) * 100
                class_name = "Legitimate" if class_val == 0 else "Fraudulent"
                report.append(f"{class_name}: {count:,} ({percentage:.2f}%)")
            report.append("")
        
        # Amount statistics
        if 'Amount' in self.df.columns:
            report.append("AMOUNT STATISTICS")
            report.append("-" * 40)
            amount_stats = self.df['Amount'].describe()
            report.append(f"Mean: ${amount_stats['mean']:.2f}")
            report.append(f"Median: ${amount_stats['50%']:.2f}")
            report.append(f"Std Dev: ${amount_stats['std']:.2f}")
            report.append(f"Min: ${amount_stats['min']:.2f}")
            report.append(f"Max: ${amount_stats['max']:.2f}")
            report.append("")
        
        # V-features info
        v_features = [col for col in self.df.columns if col.startswith('V')]
        if v_features:
            report.append("V-FEATURES INFORMATION")
            report.append("-" * 40)
            report.append(f"Total V-features: {len(v_features)}")
            report.append(f"Range: {v_features[0]} to {v_features[-1]}")
            report.append("Note: V-features are PCA-transformed numerical features")
            report.append("")
        
        # Data quality
        report.append("DATA QUALITY")
        report.append("-" * 40)
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            report.append("Missing values found:")
            for col, missing in missing_values[missing_values > 0].items():
                percentage = (missing / len(self.df)) * 100
                report.append(f"  {col}: {missing:,} ({percentage:.2f}%)")
        else:
            report.append("No missing values found")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. The dataset shows significant class imbalance - consider techniques like:")
        report.append("   - SMOTE for oversampling")
        report.append("   - Class weights in models")
        report.append("   - Stratified sampling for train/test split")
        report.append("")
        report.append("2. Amount feature shows high variance - consider:")
        report.append("   - Log transformation")
        report.append("   - Robust scaling")
        report.append("   - Outlier detection and handling")
        report.append("")
        report.append("3. V-features are already normalized - ready for ML models")
        report.append("")
        
        # Save report
        with open('plots/eda_summary_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Display report
        for line in report:
            print(line)
        
        print(f"\n Summary report saved to plots/eda_summary_report.txt")
    
    def run_complete_eda(self, sample_fraction=0.2):
        """
        Run complete EDA pipeline.
        
        Args:
            sample_fraction (float): Fraction of data to sample for analysis
        """
        print("="*80)
        print("CREDIT CARD FRAUD DATASET - COMPLETE EDA")
        print("="*80)
        
        # Load data
        if not self.load_data(sample_fraction):
            print(" Failed to load data. Exiting.")
            return
        
        # Run all analyses
        self.basic_info()
        self.feature_statistics()
        self.create_distribution_plots()
        self.create_correlation_analysis()
        self.create_feature_analysis()
        self.create_summary_report()
        
        print("\n" + "="*80)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Generated files:")
        print("  - plots/feature_distributions.png")
        print("  - plots/correlation_heatmap.png")
        print("  - plots/feature_analysis.png")
        print("  - plots/eda_summary_report.txt")
        print("\nYou can now analyze the visualizations and insights!")
    
    def close_connection(self):
        """
        Close MongoDB connection.
        """
        if self.client:
            self.client.close()
            print(" MongoDB connection closed")

def main():
    """
    Main function to run EDA.
    """
    print("Starting EDA for Credit Card Fraud Dataset")
    
    # Initialize EDA
    eda = CreditCardEDA()
    
    if eda.collection is None:
        print(" Could not connect to MongoDB. Exiting.")
        return
    
    try:
        # Run complete EDA with 20% sample for faster processing
        eda.run_complete_eda(sample_fraction=0.2)
        
    except KeyboardInterrupt:
        print("\n\n EDA interrupted by user")
    except Exception as e:
        print(f"\n Unexpected error during EDA: {e}")
    finally:
        # Always close connection
        eda.close_connection()

if __name__ == "__main__":
    main()
