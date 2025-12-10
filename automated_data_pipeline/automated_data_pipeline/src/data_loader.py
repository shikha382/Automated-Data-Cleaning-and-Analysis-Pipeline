"""
Data Loader Module
==================
This module handles loading data from various sources including CSV files and SQLite databases.

Author: Data Science Engineer
Date: December 2024
"""

import pandas as pd
import sqlite3
import os


class DataLoader:
    """
    A class to handle data loading operations from multiple sources.
    
    Attributes:
        csv_path (str): Path to the CSV file
        db_path (str): Path to the SQLite database file
    """
    
    def __init__(self, csv_path, db_path):
        """
        Initialize the DataLoader with file paths.
        
        Args:
            csv_path (str): Path to the CSV file
            db_path (str): Path to the SQLite database file
        """
        self.csv_path = csv_path
        self.db_path = db_path
    
    def load_csv_data(self):
        """
        Load data from CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing the CSV data
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
            df = pd.read_csv(self.csv_path)
            print(f"✓ Successfully loaded CSV data: {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            print(f"✗ Error loading CSV data: {str(e)}")
            raise
    
    def load_database_data(self, table_name='customers'):
        """
        Load data from SQLite database into a pandas DataFrame.
        
        Args:
            table_name (str): Name of the table to load (default: 'customers')
            
        Returns:
            pd.DataFrame: DataFrame containing the database table data
            
        Raises:
            FileNotFoundError: If the database file does not exist
            sqlite3.Error: If there's an error querying the database
        """
        try:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Database file not found: {self.db_path}")
            
            # Connect to SQLite database
            conn = sqlite3.connect(self.db_path)
            
            # Load data from table
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            
            # Close connection
            conn.close()
            
            print(f"✓ Successfully loaded database data from '{table_name}': {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            print(f"✗ Error loading database data: {str(e)}")
            raise
    
    def merge_datasets(self, df1, df2, on_column='customer_id', how='left'):
        """
        Merge two DataFrames on a common column.
        
        Args:
            df1 (pd.DataFrame): First DataFrame (left)
            df2 (pd.DataFrame): Second DataFrame (right)
            on_column (str): Column name to merge on (default: 'customer_id')
            how (str): Type of merge (default: 'left')
            
        Returns:
            pd.DataFrame: Merged DataFrame
        """
        try:
            merged_df = pd.merge(df1, df2, on=on_column, how=how)
            print(f"✓ Successfully merged datasets: {len(merged_df)} rows, {len(merged_df.columns)} columns")
            return merged_df
        
        except Exception as e:
            print(f"✗ Error merging datasets: {str(e)}")
            raise
    
    def get_data_info(self, df):
        """
        Display basic information about the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
        """
        print("\n" + "="*60)
        print("DATA INFORMATION")
        print("="*60)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn Names and Types:")
        print(df.dtypes)
        print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print("="*60 + "\n")


def main():
    """
    Main function to demonstrate the DataLoader functionality.
    """
    # Define file paths
    csv_path = '../data/raw_data.csv'
    db_path = '../sql/database.db'
    
    # Initialize DataLoader
    loader = DataLoader(csv_path, db_path)
    
    # Load CSV data
    transactions_df = loader.load_csv_data()
    loader.get_data_info(transactions_df)
    
    # Load database data
    customers_df = loader.load_database_data()
    loader.get_data_info(customers_df)
    
    # Merge datasets
    merged_df = loader.merge_datasets(transactions_df, customers_df)
    loader.get_data_info(merged_df)
    
    print("Sample of merged data:")
    print(merged_df.head())


if __name__ == "__main__":
    main()
