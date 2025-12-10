"""
Data Cleaning Module
====================
This module handles all data cleaning operations including missing value handling,
duplicate removal, and outlier detection.

Author: Data Science Engineer
Date: December 2024
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """
    A class to perform comprehensive data cleaning operations.
    
    Attributes:
        df (pd.DataFrame): The DataFrame to be cleaned
        original_shape (tuple): Original shape of the DataFrame
    """
    
    def __init__(self, df):
        """
        Initialize the DataCleaner with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to be cleaned
        """
        self.df = df.copy()
        self.original_shape = df.shape
        print(f"DataCleaner initialized with {self.original_shape[0]} rows and {self.original_shape[1]} columns")
    
    def check_missing_values(self):
        """
        Check and report missing values in the DataFrame.
        
        Returns:
            pd.DataFrame: Summary of missing values per column
        """
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percent': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("No missing values found!")
        
        print("="*60 + "\n")
        return missing_df
    
    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values using various strategies.
        
        Args:
            strategy (str): Strategy to handle missing values
                - 'auto': Automatically choose strategy based on data type
                - 'mean': Fill with mean (numeric columns)
                - 'median': Fill with median (numeric columns)
                - 'mode': Fill with mode (categorical columns)
                - 'drop': Drop rows with missing values
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        print(f"Handling missing values using '{strategy}' strategy...")
        
        if strategy == 'drop':
            before_rows = len(self.df)
            self.df = self.df.dropna()
            after_rows = len(self.df)
            print(f"✓ Dropped {before_rows - after_rows} rows with missing values")
        
        elif strategy == 'auto':
            # Handle numeric columns with median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_value = self.df[col].median()
                    self.df[col].fillna(median_value, inplace=True)
                    print(f"✓ Filled '{col}' with median: {median_value:.2f}")
            
            # Handle categorical columns with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col].fillna(mode_value, inplace=True)
                    print(f"✓ Filled '{col}' with mode: {mode_value}")
        
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    mean_value = self.df[col].mean()
                    self.df[col].fillna(mean_value, inplace=True)
                    print(f"✓ Filled '{col}' with mean: {mean_value:.2f}")
        
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_value = self.df[col].median()
                    self.df[col].fillna(median_value, inplace=True)
                    print(f"✓ Filled '{col}' with median: {median_value:.2f}")
        
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset (list): List of columns to consider for duplicates (default: all columns)
            keep (str): Which duplicate to keep ('first', 'last', False to drop all)
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        print("\n" + "="*60)
        print("DUPLICATE REMOVAL")
        print("="*60)
        
        before_rows = len(self.df)
        duplicates = self.df.duplicated(subset=subset, keep=keep).sum()
        
        if duplicates > 0:
            self.df = self.df.drop_duplicates(subset=subset, keep=keep)
            after_rows = len(self.df)
            print(f"✓ Found and removed {before_rows - after_rows} duplicate rows")
        else:
            print("No duplicate rows found!")
        
        print("="*60 + "\n")
        return self.df
    
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """
        Detect outliers using the Interquartile Range (IQR) method.
        
        Args:
            column (str): Column name to check for outliers
            multiplier (float): IQR multiplier for outlier threshold (default: 1.5)
        
        Returns:
            tuple: (lower_bound, upper_bound, outlier_count)
        """
        if column not in self.df.columns:
            print(f"✗ Column '{column}' not found in DataFrame")
            return None, None, 0
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"✗ Column '{column}' is not numeric")
            return None, None, 0
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        outlier_count = len(outliers)
        
        return lower_bound, upper_bound, outlier_count
    
    def handle_outliers(self, columns=None, method='iqr', action='cap'):
        """
        Handle outliers in specified columns.
        
        Args:
            columns (list): List of column names to check (default: all numeric columns)
            method (str): Method to detect outliers ('iqr' or 'zscore')
            action (str): Action to take ('cap', 'remove', or 'report')
        
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        print("\n" + "="*60)
        print("OUTLIER DETECTION AND HANDLING")
        print("="*60)
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_summary = []
        
        for col in columns:
            if method == 'iqr':
                lower_bound, upper_bound, outlier_count = self.detect_outliers_iqr(col)
                
                if outlier_count > 0:
                    outlier_summary.append({
                        'Column': col,
                        'Outliers': outlier_count,
                        'Lower_Bound': f"{lower_bound:.2f}",
                        'Upper_Bound': f"{upper_bound:.2f}"
                    })
                    
                    if action == 'cap':
                        # Cap outliers to bounds
                        self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                        self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                        print(f"✓ Capped {outlier_count} outliers in '{col}'")
                    
                    elif action == 'remove':
                        # Remove rows with outliers
                        before_rows = len(self.df)
                        self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                        after_rows = len(self.df)
                        print(f"✓ Removed {before_rows - after_rows} rows with outliers in '{col}'")
        
        if outlier_summary:
            print("\nOutlier Summary:")
            print(pd.DataFrame(outlier_summary).to_string(index=False))
        else:
            print("No outliers detected!")
        
        print("="*60 + "\n")
        return self.df
    
    def convert_data_types(self, type_mapping):
        """
        Convert data types of specified columns.
        
        Args:
            type_mapping (dict): Dictionary mapping column names to desired data types
        
        Returns:
            pd.DataFrame: DataFrame with converted data types
        """
        print("\n" + "="*60)
        print("DATA TYPE CONVERSION")
        print("="*60)
        
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(dtype)
                    print(f"✓ Converted '{col}' to {dtype}")
                except Exception as e:
                    print(f"✗ Error converting '{col}': {str(e)}")
        
        print("="*60 + "\n")
        return self.df
    
    def get_cleaned_data(self):
        """
        Get the cleaned DataFrame.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        print(f"\nCleaning Summary:")
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        return self.df


def main():
    """
    Main function to demonstrate the DataCleaner functionality.
    """
    # Create sample data with issues
    data = {
        'A': [1, 2, np.nan, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X']
    }
    df = pd.DataFrame(data)
    
    # Add duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    # Initialize cleaner
    cleaner = DataCleaner(df)
    
    # Check missing values
    cleaner.check_missing_values()
    
    # Handle missing values
    cleaner.handle_missing_values(strategy='auto')
    
    # Remove duplicates
    cleaner.remove_duplicates()
    
    # Handle outliers
    cleaner.handle_outliers(columns=['A'], action='cap')
    
    # Get cleaned data
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned Data:")
    print(cleaned_df)


if __name__ == "__main__":
    main()
