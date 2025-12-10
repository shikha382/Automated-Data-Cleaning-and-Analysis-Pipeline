"""
Data Analysis Module
====================
This module performs exploratory data analysis including statistical summaries,
correlation analysis, and data insights.

Author: Data Science Engineer
Date: December 2024
"""

import pandas as pd
import numpy as np


class DataAnalyzer:
    """
    A class to perform comprehensive exploratory data analysis.
    
    Attributes:
        df (pd.DataFrame): The DataFrame to be analyzed
    """
    
    def __init__(self, df):
        """
        Initialize the DataAnalyzer with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to be analyzed
        """
        self.df = df.copy()
        print(f"DataAnalyzer initialized with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
    
    def descriptive_statistics(self):
        """
        Calculate and display descriptive statistics for numeric columns.
        
        Returns:
            pd.DataFrame: Descriptive statistics summary
        """
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS")
        print("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns found for analysis!")
            return None
        
        stats_df = self.df[numeric_cols].describe()
        print(stats_df)
        print("="*80 + "\n")
        
        return stats_df
    
    def calculate_statistics(self, column):
        """
        Calculate detailed statistics for a specific column.
        
        Args:
            column (str): Column name to analyze
        
        Returns:
            dict: Dictionary containing various statistical measures
        """
        if column not in self.df.columns:
            print(f"✗ Column '{column}' not found in DataFrame")
            return None
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"✗ Column '{column}' is not numeric")
            return None
        
        stats = {
            'count': self.df[column].count(),
            'mean': self.df[column].mean(),
            'median': self.df[column].median(),
            'mode': self.df[column].mode()[0] if not self.df[column].mode().empty else None,
            'std': self.df[column].std(),
            'variance': self.df[column].var(),
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'range': self.df[column].max() - self.df[column].min(),
            'q1': self.df[column].quantile(0.25),
            'q3': self.df[column].quantile(0.75),
            'iqr': self.df[column].quantile(0.75) - self.df[column].quantile(0.25),
            'skewness': self.df[column].skew(),
            'kurtosis': self.df[column].kurtosis()
        }
        
        return stats
    
    def correlation_analysis(self, method='pearson'):
        """
        Calculate correlation matrix for numeric columns.
        
        Args:
            method (str): Correlation method ('pearson', 'kendall', or 'spearman')
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        print("\n" + "="*80)
        print(f"CORRELATION ANALYSIS ({method.upper()} METHOD)")
        print("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis!")
            return None
        
        corr_matrix = self.df[numeric_cols].corr(method=method)
        print(corr_matrix)
        print("="*80 + "\n")
        
        return corr_matrix
    
    def find_strong_correlations(self, threshold=0.7, method='pearson'):
        """
        Find pairs of columns with strong correlations.
        
        Args:
            threshold (float): Correlation threshold (default: 0.7)
            method (str): Correlation method ('pearson', 'kendall', or 'spearman')
        
        Returns:
            list: List of tuples containing strongly correlated column pairs
        """
        print(f"\nFinding correlations with |r| > {threshold}...")
        
        corr_matrix = self.correlation_analysis(method=method)
        
        if corr_matrix is None:
            return []
        
        strong_corr = []
        
        # Iterate through correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > threshold:
                    strong_corr.append((col1, col2, corr_value))
                    print(f"✓ Strong correlation: {col1} ↔ {col2} = {corr_value:.3f}")
        
        if not strong_corr:
            print(f"No correlations found with |r| > {threshold}")
        
        return strong_corr
    
    def categorical_analysis(self):
        """
        Analyze categorical columns with frequency counts and distributions.
        
        Returns:
            dict: Dictionary containing analysis for each categorical column
        """
        print("\n" + "="*80)
        print("CATEGORICAL ANALYSIS")
        print("="*80)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            print("No categorical columns found for analysis!")
            return None
        
        cat_analysis = {}
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print("-" * 40)
            
            value_counts = self.df[col].value_counts()
            value_percentages = (value_counts / len(self.df)) * 100
            
            analysis_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_percentages.round(2)
            })
            
            print(analysis_df)
            
            cat_analysis[col] = {
                'unique_values': self.df[col].nunique(),
                'most_common': value_counts.index[0],
                'most_common_count': value_counts.values[0],
                'value_counts': value_counts.to_dict()
            }
        
        print("="*80 + "\n")
        return cat_analysis
    
    def group_analysis(self, group_by_col, agg_col, agg_func='mean'):
        """
        Perform group-by analysis on the data.
        
        Args:
            group_by_col (str): Column to group by
            agg_col (str): Column to aggregate
            agg_func (str or list): Aggregation function(s) to apply
        
        Returns:
            pd.DataFrame: Grouped and aggregated data
        """
        print("\n" + "="*80)
        print(f"GROUP ANALYSIS: {agg_col} by {group_by_col}")
        print("="*80)
        
        if group_by_col not in self.df.columns or agg_col not in self.df.columns:
            print("✗ Specified columns not found in DataFrame")
            return None
        
        grouped = self.df.groupby(group_by_col)[agg_col].agg(agg_func)
        print(grouped)
        print("="*80 + "\n")
        
        return grouped
    
    def data_quality_report(self):
        """
        Generate a comprehensive data quality report.
        
        Returns:
            pd.DataFrame: Data quality metrics for each column
        """
        print("\n" + "="*80)
        print("DATA QUALITY REPORT")
        print("="*80)
        
        quality_metrics = []
        
        for col in self.df.columns:
            metrics = {
                'Column': col,
                'Data_Type': str(self.df[col].dtype),
                'Non_Null_Count': self.df[col].count(),
                'Null_Count': self.df[col].isnull().sum(),
                'Null_Percentage': f"{(self.df[col].isnull().sum() / len(self.df) * 100):.2f}%",
                'Unique_Values': self.df[col].nunique(),
                'Duplicate_Values': len(self.df[col]) - self.df[col].nunique()
            }
            quality_metrics.append(metrics)
        
        quality_df = pd.DataFrame(quality_metrics)
        print(quality_df.to_string(index=False))
        print("="*80 + "\n")
        
        return quality_df
    
    def summary_report(self):
        """
        Generate a comprehensive summary report of the analysis.
        
        Returns:
            dict: Summary statistics and insights
        """
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'total_missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': f"{(self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100):.2f}%",
            'memory_usage_kb': f"{self.df.memory_usage(deep=True).sum() / 1024:.2f}"
        }
        
        print(f"Total Rows: {summary['total_rows']}")
        print(f"Total Columns: {summary['total_columns']}")
        print(f"Numeric Columns: {summary['numeric_columns']}")
        print(f"Categorical Columns: {summary['categorical_columns']}")
        print(f"Total Missing Values: {summary['total_missing_values']}")
        print(f"Missing Percentage: {summary['missing_percentage']}")
        print(f"Memory Usage: {summary['memory_usage_kb']} KB")
        
        print("="*80 + "\n")
        
        return summary


def main():
    """
    Main function to demonstrate the DataAnalyzer functionality.
    """
    # Create sample data
    data = {
        'amount': [1000, 2000, 1500, 3000, 2500, 1800, 2200, 2800, 1600, 2400],
        'tax': [180, 360, 270, 540, 450, 324, 396, 504, 288, 432],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'status': ['Active', 'Active', 'Inactive', 'Active', 'Active', 
                   'Inactive', 'Active', 'Active', 'Inactive', 'Active']
    }
    df = pd.DataFrame(data)
    
    # Initialize analyzer
    analyzer = DataAnalyzer(df)
    
    # Descriptive statistics
    analyzer.descriptive_statistics()
    
    # Correlation analysis
    analyzer.correlation_analysis()
    
    # Categorical analysis
    analyzer.categorical_analysis()
    
    # Group analysis
    analyzer.group_analysis('category', 'amount', ['mean', 'sum', 'count'])
    
    # Data quality report
    analyzer.data_quality_report()
    
    # Summary report
    analyzer.summary_report()


if __name__ == "__main__":
    main()
