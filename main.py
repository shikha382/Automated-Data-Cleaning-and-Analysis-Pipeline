"""
Automated Data Cleaning and Analysis Pipeline
===========================================
This is the main execution script for the data science project. It orchestrates the entire
pipeline, from data loading and cleaning to analysis and visualization.

Author: Data Science Engineer
Date: December 2024
"""

import os
import pandas as pd

# Import custom modules
from data_loader import DataLoader
from data_cleaning import DataCleaner
from data_analysis import DataAnalyzer
from visualization import DataVisualizer


def main_pipeline():
    """
    Main function to execute the complete data cleaning and analysis pipeline.
    """
    print("=" * 100)
    print("🚀 STARTING AUTOMATED DATA CLEANING AND ANALYSIS PIPELINE 🚀")
    print("=" * 100)

    # --- 1. Configuration ---
    # Define file paths and settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    SQL_DIR = os.path.join(BASE_DIR, 'sql')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

    CSV_PATH = os.path.join(DATA_DIR, 'raw_data.csv')
    DB_PATH = os.path.join(SQL_DIR, 'database.db')

    # --- 2. Data Loading ---
    print("\n" + "-" * 80)
    print("STEP 1: DATA LOADING")
    print("-" * 80)

    try:
        # Initialize DataLoader
        loader = DataLoader(csv_path=CSV_PATH, db_path=DB_PATH)

        # Load transaction data from CSV
        transactions_df = loader.load_csv_data()

        # Load customer data from SQLite
        customers_df = loader.load_database_data(table_name='customers')

        # Merge datasets
        merged_df = loader.merge_datasets(transactions_df, customers_df, on_column='customer_id', how='left')

        # Display initial data info
        loader.get_data_info(merged_df)

    except Exception as e:
        print(f"✗ CRITICAL ERROR during data loading: {e}")
        return

    # --- 3. Data Cleaning ---
    print("\n" + "-" * 80)
    print("STEP 2: DATA CLEANING")
    print("-" * 80)

    try:
        # Initialize DataCleaner
        cleaner = DataCleaner(merged_df)

        # Check and handle missing values
        cleaner.check_missing_values()
        cleaner.handle_missing_values(strategy='auto')

        # Remove duplicates
        cleaner.remove_duplicates(subset=['transaction_id'], keep='first')

        # Handle outliers in numeric columns
        cleaner.handle_outliers(columns=['amount', 'tax_rate', 'credit_limit'], action='cap')

        # Convert data types for consistency
        type_mapping = {
            'transaction_date': 'datetime64[ns]',
            'registration_date': 'datetime64[ns]',
            'category': 'category',
            'region': 'category',
            'compliance_status': 'category'
        }
        cleaner.convert_data_types(type_mapping)

        # Get the cleaned DataFrame
        cleaned_df = cleaner.get_cleaned_data()

    except Exception as e:
        print(f"✗ CRITICAL ERROR during data cleaning: {e}")
        return

    # --- 4. Data Analysis ---
    print("\n" + "-" * 80)
    print("STEP 3: DATA ANALYSIS")
    print("-" * 80)

    try:
        # Initialize DataAnalyzer
        analyzer = DataAnalyzer(cleaned_df)

        # Generate a comprehensive data quality report
        analyzer.data_quality_report()

        # Perform descriptive statistics
        analyzer.descriptive_statistics()

        # Analyze categorical columns
        analyzer.categorical_analysis()

        # Perform correlation analysis
        analyzer.correlation_analysis(method='pearson')

        # Perform group-by analysis
        analyzer.group_analysis(group_by_col='category', agg_col='amount', agg_func=['mean', 'sum'])
        analyzer.group_analysis(group_by_col='region', agg_col='amount', agg_func=['mean', 'sum'])

        # Generate a final summary report
        analyzer.summary_report()

    except Exception as e:
        print(f"✗ CRITICAL ERROR during data analysis: {e}")
        return

    # --- 5. Data Visualization ---
    print("\n" + "-" * 80)
    print("STEP 4: DATA VISUALIZATION")
    print("-" * 80)

    try:
        # Initialize DataVisualizer
        visualizer = DataVisualizer(cleaned_df, output_dir=OUTPUT_DIR)

        # Create and save various plots
        visualizer.plot_distribution(column='amount', save_name='amount_distribution.png')
        visualizer.plot_bar_chart(column='category', save_name='category_distribution.png')
        visualizer.plot_bar_chart(column='region', save_name='region_distribution.png')
        visualizer.plot_correlation_heatmap(save_name='correlation_heatmap.png')
        visualizer.plot_scatter(x_col='credit_limit', y_col='amount', save_name='credit_limit_vs_amount.png')
        visualizer.plot_box_plot(columns=['amount', 'tax_rate', 'credit_limit'], save_name='numeric_box_plots.png')
        visualizer.plot_pie_chart(column='compliance_status', save_name='compliance_status_pie_chart.png')

        # Create a summary dashboard
        visualizer.create_dashboard(save_name='summary_dashboard.png')

    except Exception as e:
        print(f"✗ CRITICAL ERROR during data visualization: {e}")
        return

    print("\n" + "=" * 100)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY ✅")
    print(f"All outputs and visualizations are saved in: {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main_pipeline()
