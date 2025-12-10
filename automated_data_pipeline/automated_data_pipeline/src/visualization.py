"""
Visualization Module
====================
This module handles data visualization using Matplotlib to create insightful charts and graphs.

Author: Data Science Engineer
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Use non-interactive backend for server environments
matplotlib.use('Agg')


class DataVisualizer:
    """
    A class to create various data visualizations.
    
    Attributes:
        df (pd.DataFrame): The DataFrame to visualize
        output_dir (str): Directory to save visualization files
    """
    
    def __init__(self, df, output_dir='../outputs'):
        """
        Initialize the DataVisualizer with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to visualize
            output_dir (str): Directory to save plots (default: '../outputs')
        """
        self.df = df.copy()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        print(f"DataVisualizer initialized. Plots will be saved to: {output_dir}")
    
    def plot_distribution(self, column, bins=30, color='steelblue', save_name=None):
        """
        Create a histogram to visualize the distribution of a numeric column.
        
        Args:
            column (str): Column name to plot
            bins (int): Number of bins for histogram (default: 30)
            color (str): Color of the bars (default: 'steelblue')
            save_name (str): Filename to save the plot (optional)
        
        Returns:
            str: Path to saved plot file
        """
        if column not in self.df.columns:
            print(f"✗ Column '{column}' not found in DataFrame")
            return None
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"✗ Column '{column}' is not numeric")
            return None
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.df[column].dropna(), bins=bins, color=color, edgecolor='black', alpha=0.7)
        plt.xlabel(column, fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = self.df[column].mean()
        median_val = self.df[column].median()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        plt.legend()
        
        # Save plot
        if save_name is None:
            save_name = f'distribution_{column}.png'
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Distribution plot saved: {save_path}")
        return save_path
    
    def plot_bar_chart(self, column, top_n=10, color='coral', save_name=None):
        """
        Create a bar chart for categorical data.
        
        Args:
            column (str): Column name to plot
            top_n (int): Number of top categories to display (default: 10)
            color (str): Color of the bars (default: 'coral')
            save_name (str): Filename to save the plot (optional)
        
        Returns:
            str: Path to saved plot file
        """
        if column not in self.df.columns:
            print(f"✗ Column '{column}' not found in DataFrame")
            return None
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(value_counts)), value_counts.values, color=color, edgecolor='black', alpha=0.7)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.xlabel(column, fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} {column} Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Save plot
        if save_name is None:
            save_name = f'bar_chart_{column}.png'
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Bar chart saved: {save_path}")
        return save_path
    
    def plot_correlation_heatmap(self, save_name='correlation_heatmap.png', cmap='coolwarm'):
        """
        Create a correlation heatmap for numeric columns.
        
        Args:
            save_name (str): Filename to save the plot
            cmap (str): Colormap for the heatmap (default: 'coolwarm')
        
        Returns:
            str: Path to saved plot file
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("✗ Need at least 2 numeric columns for correlation heatmap")
            return None
        
        corr_matrix = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
        
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Correlation heatmap saved: {save_path}")
        return save_path
    
    def plot_scatter(self, x_col, y_col, color='purple', save_name=None):
        """
        Create a scatter plot for two numeric columns.
        
        Args:
            x_col (str): Column name for x-axis
            y_col (str): Column name for y-axis
            color (str): Color of the points (default: 'purple')
            save_name (str): Filename to save the plot (optional)
        
        Returns:
            str: Path to saved plot file
        """
        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"✗ One or both columns not found in DataFrame")
            return None
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[x_col], self.df[y_col], color=color, alpha=0.6, edgecolors='black', s=100)
        plt.xlabel(x_col, fontsize=12, fontweight='bold')
        plt.ylabel(y_col, fontsize=12, fontweight='bold')
        plt.title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df[x_col].dropna(), self.df[y_col].dropna(), 1)
        p = np.poly1d(z)
        plt.plot(self.df[x_col], p(self.df[x_col]), "r--", linewidth=2, label='Trend Line')
        plt.legend()
        
        # Save plot
        if save_name is None:
            save_name = f'scatter_{x_col}_vs_{y_col}.png'
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Scatter plot saved: {save_path}")
        return save_path
    
    def plot_box_plot(self, columns=None, save_name='box_plot.png'):
        """
        Create box plots for numeric columns to visualize distributions and outliers.
        
        Args:
            columns (list): List of columns to plot (default: all numeric columns)
            save_name (str): Filename to save the plot
        
        Returns:
            str: Path to saved plot file
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) == 0:
            print("✗ No numeric columns found for box plot")
            return None
        
        plt.figure(figsize=(12, 6))
        self.df[columns].boxplot(patch_artist=True)
        plt.ylabel('Values', fontsize=12, fontweight='bold')
        plt.title('Box Plot - Distribution and Outliers', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Box plot saved: {save_path}")
        return save_path
    
    def plot_pie_chart(self, column, top_n=8, save_name=None):
        """
        Create a pie chart for categorical data.
        
        Args:
            column (str): Column name to plot
            top_n (int): Number of top categories to display (default: 8)
            save_name (str): Filename to save the plot (optional)
        
        Returns:
            str: Path to saved plot file
        """
        if column not in self.df.columns:
            print(f"✗ Column '{column}' not found in DataFrame")
            return None
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(value_counts)))
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors, textprops={'fontsize': 11, 'fontweight': 'bold'})
        plt.title(f'{column} Distribution', fontsize=14, fontweight='bold')
        
        # Save plot
        if save_name is None:
            save_name = f'pie_chart_{column}.png'
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Pie chart saved: {save_path}")
        return save_path
    
    def create_dashboard(self, save_name='dashboard.png'):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            save_name (str): Filename to save the dashboard
        
        Returns:
            str: Path to saved dashboard file
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Distribution of first numeric column
        if len(numeric_cols) >= 1:
            ax1 = plt.subplot(2, 3, 1)
            self.df[numeric_cols[0]].hist(bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.set_title(f'Distribution: {numeric_cols[0]}', fontweight='bold')
            ax1.set_xlabel(numeric_cols[0])
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bar chart of first categorical column
        if len(categorical_cols) >= 1:
            ax2 = plt.subplot(2, 3, 2)
            value_counts = self.df[categorical_cols[0]].value_counts().head(8)
            value_counts.plot(kind='bar', color='coral', edgecolor='black', alpha=0.7, ax=ax2)
            ax2.set_title(f'Distribution: {categorical_cols[0]}', fontweight='bold')
            ax2.set_xlabel(categorical_cols[0])
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Box plot of numeric columns
        if len(numeric_cols) >= 1:
            ax3 = plt.subplot(2, 3, 3)
            self.df[numeric_cols[:3]].boxplot(ax=ax3, patch_artist=True)
            ax3.set_title('Box Plot - Outlier Detection', fontweight='bold')
            ax3.set_ylabel('Values')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Correlation heatmap (if multiple numeric columns)
        if len(numeric_cols) >= 2:
            ax4 = plt.subplot(2, 3, 4)
            corr_matrix = self.df[numeric_cols].corr()
            im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(corr_matrix.columns)))
            ax4.set_yticks(range(len(corr_matrix.columns)))
            ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax4.set_yticklabels(corr_matrix.columns)
            ax4.set_title('Correlation Heatmap', fontweight='bold')
            plt.colorbar(im, ax=ax4)
        
        # Plot 5: Scatter plot (if at least 2 numeric columns)
        if len(numeric_cols) >= 2:
            ax5 = plt.subplot(2, 3, 5)
            ax5.scatter(self.df[numeric_cols[0]], self.df[numeric_cols[1]], 
                       color='purple', alpha=0.6, edgecolors='black')
            ax5.set_title(f'{numeric_cols[1]} vs {numeric_cols[0]}', fontweight='bold')
            ax5.set_xlabel(numeric_cols[0])
            ax5.set_ylabel(numeric_cols[1])
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Pie chart (if categorical columns exist)
        if len(categorical_cols) >= 1:
            ax6 = plt.subplot(2, 3, 6)
            value_counts = self.df[categorical_cols[0]].value_counts().head(6)
            ax6.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3(range(len(value_counts))))
            ax6.set_title(f'{categorical_cols[0]} Distribution', fontweight='bold')
        
        # Save dashboard
        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Dashboard saved: {save_path}")
        return save_path


def main():
    """
    Main function to demonstrate the DataVisualizer functionality.
    """
    # Create sample data
    data = {
        'amount': np.random.normal(2000, 500, 100),
        'tax': np.random.normal(360, 90, 100),
        'category': np.random.choice(['Software', 'Hardware', 'Consulting', 'Training'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    }
    df = pd.DataFrame(data)
    
    # Initialize visualizer
    visualizer = DataVisualizer(df, output_dir='../outputs')
    
    # Create various plots
    visualizer.plot_distribution('amount')
    visualizer.plot_bar_chart('category')
    visualizer.plot_correlation_heatmap()
    visualizer.plot_scatter('amount', 'tax')
    visualizer.plot_box_plot(['amount', 'tax'])
    visualizer.plot_pie_chart('region')
    visualizer.create_dashboard()


if __name__ == "__main__":
    main()
