import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Set aesthetic style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def visualise_single_table(timestamp_columns, categorical_columns, numerical_columns, filename, df):
    """
    Creates meaningful visualizations based on column types in a single table.
    """

    # Check if df is None
    if df is None:
        return None
        
    print("📊 Columns confirmed. Proceeding with visualization...")
    created_figures = []  # Track created figures for return

    """

    VISUALIZATION GROUP 1: DISTRIBUTIONS
    - Histograms for numerical columns
    - Bar charts for categorical columns

    """

    # 1.1 Histograms for numerical data
    if numerical_columns:
        # Create a new figure for numerical distributions
        fig_num = plt.figure(figsize=(15, 10)).number
        created_figures.append(fig_num)
        
        print("📈 Creating histograms for numerical columns...")
        
        # Calculate optimal grid layout
        n_plots = len(numerical_columns)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        for i, column in enumerate(numerical_columns):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Check for skewness to determine best visualization
            skewness = df[column].skew()
            
            # Create a histogram with KDE
            sns.histplot(df[column].dropna(), kde=True)
            
            # Add vertical line for mean and median
            mean_val = df[column].mean()
            median_val = df[column].median()
            plt.axvline(mean_val, color='r', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='g', linestyle='-.', alpha=0.8, label=f'Median: {median_val:.2f}')
            
            # Add skewness information
            skew_label = f"Skew: {skewness:.2f}"
            if abs(skewness) > 1:
                skew_label += " (highly skewed)"
            elif abs(skewness) > 0.5:
                skew_label += " (moderately skewed)"
            else:
                skew_label += " (approximately symmetric)"
            
            plt.title(f"Distribution of {column}\n{skew_label}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.legend()
            
            # Adjust x-axis ticks if values are large
            if df[column].max() > 1000 or df[column].min() < -1000:
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        plt.tight_layout()
        plt.suptitle(f"Numerical Distributions in {filename}", fontsize=16, y=1.02)
    else:
        print("ℹ️ No numerical columns found for histogram visualization.")

    # 1.2 Bar charts for categorical data
    if categorical_columns:
        # Create a new figure for categorical distributions
        fig_num = plt.figure(figsize=(15, 10)).number
        created_figures.append(fig_num)
        
        print("📊 Creating bar charts for categorical columns...")
        
        # Calculate optimal grid layout
        n_plots = len(categorical_columns)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        for i, column in enumerate(categorical_columns):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Get value counts, limit to top 15 categories if there are many
            value_counts = df[column].value_counts()
            if len(value_counts) > 15:
                value_counts = value_counts.head(15)
                truncated = True
            else:
                truncated = False
            
            # Create bar chart
            sns.barplot(x=value_counts.index, y=value_counts.values)
            
            # Format labels
            plt.title(f"Distribution of {column}" + (" (Top 15 Categories)" if truncated else ""))
            plt.xlabel(column)
            plt.ylabel("Count")
            
            # Rotate x-labels if they're long or there are many
            if value_counts.index.astype(str).str.len().max() > 10 or len(value_counts) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Add count and percentage annotations
            total = len(df)
            for j, v in enumerate(value_counts.values):
                percentage = v / total * 100
                plt.text(j, v + (value_counts.max() * 0.02), 
                         f"{v}\n({percentage:.1f}%)", 
                         ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.suptitle(f"Categorical Distributions in {filename}", fontsize=16, y=1.02)
    

    """

    VISUALIZATION GROUP 2: TIME SERIES
    - Line charts for numerical values over time

    """

    # 2.1 Time series plots for timestamp data
    if timestamp_columns and numerical_columns:
        # Create a new figure for time series
        fig_num = plt.figure(figsize=(15, 10)).number
        created_figures.append(fig_num)
        
        print("📈 Creating time series plots...")
        
        # For each timestamp column, plot against numerical columns
        for time_col in timestamp_columns:
            # Ensure the timestamp column is properly formatted
            df_time = df.copy()
            df_time[time_col] = pd.to_datetime(df_time[time_col])
            
            # Sort by time
            df_time = df_time.sort_values(time_col)
            
            # Select up to 3 numerical columns to visualize
            if len(numerical_columns) > 3:
                # Find most interesting numerical columns (highest variation over time)
                variation_scores = []
                for num_col in numerical_columns:
                    if df_time[num_col].nunique() > 1:  # Skip constant columns
                        # Calculate rolling means and their standard deviation
                        rolling_mean = df_time[num_col].rolling(window=max(2, len(df_time)//20)).mean()
                        variation = rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else 0
                        variation_scores.append((num_col, variation))
                
                # Sort by variation and take top 3
                variation_scores.sort(key=lambda x: x[1], reverse=True)
                selected_num_cols = [col for col, _ in variation_scores[:3]]
            else:
                selected_num_cols = numerical_columns
            
            # Create subplots for each selected numerical column
            for i, num_col in enumerate(selected_num_cols):
                plt.subplot(len(selected_num_cols), 1, i+1)
                
                # Plot the time series
                plt.plot(df_time[time_col], df_time[num_col], marker='o', linestyle='-', 
                         alpha=0.7, markersize=4)
                
                # Add trend line using rolling average
                window_size = max(2, len(df_time)//20)  # Adaptive window size
                if len(df_time) > window_size:
                    rolling_avg = df_time[num_col].rolling(window=window_size).mean()
                    plt.plot(df_time[time_col], rolling_avg, 'r--', 
                             linewidth=2, label=f'Trend (rolling avg, window={window_size})')
                
                plt.title(f"{num_col} Over Time")
                plt.ylabel(num_col)
                if i == len(selected_num_cols) - 1:  # Only show x-label for bottom plot
                    plt.xlabel(time_col)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Format x-axis date labels
                plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            plt.suptitle(f"Time Series Analysis in {filename}\nTimestamp: {time_col}", 
                         fontsize=16, y=1.02)
    
    """

    VISUALIZATION GROUP 3: RELATIONSHIPS
    - Scatter plots for correlated numerical columns
    - Box plots for numerical data grouped by categorical columns

    """
    # 3.1 Scatter plots for numerical vs numerical
    if len(numerical_columns) >= 2:
        print("🔍 Finding related numerical columns for scatter plots...")
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_columns].corr().abs()
        
        # Get pairs with correlation above threshold
        corr_threshold = 0.5
        correlated_pairs = []
        
        for i in range(len(numerical_columns)):
            for j in range(i+1, len(numerical_columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value >= corr_threshold:
                    col1 = numerical_columns[i]
                    col2 = numerical_columns[j]
                    correlated_pairs.append((col1, col2, corr_value))
        
        # Sort by correlation strength (descending)
        correlated_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Plot scatter plots for the most correlated pairs (up to 3)
        if correlated_pairs:
            # Create a new figure for scatter plots
            fig_num = plt.figure(figsize=(15, 10)).number
            created_figures.append(fig_num)
            
            print(f"📊 Creating scatter plots for {min(3, len(correlated_pairs))} correlated pairs...")
            
            for i, (col1, col2, corr) in enumerate(correlated_pairs[:3]):
                plt.subplot(1, min(3, len(correlated_pairs)), i+1)
                
                # Create scatter plot with regression line
                sns.regplot(x=df[col1], y=df[col2], scatter_kws={'alpha':0.5})
                
                plt.title(f"{col1} vs {col2}\nCorrelation: {corr:.2f}")
                plt.xlabel(col1)
                plt.ylabel(col2)
                
                # Add annotation for correlation value
                plt.annotate(f"r = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            plt.suptitle(f"Correlated Numerical Features in {filename}", fontsize=16, y=1.02)
        else:
            print("ℹ️ No strongly correlated numerical columns found (correlation threshold: 0.5)")
    
    # 3.2 Box plots for numerical data by categorical
    if numerical_columns and categorical_columns:
        print("🔍 Finding meaningful relationships between categorical and numerical data...")
        
        def calculate_relationship_strength(df, cat_col, num_col):
            """Calculate how strongly a categorical variable affects a numerical one"""
            # Skip if too many unique categories or too few data points
            if df[cat_col].nunique() > 10 or df[cat_col].nunique() < 2:
                return 0
                
            try:
                # Perform ANOVA test to see if categories have different distributions
                groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
                groups = [g for g in groups if len(g) > 0]  # Filter out empty groups
                
                if len(groups) < 2:
                    return 0
                
                # Calculate F statistic and p-value
                f_val, p_val = stats.f_oneway(*groups)
                
                # Calculate effect size (eta-squared)
                # Higher values mean the categorical variable explains more variance
                df_clean = df[[cat_col, num_col]].dropna()
                categories = df_clean[cat_col].unique()
                grand_mean = df_clean[num_col].mean()
                n_total = len(df_clean)
                
                # Calculate between-group sum of squares
                ss_between = sum(len(df_clean[df_clean[cat_clean] == cat]) * 
                                (df_clean[df_clean[cat_col] == cat][num_col].mean() - grand_mean)**2 
                                for cat in categories)
                
                # Calculate total sum of squares
                ss_total = sum((df_clean[num_col] - grand_mean)**2)
                
                # Calculate eta-squared
                eta_squared = ss_between / ss_total if ss_total != 0 else 0
                
                return eta_squared
            except:
                return 0
        
        # Find strongest relationships
        relationships = []
        for cat_col in categorical_columns:
            for num_col in numerical_columns:
                strength = calculate_relationship_strength(df, cat_col, num_col)
                if strength > 0.1:  # Only include relationships with some meaningful effect
                    relationships.append((cat_col, num_col, strength))
        
        # Sort by relationship strength
        relationships.sort(key=lambda x: x[2], reverse=True)
        
        # Plot the strongest relationships (up to 3)
        if relationships:
            # Create a new figure for boxplots
            fig_num = plt.figure(figsize=(15, 10)).number
            created_figures.append(fig_num)
            
            print(f"📊 Creating box plots for {min(3, len(relationships))} categorical-numerical relationships...")
            
            for i, (cat_col, num_col, strength) in enumerate(relationships[:3]):
                plt.subplot(1, min(3, len(relationships)), i+1)
                
                # Limit to top categories if there are too many
                if df[cat_col].nunique() > 8:
                    top_cats = df[cat_col].value_counts().nlargest(8).index
                    plot_df = df[df[cat_col].isin(top_cats)]
                    truncated = True
                else:
                    plot_df = df
                    truncated = False
                
                # Create box plot
                sns.boxplot(x=cat_col, y=num_col, data=plot_df)
                
                # Add swarm plot for individual data points
                if len(plot_df) < 200:
                    sns.swarmplot(x=cat_col, y=num_col, data=plot_df, color='black', alpha=0.5)
                
                plt.title(f"Distribution of {num_col} by {cat_col}" + 
                         ("\n(Top 8 categories shown)" if truncated else ""))
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                
                # Rotate x-labels if needed
                plt.xticks(rotation=45 if len(plot_df[cat_col].unique()) > 3 else 0, ha='right')
                
                # Add annotation for relationship strength
                strength_text = f"Effect size: {strength:.2f}"
                if strength > 0.5:
                    strength_text += " (strong)"
                elif strength > 0.3:
                    strength_text += " (moderate)"
                else:
                    strength_text += " (weak)"
                    
                plt.annotate(strength_text, xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            plt.suptitle(f"Categorical-Numerical Relationships in {filename}", fontsize=16, y=1.02)
        else:
            print("ℹ️ No strong categorical-numerical relationships found")
    
    # Ensure all plots are shown
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Created {len(created_figures)} visualization figures for {filename}")
    return created_figures


def analyse_single_table(timestamp_columns, categorical_columns, numerical_columns, filename, df):
    """
    Performs statistical analysis on a single table based on column types.
    """

    # Check if df is None
    if df is None:
        return None
        
    print("📊 Columns confirmed. Proceeding with analysis...")
    created_figures = []  # Track created figures for return
    

    """

    ANALYSIS GROUP 1: BASIC SUMMARY STATISTICS

    """

    print("\n📈 === Summary Statistics ===")
    
    # 1.1 Numerical column statistics
    if numerical_columns:
        print("\n🔢 Numerical Columns:")
        summary_stats = df[numerical_columns].describe().T
        
        # Add more statistics to the summary
        summary_stats['skew'] = df[numerical_columns].skew()
        summary_stats['kurtosis'] = df[numerical_columns].kurtosis()
        summary_stats['missing'] = df[numerical_columns].isnull().sum()
        summary_stats['missing_pct'] = (df[numerical_columns].isnull().sum() / len(df) * 100).round(2)
        
        # Print enhanced summary with interpretations
        print(summary_stats)
        
        # Print interpretations for skewness and kurtosis
        print("\n📊 Distribution Analysis:")
        for col in numerical_columns:
            skew_val = summary_stats.loc[col, 'skew']
            kurt_val = summary_stats.loc[col, 'kurtosis']
            
            skew_interp = ""
            if abs(skew_val) < 0.5:
                skew_interp = "approximately symmetric"
            elif abs(skew_val) < 1:
                skew_interp = "moderately skewed"
            else:
                skew_interp = "highly skewed"
            
            if skew_val >= 0.5:
                skew_interp += " to the right (positive skew)"
            elif skew_val <= -0.5:
                skew_interp += " to the left (negative skew)"
            
            kurt_interp = ""
            if kurt_val < -0.5:
                kurt_interp = "platykurtic (flatter than normal)"
            elif kurt_val > 0.5:
                kurt_interp = "leptokurtic (more peaked than normal)"
            else:
                kurt_interp = "mesokurtic (similar to normal)"
            
            print(f"  • {col}: Distribution is {skew_interp} and {kurt_interp}")
    
    # 1.2 Categorical column statistics
    if categorical_columns:
        print("\n📝 Categorical Columns:")
        for col in categorical_columns:
            unique_count = df[col].nunique()
            most_common = df[col].value_counts().nlargest(3)
            missing = df[col].isnull().sum()
            missing_pct = round((missing / len(df) * 100), 2)
            
            print(f"\n  • {col}:")
            print(f"    - Unique values: {unique_count}")
            print(f"    - Missing values: {missing} ({missing_pct}%)")
            print(f"    - Most common values:")
            
            for val, count in most_common.items():
                pct = round((count / len(df) * 100), 2)
                print(f"      * {val}: {count} ({pct}%)")
    
    # 1.3 Timestamp column statistics
    if timestamp_columns:
        print("\n⏱️ Timestamp Columns:")
        for col in timestamp_columns:
            try:
                # Convert to datetime if not already
                df[col] = pd.to_datetime(df[col])
                
                # Calculate time statistics
                time_range = df[col].max() - df[col].min()
                earliest = df[col].min()
                latest = df[col].max()
                missing = df[col].isnull().sum()
                missing_pct = (missing / len(df) * 100).round(2)

                print(f"\n  • {col}:")
                print(f"    - Time range: {time_range}")
                print(f"    - Earliest: {earliest}")
                print(f"    - Latest: {latest}")
                print(f"    - Missing values: {missing} ({missing_pct}%)")
                
                # Time pattern analysis
                if len(df) > 10:  # Only do pattern analysis if enough data
                    by_year = df[col].dt.year.value_counts().sort_index()
                    by_month = df[col].dt.month.value_counts().sort_index()
                    by_day = df[col].dt.day_of_week.value_counts().sort_index()
                    
                    print("    - Temporal patterns:")
                    
                    if len(by_year) > 1:
                        print(f"      * Years: Data spans {len(by_year)} years")
                        print(f"        Most frequent year: {by_year.idxmax()} ({by_year.max()} records)")
                    
                    if len(by_month) > 1:
                        month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                                      7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
                        most_freq_month = month_names.get(by_month.idxmax(), by_month.idxmax())
                        print(f"      * Most frequent month: {most_freq_month} ({by_month.max()} records)")
                    
                    if len(by_day) > 1:
                        day_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
                        most_freq_day = day_names.get(by_day.idxmax(), by_day.idxmax())
                        print(f"      * Most frequent day of week: {most_freq_day} ({by_day.max()} records)")
            except Exception as e:
                print(f"    - Error analyzing timestamp: {e}")

    
    """

    ANALYSIS GROUP 2: CORRELATION ANALYSIS

    """

    # 2.1 Correlation analysis for numerical columns
    if len(numerical_columns) >= 2:
        print("\n🔄 === Correlation Analysis ===")
        correlation_matrix = df[numerical_columns].corr()
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(2))
        
        # Create heatmap figure for correlation matrix
        fig_num = plt.figure(figsize=(12, 10)).number
        created_figures.append(fig_num)
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask for upper triangle
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
        plt.title(f'Correlation Matrix Heatmap - {filename}')
        plt.tight_layout()
        
        # Find highly correlated pairs
        print("\n🔍 Highly Correlated Features:")
        corr_pairs = []
        for i in range(len(numerical_columns)):
            for j in range(i+1, len(numerical_columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.5:  # Only report strong correlations
                    col1, col2 = numerical_columns[i], numerical_columns[j]
                    corr_type = "positive" if correlation_matrix.iloc[i, j] > 0 else "negative"
                    corr_pairs.append((col1, col2, corr_value, corr_type))
        
        # Sort by correlation strength
        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if corr_pairs:
            for col1, col2, corr_value, corr_type in corr_pairs:
                strength = "very strong" if corr_value > 0.8 else "strong"
                print(f"  • {col1} and {col2}: {strength} {corr_type} correlation ({corr_value:.2f})")
        else:
            print("  • No strong correlations found (threshold: 0.5)")
    

    """

    ANALYSIS GROUP 3: MISSING VALUES ANALYSIS

    """

    print("\n🔍 === Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 
                                'Percent Missing': missing_percent.round(2)})
    
    # Only show columns with missing values
    missing_data = missing_data[missing_data['Missing Values'] > 0]
    
    if not missing_data.empty:
        print("\nColumns with missing values:")
        print(missing_data.sort_values('Missing Values', ascending=False))
        
        # Create visualization for missing values if there are any
        fig_num = plt.figure(figsize=(12, 6)).number
        created_figures.append(fig_num)
        
        # Create bar chart of missing values
        plt.subplot(1, 2, 1)
        missing_data['Percent Missing'].sort_values(ascending=False).plot(kind='bar')
        plt.title('Percentage of Missing Values by Column')
        plt.ylabel('Missing Values (%)')
        plt.xlabel('Columns')
        plt.xticks(rotation=45, ha='right')
        
        # Create heatmap of missing values patterns
        plt.subplot(1, 2, 2)
        cols_with_missing = missing_data.index.tolist()
        if len(cols_with_missing) > 10:
            cols_with_missing = cols_with_missing[:10]  # Limit to top 10 columns with most missing values
        
        sns.heatmap(df[cols_with_missing].isnull(), cmap='viridis', cbar=False, 
                   yticklabels=False)
        plt.title('Missing Values Pattern' + (' (Top 10 columns)' if len(missing_data) > 10 else ''))
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        
        plt.tight_layout()
        plt.suptitle(f'Missing Values Analysis - {filename}', fontsize=16, y=1.02)
        
        # Suggest potential reasons for missing data
        print("\n🔍 Missing Data Analysis:")
        
        # Check for patterns in missing data
        for col in missing_data.index:
            print(f"\n  • {col} ({missing_data.loc[col, 'Percent Missing']}% missing):")
            
            # Check if missing values are random or have a pattern
            if col in numerical_columns and len(numerical_columns) > 1:
                # Check if missingness correlates with other numerical variables
                for num_col in numerical_columns:
                    if num_col != col:
                        # Compare mean of num_col when col is missing vs. not missing
                        missing_mask = df[col].isnull()
                        if missing_mask.sum() > 0 and (~missing_mask).sum() > 0:
                            mean_when_missing = df.loc[missing_mask, num_col].mean()
                            mean_when_present = df.loc[~missing_mask, num_col].mean()
                            diff_percent = abs((mean_when_missing - mean_when_present) / mean_when_present * 100) if mean_when_present != 0 else 0
                            
                            if diff_percent > 20:  # If difference is substantial
                                direction = "higher" if mean_when_missing > mean_when_present else "lower"
                                print(f"    - Values in {num_col} tend to be {direction} ({diff_percent:.1f}% difference) when {col} is missing")
            
            # For categorical columns, check if missing values concentrate in certain categories
            if categorical_columns:
                for cat_col in categorical_columns:
                    missing_by_category = df.groupby(cat_col)[col].apply(lambda x: x.isnull().mean() * 100).sort_values(ascending=False)
                    if missing_by_category.max() > 1.5 * missing_data.loc[col, 'Percent Missing']:
                        top_category = missing_by_category.index[0]
                        print(f"    - Higher missing rate ({missing_by_category.iloc[0]:.1f}%) when {cat_col} is '{top_category}'")
    else:
        print("  • No missing values found in the dataset.")
    
    """

    ANALYSIS GROUP 4: OUTLIER DETECTION

    """
    
    if numerical_columns:
        print("\n🔍 === Outlier Analysis ===")
        
        # Create figure for outlier analysis
        fig_num = plt.figure(figsize=(15, 5 * min(len(numerical_columns), 4))).number
        created_figures.append(fig_num)
        
        # Calculate optimal grid layout
        n_plots = min(len(numerical_columns), 4)  # Limit to 4 columns to avoid too large figures
        
        # Identify outliers for each numerical column
        outliers_summary = {}
        
        for i, col in enumerate(numerical_columns[:n_plots]):  # Only plot the first n_plots columns
            plt.subplot(n_plots, 1, i+1)
            
            # Calculate Q1, Q3 and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outliers_count = len(outliers)
            outliers_percent = round((outliers_count / len(df) * 100), 2)
            
            # Store outlier information
            outliers_summary[col] = {
                'count': outliers_count,
                'percent': outliers_percent,
                'min': outliers.min() if not outliers.empty else None,
                'max': outliers.max() if not outliers.empty else None
            }
            
            # Create boxplot
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col} - {outliers_count} outliers ({outliers_percent}%)')
            plt.xlabel(col)
            
            # Add bounds information
            plt.axvline(x=lower_bound, color='r', linestyle='--', alpha=0.7, 
                       label=f'Lower bound: {lower_bound:.2f}')
            plt.axvline(x=upper_bound, color='r', linestyle='--', alpha=0.7,
                       label=f'Upper bound: {upper_bound:.2f}')
            plt.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Outlier Analysis - {filename}', fontsize=16, y=1.02)
        
        # Print outlier summary
        print("\nOutlier Summary:")
        for col, stats in outliers_summary.items():
            if stats['count'] > 0:
                print(f"  • {col}: {stats['count']} outliers ({stats['percent']}%)")
                print(f"    - Range: {stats['min']} to {stats['max']}")
            else:
                print(f"  • {col}: No outliers detected")
        
        # Print additional outlier analysis for columns not shown in plots
        if len(numerical_columns) > n_plots:
            print("\nAdditional columns with outliers:")
            for col in numerical_columns[n_plots:]:
                # Calculate Q1, Q3 and IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outliers_count = len(outliers)
                outliers_percent = round((outliers_count / len(df) * 100), 2)
                
                if outliers_count > 0:
                    print(f"  • {col}: {outliers_count} outliers ({outliers_percent}%)")
    
    # Return the figure numbers for potential saving
    print(f"\n✅ Analysis complete! Created {len(created_figures)} analysis figures for {filename}")
    return created_figures