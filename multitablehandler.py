import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Set aesthetic style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

def visualise_multiple_tables(tables):
    """
    Visualizes multiple tables with a focus on comparing related columns across tables.
    """

    if not tables or len(tables) < 2:
        print("❌ Need at least two tables for multi-table visualization.")
        return None
    
    print(f"🔍 Visualizing {len(tables)} tables together...")
    created_figures = []  # Track created figures for return
    
    # Identify column types for each table
    processed_tables = []
    for i, (filename, df) in enumerate(tables):
        timestamp_cols = []
        categorical_cols = []
        numerical_cols = []
        
        print(f"\n📊 Processing table {i+1}: {filename}")
        
        # Identify column types for this table
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]) or any(time_word in column.lower() for time_word in ['date', 'time', 'timestamp', 'datetime']):
                try:
                    df[column] = pd.to_datetime(df[column])
                    timestamp_cols.append(column)
                except Exception as e:
                    print(f"⚠️ Error converting {column} to datetime: {e}")
            elif pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_integer_dtype(df[column]) or pd.api.types.is_float_dtype(df[column]):
                numerical_cols.append(column)
            elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == object:
                categorical_cols.append(column)
        
        processed_tables.append({
            'filename': filename,
            'df': df,
            'timestamp_cols': timestamp_cols,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        })
        
        print(f"✅ Identified columns in {filename}:")
        print(f"  ⏱️ Timestamp: {', '.join(timestamp_cols) if timestamp_cols else 'None'}")
        print(f"  📝 Categorical: {', '.join(categorical_cols) if categorical_cols else 'None'}")
        print(f"  🔢 Numerical: {', '.join(numerical_cols) if numerical_cols else 'None'}")
    
    # Get user confirmation
    confirm = input("\n✅ Proceed with these column identifications? (y/n) >>> ").strip().lower()
    if confirm != 'y':
        print("🛑 Visualization cancelled.")
        return None
    

    """

    VISUALIZATION GROUP 1: COMMON NUMERICAL COLUMNS

    """
    # Find common numerical columns across all tables
    all_numerical_columns = [set(table['numerical_cols']) for table in processed_tables]
    common_numerical_columns = set.intersection(*all_numerical_columns) if all_numerical_columns else set()
    
    if common_numerical_columns:
        print(f"\n📈 Found {len(common_numerical_columns)} common numerical columns across all tables")
        
        # Create a new figure for distribution comparisons
        fig_num = plt.figure(figsize=(15, 12)).number
        created_figures.append(fig_num)
        
        # Calculate optimal grid layout
        n_plots = min(len(common_numerical_columns), 3)
        
        for i, column in enumerate(list(common_numerical_columns)[:3]):  # Limit to 3 plots
            plt.subplot(n_plots, 1, i+1)
            
            # Create a combined histogram with kde for the column across all tables
            for table in processed_tables:
                sns.kdeplot(table['df'][column].dropna(), label=f"{table['filename']}")
            
            plt.title(f'Distribution Comparison: {column}')
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add summary statistics annotation
            stat_text = "Stats Summary:\n"
            for table in processed_tables:
                mean_val = table['df'][column].mean()
                median_val = table['df'][column].median()
                stat_text += f"{table['filename']}: Mean={mean_val:.2f}, Median={median_val:.2f}\n"
            
            plt.annotate(stat_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                        va='top', fontsize=9)
        
        plt.tight_layout()
        plt.suptitle("Distribution Comparison of Common Numerical Columns", fontsize=16, y=1.02)
    else:
        print("ℹ️ No common numerical columns found across all tables.")
    

    """

    VISUALIZATION GROUP 2: SIMILAR COLUMNS

    """

    print("\n🔍 Looking for columns with similar statistical properties...")
    
    # Create dictionary of column statistics for all numerical columns
    column_stats = {}
    for table in processed_tables:
        for column in table['numerical_cols']:
            if not pd.isna(table['df'][column]).all():  # Skip if all values are NaN
                stats_key = f"{table['filename']}:{column}"
                column_stats[stats_key] = {
                    'mean': table['df'][column].mean(),
                    'std': table['df'][column].std(),
                    'min': table['df'][column].min(),
                    'max': table['df'][column].max(),
                    'table': table['filename'],
                    'column': column,
                    'df': table['df']
                }
    
    # Find similar columns based on mean and standard deviation
    similar_columns = []
    for key1 in column_stats:
        for key2 in column_stats:
            if key1 != key2:
                # Columns from different tables
                if column_stats[key1]['table'] != column_stats[key2]['table']:
                    # Calculate similarity score (lower is more similar)
                    # Using normalized difference in mean and std
                    mean_diff = abs(column_stats[key1]['mean'] - column_stats[key2]['mean'])
                    std_diff = abs(column_stats[key1]['std'] - column_stats[key2]['std'])
                    
                    # Normalize differences by the range
                    range1 = column_stats[key1]['max'] - column_stats[key1]['min']
                    range2 = column_stats[key2]['max'] - column_stats[key2]['min']
                    range_avg = (range1 + range2) / 2
                    
                    if range_avg > 0:  # Avoid division by zero
                        similarity = (mean_diff + std_diff) / range_avg
                        if similarity < 0.2:  # Threshold for similarity
                            similar_columns.append((key1, key2, similarity))
    
    # Sort by similarity (most similar first)
    similar_columns.sort(key=lambda x: x[2])
    
    # Visualize similar columns (up to 3 pairs)
    if similar_columns:
        # Create a new figure for similar columns
        fig_num = plt.figure(figsize=(15, 12)).number
        created_figures.append(fig_num)
        
        visualized_pairs = set()
        
        for i, (key1, key2, similarity) in enumerate(similar_columns[:6]):  # Look at top 6 to avoid duplicates
            # Create a unique pair ID (sorted to avoid duplicates)
            pair_id = tuple(sorted([key1, key2]))
            
            if pair_id in visualized_pairs or i >= 3:  # Limit to 3 plots
                continue
            
            visualized_pairs.add(pair_id)
            
            col1 = column_stats[key1]['column']
            col2 = column_stats[key2]['column']
            table1 = column_stats[key1]['table']
            table2 = column_stats[key2]['table']
            
            plt.subplot(min(len(similar_columns), 3), 1, len(visualized_pairs))
            
            # Plot KDE for both columns
            sns.kdeplot(column_stats[key1]['df'][col1].dropna(), label=f"{table1}: {col1}")
            sns.kdeplot(column_stats[key2]['df'][col2].dropna(), label=f"{table2}: {col2}")
            
            plt.title(f'Similar Columns: {table1}:{col1} vs {table2}:{col2}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add similarity information
            similarity_percent = (1-similarity) * 100
            plt.annotate(f"Similarity: {similarity_percent:.1f}%", 
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        va='top')
        
        plt.tight_layout()
        plt.suptitle("Similar Numerical Columns Across Tables", fontsize=16, y=1.02)
        
        print(f"📊 Found {len(similar_columns)} pairs of similar columns, showing top {len(visualized_pairs)}")
    else:
        print("ℹ️ No similar numerical columns found across tables.")
    
    """

    VISUALIZATION GROUP 3: CATEGORICAL COMPARISONS

    """

    # Find common categorical columns
    all_categorical_columns = [set(table['categorical_cols']) for table in processed_tables]
    common_categorical_columns = set.intersection(*all_categorical_columns) if all_categorical_columns else set()
    
    if common_categorical_columns:
        print(f"\n📊 Found {len(common_categorical_columns)} common categorical columns across all tables")
        
        # Create a new figure for categorical comparisons
        fig_num = plt.figure(figsize=(15, 15)).number
        created_figures.append(fig_num)
        
        plot_count = 0
        for column in common_categorical_columns:
            # Skip if too many categories
            if any(processed_tables[i]['df'][column].nunique() > 10 for i in range(len(processed_tables))):
                print(f"⚠️ Skipping {column} - too many categories")
                continue
                
            if plot_count >= 3:  # Limit to 3 plots
                break
                
            plot_count += 1
            plt.subplot(min(len(common_categorical_columns), 3), 1, plot_count)
            
            # Calculate proportions for each table
            bar_width = 0.8 / len(processed_tables)
            
            # Get all unique categories across all tables
            all_categories = set()
            for table in processed_tables:
                all_categories.update(table['df'][column].dropna().unique())
            
            all_categories = sorted(all_categories)
            x = np.arange(len(all_categories))
            
            # Plot bars for each table
            for i, table in enumerate(processed_tables):
                counts = table['df'][column].value_counts(normalize=True)
                values = [counts.get(cat, 0) for cat in all_categories]
                bars = plt.bar(x + i*bar_width - 0.4 + bar_width/2, values, width=bar_width, 
                        label=table['filename'], alpha=0.7)
                
                # Add data labels for significant proportions
                for j, v in enumerate(values):
                    if v > 0.1:  # Only show labels for significant proportions (>10%)
                        plt.text(x[j] + i*bar_width - 0.4 + bar_width/2, v + 0.02, 
                                f"{v:.1%}", ha='center', va='bottom', fontsize=8, rotation=90)
            
            plt.title(f'Category Distribution Comparison: {column}')
            plt.xlabel(column)
            plt.ylabel('Proportion')
            plt.xticks(x, all_categories, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if plot_count > 0:
            plt.tight_layout()
            plt.suptitle("Categorical Distribution Comparison", fontsize=16, y=1.02)
        else:
            plt.close()
            created_figures.remove(fig_num)
    
    """

    VISUALIZATION GROUP 4: TIME SERIES COMPARISONS

    """

    # Check if all tables have timestamp columns
    tables_with_timestamps = [table for table in processed_tables if table['timestamp_cols']]
    
    if len(tables_with_timestamps) >= 2:
        print("\n📈 Comparing time series data across tables...")
        
        # Create a new figure for time series comparisons
        fig_num = plt.figure(figsize=(15, 12)).number
        created_figures.append(fig_num)
        
        plot_count = 0
        # For each pair of tables with timestamps
        for i in range(len(tables_with_timestamps)):
            for j in range(i+1, len(tables_with_timestamps)):
                if plot_count >= 3:  # Limit to 3 plots
                    break
                    
                table1 = tables_with_timestamps[i]
                table2 = tables_with_timestamps[j]
                
                # For simplicity, use the first timestamp column from each table
                time_col1 = table1['timestamp_cols'][0]
                time_col2 = table2['timestamp_cols'][0]
                
                # Find common numerical columns between these two tables
                common_num_cols = set(table1['numerical_cols']).intersection(set(table2['numerical_cols']))
                
                if common_num_cols:
                    for num_col in list(common_num_cols)[:1]:  # Just use the first common column for clarity
                        plot_count += 1
                        plt.subplot(min(3, plot_count), 1, plot_count)
                        
                        # Create sorted copies of the data
                        df1_sorted = table1['df'].sort_values(time_col1)
                        df2_sorted = table2['df'].sort_values(time_col2)
                        
                        # Plot time series for both tables
                        plt.plot(df1_sorted[time_col1], df1_sorted[num_col], 'o-',
                                markersize=4, alpha=0.7, label=f"{table1['filename']}: {num_col}")
                        plt.plot(df2_sorted[time_col2], df2_sorted[num_col], 's-',
                                markersize=4, alpha=0.7, label=f"{table2['filename']}: {num_col}")
                        
                        plt.title(f'Time Series Comparison: {num_col}')
                        plt.xlabel('Time')
                        plt.ylabel(num_col)
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Add trend lines
                        window_size1 = max(2, len(df1_sorted)//10)
                        window_size2 = max(2, len(df2_sorted)//10)
                        
                        if len(df1_sorted) > window_size1:
                            rolling_avg1 = df1_sorted[num_col].rolling(window=window_size1).mean()
                            plt.plot(df1_sorted[time_col1], rolling_avg1, 'r--', 
                                    linewidth=2, label=f"Trend {table1['filename']}")
                            
                        if len(df2_sorted) > window_size2:
                            rolling_avg2 = df2_sorted[num_col].rolling(window=window_size2).mean()
                            plt.plot(df2_sorted[time_col2], rolling_avg2, 'b--', 
                                    linewidth=2, label=f"Trend {table2['filename']}")
                            
                        plt.legend()
                        
                        # Format x-axis date labels
                        plt.gcf().autofmt_xdate()
        
        if plot_count > 0:
            plt.tight_layout()
            plt.suptitle("Time Series Comparison Across Tables", fontsize=16, y=1.02)
        else:
            plt.close()
            created_figures.remove(fig_num)
            print("ℹ️ No suitable time series comparisons found.")
    
    # Show all plots
    plt.tight_layout()
    print(f"\n✅ Created {len(created_figures)} visualization figures")
    
    return created_figures


def analyse_multiple_tables(tables):

    """
    Analyzes multiple tables, focusing on comparisons and relationships between them.
    """

    if not tables or len(tables) < 2:
        print("❌ Need at least two tables for multi-table analysis.")
        return None
    
    print(f"🔍 Analyzing {len(tables)} tables together...")
    created_figures = []  # Track created figures for return
    
    # Identify column types for each table (similar to visualization function)
    processed_tables = []
    for i, (filename, df) in enumerate(tables):
        timestamp_cols = []
        categorical_cols = []
        numerical_cols = []
        
        print(f"\n📊 Processing table {i+1}: {filename}")
        
        # Identify column types for this table
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]) or any(time_word in column.lower() for time_word in ['date', 'time', 'timestamp', 'datetime']):
                try:
                    df[column] = pd.to_datetime(df[column])
                    timestamp_cols.append(column)
                except Exception as e:
                    print(f"⚠️ Error converting {column} to datetime: {e}")
            elif pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_integer_dtype(df[column]) or pd.api.types.is_float_dtype(df[column]):
                numerical_cols.append(column)
            elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == object:
                categorical_cols.append(column)
        
        processed_tables.append({
            'filename': filename,
            'df': df,
            'timestamp_cols': timestamp_cols,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        })
        
        print(f"✅ Identified columns in {filename}:")
        print(f"  ⏱️ Timestamp: {', '.join(timestamp_cols) if timestamp_cols else 'None'}")
        print(f"  📝 Categorical: {', '.join(categorical_cols) if categorical_cols else 'None'}")
        print(f"  🔢 Numerical: {', '.join(numerical_cols) if numerical_cols else 'None'}")
    
    # Get user confirmation
    confirm = input("\n✅ Proceed with these column identifications? (y/n) >>> ").strip().lower()
    if confirm != 'y':
        print("🛑 Analysis cancelled.")
        return None
    
    """

    ANALYSIS GROUP 1: STATISTICAL COMPARISON

    """

    # Find common numerical columns across all tables
    all_numerical_columns = [set(table['numerical_cols']) for table in processed_tables]
    common_numerical_columns = set.intersection(*all_numerical_columns) if all_numerical_columns else set()
    
    if common_numerical_columns:
        print(f"\n📈 === Statistical Comparison of Common Numerical Columns ===")
        
        # Create a figure for boxplot comparisons
        fig_num = plt.figure(figsize=(15, 5 * min(len(common_numerical_columns), 3))).number
        created_figures.append(fig_num)
        
        # For each common numerical column, create comparative boxplots
        for i, column in enumerate(list(common_numerical_columns)[:3]):  # Limit to 3
            plt.subplot(min(len(common_numerical_columns), 3), 1, i+1)
            
            print(f"\n📊 Statistics for '{column}':")
            column_stats = []
            data = []
            labels = []
            
            # Print statistics and prepare boxplot data
            for table in processed_tables:
                stats = {
                    'Table': table['filename'],
                    'Mean': table['df'][column].mean(),
                    'Median': table['df'][column].median(),
                    'Std Dev': table['df'][column].std(),
                    'Min': table['df'][column].min(),
                    'Max': table['df'][column].max(),
                    'Skewness': table['df'][column].skew()
                }
                column_stats.append(stats)
                
                # Print statistics
                print(f"  • {table['filename']}:")
                print(f"    Mean={stats['Mean']:.2f}, Median={stats['Median']:.2f}, StdDev={stats['Std Dev']:.2f}")
                print(f"    Range: {stats['Min']:.2f} to {stats['Max']:.2f}, Skewness: {stats['Skewness']:.2f}")
                
                # Add data for boxplot
                data.append(table['df'][column].dropna())
                labels.append(table['filename'])
            
            # Create boxplot
            box = plt.boxplot(data, labels=labels, patch_artist=True)
            
            # Add colors to boxes
            colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightcyan']
            for patch, color in zip(box['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
            
            plt.title(f'Box Plot Comparison: {column}')
            plt.ylabel(column)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add mean markers
            for i, d in enumerate(data):
                plt.plot(i+1, d.mean(), 'rd', markersize=8, label='Mean' if i==0 else "")
            
            if i == 0:  # Only add legend for first plot
                plt.legend()
            
            # Run statistical tests
            if len(processed_tables) > 1:
                print(f"\n  Statistical significance tests for '{column}':")
                
                significant_pairs = []
                for i in range(len(processed_tables)):
                    for j in range(i+1, len(processed_tables)):
                        table1 = processed_tables[i]
                        table2 = processed_tables[j]
                        
                        # Perform t-test if both have data
                        if not table1['df'][column].dropna().empty and not table2['df'][column].dropna().empty:
                            t_stat, p_val = stats.ttest_ind(
                                table1['df'][column].dropna(), 
                                table2['df'][column].dropna(),
                                equal_var=False  # Welch's t-test for unequal variances
                            )
                            
                            sig_status = "significantly different" if p_val < 0.05 else "not significantly different"
                            effect_size = abs(table1['df'][column].mean() - table2['df'][column].mean()) / \
                                         ((table1['df'][column].std()**2 + table2['df'][column].std()**2) / 2)**0.5
                            
                            effect_strength = ""
                            if effect_size > 0.8:
                                effect_strength = "large effect"
                            elif effect_size > 0.5:
                                effect_strength = "medium effect"
                            elif effect_size > 0.2:
                                effect_strength = "small effect"
                            else:
                                effect_strength = "negligible effect"
                            
                            print(f"    • {table1['filename']} vs {table2['filename']}: {sig_status} (p={p_val:.4f}, {effect_strength})")
                            
                            if p_val < 0.05:
                                significant_pairs.append((i+1, j+1))
                
                # Add annotations for significant differences
                y_max = max([d.max() for d in data if len(d) > 0])
                y_height = y_max * 1.05
                
                for i, j in significant_pairs:
                    x1, x2 = i, j
                    y = y_height + (y_max * 0.05 * significant_pairs.index((i, j)))
                    plt.plot([x1, x1, x2, x2], [y, y + y_max * 0.02, y + y_max * 0.02, y], 'k-', linewidth=1)
                    plt.text((x1 + x2) / 2, y + y_max * 0.025, "*", ha='center')
        
        plt.tight_layout()
        plt.suptitle("Statistical Comparison of Numerical Columns", fontsize=16, y=1.02)
    else:
        print("\nℹ️ No common numerical columns found across all tables.")
    

    """

    ANALYSIS GROUP 2: CORRELATION COMPARISON

    """

    print("\n🔄 === Correlation Patterns Comparison ===")
    
    # Create a figure for correlation matrices
    if any(len(table['numerical_cols']) >= 2 for table in processed_tables):
        fig_num = plt.figure(figsize=(15, 10 * min(len(processed_tables), 2))).number
        created_figures.append(fig_num)
        
        plot_count = 0
        for table in processed_tables:
            if len(table['numerical_cols']) >= 2:
                plot_count += 1
                if plot_count > 2:  # Limit to 2 tables
                    break
                
                plt.subplot(min(len(processed_tables), 2), 1, plot_count)
                
                # Calculate correlation matrix
                correlation_matrix = table['df'][table['numerical_cols']].corr()
                print(f"\n📊 Correlation matrix for {table['filename']}:")
                print(correlation_matrix.round(2))
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                
                # Create heatmap for correlation matrix
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
                plt.title(f'Correlation Matrix: {table["filename"]}')
                
                # Find and print strongest correlations
                strong_correlations = []
                for i in range(len(table['numerical_cols'])):
                    for j in range(i+1, len(table['numerical_cols'])):
                        corr_value = abs(correlation_matrix.iloc[i, j])
                        if corr_value > 0.5:  # Only report strong correlations
                            col1, col2 = table['numerical_cols'][i], table['numerical_cols'][j]
                            corr_type = "positive" if correlation_matrix.iloc[i, j] > 0 else "negative"
                            strong_correlations.append((col1, col2, corr_value, corr_type))
                
                # Sort by correlation strength
                strong_correlations.sort(key=lambda x: x[2], reverse=True)
                
                if strong_correlations:
                    print(f"  • Strong correlations in {table['filename']}:")
                    for col1, col2, corr_value, corr_type in strong_correlations:
                        print(f"    - {col1} and {col2}: {corr_type} correlation ({corr_value:.2f})")
                else:
                    print(f"  • No strong correlations found in {table['filename']}")
        
        if plot_count > 0:
            plt.tight_layout()
            plt.suptitle("Correlation Patterns Across Tables", fontsize=16, y=1.02)
        else:
            plt.close()
            created_figures.remove(fig_num)
    
    """

    ANALYSIS GROUP 3: CATEGORICAL-NUMERICAL RELATIONSHIP COMPARISON

    """

    # Find common categorical columns
    all_categorical_columns = [set(table['categorical_cols']) for table in processed_tables]
    common_categorical_columns = set.intersection(*all_categorical_columns) if all_categorical_columns else set()
    
    if common_numerical_columns and common_categorical_columns:
        print("\n📊 === Categorical-Numerical Relationships ===")
        
        # Filter to categorical columns with reasonable number of categories
        usable_cat_cols = [col for col in common_categorical_columns 
                          if not any(processed_tables[i]['df'][col].nunique() > 8 
                                    for i in range(len(processed_tables)))]
        
        if usable_cat_cols and len(usable_cat_cols) > 0:
            # Create a figure for categorical-numerical relationships
            fig_num = plt.figure(figsize=(15, 10 * min(len(usable_cat_cols), 2))).number
            created_figures.append(fig_num)
            
            cat_num_pairs = []
            # Find the strongest categorical-numerical relationships
            for cat_col in usable_cat_cols:
                for num_col in common_numerical_columns:
                    # Calculate effect sizes across tables
                    effect_sizes = []
                    
                    for table in processed_tables:
                        # Skip if too many NaNs
                        if table['df'][cat_col].isna().mean() > 0.2 or table['df'][num_col].isna().mean() > 0.2:
                            continue
                            
                        try:
                            # Calculate eta-squared for relationship strength
                            categories = table['df'][cat_col].dropna().unique()
                            grand_mean = table['df'][num_col].mean()
                            
                            # Calculate between-group sum of squares
                            ss_between = sum(
                                len(table['df'][table['df'][cat_col] == cat]) * 
                                (table['df'][table['df'][cat_col] == cat][num_col].mean() - grand_mean)**2 
                                for cat in categories if len(table['df'][table['df'][cat_col] == cat]) > 0
                            )
                            
                            # Calculate total sum of squares
                            ss_total = sum((table['df'][num_col] - grand_mean)**2)
                            
                            # Calculate eta-squared
                            eta_squared = ss_between / ss_total if ss_total != 0 else 0
                            effect_sizes.append(eta_squared)
                        except:
                            pass
                    
                    # If we have effect sizes from multiple tables, calculate average
                    if effect_sizes and len(effect_sizes) >= len(processed_tables) / 2:
                        avg_effect = sum(effect_sizes) / len(effect_sizes)
                        if avg_effect > 0.1:  # Only include if average effect is substantial
                            cat_num_pairs.append((cat_col, num_col, avg_effect))
            
            # Sort by effect size and take top pairs
            cat_num_pairs.sort(key=lambda x: x[2], reverse=True)
            cat_num_pairs = cat_num_pairs[:2]  # Limit to top 2
            
            if cat_num_pairs:
                print(f"📊 Found {len(cat_num_pairs)} strong categorical-numerical relationships")
                
                for i, (cat_col, num_col, effect) in enumerate(cat_num_pairs):
                    plt.subplot(len(cat_num_pairs), 1, i+1)
                    
                    # Create subplots in a row for each table
                    fig, axes = plt.subplots(1, len(processed_tables), figsize=(15, 5), sharey=True)
                    
                    # Create box plots for each table
                    for j, table in enumerate(processed_tables):
                        try:
                            if isinstance(axes, np.ndarray):
                                ax = axes[j]
                            else:
                                ax = axes  # If only one subplot
                                
                            sns.boxplot(x=cat_col, y=num_col, data=table['df'], ax=ax)
                            ax.set_title(f'{table["filename"]}')
                            ax.set_xlabel(cat_col)
                            
                            if j == 0:  # Only show y-label for first plot
                                ax.set_ylabel(num_col)
                            else:
                                ax.set_ylabel('')
                                
                            ax.tick_params(axis='x', rotation=45)
                        except Exception as e:
                            print(f"⚠️ Error creating boxplot for {table['filename']}: {e}")
                    
                    # Add overall title
                    fig.suptitle(f'Distribution of {num_col} by {cat_col} (Effect: {effect:.2f})', fontsize=14)
                    plt.tight_layout()
                    
                    # Add this figure to created_figures
                    created_figures.append(fig.number)
                    
                    # Print analysis of the relationship
                    print(f"\n📊 Analysis of {num_col} by {cat_col}:")
                    print(f"  • Overall effect size: {effect:.2f} ({get_effect_strength(effect)})")
                    
                    for table in processed_tables:
                        # Calculate means for each category
                        category_means = table['df'].groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                        if len(category_means) > 0:
                            top_cat = category_means.index[0]
                            bottom_cat = category_means.index[-1]
                            diff_pct = (category_means.max() - category_means.min()) / category_means.mean() * 100 if category_means.mean() != 0 else 0
                            
                            print(f"  • In {table['filename']}:")
                            print(f"    - Highest {num_col} in {cat_col}='{top_cat}' (avg: {category_means.max():.2f})")
                            print(f"    - Lowest {num_col} in {cat_col}='{bottom_cat}' (avg: {category_means.min():.2f})")
                            print(f"    - Difference: {diff_pct:.1f}% from mean")
            else:
                print("ℹ️ No strong categorical-numerical relationships found")
                plt.close()
                created_figures.remove(fig_num)
        else:
            print("ℹ️ No suitable categorical columns for analysis (too many categories)")
    
    """

    ANALYSIS GROUP 4: MISSING VALUES COMPARISON

    """
    
    print("\n🔍 === Missing Values Comparison ===")
    
    # Create a DataFrame to compare missing values percentage across tables
    missing_comparison = pd.DataFrame()
    
    for table in processed_tables:
        missing_vals = table['df'].isnull().sum()
        missing_percent = (missing_vals / len(table['df'])) * 100
        missing_comparison[table['filename']] = missing_percent
    
    # Only show rows with at least some missing values
    missing_rows = (missing_comparison > 0).any(axis=1)
    if missing_rows.any():
        print("\n📊 Missing values by column:")
        print(missing_comparison[missing_rows].round(2))
        
        # Create a figure for missing values comparison
        fig_num = plt.figure(figsize=(12, 8)).number
        created_figures.append(fig_num)
        
        # Visualize missing values comparison
        missing_comparison[missing_rows].plot(kind='bar')
        plt.title('Missing Values Percentage Comparison')
        plt.ylabel('Percent Missing (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Table')
        
        # Add summary of missing data
        print("\n📊 Missing data summary:")
        for table in processed_tables:
            total_missing = table['df'].isnull().sum().sum()
            total_cells = table['df'].size
            missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
            
            print(f"  • {table['filename']}: {total_missing} missing values ({missing_percent:.2f}% of all data)")
            
            # Identify columns with high missing percentages
            high_missing_cols = []
            for col in table['df'].columns:
                missing_pct = table['df'][col].isnull().mean() * 100
                if missing_pct > 20:
                    high_missing_cols.append((col, missing_pct))
            
            if high_missing_cols:
                print(f"    - Columns with high missing rates:")
                for col, pct in sorted(high_missing_cols, key=lambda x: x[1], reverse=True):
                    print(f"      * {col}: {pct:.1f}% missing")
        
        plt.tight_layout()
    else:
        print("ℹ️ No missing values found in any table.")
    
    # Return created figures for saving
    print(f"\n✅ Analysis complete! Created {len(created_figures)} analysis figures")
    return created_figures


def get_effect_strength(effect_size):
    """Helper function to interpret effect sizes"""
    if effect_size > 0.5:
        return "strong effect"
    elif effect_size > 0.3:
        return "moderate effect"
    elif effect_size > 0.1:
        return "weak effect"
    else:
        return "negligible effect"