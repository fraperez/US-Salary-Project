import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns
from scipy.stats import shapiro,f_oneway,pearsonr,spearmanr
from scipy.stats import ttest_ind


def analyze_numerical_variable(df, feature, bins=35):
    """
    Performs univariate analysis on a numerical feature:
    - Descriptive statistics
    - Histogram + KDE
    - Boxplot
    - Shapiro-Wilk test for normality
    - Outlier detection using the IQR method

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    feature : str
        The numerical column to analyze.
    bins : int
        Number of bins for the histogram.

    Returns
    -------
    list
        Indexes of detected outliers (optional).
    """
    print(f"\n===== Analyzing: {feature} =====")

    # Descriptive statistics
    summary_stats = df[feature].describe()
    print("\nDescriptive statistics:")
    print(summary_stats)

    # Histogram and boxplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(df[feature], kde=True, bins=bins, color='steelblue', ax=axes[0])
    axes[0].set_title(f'{feature} - Histogram + KDE')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")

    sns.boxplot(y=df[feature], ax=axes[1], color='skyblue')
    axes[1].set_title(f'{feature} - Boxplot')
    axes[1].set_ylabel(feature)

    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test for normality
    stat, p = shapiro(df[feature].dropna())
    print("\nShapiro-Wilk Normality Test:")
    print(f"Statistic = {stat:.4f}, p-value = {p:.4f}")
    if p > 0.05:
        print(f"✅ {feature} appears to follow a normal distribution.")
    else:
        print(f"❌ {feature} does not follow a normal distribution.")

    # Outlier detection (IQR method)
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

    print(f"\nOutliers detected: {len(outliers)}")
    
    return outliers.index.tolist()

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def normalize_column(df, column, method='standard'):
    """
    Normalizes or scales a numeric column using a selected method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The name of the column to normalize.
    method : str
        One of 'standard', 'minmax', 'robust', or 'log'.

    Returns
    -------
    pd.Series
        A new Series with the normalized values.
    """
    data = df[[column]].dropna()  # keep 2D format for scalers

    if method == 'standard':
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
    elif method == 'robust':
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data)
    elif method == 'log':
        # Add 1 to avoid log(0), safe for positive salary data
        scaled = np.log1p(data[column].values)  # Esto genera un array
    else:
        raise ValueError("Invalid method. Choose from 'standard', 'minmax', 'robust', or 'log'.")

    scaled_series = pd.Series(scaled.flatten(), index=data.index, name=f"{column}_{method}_scaled")

    print(f"\n'{column}' scaled using '{method}' method.")
    print(f"Min: {scaled_series.min():.2f}, Max: {scaled_series.max():.2f}, Mean: {scaled_series.mean():.2f}, Std: {scaled_series.std():.2f}")

    return scaled_series

from scipy.stats import pearsonr, spearmanr, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
def compare_numerical_variables(df, var1, var2, bins=5, categorize=False):
    """
    Compares two numerical variables through visual and statistical methods.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    var1 : str
        First numerical variable (predictor).
    var2 : str
        Second numerical variable (typically the target).
    bins : int
        Number of quantile bins to categorize var1 if needed.
    categorize : bool
        If True, creates and displays boxplot of categorized var1 vs var2.

    Returns
    -------
    None
    """
    print(f"\n===== Comparing: {var1} vs {var2} =====")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    sns.regplot(x=df[var1], y=df[var2], scatter_kws={'alpha': 0.5}, ax=axes[0])
    axes[0].set_title(f'Regression: {var1} vs {var2}')

    sns.histplot(df[var1], bins=20, kde=True, color='blue', ax=axes[1])
    axes[1].set_title(f'Histogram: {var1}')

    sns.histplot(df[var2], bins=20, kde=True, color='red', ax=axes[2])
    axes[2].set_title(f'Histogram: {var2}')

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig("../images/image.png", bbox_inches="tight", dpi=300)
    plt.show()

    # Correlation tests
    pearson_corr, pearson_p = pearsonr(df[var1].dropna(), df[var2].dropna())
    spearman_corr, spearman_p = spearmanr(df[var1].dropna(), df[var2].dropna())

    print("\nCorrelation:")
    print(f"Pearson:  r = {pearson_corr:.4f}  | p-value = {pearson_p:.4f}")
    print(f"Spearman: r = {spearman_corr:.4f}  | p-value = {spearman_p:.4f}")

    if pearson_p < 0.05:
        print(f"✅ Significant linear correlation between {var1} and {var2} (Pearson).")
    else:
        print(f"❌ No significant linear correlation (Pearson).")

    if spearman_p < 0.05:
        print(f"✅ Significant monotonic correlation between {var1} and {var2} (Spearman).")
    else:
        print(f"❌ No significant monotonic correlation (Spearman).")

    # Optional ANOVA after binning
    df = df.copy()
    df['var1_binned'] = pd.qcut(df[var1], bins, duplicates='drop')

    groups = [df[var2][df['var1_binned'] == category] for category in df['var1_binned'].unique()]
    f_stat, p_value = f_oneway(*groups)

    print("\nANOVA Test:")
    print(f"F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

    if p_value < 0.05:
        print(f"✅ Categorizing {var1} reveals significant differences in {var2}.")
    else:
        print(f"❌ No significant differences in {var2} across {var1} bins.")

    if categorize:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df['var1_binned'], y=df[var2])
        plt.title(f'{var2} by {var1} bins')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    df.drop(columns=['var1_binned'], inplace=True, errors='ignore')

from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_categorical_vs_target(df, cat_var, target_var, category_names=None, min_frequency=1):
    """
    Analyzes the relationship between a categorical variable and a numerical target variable.
    Includes summary stats, ANOVA test, and 3 plots:
        1. Category frequency
        2. Target distribution (boxplot)
        3. Mean target per category
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    cat_var : str
        The name of the categorical variable.
    target_var : str
        The name of the target numeric variable.
    category_names : dict, optional
        Optional mapping for category labels.
    min_frequency : float
        Minimum frequency (proportion) required to keep a category.
    """
    df_copy = df.copy()

    # Drop missing
    df_copy = df_copy.dropna(subset=[cat_var, target_var])

    if category_names:
        df_copy[cat_var] = df_copy[cat_var].map(category_names)

    

    print(f"\n===== Analysis: '{cat_var}' vs '{target_var}' =====")
    print(f"Unique categories: {df_copy[cat_var].nunique()}\n")

    # Filter rare categories
    value_counts = df_copy[cat_var].value_counts(normalize=True)
    valid_categories = value_counts[value_counts >= min_frequency].index
    df_copy = df_copy[df_copy[cat_var].isin(valid_categories)]

    # Summary stats
    
    grouped_stats = df_copy.groupby(cat_var)[target_var].agg(['count', 'mean', 'std', 'min', 'max'])
    ordered_cats = grouped_stats.sort_values('mean', ascending=False).index

    print("Descriptive statistics:")
    print(grouped_stats.loc[ordered_cats])

    # ANOVA
    groups = [df_copy[target_var][df_copy[cat_var] == cat] for cat in ordered_cats]
    f_stat, p_value = f_oneway(*groups)
    print(f"\nANOVA: F = {f_stat:.4f}, p-value = {p_value:.4f}")

    if p_value < 0.05:
        print(f"✅ '{cat_var}' is a significant predictor of '{target_var}' (p < 0.05).")
    else:
        print(f"❌ '{cat_var}' is NOT a significant predictor of '{target_var}' (p ≥ 0.05).")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Frequency
    cat_counts = df_copy[cat_var].value_counts().loc[ordered_cats]
    sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=axes[0], palette="husl")
    axes[0].set_title("Category Frequency")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom')

    # Plot 2: Boxplot
    sns.boxplot(x=df_copy[cat_var], y=df_copy[target_var], order=ordered_cats, ax=axes[1], palette="husl")
    axes[1].set_title(f"{target_var} Distribution by {cat_var}")
    axes[1].tick_params(axis='x', rotation=30)

    # Plot 3: Mean target
    mean_values = grouped_stats['mean'].loc[ordered_cats]
    sns.barplot(x=mean_values.index, y=mean_values.values, ax=axes[2], palette="husl")
    axes[2].set_title(f"Mean {target_var} by {cat_var}")
    axes[2].set_ylabel(f"Mean {target_var}")
    axes[2].tick_params(axis='x', rotation=30)
    for p in axes[2].patches:
        axes[2].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def compare_categorical_with_targets(df, cat_var, target1, target2, category_names=None, min_frequency=0.01,image_name = None):
    """
    Compares the relationship between a categorical variable and two versions of a target variable
    (e.g., original and transformed). Displays ANOVA results and side-by-side visualizations.
    """

    df_copy = df.copy()
    if category_names:
        df_copy[cat_var] = df_copy[cat_var].map(category_names)

    df_copy = df_copy.dropna(subset=[cat_var, target1, target2])

    # Remove rare categories
    valid_cats = df_copy[cat_var].value_counts(normalize=True)
    valid_cats = valid_cats[valid_cats >= min_frequency].index
    df_copy = df_copy[df_copy[cat_var].isin(valid_cats)]

    # Convert to category and clean unused
    if not pd.api.types.is_categorical_dtype(df_copy[cat_var]):
        df_copy[cat_var] = df_copy[cat_var].astype("category")
    df_copy[cat_var] = df_copy[cat_var].cat.remove_unused_categories()

    print(f"\n===== Comparison: '{cat_var}' vs '{target1}' and '{target2}' =====")
    print(f"Unique categories analyzed: {df_copy[cat_var].nunique()}\n")

    # Summary stats
    grouped1 = df_copy.groupby(cat_var)[target1].agg(['count', 'mean', 'std', 'min', 'max'])
    grouped2 = df_copy.groupby(cat_var)[target2].agg(['mean', 'std'])
    summary = grouped1.join(grouped2, lsuffix='_orig', rsuffix='_trans')
    summary = summary.sort_values("mean_orig", ascending=False)

    print("Descriptive statistics (sorted by original target):")
    print(grouped1.sort_values("mean", ascending=False))

    # ANOVA tests
    categories = summary.index.tolist()


    groups1 = [df_copy[target1][df_copy[cat_var] == cat] for cat in categories]
    groups2 = [df_copy[target2][df_copy[cat_var] == cat] for cat in categories]
    f1, p1 = f_oneway(*groups1)
    f2, p2 = f_oneway(*groups2)
    print(f"\nANOVA on {target1}: F = {f1:.4f}, p = {p1:.4f} → {'✅' if p1 < 0.05 else '❌'}")
    print(f"ANOVA on {target2}: F = {f2:.4f}, p = {p2:.4f} → {'✅' if p2 < 0.05 else '❌'}")

    # Palette
    palette_colors = sns.color_palette("husl", len(categories))
    freq_counts = df_copy[cat_var].value_counts().reindex(categories)


    # 1. Obtener categorías ordenadas según salario
    categories = summary.sort_values("mean_orig", ascending=False).index.tolist()

    # 2. Convertirlas a string (si es necesario para los gráficos)
    categories = [str(c) for c in categories]

    # 3. Asegurarse de que los índices estén en string también
    summary.index = summary.index.astype(str)
    freq_counts.index = freq_counts.index.astype(str)

    # 4. Crear la paleta de colores consistente
    palette_colors = sns.color_palette("husl", len(categories))
    palette_dict = dict(zip(categories, palette_colors))

    # 5. Usar categories en todos los plots
    barplot_df1 = summary.loc[categories].reset_index()

    # --- Frequency Plot ---
    # Reorder frequency counts to match salary order

    # Frequency plot (ordered by salary)
    plt.figure(figsize=(11, 3))

    ax = sns.barplot(x=freq_counts.index, y=freq_counts.values, palette=palette_dict, order=categories)
    ax.set_title("Category Frequency (ordered by avg salary)")
    ax.set_xlabel(cat_var)
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=30, labelsize=9)

    # Add value labels
    max_height = max([p.get_height() for p in ax.patches])
    ax.set_ylim(0, max_height * 1.15)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height + (0.01 * height)),
                    ha='center', va='bottom', fontsize=9)


    plt.tight_layout()
    plt.show()


    # --- Visuals for Original Salary ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Boxplot target1
    sns.boxplot(x=cat_var, y=target1, data=df_copy, order=categories, ax=axes[0], palette=palette_dict)
    axes[0].set_title(f"{target1} by {cat_var}")
    axes[0].tick_params(axis='x', rotation=30)

    # Create a DataFrame ordered by salary for the barplot
    barplot_df1 = summary.loc[categories].reset_index()

    sns.barplot(data=barplot_df1, x=cat_var, y="mean_orig", ax=axes[1], palette=palette_dict, order=categories)
    axes[1].set_title(f"Mean {target1} by {cat_var}")
    axes[1].tick_params(axis='x', rotation=30)
    max_height = max([p.get_height() for p in axes[1].patches])
    axes[1].set_ylim(0, max_height * 1.15)

    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height + (0.01 * height)),
                        ha='center', va='bottom', fontsize=9)


    plt.tight_layout()
    if image_name is not None:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/{image_name}.png", dpi=300)
    plt.show()

    # --- Visuals for Transformed Salary ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))

    # Boxplot target2
    sns.boxplot(x=cat_var, y=target2, data=df_copy, order=categories, ax=axes[0], palette=palette_dict)
    axes[0].set_title(f"{target2} by {cat_var}")
    axes[0].tick_params(axis='x', rotation=30)

    # Create a DataFrame ordered by transformed salary for the barplot
    barplot_df2 = summary.loc[categories].reset_index()

    sns.barplot(data=barplot_df2, x=cat_var, y="mean_trans", ax=axes[1], palette=palette_dict, order=categories)
    axes[1].set_title(f"Mean {target2} by {cat_var}")
    axes[1].tick_params(axis='x', rotation=30)
    max_height = max([p.get_height() for p in axes[1].patches])
    axes[1].set_ylim(0, max_height * 1.15)

    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height + (0.01 * height)),
                        ha='center', va='bottom', fontsize=9)


    plt.tight_layout()
    
    plt.show()

def analyze_binary_vs_target(df, var, target_orig, target_trans, category_names=None):
    """
    Analyze binary categorical variable vs numerical target.
    Performs t-test on transformed target and if significant, plots original target distributions.

    Parameters:
    - df: DataFrame.
    - var: Binary categorical column.
    - target_orig: Original target column (e.g., avg_salary).
    - target_trans: Transformed target column (e.g., avg_salary_boxcox).
    - category_names: Optional dictionary to rename binary categories.

    Returns:
    - Tuple (column name, mean difference, std deviation) if significant, else empty string.
    """
    df_copy = df.copy()
    if category_names:
        df_copy[var] = df_copy[var].map(category_names)

    print(f"\n===== Analysis of {var} vs {target_trans} (Binary - Transformed) =====\n")
    print("Category Counts:")
    print(df_copy[var].value_counts())

    group1 = df_copy[df_copy[var] == 1][target_trans]
    group2 = df_copy[df_copy[var] == 0][target_trans]

    # T-test on transformed target
    stat, p_value = ttest_ind(group1, group2, equal_var=False)
    print(f"\nT-test on {target_trans}: Statistic = {stat:.4f}, p-value = {p_value:.4f}")

    if p_value >= 0.05:
        print(f"❌ {var} does not show a statistically significant relationship with {target_trans}.")
        return ""

    print(f"✅ {var} is a significant predictor of {target_trans}.\n")

    # Summary stats using original target
    group1_orig = df_copy[df_copy[var] == 1][target_orig]
    group2_orig = df_copy[df_copy[var] == 0][target_orig]

    mean_diff = group1_orig.mean() - group2_orig.mean()
    std_diff = np.sqrt(group1_orig.var()/len(group1_orig) + group2_orig.var()/len(group2_orig))

    print(f"Mean Difference (original target): {mean_diff:.2f}")
    print(f"Std. Deviation of Difference (original target): {std_diff:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    sns.countplot(x=var, data=df_copy, palette="coolwarm", ax=axes[0])
    axes[0].set_title("Category Count")
    axes[0].tick_params(axis='x', rotation=30)

    sns.boxplot(x=var, y=target_orig, data=df_copy, palette="coolwarm", ax=axes[1])
    axes[1].set_title(f"{target_orig} by {var}")
    axes[1].tick_params(axis='x', rotation=30)

    sns.histplot(group1_orig, label='1', kde=True, color='blue', ax=axes[2])
    sns.histplot(group2_orig, label='0', kde=True, color='red', ax=axes[2])
    axes[2].legend()
    axes[2].set_title(f"{target_orig} Distribution by {var}")

    plt.tight_layout()
    plt.show()

    return var, mean_diff, std_diff

def analyze_and_filter_tools(
    df, 
    df_binary, 
    significant_tools, 
    target_col='avg_salary', 
    transformed_col='avg_salary_boxcox', 
    show_plots=True
):
    counts = []
    tools = []

    # Filter binary columns that exist in df
    binary_cols_in_df = [col for col in df_binary.columns if col in df.columns]

    # Count tool mentions
    for column in binary_cols_in_df:
        count = df[column].astype(int).sum()
        counts.append(count)
        tools.append(column)

    # Create DataFrame of tools
    df_tools = pd.DataFrame({"Tool": tools, "Count": counts})
    df_tools["Significant"] = df_tools["Tool"].isin(significant_tools)
    df_tools = df_tools.sort_values(by="Count", ascending=False).reset_index(drop=True)

    # Set colors based on significance
    colors = ["green" if tool in significant_tools else "red" for tool in df_tools["Tool"]]

    # Plot 1: Tool frequency
    if show_plots:
        plt.figure(figsize=(12, 4))
        ax = sns.barplot(x="Tool", y="Count", data=df_tools, palette=colors)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.title("Most Requested Data Science & Analyst Tools in Job Descriptions")
        plt.ylabel("Count of Job Listings")
        plt.xlabel("Tools")
        for p in ax.patches:
            ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, color='black')
        plt.tight_layout()
        os.makedirs("images", exist_ok=True)
        plt.savefig("../images/tool_frequency.png", bbox_inches="tight", dpi=300)
        plt.show()

    # Drop non-significant tools
    non_significant = df_tools[~df_tools["Significant"]]["Tool"].tolist()
    tools_to_drop = [col for col in non_significant if col in df.columns]
    df_filtered = df.drop(columns=tools_to_drop)
    num_removed = len(tools_to_drop)
    print(f"{num_removed} non-significant tools were removed.")

    # Calculate salary stats for significant tools
    data_summary = []
    for tool in significant_tools:
        if tool in df_filtered.columns:
            mask = df_filtered[tool] == 1
            mean_salary = df_filtered.loc[mask, target_col].mean()
            std_salary = df_filtered.loc[mask, target_col].std()
            mean_trans = df_filtered.loc[mask, transformed_col].mean()
            std_trans = df_filtered.loc[mask, transformed_col].std()
            data_summary.append({
                "Tool": tool,
                "Mean": mean_salary,
                "Std Dev": std_salary,
                "Mean_trans": mean_trans,
                "Std Dev_trans": std_trans
            })

    df_significant = pd.DataFrame(data_summary).sort_values(by="Mean", ascending=False)

    # Plot 2: Mean salary difference from overall mean
    if show_plots:
        overall_mean = df[target_col].mean()
        df_significant["Diff From Mean"] = df_significant["Mean"] - overall_mean

        plt.figure(figsize=(12, 4))
        colors_diff = ['green' if val >= 0 else 'red' for val in df_significant["Diff From Mean"]]
        ax2 = sns.barplot(x="Tool", y="Diff From Mean", data=df_significant, palette=colors_diff)

        plt.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.title("Difference from Overall Mean Salary for Significant Tools")
        plt.ylabel("Salary Difference")
        plt.xlabel("Tools")
        for p in ax2.patches:
            ax2.annotate(f'{p.get_height():+.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, color='black')
        plt.tight_layout()
        os.makedirs("images", exist_ok=True)
        plt.savefig("../images/tool_salary_impact.png", bbox_inches="tight", dpi=300)
        plt.show()

    return df_filtered, num_removed, df_significant
def basic_data_checks(df, remove_duplicates=True):
    """
    Performs basic data quality checks on the DataFrame:
    
    - Checks for missing values and shows columns affected.
    - Identifies and optionally removes duplicate rows.
    - Detects columns with only one unique value (non-informative).
    - Displays the DataFrame shape and column data types.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to check.
    remove_duplicates : bool, default=True
        Whether to drop duplicate rows from the DataFrame.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame (if duplicates are removed).
    """
    print("📐 Initial DataFrame shape:", df.shape)

    # Missing values
    missing = df.isna().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("✅ No missing values found.")
    else:
        print(f"⚠️ Total missing values: {total_missing}")
        print("Missing values by column:")
        print(missing[missing > 0])

    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        print("✅ No duplicate rows found.")
    else:
        print(f"⚠️ Found {duplicate_count} duplicate rows.")
        if remove_duplicates:
            df.drop_duplicates(inplace=True)
            print("🧹 Duplicates removed.")
            print("📏 New shape:", df.shape)

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if constant_cols:
        print("\n🛑 Columns with constant values (non-informative):")
        print(constant_cols)

    # Data types
    print("\n🔍 Data types:")
    print(df.dtypes)

    return df


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_tool_usage_by_job_category(df_analysis, df_binary, target_col='avg_salary', usage_threshold=40, show_plot=True):
    # 1. Determine which binary tool columns are in df_analysis
    binary_tool_columns = [col for col in df_binary.columns if col in df_analysis.columns]

    # 2. Compute average salary per job category
    category_salary_mean = df_analysis.groupby("Job Category")[target_col].mean().sort_values(ascending=False)

    # 3. Compute average tool usage per category (in %)
    tool_heatmap_data = df_analysis.groupby("Job Category")[binary_tool_columns].mean().T * 100

    # 4. Filter tools with total usage above threshold
    total_usage = df_analysis[binary_tool_columns].sum()
    tool_heatmap_data = tool_heatmap_data.loc[total_usage[total_usage > usage_threshold].index]

    # 5. Reorder job categories based on average salary
    ordered_categories = category_salary_mean.index
    tool_heatmap_data = tool_heatmap_data[ordered_categories.intersection(tool_heatmap_data.columns)]

    # 6. Plot the heatmap
    if show_plot:
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            tool_heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            cbar_kws={'label': '% of Job Listings'}
        )
        plt.title("Tool Prevalence by Job Category (%)\n(Ordered by Avg. Salary)", fontsize=14)
        plt.xlabel("Job Category (highest avg. salary → left)")
        plt.ylabel("Tool")
        plt.tight_layout()
        
        plt.show()

    return tool_heatmap_data, category_salary_mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tool_heatmap_by_job_category(
    df, 
    df_binary, 
    target_col='avg_salary', 
    job_category_col='Job Category', 
    min_usage=40,
    show_plot=True
):
    # Step 1: Use only binary tool columns that are present in df
    binary_tool_cols = [col for col in df_binary.columns if col in df.columns]

    # Step 2: Compute average salary per job category
    category_salary_mean = df.groupby(job_category_col)[target_col].mean().sort_values(ascending=False)

    # Step 3: Compute mean tool usage per job category (in %)
    tool_usage = df.groupby(job_category_col)[binary_tool_cols].mean().T * 100

    # Step 4: Filter tools with total usage above the threshold
    total_usage = df[binary_tool_cols].sum()
    tool_usage = tool_usage.loc[total_usage[total_usage > min_usage].index]

    # Step 5: Order tools by total usage (same order as barplot)
    overall_usage_order = total_usage.sort_values(ascending=False).index
    tool_usage = tool_usage.loc[overall_usage_order.intersection(tool_usage.index)]


    # Step 6: Reorder job categories (columns) by salary
    tool_usage = tool_usage[category_salary_mean.index.intersection(tool_usage.columns)]

    # Step 7: Plot heatmap
    if show_plot:
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            tool_usage,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            cbar_kws={'label': '% of Job Listings'}
        )
        plt.title("Tool Prevalence by Job Category (%)\n(Tools ordered by overall usage)", fontsize=14)
        plt.xlabel("Job Category (highest avg. salary → left)")
        plt.ylabel("Tool")
        plt.tight_layout()
        os.makedirs("images", exist_ok=True)
        plt.savefig("../images/tool_usage_heatmap.png", bbox_inches="tight", dpi=300)
        plt.show()

    return tool_usage
