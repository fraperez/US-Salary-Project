import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Modeling
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve




def analyze_numeric_correlations(df, method='pearson', figsize=(11, 4), cmap='coolwarm', show_matrix=True):
    """
    Analyzes correlations between numerical variables with more than two unique values and displays a heatmap.
    
    Parameters:
    - df: pandas DataFrame
    - method: correlation method, 'pearson' (default) or 'spearman'
    - figsize: size of the heatmap figure (width, height)
    - cmap: colormap for the heatmap
    - show_matrix: if True, prints the correlation matrix
    """
    if method not in ['pearson', 'spearman']:
        raise ValueError("Method must be 'pearson' or 'spearman'")

    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Keep only columns with more than two unique values
    filtered_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() > 2]

    if len(filtered_cols) < 2:
        print("Not enough numeric columns with more than two unique values.")
        return

    print(f"Numeric columns analyzed ({len(filtered_cols)}): {filtered_cols}")

    # Filtered DataFrame
    filtered_df = numeric_df[filtered_cols]

    # Compute correlation matrix
    corr_matrix = filtered_df.corr(method=method)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        cbar_kws={"shrink": 0.75}
    )
    plt.title(f"Correlation Matrix ({method.title()}) of Numerical Variables", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    #if show_matrix:
        #return corr_matrix

from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cramers_v_matrix(df, cat_cols, figsize=(10, 8), cmap='coolwarm', max_categories=20,
                     show_matrix=True, threshold=0.6, return_pairs=True):
    """
    Computes and visualizes Cramér's V association matrix between categorical variables.
    
    Parameters:
    - df: pandas DataFrame
    - cat_cols: list of categorical columns to evaluate
    - figsize: heatmap figure size
    - cmap: heatmap color palette
    - max_categories: exclude columns with too many unique categories
    - show_matrix: if True, shows the heatmap
    - threshold: return pairs with Cramér's V > threshold
    - return_pairs: if True, returns the high-correlation pairs
    
    Returns:
    - matrix: DataFrame of Cramér's V values
    - high_v_pairs: list of tuples (col1, col2, score) above the threshold
    """
    if len(cat_cols) < 2:
        print("Not enough categorical columns to compute Cramér's V.")
        return

    # Filter out columns with too many unique values
    filtered_cols = [col for col in cat_cols if df[col].nunique() <= max_categories]

    print(f"Analyzing {len(filtered_cols)} categorical columns: {filtered_cols}")

    results = pd.DataFrame(index=filtered_cols, columns=filtered_cols, dtype=float)
    high_v_pairs = []

    for i, col1 in enumerate(filtered_cols):
        for j, col2 in enumerate(filtered_cols):
            if j < i:
                continue  # avoid duplicate pairs (upper triangle only)
            if col1 == col2:
                results.loc[col1, col2] = 1.0
            else:
                confusion = pd.crosstab(df[col1], df[col2])
                chi2 = chi2_contingency(confusion, correction=False)[0]
                n = confusion.sum().sum()
                phi2 = chi2 / n
                r, k = confusion.shape
                v = np.sqrt(phi2 / min(k - 1, r - 1))
                results.loc[col1, col2] = round(v, 3)
                results.loc[col2, col1] = round(v, 3)

                if v >= threshold:
                    high_v_pairs.append((col1, col2, round(v, 3)))

    # Plot heatmap
    if show_matrix:
        plt.figure(figsize=figsize)
        sns.heatmap(results, annot=True, fmt=".2f", cmap=cmap, square=True, cbar_kws={"shrink": 0.75})
        plt.title("Cramér's V Association Matrix for Categorical Variables", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    if return_pairs:
        return results, sorted(high_v_pairs, key=lambda x: -x[2])
    else:
        return results
from scipy import stats
import pandas as pd

def compare_categorical_predictors_in_pairs(df, pairs, target, verbose=True):
    """
    Compares pairs of categorical variables using ANOVA and eta squared (R²)
    to determine which variable has more predictive power over the target.
    
    Parameters:
    - df: pandas DataFrame
    - pairs: list of tuples (cat1, cat2, score), such as from Cramér’s V
    - target: numeric target variable
    - verbose: if True, prints comparison summaries
    
    Returns:
    - to_remove: list of columns suggested for removal
    - summary_results: DataFrame with comparison metrics
    """
    to_remove = []
    all_results = []

    for cat1, cat2, _ in pairs:
        results = []

        for cat in [cat1, cat2]:
            groups = [df[df[cat] == val][target].dropna() for val in df[cat].dropna().unique()]
            
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                grand_mean = df[target].mean()
                ss_between = sum([(grp.mean() - grand_mean)**2 * len(grp) for grp in groups])
                ss_total = sum((df[target] - grand_mean)**2)
                eta_squared = ss_between / ss_total

                results.append({
                    'Variable': cat,
                    'ANOVA p-value': round(p_val, 4),
                    'Explained Variance (R²)': round(eta_squared, 3)
                })
            else:
                results.append({
                    'Variable': cat,
                    'ANOVA p-value': None,
                    'Explained Variance (R²)': None
                })

        # Sort by R²
        comp_df = pd.DataFrame(results).sort_values(by='Explained Variance (R²)', ascending=False)
        all_results.append(comp_df)

        if verbose:
            print(f"\n🔎 Comparing: {cat1} vs {cat2}")
            print(comp_df)

        # Suggest removal of weaker variable
        if pd.notna(comp_df["Explained Variance (R²)"].iloc[-1]):
            to_remove.append(comp_df["Variable"].iloc[-1])

    # Combine all individual summaries
    summary_results = pd.concat(all_results, ignore_index=True).drop_duplicates()

    return to_remove, summary_results

def analyze_binary_correlations(df, figsize=(10, 8), cmap='coolwarm'):
    """
    Analiza correlación entre columnas binarias (0/1) con Phi coefficient.
    Muestra matriz de correlación y heatmap.
    """
    # Detectar columnas binarias
    binary_cols = [col for col in df.columns if df[col].dropna().nunique() == 2 and set(df[col].dropna().unique()).issubset({0,1})]
    
    if len(binary_cols) < 2:
        print("No hay suficientes columnas binarias (0/1) para analizar.")
        return
    
    print(f"Columnas binarias analizadas ({len(binary_cols)}): {binary_cols}")
    
    binary_df = df[binary_cols]
    
    # Matriz de correlación (Phi = Pearson en binarias)
    corr_matrix = binary_df.corr(method='pearson')
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, square=True, cbar_kws={"shrink": 0.75})
    plt.title("Matriz de Correlación entre Binarias (Phi coefficient)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def analyze_numeric_preprocessing(df, z_thresh=3):
    """
    Analiza variables numéricas: outliers, normalidad, distribución.
    
    - z_thresh: umbral para Z-score (default 3).
    """
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if df[col].nunique() > 2]
    
    if not numeric_cols:
        print("No hay columnas numéricas suficientes para analizar.")
        return
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        print(f"\n===== Análisis de '{col}' =====")
        
        # Outliers - Z-score
        z_scores = np.abs(stats.zscore(data))
        outliers_z = (z_scores > z_thresh).sum()
        
        # Outliers - IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
        
        print(f"Outliers (Z > {z_thresh}): {outliers_z}")
        print(f"Outliers (IQR): {outliers_iqr}")
    

def print_metrics(y_true, y_pred, dataset='Test'):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset} RMSE: {rmse:.2f}")
    print(f"{dataset} MAE : {mae:.2f}")
    print(f"{dataset} R²   : {r2:.3f}")

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(y_train, y_train_pred, y_test, y_test_pred, model_name="Model", 
                   boxcox_transformer=None, transform_back=False):
    """
    Evaluates a regression model on train and test sets.

    If transform_back=True and boxcox_transformer is provided,
    predictions and actual values are inverse-transformed using PowerTransformer.
    """

    def safe_inverse(array):
        array = array.values.reshape(-1, 1) if isinstance(array, pd.Series) else array.reshape(-1, 1)
        array = np.where(array <= 0, 1e-6, array)
        return boxcox_transformer.inverse_transform(array).flatten()

    if transform_back and boxcox_transformer is not None:
        y_train = safe_inverse(y_train)
        y_train_pred = safe_inverse(y_train_pred)
        y_test = safe_inverse(y_test)
        y_test_pred = safe_inverse(y_test_pred)

    results = {
        'Model': model_name,
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train RMSE': mean_squared_error(y_train, y_train_pred, squared=False),
        'Test RMSE': mean_squared_error(y_test, y_test_pred, squared=False),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred)
    }

    return pd.DataFrame([results]).style.format({
        'Train MAE': "{:.2f}",
        'Test MAE': "{:.2f}",
        'Train RMSE': "{:.2f}",
        'Test RMSE': "{:.2f}",
        'Train R²': "{:.3f}",
        'Test R²': "{:.3f}"
    })

def plot_learning_curve(estimator, X, y, title, transformer=None):
    """
    Generates a learning curve for a given estimator and dataset.
    If transformer is provided (e.g. BoxCox), it applies inverse_transform to y-axis values.
    """
    plt.figure(figsize=(8, 5))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_root_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42)

    # Convert negative RMSE to positive
    train_rmse = -train_scores.mean(axis=1)
    test_rmse = -test_scores.mean(axis=1)

    plt.plot(train_sizes, train_rmse, 'o-', label="Train RMSE")
    plt.plot(train_sizes, test_rmse, 'o-', label="Validation RMSE")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()