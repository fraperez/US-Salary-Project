import re
import numpy as np
import pandas as pd

import re

def title_simplifier(title, debug=False):
    """
    Simplifies job titles into standardized categories using prioritized regex patterns.

    Args:
        title (str): The job title to simplify.
        debug (bool): If True, prints unmatched or suspicious titles.

    Returns:
        str: Simplified job category.
    """
    if not isinstance(title, str):
        return "na"

    title = title.lower()

    patterns = {
        "mle": [
            r"\bmachine\s+learning\b",
            r"\bmle\b",
            r"\bml\s+engineer\b",
            r"\bdeep\s+learning\b"
        ],
        "data scientist": [
            r"\bdata\s+scientists?\b",
            r"\bdata\s+science\b",
            r"\bdatascientist\b",
            r"\bdatasci\b"
        ],
        "data engineer": [
            r"\bdata\s+engineer\b",
            r"\bdata\b.*\bengineer\b"
        ],
        "data analyst": [
            r"\bdata\s+analyst\b",
            r"\bdata\b.*\banalyst\b",
            r"\bdata\s+analytics?\b"
        ],
        "scientist": [
            r"\b\w*\s*scientist\b(?!.*data)"  # Avoid misclassifying 'data scientist'
        ]
    }

    for category, regex_list in patterns.items():
        if any(re.search(pattern, title, flags=re.IGNORECASE) for pattern in regex_list):
            return category

    if debug:
        print(f"🔍 Unmatched title: {title}")

    return "na"

def extract_experience(description):
    """
    Extracts the required years of experience from a job description and identifies potential outliers.

    Args:
        description (str): The job description text.

    Returns:
        tuple:
            - experience_years (float or np.nan): Extracted number of years, or NaN if not found.
            - matched_text (str or None): The exact matched phrase for reference.
    """
    if not isinstance(description, str):
        return (np.nan, None)

    # Regex to match phrases like: "3 years of experience", "2 to 4 years", "5+ years", etc.
    pattern = r'(\d{1,2})(\+)?\s*(?:-|to)?\s*(\d{0,2})?\s*(?:\+)?\s*years? of experience'
    match = re.search(pattern, description, flags=re.IGNORECASE)

    if match:
        first_num = match.group(1)
        second_num = match.group(3)

        try:
            if second_num:
                experience_years = (int(first_num) + int(second_num)) / 2
            else:
                experience_years = int(first_num)

            # Flag potential outliers
            if experience_years > 30:
                print(f"⚠️ Potential outlier: {experience_years} years | Phrase: '{match.group(0)}'")

            return (experience_years, match.group(0))

        except ValueError:
            return (np.nan, match.group(0))

    return (np.nan, None)

def remove_and_recalculate_experience(df, incorrect_indices):
    """
    Removes incorrect experience fragments from 'Job Description' for specified indices 
    and recalculates the years of experience.

    Args:
        df (pd.DataFrame): DataFrame with 'Job Description', 'Experience Extract', and 'Years of Experience' columns.
        incorrect_indices (list): List of row indices with incorrect experience extraction.

    Returns:
        pd.DataFrame: Updated DataFrame with corrected 'Years of Experience' and 'Experience Extract' values.
    """
    df_updated = df.copy()

    for idx in incorrect_indices:
        if pd.notna(df_updated.loc[idx, "Experience Extract"]):
            # Remove incorrect experience phrase from the job description
            df_updated.loc[idx, "Job Description"] = df_updated.loc[idx, "Job Description"].replace(
                df_updated.loc[idx, "Experience Extract"], "", 1
            )
            # Reset extracted experience
            df_updated.loc[idx, "Experience Extract"] = None
            df_updated.loc[idx, "Years of Experience"] = None

    # Recalculate experience using the improved description
    df_updated[["Years of Experience", "Experience Extract"]] = df_updated["Job Description"].apply(
        lambda desc: pd.Series(extract_experience(desc))
    )

    return df_updated

def broader_experience_extraction(description):
    """
    Searches for qualitative experience-related phrases in job descriptions 
    when a specific number of years is not mentioned.

    Args:
        description (str): The job description text.

    Returns:
        str or None: The matched phrase indicating experience, or None if not found.
    """
    if not isinstance(description, str):
        return None

    patterns = [
        r'(strong|proven|extensive|significant)\s+experience',
        r'(background in|expertise in|proficiency in)\s+\w+',
        r'(track record of|hands-on experience with)\s+\w+',
        r'experience with\s+\w+',
        r'(working knowledge of|practical experience with)\s+\w+'
    ]

    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return None

def categorize_broader_experience(text):
    """
    Categorizes qualitative experience phrases into levels.

    Args:
        text (str): Matched experience phrase.

    Returns:
        str: Experience category.
    """
    if not isinstance(text, str):
        return "Unknown"
    
    text_lower = text.lower()
    if any(word in text_lower for word in ["extensive", "proven", "significant", "strong"]):
        return "Strong Experience"
    elif any(word in text_lower for word in ["background in", "expertise in", "working knowledge of"]):
        return "Some Experience"
    elif any(word in text_lower for word in ["experience with", "hands-on experience"]):
        return "General Experience"
    else:
        return "Unknown"


def categorize_experience(years):
    """
    Categorizes numeric years of experience into defined experience levels.

    Args:
        years (float or int): Numeric value of years of experience.

    Returns:
        str: Experience level category.
    """
    if pd.isna(years):
        return "Unknown"
    try:
        years = float(years)
        if years <= 2:
            return "Entry Level"
        elif 3 <= years <= 5:
            return "Junior"
        elif 6 <= years <= 9:
            return "Mid Level"
        else:
            return "Senior"
    except:
        return "Unknown"


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


def reassign_industry(industry):
    """
    Reassigns industries into broader categories to reduce the number of unique industry groups.
    """

    if not isinstance(industry, str):
        industry = "-1"
    else:
        industry = industry.strip()

    industry_mapping = {
        "Biotech & Pharmaceuticals": "Health & Pharmaceuticals",
        "Health Care Services & Hospitals": "Health & Pharmaceuticals",
        "Health Care Products Manufacturing": "Health & Pharmaceuticals",
        "Health, Beauty, & Fitness": "Health & Pharmaceuticals",
        
        "Insurance Carriers": "Finance & Insurance",
        "Banks & Credit Unions": "Finance & Insurance",
        "Investment Banking & Asset Management": "Finance & Insurance",
        "Financial Analytics & Research": "Finance & Insurance",
        "Lending": "Finance & Insurance",
        "Insurance Agencies & Brokerages": "Finance & Insurance",
        "Stock Exchanges": "Finance & Insurance",
        "Financial Transaction Processing": "Finance & Insurance",
        "Brokerage Services": "Finance & Insurance",
        
        "Computer Hardware & Software": "Technology",
        "IT Services": "Technology",
        "Enterprise Software & Network Solutions": "Technology",
        "Internet": "Technology",
        "Video Games": "Technology",
        "Telecommunications Services": "Technology",
        "Telecommunications Manufacturing": "Technology",
        "TV Broadcast & Cable Networks": "Technology",
        
        "Consulting": "Professional Services",
        "Architectural & Engineering Services": "Professional Services",
        "Accounting": "Professional Services",
        "Staffing & Outsourcing": "Professional Services",
        "Education Training Services": "Professional Services",
        "Research & Development": "Professional Services",
        
        "Aerospace & Defense": "Manufacturing & Engineering",
        "Industrial Manufacturing": "Manufacturing & Engineering",
        "Transportation Equipment Manufacturing": "Manufacturing & Engineering",
        "Consumer Products Manufacturing": "Manufacturing & Engineering",
        "Food & Beverage Manufacturing": "Manufacturing & Engineering",
        
        "Advertising & Marketing": "Media & Entertainment",
        "Motion Picture Production & Distribution": "Media & Entertainment",
        "Auctions & Galleries": "Media & Entertainment",
        
        "Energy": "Energy & Utilities",
        "Gas Stations": "Energy & Utilities",
        "Mining": "Energy & Utilities",
        
        "Real Estate": "Real Estate & Construction",
        "Construction": "Real Estate & Construction",
        
        "Colleges & Universities": "Education",
        "K-12 Education": "Education",
        
        "Department, Clothing, & Shoe Stores": "Retail & Consumer Services",
        "Beauty & Personal Accessories Stores": "Retail & Consumer Services",
        "Other Retail Stores": "Retail & Consumer Services",
        "Wholesale": "Retail & Consumer Services",
        "Sporting Goods Stores": "Retail & Consumer Services",
        
        "Security Services": "Government & Public Services",
        "Federal Agencies": "Government & Public Services",
        "Social Assistance": "Government & Public Services",
        "Religious Organizations": "Government & Public Services",
        
        "Transportation Management": "Transportation & Logistics",
        "Trucking": "Transportation & Logistics",
        "Logistics & Supply Chain": "Transportation & Logistics",
        "Travel Agencies": "Transportation & Logistics",
        
        "Gambling": "Hospitality & Leisure",
        "Consumer Product Rental": "Hospitality & Leisure",
        
        "-1": "Unknown"
    }
    
    return industry_mapping.get(industry, "Other")

def reassign_sector(sector):
    """
    Reassigns raw sector labels into broader and more consistent categories.
    """
    mapping = {
        "Information Technology": "Technology",
        "Telecommunications": "Technology",
        
        "Biotech & Pharmaceuticals": "Health & Pharmaceuticals",
        "Health Care": "Health & Pharmaceuticals",
        
        "Insurance": "Finance & Insurance",
        "Finance": "Finance & Insurance",
        
        "Business Services": "Professional Services",
        "Consumer Services": "Professional Services",
        "Non-Profit": "Professional Services",
        
        "Aerospace & Defense": "Manufacturing & Engineering",
        "Manufacturing": "Manufacturing & Engineering",
        "Retail": "Retail",
        "Media": "Media & Entertainment",
        "Travel & Tourism": "Hospitality & Leisure",
        "Transportation & Logistics": "Transportation",
        "Education": "Education",
        "Oil, Gas, Energy & Utilities": "Energy & Utilities",
        "Real Estate": "Real Estate & Construction",
        "Government": "Government & Public Services",
        "other": "Other",
        "-1": "Unknown"
    }

    return mapping.get(sector, "Other")
