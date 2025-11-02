import pandas as pd
import json

def rule_based_cleaning_processor(df: pd.DataFrame, rules: dict = {}):
    print(f"Applying rules to {len(df)} rows...")
    original_count = len(df)
    
    for column, rule in rules.items():
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataset")
            continue
            
        if len(rule) >= 2 and rule[0] and rule[1]:  # Numeric range (floor, ceil)
            try:
                floor_val = float(rule[0])
                ceil_val = float(rule[1])
                before_count = len(df)
                df = df[(df[column].isnull()) | (df[column].between(floor_val, ceil_val))]
                after_count = len(df)
                print(f"Applied range filter to '{column}': {floor_val}-{ceil_val}, removed {before_count - after_count} rows")
            except (ValueError, TypeError) as e:
                print(f"Error applying numeric filter to '{column}': {e}")
        elif len(rule) > 0:  # Categorical filter
            try:
                # Keep only selected categories
                selected_categories = [r for r in rule if r]  # Remove empty strings
                if selected_categories:
                    before_count = len(df)
                    df = df[(df[column].isnull()) | (df[column].isin(selected_categories))]
                    after_count = len(df)
                    print(f"Applied category filter to '{column}': kept {selected_categories}, removed {before_count - after_count} rows")
            except Exception as e:
                print(f"Error applying categorical filter to '{column}': {e}")
    
    final_count = len(df)
    print(f"Rule-based cleaning completed: {original_count} -> {final_count} rows")
    return df

def rule_based_cleaning(df , file_path ):
    with open(file_path , 'r') as file:
        rules = json.load(file)
    return rule_based_cleaning_processor(df,rules)