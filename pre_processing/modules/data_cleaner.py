import hashlib

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from pre_processing.modules.pre_processing_utils import infer_column_type


class DataCleaner:
    def __init__(self):
        self.type_validation_report = {}
        self.missing_info = {}
        self.duplicate_indices = []
    

    def identify_duplicate_rows(self, df):
        """
        Identify and remove duplicate rows using row hashing.
        """
        def hash_row(row):
            row_str = ''.join(str(x) for x in row)
            return hashlib.md5(row_str.encode()).hexdigest()
        
        df_copy = df.copy()
        df_copy['_row_hash'] = df_copy.apply(hash_row, axis=1)
        
        # Find duplicates (keep=False marks all duplicates, including first occurrence)
        duplicate_mask = df_copy.duplicated(subset=['_row_hash'], keep='first')
        self.duplicate_indices = df_copy[duplicate_mask].index.tolist()
        
        # Remove duplicates and the hash column
        df_unique = df_copy.drop_duplicates(subset=['_row_hash']).drop(columns=['_row_hash'])
        
        print(f"Found {len(self.duplicate_indices)} duplicate rows")
        return df_unique
    

    def infer_and_validate_column_types(self, df):
        """
        Infer and validate data types for each column with regex handling.
        """
        def is_datetime(column) -> bool:
            try:
                pd.to_datetime(column)
                return True
            except (ValueError, TypeError):
                return False
            
        import pandas as pd

        column_dtype = {}
        
        for column in df.columns:
            if is_datetime(df[column]):
                converted_series = df[column]
                column_dtype[column] = "datetime"  
            else:
                converted_series, inferred_type = infer_column_type(df[column])
                column_dtype[column] = inferred_type  
                invalid_indices = df[column].index[pd.isna(converted_series)]
                self.type_validation_report[column] = {
                    'inferred_type': inferred_type,
                    'invalid_entries_count': len(invalid_indices),
                    'invalid_entry_indices': list(invalid_indices)
                }
                if len(invalid_indices) > 0:
                    print(f"Column '{column}': {len(invalid_indices)} invalid entries for type '{inferred_type}'")
                df[column] = converted_series

        numeric = ['int', 'float']
        numeric_cols = [col for col in df.columns if (column_dtype[col] in numeric)]
        missing_percentages = df.isna().sum() / len(df) * 100
        high_missing_numeric_cols = [col for col in numeric_cols if missing_percentages[col] >= 5]

        if high_missing_numeric_cols:
            if not df[high_missing_numeric_cols].isnull().all().all():
                imputer = KNNImputer(n_neighbors=5)
                scaler = StandardScaler()
                null_mask = df[high_missing_numeric_cols].isnull()
                scaled_data = scaler.fit_transform(df[high_missing_numeric_cols])
                imputed_scaled_data = imputer.fit_transform(scaled_data)
                df[high_missing_numeric_cols]= scaler.inverse_transform(imputed_scaled_data)
                # imputed_values = imputer.fit_transform(df[high_missing_numeric_cols])
                df.loc[:, high_missing_numeric_cols] = df[high_missing_numeric_cols].where(~null_mask, imputed_scaled_data)

        for column in df.columns:
            if column_dtype[column] == "datetime":
                    df[column] = df[column].ffill().bfill()
            elif column_dtype[column] == "object":
                    mode_value = df[column].mode()
                    if  not mode_value.empty:
                        df[column] = df[column].fillna(mode_value.iloc[0])
            elif column_dtype[column] in numeric:
                if missing_percentages[column] < 5:
                    # Check skewness
                    skewness = skew(df[column].dropna())
                    if abs(skewness) > 0.5:
                        # Use median for skewed data
                        df[column] = df[column].fillna(df[column].median())
                    else:
                        df[column] = df[column].fillna(df[column].mean())            
        
        return df,column_dtype
    
    def detect_missing_values(self, df):
        """
        Detect missing values and return their details.
        """
        print("Detecting missing values...")
        total_missing = 0
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_indices = df[df[column].isnull()].index.tolist()
                missing_percentage = (missing_count / len(df)) * 100
                
                self.missing_info[column] = {
                    'total_missing': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2),
                    'missing_indices': missing_indices
                }
                total_missing += missing_count
                print(f"Column '{column}': {missing_count} missing values ({missing_percentage:.2f}%)")
        
        if total_missing == 0:
            print("No missing values found in the dataset")
        else:
            print(f"Total missing values found: {total_missing}")
            
        return self.missing_info
    
