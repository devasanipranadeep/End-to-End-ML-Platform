import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
import re
warnings.filterwarnings('ignore')

def detect_unwanted_columns(df, target_column=None):
    """Detect unwanted columns like IDs, unique numbers, and high-cardinality columns"""
    unwanted_columns = []
    reasons = {}
    
    for column in df.columns:
        # Skip target column
        if column == target_column:
            continue
            
        col_data = df[column]
        reason = []
        
        # Check for ID-like columns (common patterns)
        id_patterns = [
            r'id$', r'_id$', r'ID$', r'Id$', r'identifier',
            r'code$', r'Code$', r'number$', r'Number$',
            r'ssn$', r'SSN$', r'passport', r'employee',
            r'customer', r'client', r'user', r'account',
            r'candidate', r'student', r'patient', r'member',
            r'record', r'entry', r'transaction', r'order',
            r'product', r'item', r'entity', r'reference',
            r'^candidate_id', r'^user_id', r'^student_id'  # Exact matches
        ]
        
        for pattern in id_patterns:
            if re.search(pattern, column, re.IGNORECASE):
                reason.append("ID-like column name")
                break
        
        # Check for unique or near-unique values
        unique_ratio = col_data.nunique() / len(col_data)
        if unique_ratio > 0.95:  # 95% or more unique values
            reason.append(f"High uniqueness ({unique_ratio:.1%})")
        elif unique_ratio > 0.90:  # 90% or more unique values
            reason.append(f"Very high uniqueness ({unique_ratio:.1%})")
        
        # Check for sequential numbers (like auto-increment IDs)
        if col_data.dtype in ['int64', 'int32', 'float64', 'float32']:
            if col_data.nunique() == len(col_data):
                sorted_vals = col_data.sort_values()
                if all(sorted_vals.iloc[i+1] - sorted_vals.iloc[i] == 1 for i in range(len(sorted_vals)-1)):
                    reason.append("Sequential integer values")
                else:
                    # Check for near-sequential (common in IDs)
                    diffs = sorted_vals.iloc[1:].values - sorted_vals.iloc[:-1].values
                    if np.all(diffs > 0) and np.mean(diffs) < 100:
                        reason.append("Increasing numeric sequence (likely ID)")
        
        # Check for columns with very high cardinality that are likely IDs
        if col_data.nunique() > 1000 and col_data.dtype in ['int64', 'int32']:
            if unique_ratio > 0.8:
                reason.append(f"High cardinality numeric ({col_data.nunique()} unique values)")
        
        # Check for constant columns (no variation)
        if col_data.nunique() == 1:
            reason.append("Constant value")
        
        # Check for high cardinality categorical columns
        if col_data.dtype == 'object' and col_data.nunique() > 100:
            reason.append(f"High cardinality categorical ({col_data.nunique()} unique values)")
        
        # Check for columns with too many missing values
        missing_ratio = col_data.isnull().sum() / len(col_data)
        if missing_ratio > 0.7:  # More than 70% missing
            reason.append(f"High missing values ({missing_ratio:.1%})")
        
        # Check for columns that are likely timestamps/dates that might not be useful
        if col_data.dtype == 'object':
            sample_values = col_data.dropna().head(10)
            if sample_values.str.match(r'\d{4}-\d{2}-\d{2}').all():
                reason.append("Date column (may not be useful for basic ML)")
        
        if reason:
            unwanted_columns.append(column)
            reasons[column] = reason
    
    return unwanted_columns, reasons

def remove_unwanted_columns(df, columns_to_remove):
    """Remove unwanted columns from the dataset"""
    df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
    return df_cleaned

def detect_outliers(df, column, method='iqr'):
    """Detect outliers in a column"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[z_scores > 3]
    return outliers

def handle_missing_values(df, strategy, columns=None):
    """Handle missing values in the dataset"""
    df_processed = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['object', 'category']:
                if strategy == 'mode':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df_processed[col] = df_processed[col].fillna(mode_val)
                elif strategy == 'drop':
                    df_processed = df_processed.dropna(subset=[col])
            else:
                if strategy == 'mean':
                    df_processed[col] = df_processed[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df_processed[col] = df_processed[col].fillna(df[col].median())
                elif strategy == 'mode':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    df_processed[col] = df_processed[col].fillna(mode_val)
                elif strategy == 'drop':
                    df_processed = df_processed.dropna(subset=[col])
    
    return df_processed

def encode_categorical(df, encoding_method, columns=None):
    """Encode categorical variables"""
    df_processed = df.copy()
    
    if columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_columns = columns
    
    # Remove target column from encoding if it exists
    target_col = st.session_state.get('target_column')
    if target_col and target_col in categorical_columns:
        categorical_columns.remove(target_col)
    
    for col in categorical_columns:
        if col in df_processed.columns:
            if encoding_method == 'label':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                st.session_state.preprocessing_steps.append(f"Label encoded {col}")
            elif encoding_method == 'onehot':
                # Get dummy variables
                dummies = pd.get_dummies(df_processed[col], prefix=col)
                df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)
                st.session_state.preprocessing_steps.append(f"One-hot encoded {col}")
    
    return df_processed

def scale_features(df, scaling_method, columns=None):
    """Scale numerical features"""
    df_processed = df.copy()
    
    if columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numerical_columns = columns
    
    # Remove target column from scaling if it exists
    target_col = st.session_state.get('target_column')
    if target_col and target_col in numerical_columns:
        numerical_columns.remove(target_col)
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
        df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
        st.session_state.preprocessing_steps.append(f"Standard scaled {len(numerical_columns)} numerical columns")
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
        st.session_state.preprocessing_steps.append(f"Min-Max scaled {len(numerical_columns)} numerical columns")
    
    return df_processed

def handle_outliers(df, method, columns=None):
    """Handle outliers in the dataset"""
    df_processed = df.copy()
    
    if columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numerical_columns = columns
    
    # Remove target column from outlier handling if it exists
    if st.session_state.target_column in numerical_columns:
        numerical_columns.remove(st.session_state.target_column)
    
    outliers_removed = 0
    
    for col in numerical_columns:
        if col in df_processed.columns:
            outliers = detect_outliers(df_processed, col, method='iqr')
            if len(outliers) > 0:
                if method == 'remove':
                    df_processed = df_processed.drop(outliers.index)
                    outliers_removed += len(outliers)
                elif method == 'cap':
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                    outliers_removed += len(outliers)
    
    if outliers_removed > 0:
        st.session_state.preprocessing_steps.append(f"Handled {outliers_removed} outliers using {method} method")
    
    return df_processed

def preprocessing_page():
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first in the Data Input section.")
        return
    
    # Show original data info
    st.markdown("### Original Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", st.session_state.data.shape[0])
    with col2:
        st.metric("Columns", st.session_state.data.shape[1])
    with col3:
        st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
    
    # Initialize processed data if not exists
    if st.session_state.processed_data is None:
        st.session_state.processed_data = st.session_state.data.copy()
    
    # Current Dataset Display
    st.markdown("### 📊 Current Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", st.session_state.processed_data.shape[0])
    with col2:
        st.metric("Columns", st.session_state.processed_data.shape[1])
    with col3:
        removed_cols = st.session_state.data.shape[1] - st.session_state.processed_data.shape[1]
        st.metric("Columns Removed", removed_cols)
    
    # Show current dataset preview
    st.markdown("### 📋 Current Dataset Preview")
    st.dataframe(st.session_state.processed_data.head(10))
    
    # Show current columns
    with st.expander("📋 Current Columns List", expanded=False):
        current_cols = st.session_state.processed_data.columns.tolist()
        st.write(f"Total columns: {len(current_cols)}")
        for i, col in enumerate(current_cols, 1):
            st.write(f"{i}. **{col}** ({st.session_state.processed_data[col].dtype})")
    
    # Reset button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Reset Data", help="Reset processed data to original", key="reset_data_top"):
            st.session_state.processed_data = st.session_state.data.copy()
            st.session_state.preprocessing_steps = []
            st.success("Processed data reset to original!")
            st.rerun()
    with col2:
        st.write("")
    
    # Unwanted Columns Detection and Removal
    st.markdown("### 🗑️ Remove Unwanted Columns")
    
    # Detect unwanted columns
    target_col = st.session_state.get('target_column', None)
    
    # Debug: Show detection process
    with st.expander("🔍 Column Detection Debug"):
        st.write("**Target Column:**", target_col)
        st.write("**Analyzing columns:**")
        
        for col in st.session_state.processed_data.columns:
            if col == target_col:
                st.write(f"🎯 **{col}**: SKIPPED (target column)")
                continue
                
            col_data = st.session_state.processed_data[col]
            reasons = []
            
            # Check ID patterns
            id_patterns = [
                r'id$', r'_id$', r'ID$', r'Id$', r'identifier',
                r'code$', r'Code$', r'number$', r'Number$',
                r'ssn$', r'SSN$', r'passport', r'employee',
                r'customer', r'client', r'user', r'account',
                r'candidate', r'student', r'patient', r'member',
                r'record', r'entry', r'transaction', r'order',
                r'product', r'item', r'entity', r'reference',
                r'^candidate_id', r'^user_id', r'^student_id'
            ]
            
            for pattern in id_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    reasons.append("ID-like column name")
                    st.write(f"🔍 **{col}**: ID pattern matched - {pattern}")
                    break
            
            # Check uniqueness
            unique_ratio = col_data.nunique() / len(col_data)
            if unique_ratio > 0.95:
                reasons.append(f"High uniqueness ({unique_ratio:.1%})")
                st.write(f"🔍 **{col}**: High uniqueness - {unique_ratio:.1%}")
            
            if not reasons:
                st.write(f"✅ **{col}**: OK")
            else:
                st.write(f"⚠️ **{col}**: {', '.join(reasons)}")
    
    unwanted_cols, unwanted_reasons = detect_unwanted_columns(st.session_state.processed_data, target_col)
    
    if unwanted_cols:
        st.warning(f"Found {len(unwanted_cols)} potentially unwanted columns:")
        
        # Show detected unwanted columns with reasons
        for col in unwanted_cols:
            reasons_text = ", ".join(unwanted_reasons[col])
            st.write(f"• **{col}**: {reasons_text}")
        
        # Option to select which columns to remove
        selected_for_removal = st.multiselect(
            "Select columns to remove:",
            options=unwanted_cols,
            default=unwanted_cols,
            help="Select the unwanted columns you want to remove from the dataset"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove Selected Columns", type="primary", key="remove_selected_columns"):
                if selected_for_removal:
                    # Store original shape for comparison
                    original_shape = st.session_state.processed_data.shape
                    
                    # Remove the selected columns
                    st.session_state.processed_data = remove_unwanted_columns(
                        st.session_state.processed_data, 
                        selected_for_removal
                    )
                    st.session_state.preprocessing_steps.append(f"Removed {len(selected_for_removal)} unwanted columns: {', '.join(selected_for_removal)}")
                    
                    # Show success message with details
                    new_shape = st.session_state.processed_data.shape
                    st.success(f"✅ Removed {len(selected_for_removal)} columns: {', '.join(selected_for_removal)}")
                    st.info(f"📊 Dataset shape changed from {original_shape} to {new_shape}")
                    
                    # Display updated dataset preview
                    st.markdown("### 📋 Updated Dataset Preview")
                    st.dataframe(st.session_state.processed_data.head(10))
                    
                    # Show remaining columns
                    st.markdown("### 📋 Remaining Columns")
                    remaining_cols = st.session_state.processed_data.columns.tolist()
                    st.write(f"Total columns remaining: {len(remaining_cols)}")
                    st.write(remaining_cols)
                else:
                    st.info("No columns selected for removal.")
        
        with col2:
            if st.button("🔍 Force Detect candidate_id", type="secondary", key="force_detect_candidate_id"):
                # Force remove candidate_id if it exists
                if 'candidate_id' in st.session_state.processed_data.columns:
                    # Store original shape for comparison
                    original_shape = st.session_state.processed_data.shape
                    
                    st.session_state.processed_data = remove_unwanted_columns(
                        st.session_state.processed_data, 
                        ['candidate_id']
                    )
                    st.session_state.preprocessing_steps.append("Force removed candidate_id column")
                    
                    # Show success message with details
                    new_shape = st.session_state.processed_data.shape
                    st.success("✅ Force removed candidate_id column!")
                    st.info(f"📊 Dataset shape changed from {original_shape} to {new_shape}")
                    
                    # Display updated dataset preview
                    st.markdown("### 📋 Updated Dataset Preview")
                    st.dataframe(st.session_state.processed_data.head(10))
                    
                    # Show remaining columns
                    st.markdown("### 📋 Remaining Columns")
                    remaining_cols = st.session_state.processed_data.columns.tolist()
                    st.write(f"Total columns remaining: {len(remaining_cols)}")
                    st.write(remaining_cols)
                else:
                    st.info("candidate_id column not found.")
    else:
        st.success("✅ No unwanted columns detected!")
    
    # Manual column removal
    with st.expander("🔧 Manual Column Removal"):
        st.write("Manually select any additional columns to remove:")
        
        all_columns = st.session_state.processed_data.columns.tolist()
        # Remove target column from options
        if target_col and target_col in all_columns:
            all_columns.remove(target_col)
        
        manual_removal = st.multiselect(
            "Select additional columns to remove:",
            options=all_columns,
            help="Select any other columns you want to remove manually"
        )
        
        if st.button("Remove Manually Selected Columns", key="remove_manual_columns"):
            if manual_removal:
                # Store original shape for comparison
                original_shape = st.session_state.processed_data.shape
                
                st.session_state.processed_data = remove_unwanted_columns(
                    st.session_state.processed_data, 
                    manual_removal
                )
                st.session_state.preprocessing_steps.append(f"Manually removed {len(manual_removal)} columns: {', '.join(manual_removal)}")
                
                # Show success message with details
                new_shape = st.session_state.processed_data.shape
                st.success(f"✅ Removed {len(manual_removal)} columns: {', '.join(manual_removal)}")
                st.info(f"📊 Dataset shape changed from {original_shape} to {new_shape}")
                
                # Display updated dataset preview
                st.markdown("### 📋 Updated Dataset Preview")
                st.dataframe(st.session_state.processed_data.head(10))
                
                # Show remaining columns
                st.markdown("### 📋 Remaining Columns")
                remaining_cols = st.session_state.processed_data.columns.tolist()
                st.write(f"Total columns remaining: {len(remaining_cols)}")
                st.write(remaining_cols)
            else:
                st.info("No columns selected for manual removal.")
    
    st.markdown("---")
    
    # Preprocessing options
    st.markdown("### Preprocessing Options")
    
    # Missing Values
    with st.expander("🔍 Handle Missing Values", expanded=True):
        missing_cols = st.session_state.processed_data.columns[st.session_state.processed_data.isnull().any()].tolist()
        
        if missing_cols:
            st.write(f"Columns with missing values: {', '.join(missing_cols)}")
            
            col1, col2 = st.columns(2)
            with col1:
                missing_strategy = st.selectbox(
                    "Select strategy:",
                    ["mean", "median", "mode", "drop"]
                )
            with col2:
                apply_missing = st.button("Apply Missing Value Handling", key="apply_missing_values")
            
            if apply_missing:
                st.session_state.processed_data = handle_missing_values(
                    st.session_state.processed_data, missing_strategy
                )
                st.success(f"Missing values handled using {missing_strategy} strategy")
                st.rerun()
        else:
            st.success("No missing values found in the dataset!")
    
    # Categorical Encoding
    with st.expander("🔤 Encode Categorical Variables"):
        categorical_cols = st.session_state.processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.write(f"Categorical columns: {', '.join(categorical_cols)}")
            
            col1, col2 = st.columns(2)
            with col1:
                encoding_method = st.selectbox(
                    "Select encoding method:",
                    ["label", "onehot"]
                )
            with col2:
                apply_encoding = st.button("Apply Categorical Encoding", key="apply_categorical_encoding")
            
            if apply_encoding:
                st.session_state.processed_data = encode_categorical(
                    st.session_state.processed_data, encoding_method
                )
                st.success(f"Categorical variables encoded using {encoding_method} encoding")
                st.rerun()
        else:
            st.success("No categorical columns found in the dataset!")
    
    # Feature Scaling
    with st.expander("📏 Feature Scaling"):
        numerical_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from scaling if it exists
        if st.session_state.target_column and st.session_state.target_column in numerical_cols:
            numerical_cols.remove(st.session_state.target_column)
        
        if numerical_cols:
            st.write(f"Numerical columns for scaling: {', '.join(numerical_cols)}")
            st.info(f"📊 Total numerical columns available: {len(numerical_cols)}")
            
            col1, col2 = st.columns(2)
            with col1:
                scaling_method = st.selectbox(
                    "Select scaling method:",
                    ["none", "standard", "minmax"]
                )
            with col2:
                apply_scaling = st.button("Apply Feature Scaling", key="apply_feature_scaling")
            
            if apply_scaling and scaling_method != "none":
                st.session_state.processed_data = scale_features(
                    st.session_state.processed_data, scaling_method
                )
                st.success(f"Features scaled using {scaling_method} scaling")
                st.rerun()
        else:
            st.success("No numerical columns found for scaling!")
    
    # Outlier Detection and Handling
    with st.expander("🎯 Outlier Detection & Handling"):
        numerical_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from outlier handling
        if st.session_state.target_column in numerical_cols:
            numerical_cols.remove(st.session_state.target_column)
        
        if numerical_cols:
            st.write(f"Checking for outliers in: {', '.join(numerical_cols)}")
            
            # Show outliers for each numerical column
            outlier_info = {}
            for col in numerical_cols:
                outliers = detect_outliers(st.session_state.processed_data, col)
                outlier_info[col] = len(outliers)
            
            outlier_df = pd.DataFrame(list(outlier_info.items()), columns=['Column', 'Outliers'])
            st.dataframe(outlier_df)
            
            col1, col2 = st.columns(2)
            with col1:
                outlier_method = st.selectbox(
                    "Select outlier handling method:",
                    ["none", "remove", "cap"]
                )
            with col2:
                apply_outliers = st.button("Apply Outlier Handling", key="apply_outlier_handling")
            
            if apply_outliers and outlier_method != "none":
                st.session_state.processed_data = handle_outliers(
                    st.session_state.processed_data, outlier_method
                )
                st.success(f"Outliers handled using {outlier_method} method")
                st.rerun()
        else:
            st.success("No numerical columns found for outlier detection!")
    
    # Show preprocessing steps
    if st.session_state.preprocessing_steps:
        st.markdown("### Applied Preprocessing Steps")
        for i, step in enumerate(st.session_state.preprocessing_steps, 1):
            st.write(f"{i}. {step}")
    
    # Show processed data info
    st.markdown("### Processed Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", st.session_state.processed_data.shape[0])
    with col2:
        st.metric("Columns", st.session_state.processed_data.shape[1])
    with col3:
        st.metric("Missing Values", st.session_state.processed_data.isnull().sum().sum())
    
    # Data preview
    st.markdown("### Processed Dataset Preview")
    st.dataframe(st.session_state.processed_data.head(10))
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Reset to Original Data", type="secondary", key="reset_to_original"):
            st.session_state.processed_data = st.session_state.data.copy()
            st.session_state.preprocessing_steps = []
            st.rerun()
    
    with col2:
        if st.button("Use Processed Data", type="primary", key="use_processed_data"):
            st.session_state.data = st.session_state.processed_data.copy()
            st.success("Processed data is now being used for further analysis!")
            st.rerun()
    
    with col3:
        if st.button("Download Processed Data", type="secondary", key="download_processed_data"):
            csv = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="processed_dataset.csv",
                mime="text/csv"
            )
    
    # Visual comparison
    if len(st.session_state.preprocessing_steps) > 0:
        st.markdown("### Data Distribution Comparison")
        
        numerical_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            selected_col = st.selectbox("Select column to visualize:", numerical_cols)
            
            if selected_col:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Original data
                if selected_col in st.session_state.data.columns:
                    axes[0].hist(st.session_state.data[selected_col].dropna(), bins=30, alpha=0.7)
                    axes[0].set_title(f'Original {selected_col}')
                    axes[0].set_xlabel(selected_col)
                    axes[0].set_ylabel('Frequency')
                
                # Processed data
                if selected_col in st.session_state.processed_data.columns:
                    axes[1].hist(st.session_state.processed_data[selected_col].dropna(), bins=30, alpha=0.7)
                    axes[1].set_title(f'Processed {selected_col}')
                    axes[1].set_xlabel(selected_col)
                    axes[1].set_ylabel('Frequency')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # Debug Information
    with st.expander("🔍 Debug Information"):
        st.write("**Original Data Shape:**", st.session_state.data.shape)
        st.write("**Processed Data Shape:**", st.session_state.processed_data.shape)
        st.write("**Original Columns:**", list(st.session_state.data.columns))
        st.write("**Processed Columns:**", list(st.session_state.processed_data.columns))
        st.write("**Target Column:**", st.session_state.get('target_column', 'Not set'))
        st.write("**Preprocessing Steps:**", st.session_state.preprocessing_steps)
        
        # Data integrity check
        st.write("**Data Integrity Check:**")
        st.write("- Processed data is original data copy:", st.session_state.processed_data is not st.session_state.data)
        st.write("- Processed data equals original data:", st.session_state.processed_data.equals(st.session_state.data))
        st.write("- Missing values in processed data:", st.session_state.processed_data.isnull().sum().sum())
        st.write("- Data types in processed data:", dict(st.session_state.processed_data.dtypes))
    
    # Quick Fix Section
    st.markdown("### 🔧 Quick Fixes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Processed Data", help="Reset processed data to original", key="reset_processed_data_qf"):
            st.session_state.processed_data = st.session_state.data.copy()
            st.session_state.preprocessing_steps = []
            st.success("Processed data reset to original!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Remove All IDs", help="Remove all ID-like columns", key="remove_all_ids"):
            id_columns = []
            for col in st.session_state.processed_data.columns:
                if any(pattern in col.lower() for pattern in ['id', 'candidate', 'user', 'customer', 'employee']):
                    if col != st.session_state.get('target_column'):
                        id_columns.append(col)
            
            if id_columns:
                st.session_state.processed_data = st.session_state.processed_data.drop(columns=id_columns)
                st.session_state.preprocessing_steps.append(f"Removed ID columns: {', '.join(id_columns)}")
                st.success(f"Removed ID columns: {', '.join(id_columns)}")
                st.rerun()
            else:
                st.info("No ID columns found to remove.")
    
    with col3:
        if st.button("🧹 Clean All Data", help="Apply all common preprocessing steps", key="clean_all_data"):
            try:
                df_clean = st.session_state.processed_data.copy()
                
                # Remove ID columns
                id_columns = []
                for col in df_clean.columns:
                    if any(pattern in col.lower() for pattern in ['id', 'candidate', 'user', 'customer', 'employee']):
                        if col != st.session_state.get('target_column'):
                            id_columns.append(col)
                
                if id_columns:
                    df_clean = df_clean.drop(columns=id_columns)
                
                # Handle missing values
                for col in df_clean.columns:
                    if df_clean[col].isnull().sum() > 0:
                        if df_clean[col].dtype in ['object', 'category']:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
                        else:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
                # Encode categorical
                cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in cat_cols:
                    if col != st.session_state.get('target_column'):
                        df_clean[col] = pd.factorize(df_clean[col])[0]
                
                st.session_state.processed_data = df_clean
                steps = ["Applied comprehensive cleaning"]
                if id_columns:
                    steps[0] += f" (removed IDs: {', '.join(id_columns)})"
                st.session_state.preprocessing_steps = steps
                st.success("Comprehensive cleaning applied!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during cleaning: {str(e)}")
    
    # Current Dataset Status
    st.markdown("---")
    st.markdown("### 📊 Current Dataset Status")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Rows", st.session_state.processed_data.shape[0])
    with col2:
        st.metric("Current Columns", st.session_state.processed_data.shape[1])
    with col3:
        st.metric("Columns Removed", st.session_state.data.shape[1] - st.session_state.processed_data.shape[1])
    with col4:
        st.metric("Missing Values", st.session_state.processed_data.isnull().sum().sum())
    
    # Show preprocessing steps
    if st.session_state.preprocessing_steps:
        st.markdown("### 📋 Applied Preprocessing Steps")
        for i, step in enumerate(st.session_state.preprocessing_steps, 1):
            st.write(f"{i}. {step}")
    
    # Show current columns
    st.markdown("### 📋 Current Columns")
    st.dataframe(pd.DataFrame({
        'Column Name': st.session_state.processed_data.columns,
        'Data Type': st.session_state.processed_data.dtypes.values,
        'Non-Null Count': st.session_state.processed_data.count().values
    }))
    
    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous: Data Input", type="secondary", key="nav_prev_data_input"):
            st.session_state.explicit_navigation = "📊 Data Input"
            st.rerun()
    
    with col2:
        if st.button("💾 Save Progress", type="primary", key="nav_save_progress"):
            st.success("✅ Preprocessing progress saved!")
            st.info("💡 Your processed data and steps have been saved to session.")
    
    with col3:
        if st.button("➡️ Next: EDA", type="primary", key="nav_next_eda"):
            if st.session_state.processed_data is not None:
                st.session_state.current_phase = "eda"
                st.success("✅ Preprocessing completed! Moving to Exploratory Data Analysis...")
                st.session_state.explicit_navigation = "📈 Exploratory Data Analysis"
                st.rerun()
            else:
                st.error("⚠️ Please complete preprocessing first!")
