import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
try:
    import mysql.connector
except ImportError:
    mysql = None
try:
    import psycopg2
except ImportError:
    psycopg2 = None
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def validate_dataset(df):
    """Validate the uploaded dataset for ML readiness"""
    validation_results = {
        'is_valid': True,
        'issues': [],
        'recommendations': []
    }
    
    # Check if dataset is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Dataset is empty")
        return validation_results
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        missing_cols = missing_values[missing_values > 0].index.tolist()
        validation_results['issues'].append(f"Missing values found in: {', '.join(missing_cols)}")
        validation_results['recommendations'].append("Handle missing values in preprocessing before model training")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation_results['issues'].append(f"Found {duplicates} duplicate rows")
        validation_results['recommendations'].append("Remove duplicates in preprocessing to improve model performance")
    
    # Check dataset size for ML
    if len(df) < 100:
        validation_results['issues'].append("Very small dataset (<100 samples)")
        validation_results['recommendations'].append("Small datasets may lead to overfitting - consider data augmentation or simpler models")
    elif len(df) > 100000:
        validation_results['recommendations'].append("Large dataset detected - consider using efficient algorithms like Random Forest or XGBoost")
    
    # Check feature-to-sample ratio
    n_features = len(df.columns)
    if n_features > len(df) / 10:
        validation_results['recommendations'].append("High feature-to-sample ratio - consider feature selection or dimensionality reduction")
    
    # Check for ML-ready data structure
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(categorical_cols) > 0:
        validation_results['recommendations'].append(f"Found {len(categorical_cols)} categorical features - encoding required for ML models")
    
    if len(numeric_cols) < 2:
        validation_results['issues'].append("Insufficient numeric features for ML modeling")
        validation_results['recommendations'].append("Consider feature engineering or selecting appropriate target variable")
    
    return validation_results

def connect_to_database(db_type, host, port, database, username, password, table=None):
    """Connect to various database types"""
    try:
        if db_type == "MySQL":
            if mysql is None:
                return None, "MySQL connector is not installed. Please install mysql-connector-python."
            conn = mysql.connector.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
        elif db_type == "PostgreSQL":
            if psycopg2 is None:
                return None, "PostgreSQL connector is not installed. Please install psycopg2-binary."
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
        elif db_type == "SQLite":
            conn = sqlite3.connect(database)
        else:
            return None, "Unsupported database type"
        
        if table:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
        else:
            # Get list of tables
            if db_type == "SQLite":
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            else:
                tables = pd.read_sql("SHOW TABLES", conn)
            return conn, tables['name'].tolist() if 'name' in tables.columns else tables.iloc[:, 0].tolist()
        
        conn.close()
        return df, None
        
    except Exception as e:
        return None, str(e)

def data_input_page():
    st.markdown('<h2 class="section-header">📊 Data Input</h2>', unsafe_allow_html=True)
    
    # Data source selection
    data_source = st.radio("Select Data Source:", ["File Upload", "Database Connection"])
    
    if data_source == "File Upload":
        st.markdown("### Upload Dataset")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel files"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    # Try different encodings
                    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                    df = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if df is None:
                        st.error("Could not read the CSV file with any supported encoding")
                        return
                        
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                
                # Store in session state
                st.session_state.data = df
                
                # Show dataset info
                st.success("Dataset loaded successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                with col4:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Display dataset preview
                st.markdown("### Dataset Preview")
                st.dataframe(df.head(10))
                
                # Column information
                st.markdown("### Column Information")
                col_info = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': df.dtypes.astype(str).values,
                    'Non-Null Count': df.count().values,
                    'Unique Values': [df[col].nunique() for col in df.columns],
                    'Missing Values': df.isnull().sum().values
                })
                st.dataframe(col_info)
                
                # Validation
                st.markdown("### Dataset Validation")
                validation = validate_dataset(df)
                
                if validation['is_valid']:
                    st.success("✅ Dataset is valid for ML processing")
                else:
                    st.error("❌ Dataset has issues that need to be addressed")
                
                if validation['issues']:
                    st.markdown("**Issues Found:**")
                    for issue in validation['issues']:
                        st.write(f"⚠️ {issue}")
                
                if validation['recommendations']:
                    st.markdown("**Recommendations:**")
                    for rec in validation['recommendations']:
                        st.write(f"💡 {rec}")
                
                # Target column selection
                st.markdown("### Select Target Column")
                target_col = st.selectbox(
                    "Choose the target variable (what you want to predict):",
                    options=["None"] + df.columns.tolist(),
                    help="Select the column you want to predict"
                )
                
                if target_col != "None":
                    st.session_state.target_column = target_col
                    st.info(f"Target column set to: **{target_col}**")
                    
                    # Show target column statistics
                    target_stats = df[target_col].describe()
                    st.markdown("**Target Column Statistics:**")
                    st.dataframe(pd.DataFrame(target_stats).T)
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    else:  # Database Connection
        st.markdown("### Database Connection")
        
        # Database type selection
        db_type = st.selectbox("Database Type:", ["MySQL", "PostgreSQL", "SQLite"])
        
        # Connection parameters
        col1, col2 = st.columns(2)
        
        with col1:
            if db_type != "SQLite":
                host = st.text_input("Host:", value="localhost")
                port = st.number_input("Port:", value=3306 if db_type == "MySQL" else 5432)
                username = st.text_input("Username:")
                password = st.text_input("Password:", type="password")
            database = st.text_input("Database Name:")
        
        with col2:
            if db_type == "SQLite":
                st.info("For SQLite, only database name (file path) is required")
            table = st.text_input("Table Name (optional):", help="Leave empty to see all tables")
        
        # Connect button
        if st.button("Connect to Database"):
            if db_type == "SQLite":
                if not database:
                    st.error("Please enter database name")
                    return
                conn, result = connect_to_database(db_type, None, None, database, None, None, table)
            else:
                if not all([host, username, password, database]):
                    st.error("Please fill all connection parameters")
                    return
                conn, result = connect_to_database(db_type, host, port, database, username, password, table)
            
            if conn is None:
                if isinstance(result, list):
                    # Show available tables
                    st.success("Connected successfully!")
                    st.markdown("### Available Tables:")
                    for table_name in result:
                        if st.button(f"Load {table_name}", key=f"table_{table_name}"):
                            conn, df = connect_to_database(db_type, host, port, database, username, password, table_name)
                            if df is not None:
                                st.session_state.data = df
                                st.success(f"Table {table_name} loaded successfully!")
                                st.dataframe(df.head())
                                st.rerun()
                else:
                    st.error(f"Connection failed: {result}")
            else:
                # Data loaded successfully
                st.session_state.data = result
                st.success("Data loaded from database successfully!")
                st.dataframe(result.head())
                st.rerun()
    
    # Show current dataset info if available
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("### 📊 Current Dataset in Session")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Clear Dataset", type="secondary"):
                st.session_state.data = None
                st.session_state.processed_data = None
                st.session_state.target_column = None
                st.session_state.problem_type = None
                st.session_state.trained_models = {}
                st.session_state.preprocessing_steps = []
                st.rerun()
        
        with col2:
            if st.button("Download Current Dataset", type="primary"):
                csv = st.session_state.data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="dataset.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("➡️ Next: Preprocessing", type="primary", key="nav_next_preprocessing"):
                if st.session_state.data is not None:
                    st.session_state.current_phase = "preprocessing"
                    st.success("✅ Data input completed! Moving to preprocessing...")
                    # Use consistent navigation system
                    st.session_state.explicit_navigation = "🔧 Data Preprocessing"
                    st.rerun()
                else:
                    st.error("⚠️ Please upload a dataset first!")
