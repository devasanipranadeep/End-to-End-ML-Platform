import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime

# Fix compatibility issues for Streamlit Cloud
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Safe dataframe display function
def safe_display_dataframe(df):
    """Safely display dataframe with Arrow compatibility"""
    if df is None:
        return df
    
    # Create a copy to avoid modifying original
    df_safe = df.copy()
    
    try:
        # Convert all columns to Arrow-compatible types
        for col in df_safe.columns:
            dtype_str = str(df_safe[col].dtype)
            
            # Handle pandas StringDtype (the main culprit)
            if 'string' in dtype_str.lower():
                df_safe[col] = df_safe[col].astype(str)
            
            # Handle object dtype (mixed types)
            elif df_safe[col].dtype == 'object':
                # Check if it contains mixed types
                try:
                    # Try to convert to string first
                    df_safe[col] = df_safe[col].astype(str)
                except:
                    # If that fails, convert each element individually
                    df_safe[col] = df_safe[col].apply(lambda x: str(x) if pd.notna(x) else '')
            
            # Handle categorical data
            elif 'category' in dtype_str:
                df_safe[col] = df_safe[col].astype(str)
        
        return df_safe
    except Exception as e:
        # If conversion fails, return original dataframe
        # Streamlit will handle Arrow conversion automatically
        return df

# Apply safe conversion to all dataframes before display
def safe_st_dataframe(df, **kwargs):
    """Wrapper for st.dataframe with Arrow compatibility"""
    if df is not None and not df.empty:
        df_safe = safe_display_dataframe(df)
        return st.dataframe(df_safe, **kwargs)
    else:
        return st.dataframe(df, **kwargs)

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, r2_score, 
                           mean_absolute_error, mean_squared_error)
from scipy import stats

# Database libraries
import sqlite3

# Optional database connectors (commented out for cloud compatibility)
# Uncomment these if deploying on platforms with system dependencies or use pure Python alternatives

# PostgreSQL options:
# try:
#     import psycopg2  # Original (requires system libraries)
# except ImportError:
#     psycopg2 = None
# try:
#     import psycopg2cffi  # Pure Python alternative
# except ImportError:
#     psycopg2cffi = None

# MySQL options:
# try:
#     import mysql.connector  # Original (requires system libraries)
# except ImportError:
#     mysql = None
# try:
#     import pymysql  # Pure Python alternative
# except ImportError:
#     pymysql = None

# Other databases:
# try:
#     import pymongo  # MongoDB (pure Python)
# except ImportError:
#     pymongo = None
# try:
#     import redis  # Redis (pure Python)
# except ImportError:
#     redis = None

# Set page config
st.set_page_config(
    page_title="ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Data Input"

# Title
st.markdown('<h1 class="main-header">🤖 End-to-End ML Platform</h1>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")

_pages = [
    "📊 Data Input",
    "🔧 Data Preprocessing",
    "📈 Exploratory Data Analysis",
    "🎯 Problem Type Detection",
    "🤖 Model Recommendation",
    "🚀 Model Training",
    "📊 Model Evaluation",
    "🏆 Model Comparison",
    "⚙️ Advanced Features"
]

# Initialize current page if not set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Data Input"

# Handle explicit navigation from buttons (highest priority)
if 'explicit_navigation' in st.session_state:
    st.session_state.current_page = st.session_state.explicit_navigation
    del st.session_state.explicit_navigation

# Get current page
page = st.session_state.current_page

# Resolve current page index safely for sidebar
_page_index = _pages.index(page) if page in _pages else 0

# Sidebar selectbox - uses on_change to avoid overriding button navigation
def _on_sidebar_change():
    st.session_state.current_page = st.session_state.sidebar_page_selectbox

# Keep the selectbox display in sync with current_page (button navigation doesn't
# update the widget key automatically — only the index param is ignored when key exists)
st.session_state.sidebar_page_selectbox = page

st.sidebar.selectbox(
    "Select a page",
    _pages,
    index=_page_index,
    key="sidebar_page_selectbox",
    on_change=_on_sidebar_change
)

# Always use current_page as the source of truth
page = st.session_state.current_page

# Debug navigation state (helpful for troubleshooting)
if st.sidebar.checkbox("Show Navigation Debug", key="show_nav_debug"):
    st.sidebar.write("**Navigation Debug Info:**")
    st.sidebar.write(f"Current page: {page}")
    st.sidebar.write(f"Session keys: {list(st.session_state.keys())}")
    if 'explicit_navigation' in st.session_state:
        st.sidebar.write(f"Explicit navigation target: {st.session_state.explicit_navigation}")
    if 'navigation_target' in st.session_state:
        st.sidebar.write(f"Old navigation target: {st.session_state.navigation_target}")
    
    if st.sidebar.button("Clear All Navigation State"):
        keys_to_clear = ['explicit_navigation', 'navigation_target']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.success("All navigation state cleared!")
        st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔒 Disable Navigation (Stay on Current Page)", help="Temporarily disable all navigation to prevent unwanted redirects"):
        keys_to_clear = ['explicit_navigation', 'navigation_target']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.warning("Navigation disabled - use sidebar to navigate manually")
        st.rerun()


# Sidebar Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Current Status")

if st.session_state.data is not None:
    st.sidebar.success("✅ Dataset Loaded")
    st.sidebar.write(f"📊 **Shape:** {st.session_state.data.shape}")
    
    if st.session_state.target_column:
        st.sidebar.info(f"🎯 **Target Column:** {st.session_state.target_column}")
        
        if st.session_state.problem_type:
            st.sidebar.info(f"📈 **Problem Type:** {st.session_state.problem_type}")
    else:
        st.sidebar.warning("⚠️ No target column selected")
else:
    st.sidebar.warning("⚠️ No dataset loaded")

if st.session_state.trained_models:
    st.sidebar.success(f"✅ {len(st.session_state.trained_models)} models trained")

# Progress Indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Workflow Progress")

phases = [
    ("📊 Data Input", "data_input"),
    ("🔧 Preprocessing", "preprocessing"), 
    ("📈 EDA", "eda"),
    ("🎯 Problem Detection", "problem_detection"),
    ("🤖 Model Recommendation", "model_recommendation"),
    ("🚀 Model Training", "training"),
    ("📊 Model Evaluation", "evaluation"),
    ("🏆 Model Comparison", "comparison"),
    ("⚙️ Advanced Features", "advanced")
]

current_phase_index = 0
for i, (phase_name, phase_key) in enumerate(phases):
    # Check if this phase is completed
    is_completed = False
    if phase_key == "data_input" and st.session_state.data is not None:
        is_completed = True
    elif phase_key == "preprocessing" and st.session_state.processed_data is not None and st.session_state.preprocessing_steps:
        is_completed = True
    elif phase_key == "eda" and st.session_state.data is not None:
        is_completed = True
    elif phase_key == "problem_detection" and st.session_state.problem_type is not None:
        is_completed = True
    elif phase_key == "model_recommendation" and 'selected_models' in st.session_state:
        is_completed = True
    elif phase_key == "training" and st.session_state.trained_models:
        is_completed = True
    
    # Check if this is the current phase
    is_current = page == phase_name
    
    if is_completed:
        st.sidebar.markdown(f"✅ {phase_name}")
    elif is_current:
        st.sidebar.markdown(f"🔄 {phase_name}")
        current_phase_index = i
    else:
        st.sidebar.markdown(f"⏳ {phase_name}")

# Progress bar
progress = (current_phase_index + 1) / len(phases)
st.sidebar.progress(progress)
st.sidebar.write(f"Progress: {progress:.1%}")

# Navigation helper function - can be imported by other modules
def navigate_to_page(page_name):
    """Reliably navigate to a specific page - works on Streamlit Cloud"""
    if page_name in _pages:  # Validate page name
        st.session_state.current_page = page_name
        st.session_state.explicit_navigation = page_name
        st.rerun()
    else:
        st.error(f"Invalid page: {page_name}")

# Make navigation function available globally
def safe_navigate(page_name):
    """Safe navigation that works across all Streamlit deployments"""
    if page_name in _pages:  # Validate page name
        st.session_state.current_page = page_name
        st.session_state.explicit_navigation = page_name
        st.rerun()
    else:
        st.error(f"Invalid page: {page_name}")

# Import modules
from data_input import data_input_page
from preprocessing import preprocessing_page
from eda import eda_page
from problem_detection import problem_detection_page
from model_recommendation import model_recommendation_page
from training import training_page
from evaluation import evaluation_page
from comparison import comparison_page
from advanced import advanced_page

# Page routing
if page == "📊 Data Input":
    data_input_page()
elif page == "🔧 Data Preprocessing":
    preprocessing_page()
elif page == "📈 Exploratory Data Analysis":
    eda_page()
elif page == "🎯 Problem Type Detection":
    problem_detection_page()
elif page == "🤖 Model Recommendation":
    model_recommendation_page()
elif page == "🚀 Model Training":
    training_page()
elif page == "📊 Model Evaluation":
    evaluation_page()
elif page == "🏆 Model Comparison":
    comparison_page()
elif page == "⚙️ Advanced Features":
    advanced_page()

# Footer
st.markdown("---")
st.markdown("<center><p>Built with ❤️ using Streamlit | ML Platform © 2026</p></center>", unsafe_allow_html=True)
