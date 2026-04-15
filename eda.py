import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def generate_data_summary(df):
    """Generate comprehensive data summary"""
    summary = {}
    
    # Basic info
    summary['shape'] = df.shape
    summary['memory_usage'] = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # Data types — convert dtype keys to plain strings so Plotly can JSON-serialize them
    summary['dtypes'] = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}
    
    # Missing values
    missing_data = df.isnull().sum()
    summary['missing_values'] = missing_data[missing_data > 0].to_dict()
    summary['missing_percentage'] = (missing_data / len(df) * 100).round(2).to_dict()
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().round(2)
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary['categorical_stats'] = {}
        for col in categorical_cols:
            summary['categorical_stats'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0
            }
    
    return summary

def create_correlation_matrix(df):
    """Create correlation matrix for numerical columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        return correlation_matrix
    else:
        return None

def detect_feature_importance_hints(df, target_col=None):
    """Detect ML-specific hints about feature importance"""
    hints = []
    
    if target_col and target_col in df.columns:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
            
            # Top predictive features for ML
            top_features = correlations[1:6]  # Exclude target itself
            if len(top_features) > 0:
                hints.append(f"Strong predictive features for ML: {', '.join(top_features.head(3).index.tolist())}")
            
            # Features that may need removal for ML
            low_corr_features = correlations[correlations < 0.1].index.tolist()
            if len(low_corr_features) > 0:
                hints.append(f"Consider feature selection - low predictive features: {', '.join(low_corr_features[:3])}")
    
    # Categorical features that need encoding for ML
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5:
            hints.append(f"High cardinality in {col} - consider target encoding or feature grouping for ML")
    
    return hints

def eda_page():
    st.markdown('<h2 class="section-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first in the Data Input section.")
        return
    
    df = st.session_state.data
    
    # Show current target column if set
    if st.session_state.target_column:
        st.info(f"🎯 **Current Target Column:** {st.session_state.target_column}")
    else:
        st.warning("⚠️ No target column selected. Please select a target column in Data Input or Problem Detection section.")
    
    # Generate comprehensive summary
    st.markdown("### 📊 Dataset Summary")
    summary = generate_data_summary(df)
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", summary['shape'][0])
    with col2:
        st.metric("Columns", summary['shape'][1])
    with col3:
        st.metric("Memory Usage", f"{summary['memory_usage']:.2f} MB")
    with col4:
        st.metric("Missing Values", len(summary['missing_values']))
    
    # Data types distribution
    st.markdown("### 📋 Data Types Distribution")
    dtype_df = pd.DataFrame(list(summary['dtypes'].items()), columns=['Data Type', 'Count'])
    # Convert to native Python types for Plotly
    dtype_df['Count'] = dtype_df['Count'].astype(int)
    fig = px.pie(dtype_df, values='Count', names='Data Type', title='Data Types Distribution', height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values analysis
    if summary['missing_values']:
        st.markdown("### 🔍 Missing Values Analysis")
        
        missing_df = pd.DataFrame({
            'Column': list(summary['missing_values'].keys()),
            'Missing Count': list(summary['missing_values'].values()),
            'Missing Percentage': [summary['missing_percentage'][col] for col in summary['missing_values'].keys()]
        })
        # Convert to native Python types for Plotly
        missing_df['Missing Count'] = missing_df['Missing Count'].astype(int)
        missing_df['Missing Percentage'] = missing_df['Missing Percentage'].astype(float)
        
        fig = px.bar(missing_df, x='Column', y='Missing Percentage', 
                    title='Missing Values Percentage by Column',
                    labels={'Missing Percentage': 'Missing %'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(missing_df)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Numerical features analysis
    if 'numeric_stats' in summary:
        st.markdown("### 🔢 Numerical Features Statistics")
        st.dataframe(summary['numeric_stats'])
        
        # Distribution plots for numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.markdown("### 📈 Feature Distributions")
            
            # Select columns to visualize
            selected_cols = st.multiselect(
                "Select columns to visualize:",
                numeric_cols,
                default=numeric_cols
            )
            
            if selected_cols:
                # Create subplots
                n_cols = min(3, len(selected_cols))
                n_rows = (len(selected_cols) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=selected_cols,
                    specs=[[{"secondary_y": False}] * n_cols] * n_rows
                )
                
                for i, col in enumerate(selected_cols):
                    row = (i // n_cols) + 1
                    col_idx = (i % n_cols) + 1
                    
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(x=df[col].dropna(), name=col, nbinsx=30),
                        row=row, col=col_idx
                    )
                
                fig.update_layout(
                    height=280 * n_rows,
                    title_text="Feature Distributions",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Categorical features analysis
    if 'categorical_stats' in summary:
        st.markdown("### 📝 Categorical Features Statistics")
        
        cat_stats_data = []
        for col, stats in summary['categorical_stats'].items():
            cat_stats_data.append({
                'Column': col,
                'Unique Values': stats['unique_values'],
                'Most Frequent': stats['most_frequent'],
                'Frequency': stats['frequency']
            })
        
        cat_stats_df = pd.DataFrame(cat_stats_data)
        st.dataframe(cat_stats_df)
        
        # Categorical feature plots
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) > 0:
            st.markdown("### 📊 Categorical Feature Distributions")
            
            selected_cat_col = st.selectbox("Select categorical column to visualize:", categorical_cols)
            
            if selected_cat_col:
                # Value counts plot
                value_counts = df[selected_cat_col].value_counts().head(10)
                
                # Convert to native Python types for Plotly
                x_values = value_counts.index.astype(str).tolist()  # Convert to strings
                y_values = value_counts.values.astype(int).tolist()  # Convert to int
                
                fig = px.bar(
                    x=x_values,
                    y=y_values,
                    title=f'Distribution of {selected_cat_col}',
                    labels={'x': selected_cat_col, 'y': 'Count'},
                    height=350
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    correlation_matrix = create_correlation_matrix(df)
    
    if correlation_matrix is not None:
        # Heatmap
        # Convert correlation matrix to float for Plotly
        corr_matrix_float = correlation_matrix.astype(float)
        fig = px.imshow(
            corr_matrix_float,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        st.markdown("#### Strong Correlations (|r| > 0.7)")
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': round(corr_value, 3)
                    })
        
        if strong_correlations:
            strong_corr_df = pd.DataFrame(strong_correlations)
            st.dataframe(strong_corr_df)
        else:
            st.info("No strong correlations (|r| > 0.7) found.")
    else:
        st.info("Not enough numerical columns for correlation analysis.")
    
    # Pair plots (sample for performance)
    if st.checkbox("Show Pair Plots (may be slow for large datasets)"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Sample data if too large
            plot_df = df[numeric_cols].sample(min(1000, len(df))) if len(df) > 1000 else df[numeric_cols]
            
            # Create pair plot
            fig = px.scatter_matrix(
                plot_df,
                dimensions=numeric_cols[:min(5, len(numeric_cols))],  # Limit to 5 features
                title="Pair Plot of Numerical Features"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Target variable analysis (if specified)
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        st.markdown(f"### 🎯 Target Variable Analysis: {st.session_state.target_column}")
        
        target_col = st.session_state.target_column
        target_data = df[target_col]
        
        # Target variable statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Values", target_data.nunique())
        with col2:
            st.metric("Missing Values", target_data.isnull().sum())
        
        # Target distribution
        if target_data.dtype in ['object', 'category']:
            # Categorical target
            value_counts = target_data.value_counts()
            
            # Convert to native Python types for Plotly
            values = value_counts.values.astype(int).tolist()
            names = value_counts.index.astype(str).tolist()
            
            fig = px.pie(
                values=values,
                names=names,
                title=f'Distribution of {target_col}',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(target_data) * 100).round(2)
            }))
        else:
            # Numerical target
            # Convert to native Python types for Plotly
            target_values = target_data.dropna().astype(float).tolist()
            fig = px.histogram(
                x=target_values,
                nbins=30,
                title=f'Distribution of {target_col}',
                labels={'x': target_col, 'y': 'Frequency'},
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            stats_df = pd.DataFrame(target_data.describe()).round(2)
            st.dataframe(stats_df.T)
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    insights = []
    
    # Dataset characteristics
    if summary['shape'][0] > 10000:
        insights.append("📊 Large dataset detected (>10,000 rows)")
    elif summary['shape'][0] < 1000:
        insights.append("📊 Small dataset detected (<1,000 rows) - consider data augmentation")
    
    # Missing data insights
    missing_percentage = (sum(summary['missing_values'].values()) / (summary['shape'][0] * summary['shape'][1])) * 100
    if missing_percentage > 20:
        insights.append("⚠️ High percentage of missing data detected")
    elif missing_percentage > 0:
        insights.append("ℹ️ Some missing data present - consider imputation")
    
    # Feature insights
    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
    categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
    
    if numeric_count > categorical_count:
        insights.append(f"🔢 Dataset is predominantly numerical ({numeric_count} numerical, {categorical_count} categorical)")
    else:
        insights.append(f"📝 Dataset is predominantly categorical ({categorical_count} categorical, {numeric_count} numerical)")
    
    # Target variable insights
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        target_data = df[st.session_state.target_column]
        if target_data.dtype in ['object', 'category']:
            if target_data.nunique() == 2:
                insights.append("🎯 Binary classification problem detected")
            else:
                insights.append(f"🎯 Multi-class classification problem detected ({target_data.nunique()} classes)")
        else:
            insights.append("🎯 Regression problem detected")
    
    # Correlation insights
    if correlation_matrix is not None and strong_correlations:
        insights.append(f"🔗 Found {len(strong_correlations)} strong feature correlations")
    
    # Display insights
    for insight in insights:
        st.info(insight)
    
    # ML-specific feature insights
    auto_hints = detect_feature_importance_hints(df, st.session_state.target_column)
    if auto_hints:
        st.markdown("### 🤖 ML Feature Insights")
        for hint in auto_hints:
            st.write(f"💡 {hint}")
    
    # Download EDA report
    st.markdown("### 📥 Download EDA Report")
    
    if st.button("Generate EDA Summary Report"):
        # Create a summary report
        report_lines = []
        report_lines.append("EXPLORATORY DATA ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {pd.Timestamp.now()}")
        report_lines.append("")
        
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 20)
        report_lines.append(f"Shape: {summary['shape'][0]} rows, {summary['shape'][1]} columns")
        report_lines.append(f"Memory Usage: {summary['memory_usage']:.2f} MB")
        report_lines.append(f"Missing Values: {len(summary['missing_values'])}")
        report_lines.append("")
        
        if summary['missing_values']:
            report_lines.append("MISSING VALUES")
            report_lines.append("-" * 20)
            for col, count in summary['missing_values'].items():
                pct = summary['missing_percentage'][col]
                report_lines.append(f"{col}: {count} ({pct}%)")
            report_lines.append("")
        
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-" * 20)
        for insight in insights:
            report_lines.append(insight.replace("📊 ", "").replace("⚠️ ", "").replace("ℹ️ ", "").replace("🔢 ", "").replace("📝 ", "").replace("🎯 ", "").replace("🔗 ", ""))
        
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="Download EDA Report",
            data=report_text,
            file_name="eda_report.txt",
            mime="text/plain"
        )
    
    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous: Preprocessing", type="secondary"):
            st.session_state.explicit_navigation = "🔧 Data Preprocessing"
            st.rerun()
    
    with col2:
        if st.button("💾 Save Progress", type="primary"):
            st.success("✅ EDA progress saved!")
            st.info("💡 Your exploratory analysis insights have been saved to session.")
    
    with col3:
        if st.button("➡️ Next: Problem Detection", type="primary"):
            if st.session_state.data is not None:
                st.session_state.current_phase = "problem_detection"
                st.success("✅ EDA completed! Moving to Problem Type Detection...")
                st.session_state.explicit_navigation = "🎯 Problem Type Detection"
                st.rerun()
            else:
                st.error("⚠️ Please complete EDA first!")
