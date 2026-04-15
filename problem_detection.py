import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def analyze_target_variable(df, target_column):
    """Analyze target variable to determine problem type"""
    if target_column not in df.columns:
        return None, "Target column not found in dataset"
    
    target_data = df[target_column].dropna()
    
    if len(target_data) == 0:
        return None, "Target column has no valid data"
    
    analysis = {
        'column_name': target_column,
        'data_type': str(target_data.dtype),
        'unique_values': target_data.nunique(),
        'total_values': len(target_data),
        'missing_values': df[target_column].isnull().sum(),
        'missing_percentage': (df[target_column].isnull().sum() / len(df)) * 100
    }
    
    # Determine if it's classification or regression
    # Check dtype.name to catch numpy bool_ ('bool'), pandas Categorical, and object types
    dtype_name = target_data.dtype.name
    if dtype_name in ('object', 'category', 'bool', 'boolean') or str(target_data.dtype) in ('object', 'bool'):
        # Categorical - Classification
        analysis['problem_type'] = 'Classification'
        analysis['classification_type'] = 'Binary' if target_data.nunique() == 2 else 'Multi-class'
        analysis['classes'] = target_data.value_counts().to_dict()
        analysis['class_distribution'] = (target_data.value_counts() / len(target_data) * 100).round(2).to_dict()
        
        # Check for class imbalance
        max_class_pct = max(analysis['class_distribution'].values())
        if max_class_pct > 70:
            analysis['class_balance'] = 'Imbalanced'
            analysis['imbalance_severity'] = 'Severe' if max_class_pct > 85 else 'Moderate'
        else:
            analysis['class_balance'] = 'Balanced'
            analysis['imbalance_severity'] = 'None'
    
    else:
        # Safely check if numeric (np.issubdtype can raise TypeError on some dtypes)
        try:
            is_numeric = np.issubdtype(target_data.dtype, np.number)
        except (TypeError, AttributeError):
            is_numeric = False

        if is_numeric:
            # Numerical - could be classification or regression
            unique_count = target_data.nunique()
            total_count = len(target_data)
            unique_ratio = unique_count / total_count

            # Heuristics to determine if it's actually classification
            if unique_count <= 20 or unique_ratio < 0.05:
                # Likely classification with numeric labels
                analysis['problem_type'] = 'Classification'
                analysis['classification_type'] = 'Binary' if target_data.nunique() == 2 else 'Multi-class'
                analysis['classes'] = target_data.value_counts().to_dict()
                analysis['class_distribution'] = (target_data.value_counts() / len(target_data) * 100).round(2).to_dict()
                analysis['note'] = 'Numeric target detected but appears to be classification based on low unique value count'
                # Check for class imbalance (same logic as categorical branch)
                max_class_pct = max(analysis['class_distribution'].values())
                if max_class_pct > 70:
                    analysis['class_balance'] = 'Imbalanced'
                    analysis['imbalance_severity'] = 'Severe' if max_class_pct > 85 else 'Moderate'
                else:
                    analysis['class_balance'] = 'Balanced'
                    analysis['imbalance_severity'] = 'None'
            else:
                # Regression
                analysis['problem_type'] = 'Regression'
                analysis['statistics'] = {
                    'mean': target_data.mean(),
                    'median': target_data.median(),
                    'std': target_data.std(),
                    'min': target_data.min(),
                    'max': target_data.max(),
                    'range': target_data.max() - target_data.min(),
                    'skewness': target_data.skew(),
                    'kurtosis': target_data.kurtosis()
                }

                # Check distribution characteristics
                skewness = analysis['statistics']['skewness']
                if abs(skewness) > 1:
                    analysis['distribution'] = 'Highly Skewed'
                elif abs(skewness) > 0.5:
                    analysis['distribution'] = 'Moderately Skewed'
                else:
                    analysis['distribution'] = 'Approximately Normal'
        else:
            # Fallback: treat any unrecognised dtype as classification
            analysis['problem_type'] = 'Classification'
            analysis['classification_type'] = 'Binary' if target_data.nunique() == 2 else 'Multi-class'
            analysis['classes'] = target_data.astype(str).value_counts().to_dict()
            analysis['class_distribution'] = (target_data.astype(str).value_counts() / len(target_data) * 100).round(2).to_dict()
            max_class_pct = max(analysis['class_distribution'].values())
            if max_class_pct > 70:
                analysis['class_balance'] = 'Imbalanced'
                analysis['imbalance_severity'] = 'Severe' if max_class_pct > 85 else 'Moderate'
            else:
                analysis['class_balance'] = 'Balanced'
                analysis['imbalance_severity'] = 'None'

    return analysis, None

def recommend_problem_type(analysis):
    """Provide recommendations based on analysis"""
    recommendations = []
    
    if analysis['problem_type'] == 'Classification':
        if analysis['classification_type'] == 'Binary':
            recommendations.append("✅ Binary classification problem detected")
            recommendations.append("📊 Consider using: Logistic Regression, Random Forest, SVM")
        else:
            recommendations.append(f"✅ Multi-class classification problem detected ({analysis['unique_values']} classes)")
            recommendations.append("📊 Consider using: Random Forest, Gradient Boosting, Neural Networks")
        
        if analysis['class_balance'] == 'Imbalanced':
            recommendations.append(f"⚠️ Class imbalance detected ({analysis['imbalance_severity']})")
            recommendations.append("💡 Consider: SMOTE, class weights, or different evaluation metrics")
    
    elif analysis['problem_type'] == 'Regression':
        recommendations.append("✅ Regression problem detected")
        recommendations.append("📊 Consider using: Linear Regression, Random Forest, Gradient Boosting")
        
        if analysis['distribution'] != 'Approximately Normal':
            recommendations.append(f"⚠️ Target distribution is {analysis['distribution'].lower()}")
            recommendations.append("💡 Consider: Log transformation or robust regression methods")
    
    # General recommendations
    if analysis['missing_percentage'] > 10:
        recommendations.append(f"⚠️ High missing values in target ({analysis['missing_percentage']:.1f}%)")
        recommendations.append("💡 Consider imputation or removing rows with missing target values")
    
    if analysis['total_values'] < 1000:
        recommendations.append("⚠️ Small dataset size detected")
        recommendations.append("💡 Consider cross-validation and simpler models to avoid overfitting")
    
    return recommendations

def problem_detection_page():
    st.markdown('<h2 class="section-header">🎯 Problem Type Detection</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first in the Data Input section.")
        return
    
    df = st.session_state.data
    
    # Target column selection
    st.markdown("### Target Column Selection")
    
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        # Show the already selected target column
        st.info(f"🎯 **Target Column Selected:** {st.session_state.target_column}")
        
        # Debug info
        if st.checkbox("Show Debug Info"):
            st.write(f"Current target in session: {st.session_state.target_column}")
            st.write(f"Available columns: {df.columns.tolist()}")
        
        # Force refresh button
        if st.button("🔄 Refresh Target Display"):
            st.rerun()
        
        # Option to change target column
        if st.checkbox("Change Target Column"):
            # Find current target index for proper default
            current_target = st.session_state.target_column
            available_cols = [col for col in df.columns if col != current_target]
            
            if available_cols:
                default_index = 0
            else:
                default_index = None
                
            target_column = st.selectbox(
                "Choose a different target variable:",
                options=available_cols,
                index=default_index,
                help="Select the column you want to predict"
            )
            
            # Add confirmation button to prevent accidental changes
            col1, col2 = st.columns(2)
            with col1:
                st.write("Selected:", target_column if target_column else "None")
            with col2:
                confirm_change = st.button("Confirm Change", type="primary")
            
            if confirm_change and target_column and target_column != st.session_state.target_column:
                st.session_state.target_column = target_column
                st.success(f"Target column updated to: {target_column}")
                st.rerun()
            elif confirm_change and target_column == st.session_state.target_column:
                st.info("Selected the same target column - no change needed.")
            elif confirm_change and not target_column:
                st.warning("Please select a target column before confirming.")
    else:
        # No target column selected yet
        st.warning("⚠️ No target column selected. Please select a target column in the Data Input section first.")
        
        # Get current target column or default to first column
        current_target = st.session_state.get('target_column', None)
        if current_target and current_target in df.columns:
            default_index = df.columns.tolist().index(current_target) + 1  # +1 because of "None" option
        else:
            default_index = 0
        
        target_column = st.selectbox(
            "Choose the target variable (what you want to predict):",
            options=["None"] + df.columns.tolist(),
            index=default_index,
            help="Select the column you want to predict"
        )
        
        if target_column != "None":
            st.session_state.target_column = target_column
            st.success(f"Target column set to: {target_column}")
            st.rerun()
        else:
            st.info("Please select a target column to analyze the problem type.")
            return
    
    # Use the current target column
    target_column = st.session_state.target_column
    
    # Analyze target variable
    analysis, error = analyze_target_variable(df, target_column)
    
    if error:
        st.error(error)
        return
    
    # Save problem type to session state
    st.session_state.problem_type = analysis['problem_type']
    
    # Display analysis results
    st.markdown("### 📊 Analysis Results")
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Type", analysis['data_type'])
    with col2:
        st.metric("Unique Values", analysis['unique_values'])
    with col3:
        st.metric("Total Values", analysis['total_values'])
    with col4:
        st.metric("Missing Values", f"{analysis['missing_values']} ({analysis['missing_percentage']:.1f}%)")
    
    # Problem type determination
    st.markdown("### 🎯 Problem Type Determination")
    
    problem_type = analysis['problem_type']
    classification_type = analysis.get('classification_type', '')
    
    # Display problem type with visual indicator
    col1, col2 = st.columns([1, 2])
    with col1:
        if problem_type == "Classification":
            if classification_type == "Binary":
                st.success("🎯 Binary Classification")
            else:
                st.success("🎯 Multi-class Classification")
        else:
            st.success("🎯 Regression")
    
    with col2:
        st.markdown(f"**Problem Type:** {problem_type}")
        if classification_type:
            st.markdown(f"**Classification Type:** {classification_type}")
        st.markdown(f"**Target Column:** {target_column}")
    
    # Detailed analysis based on problem type
    if problem_type == "Classification":
        st.markdown("### 📈 Classification Analysis")
        
        # Class distribution
        classes = analysis['classes']
        class_dist = analysis['class_distribution']
        
        # Create dataframe for display
        class_df = pd.DataFrame({
            'Class': list(classes.keys()),
            'Count': list(classes.values()),
            'Percentage': list(class_dist.values())
        })
        
        st.dataframe(class_df)
        
        # Visualize class distribution
        # Convert to native Python types for Plotly
        x_values = class_df['Class'].astype(str).tolist()
        y_values = class_df['Count'].astype(int).tolist()
        
        fig = px.bar(
            x=x_values,
            y=y_values,
            title=f'Class Distribution for {target_column}',
            labels={'x': 'Class', 'y': 'Count'}
        )
        
        # Add percentage annotations
        # Convert to native Python types for Plotly
        percentages = [float(p) for p in class_df['Percentage']]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='text',
            text=[f"{p}%" for p in percentages],
            textposition='top center',
            showlegend=False
        ))
        
        st.plotly_chart(fig, width='stretch')
        
        # Pie chart for class distribution
        # Convert to native Python types for Plotly
        values = class_df['Count'].astype(int).tolist()
        names = class_df['Class'].astype(str).tolist()
        
        fig_pie = px.pie(
            values=values,
            names=names,
            title=f'Class Distribution (Pie Chart) - {target_column}'
        )
        st.plotly_chart(fig_pie, width='stretch')
        
        # Class balance analysis
        st.markdown("#### ⚖️ Class Balance Analysis")
        balance_status = analysis['class_balance']
        imbalance_severity = analysis['imbalance_severity']
        
        if balance_status == "Balanced":
            st.success("✅ Classes are well balanced")
        else:
            st.warning(f"⚠️ Classes are {balance_status.lower()} ({imbalance_severity.lower()} imbalance)")
            
            # Recommendations for imbalanced data
            st.markdown("**Recommendations for imbalanced data:**")
            st.write("- Use stratified train-test split")
            st.write("- Consider class weights in models")
            st.write("- Try oversampling (SMOTE) or undersampling techniques")
            st.write("- Use appropriate evaluation metrics (F1-score, AUC-ROC)")
        
        # Target variable preview
        st.markdown("### 🔍 Target Variable Preview")
        st.dataframe(df[target_column].value_counts().to_frame('Count'))
    
    else:  # Regression
        st.markdown("### 📈 Regression Analysis")
        
        stats = analysis['statistics']
        
        # Statistics table
        stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
        stats_df['Value'] = stats_df['Value'].round(4)
        st.dataframe(stats_df)
        
        # Distribution visualization
        target_data = df[target_column].dropna()
        
        # Histogram
        # Convert to native Python types for Plotly
        target_values = target_data.astype(float).tolist()
        fig = px.histogram(
            x=target_values,
            nbins=30,
            title=f'Distribution of {target_column}',
            labels={'x': target_column, 'y': 'Frequency'}
        )
        
        # Add mean and median lines
        fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {stats['mean']:.2f}")
        fig.add_vline(x=stats['median'], line_dash="dash", line_color="green", 
                     annotation_text=f"Median: {stats['median']:.2f}")
        
        st.plotly_chart(fig, width='stretch')
        
        # Box plot
        # Convert to native Python types for Plotly
        target_values = target_data.astype(float).tolist()
        fig_box = px.box(
            y=target_values,
            title=f'Box Plot of {target_column}',
            labels={'y': target_column}
        )
        st.plotly_chart(fig_box, width='stretch')
        
        # Distribution analysis
        st.markdown("#### 📊 Distribution Analysis")
        distribution = analysis['distribution']
        
        if distribution == "Approximately Normal":
            st.success("✅ Target distribution is approximately normal")
        elif distribution == "Moderately Skewed":
            st.warning(f"⚠️ Target distribution is {distribution.lower()}")
            st.write("💡 Consider applying transformations (log, square root) to normalize the distribution")
        else:
            st.error(f"❌ Target distribution is {distribution.lower()}")
            st.write("💡 Strongly recommended to apply transformations or use robust regression methods")
        
        # Outlier detection
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
        outlier_percentage = (len(outliers) / len(target_data)) * 100
        
        st.markdown("#### 🎯 Outlier Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Outliers Detected", len(outliers))
        with col2:
            st.metric("Outlier Percentage", f"{outlier_percentage:.2f}%")
        
        if outlier_percentage > 5:
            st.warning("⚠️ High percentage of outliers detected")
            st.write("💡 Consider outlier handling techniques or robust regression methods")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    recommendations = recommend_problem_type(analysis)
    
    for rec in recommendations:
        st.info(rec)
    
    # Model suitability preview
    st.markdown("### 🤖 Suitable Models")
    
    if problem_type == "Classification":
        if classification_type == "Binary":
            models = [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors (KNN)",
                "XGBoost"
            ]
        else:
            models = [
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting (XGBoost)",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors (KNN)",
                "Neural Networks"
            ]
    else:  # Regression
        models = [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Regression (SVR)"
        ]
    
    # Display models in columns
    cols = st.columns(3)
    for i, model in enumerate(models):
        with cols[i % 3]:
            st.write(f"✅ {model}")
    
    # Confirm and proceed button
    st.markdown("---")
    st.markdown("### ✅ Confirm Problem Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Confirm Problem Type", type="primary"):
            st.session_state.problem_type = problem_type
            st.success(f"Problem type confirmed: {problem_type}")
            st.balloons()
    
    with col2:
        if st.button("Analyze Different Target"):
            st.info("Select a different target column from the dropdown above")
    
    # Show current session state
    if st.session_state.problem_type:
        st.markdown("### 📋 Current Session State")
        st.write(f"**Target Column:** {st.session_state.target_column}")
        st.write(f"**Problem Type:** {st.session_state.problem_type}")
        st.write("✅ Ready for model recommendation and training!")
    
    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous: EDA", type="secondary", key="nav_prev_from_problem_detection"):
            st.session_state.explicit_navigation = "📈 Exploratory Data Analysis"
            st.rerun()
    
    with col2:
        if st.button("💾 Save Progress", type="primary", key="nav_save_problem_detection"):
            st.success("✅ Problem detection progress saved!")
            st.info("💡 Your target column and problem type have been saved to session.")
    
    with col3:
        if st.button("➡️ Next: Model Recommendation", type="primary", key="nav_next_from_problem_detection"):
            if st.session_state.target_column and st.session_state.problem_type:
                st.session_state.current_phase = "model_recommendation"
                st.success("✅ Problem detection completed! Moving to Model Recommendation...")
                st.session_state.explicit_navigation = "🤖 Model Recommendation"
                st.rerun()
            else:
                st.error("⚠️ Please complete problem type detection first!")
