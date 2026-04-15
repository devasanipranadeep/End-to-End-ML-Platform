import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def get_dataset_characteristics(df, target_column):
    """Analyze dataset characteristics for model recommendation"""
    characteristics = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,  # Exclude target
        'n_numeric_features': len(df.select_dtypes(include=[np.number]).columns) - 1 if df[target_column].dtype in [np.number] else len(df.select_dtypes(include=[np.number]).columns),
        'n_categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
        'has_missing_values': df.isnull().any().any(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'target_type': 'categorical' if df[target_column].dtype in ['object', 'category'] else 'numeric'
    }
    
    # Target specific characteristics
    target_data = df[target_column].dropna()
    if characteristics['target_type'] == 'categorical':
        characteristics['n_classes'] = target_data.nunique()
        characteristics['is_binary'] = target_data.nunique() == 2
        characteristics['class_balance'] = target_data.value_counts().min() / target_data.value_counts().max()
    else:
        characteristics['target_range'] = float(target_data.max() - target_data.min())
        characteristics['target_std'] = float(target_data.std())
        # Numeric target that is actually Classification (e.g. 0/1 labels)
        unique_count = target_data.nunique()
        unique_ratio = unique_count / len(target_data)
        if unique_count <= 20 or unique_ratio < 0.05:
            characteristics['n_classes'] = unique_count
            characteristics['is_binary'] = unique_count == 2
            vc = target_data.value_counts()
            characteristics['class_balance'] = float(vc.min() / vc.max())
    
    return characteristics

def recommend_classification_models(characteristics):
    """Recommend models for classification problems"""
    models = []
    
    # Logistic Regression
    models.append({
        'name': 'Logistic Regression',
        'suitability': 'High' if characteristics['is_binary'] else 'Medium',
        'pros': [
            'Fast training and prediction',
            'Interpretable coefficients',
            'Good baseline model',
            'Works well with linear relationships'
        ],
        'cons': [
            'Assumes linear relationship',
            'May not capture complex patterns',
            'Sensitive to outliers'
        ],
        'best_for': 'Binary classification, interpretable models, baseline comparisons',
        'complexity': 'Low',
        'data_requirements': 'Moderate sample size, linearly separable data'
    })
    
    # Decision Tree
    models.append({
        'name': 'Decision Tree',
        'suitability': 'High',
        'pros': [
            'Easy to interpret and visualize',
            'Handles both numerical and categorical data',
            'Non-parametric - no assumptions about data distribution',
            'Captures non-linear relationships'
        ],
        'cons': [
            'Prone to overfitting',
            'Unstable - small changes can create different trees',
            'May create biased trees if classes are imbalanced'
        ],
        'best_for': 'Quick insights, non-linear relationships, mixed data types',
        'complexity': 'Low',
        'data_requirements': 'Small to medium datasets, needs pruning'
    })
    
    # Random Forest
    models.append({
        'name': 'Random Forest',
        'suitability': 'Very High',
        'pros': [
            'High accuracy and robust performance',
            'Handles overfitting well',
            'Provides feature importance',
            'Works with missing values'
        ],
        'cons': [
            'Less interpretable (black box)',
            'Slower training and prediction',
            'Requires more memory'
        ],
        'best_for': 'High accuracy requirements, complex datasets, feature importance',
        'complexity': 'Medium',
        'data_requirements': 'Medium to large datasets, benefits from more features'
    })
    
    # Support Vector Machine (SVM)
    models.append({
        'name': 'Support Vector Machine (SVM)',
        'suitability': 'High' if characteristics['n_samples'] < 100000 else 'Medium',
        'pros': [
            'Effective in high dimensional spaces',
            'Works well with clear margin of separation',
            'Memory efficient',
            'Versatile with different kernels'
        ],
        'cons': [
            'Poor performance on large datasets',
            'Sensitive to parameter selection',
            'Doesn\'t work well with noisy data'
        ],
        'best_for': 'High-dimensional data, clear margins, smaller datasets',
        'complexity': 'Medium',
        'data_requirements': 'Clean data, proper scaling, smaller datasets'
    })
    
    # K-Nearest Neighbors (KNN)
    models.append({
        'name': 'K-Nearest Neighbors (KNN)',
        'suitability': 'Medium',
        'pros': [
            'Simple and intuitive',
            'No training phase',
            'Works well for multi-class problems',
            'Non-parametric'
        ],
        'cons': [
            'Slow prediction on large datasets',
            'Sensitive to irrelevant features',
            'Requires feature scaling',
            'Curse of dimensionality'
        ],
        'best_for': 'Multi-class problems, smaller datasets, when training time is critical',
        'complexity': 'Low',
        'data_requirements': 'Scaled features, smaller datasets, relevant features'
    })
    
    # XGBoost
    models.append({
        'name': 'XGBoost',
        'suitability': 'Very High',
        'pros': [
            'State-of-the-art performance',
            'Handles missing values automatically',
            'Built-in regularization',
            'Parallel processing'
        ],
        'cons': [
            'Complex to tune',
            'Can overfit if not tuned properly',
            'Less interpretable',
            'Longer training time'
        ],
        'best_for': 'Competitions, high accuracy requirements, structured data',
        'complexity': 'High',
        'data_requirements': 'Medium to large datasets, benefits from feature engineering'
    })
    
    return models

def recommend_regression_models(characteristics):
    """Recommend models for regression problems"""
    models = []
    
    # Linear Regression
    models.append({
        'name': 'Linear Regression',
        'suitability': 'High' if characteristics['n_features'] < characteristics['n_samples'] else 'Medium',
        'pros': [
            'Simple and interpretable',
            'Fast training and prediction',
            'Provides coefficient insights',
            'Good baseline model'
        ],
        'cons': [
            'Assumes linear relationship',
            'Sensitive to outliers',
            'Requires features to be independent',
            'May underfit complex data'
        ],
        'best_for': 'Linear relationships, interpretable models, baseline comparisons',
        'complexity': 'Low',
        'data_requirements': 'Linear relationships, no multicollinearity, normal residuals'
    })
    
    # Ridge Regression
    models.append({
        'name': 'Ridge Regression',
        'suitability': 'High',
        'pros': [
            'Handles multicollinearity well',
            'Reduces overfitting',
            'Stable coefficients',
            'Works well with many features'
        ],
        'cons': [
            'Less interpretable than linear regression',
            'Requires hyperparameter tuning',
            'Doesn\'t perform feature selection'
        ],
        'best_for': 'Multicollinearity, many features, preventing overfitting',
        'complexity': 'Low',
        'data_requirements': 'Correlated features, needs alpha tuning'
    })
    
    # Lasso Regression
    models.append({
        'name': 'Lasso Regression',
        'suitability': 'High',
        'pros': [
            'Performs automatic feature selection',
            'Creates sparse models',
            'Handles multicollinearity',
            'Interpretable results'
        ],
        'cons': [
            'Can be unstable with correlated features',
            'Requires hyperparameter tuning',
            'May select too few features'
        ],
        'best_for': 'Feature selection, sparse models, interpretable results',
        'complexity': 'Low',
        'data_requirements': 'Needs alpha tuning, works with correlated features'
    })
    
    # Decision Tree
    models.append({
        'name': 'Decision Tree',
        'suitability': 'High',
        'pros': [
            'Captures non-linear relationships',
            'Easy to interpret',
            'No assumptions about data',
            'Handles mixed data types'
        ],
        'cons': [
            'Prone to overfitting',
            'Unstable predictions',
            'May create biased trees'
        ],
        'best_for': 'Non-linear relationships, quick insights, mixed data types',
        'complexity': 'Low',
        'data_requirements': 'Needs pruning, smaller datasets'
    })
    
    # Random Forest
    models.append({
        'name': 'Random Forest',
        'suitability': 'Very High',
        'pros': [
            'High accuracy and robustness',
            'Handles non-linear relationships',
            'Provides feature importance',
            'Resistant to overfitting'
        ],
        'cons': [
            'Less interpretable',
            'Slower prediction',
            'Requires more memory'
        ],
        'best_for': 'High accuracy, complex relationships, feature importance',
        'complexity': 'Medium',
        'data_requirements': 'Medium to large datasets'
    })
    
    # Support Vector Regression (SVR)
    models.append({
        'name': 'Support Vector Regression (SVR)',
        'suitability': 'Medium',
        'pros': [
            'Effective in high dimensions',
            'Robust to outliers',
            'Versatile kernels',
            'Good generalization'
        ],
        'cons': [
            'Computationally intensive',
            'Requires feature scaling',
            'Sensitive to parameters',
            'Not suitable for large datasets'
        ],
        'best_for': 'High-dimensional data, non-linear relationships, smaller datasets',
        'complexity': 'Medium',
        'data_requirements': 'Scaled features, parameter tuning, smaller datasets'
    })
    
    return models

def rank_models(models, characteristics):
    """Rank models based on dataset characteristics"""
    ranked_models = []
    
    for model in models:
        score = 0
        
        # Base score from suitability
        suitability_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
        score += suitability_scores.get(model['suitability'], 2) * 25
        
        # Dataset size considerations
        if characteristics['n_samples'] < 1000:
            if model['complexity'] == 'Low':
                score += 15
            elif model['complexity'] == 'High':
                score -= 10
        elif characteristics['n_samples'] > 100000:
            if model['name'] in ['K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)', 'Support Vector Regression (SVR)']:
                score -= 15
            elif model['name'] in ['Random Forest', 'XGBoost']:
                score += 10
        
        # Feature count considerations
        if characteristics['n_features'] > 100:
            if 'feature selection' in model['best_for'].lower() or 'feature importance' in model['best_for'].lower():
                score += 10
            elif model['name'] == 'K-Nearest Neighbors (KNN)':
                score -= 15
        
        # Missing values consideration
        if characteristics['has_missing_values']:
            if 'missing values' in model['pros'][2].lower():
                score += 10
        
        # Categorical features consideration
        if characteristics['n_categorical_features'] > 0:
            if 'categorical' in model['pros'][1].lower():
                score += 10
        
        ranked_models.append({
            **model,
            'recommendation_score': score
        })
    
    # Sort by recommendation score
    ranked_models.sort(key=lambda x: x['recommendation_score'], reverse=True)
    
    return ranked_models

def model_recommendation_page():
    st.markdown('<h2 class="section-header">🤖 Model Recommendation System</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first in the Data Input section.")
        return
    
    if st.session_state.target_column is None:
        st.warning("Please select a target column in the Problem Type Detection section.")
        return
    
    if st.session_state.problem_type is None:
        st.warning("Please confirm the problem type in the Problem Type Detection section.")
        return
    
    df = st.session_state.data
    target_column = st.session_state.target_column
    problem_type = st.session_state.problem_type
    
    # Analyze dataset characteristics
    st.markdown("### 📊 Dataset Analysis")
    characteristics = get_dataset_characteristics(df, target_column)
    
    # Display dataset characteristics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", characteristics['n_samples'])
    with col2:
        st.metric("Features", characteristics['n_features'])
    with col3:
        st.metric("Numeric Features", characteristics['n_numeric_features'])
    with col4:
        st.metric("Categorical Features", characteristics['n_categorical_features'])
    
    # Additional characteristics
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Missing Values:** {'Yes' if characteristics['has_missing_values'] else 'No'}")
        st.write(f"**Missing Percentage:** {characteristics['missing_percentage']:.2f}%")
    with col2:
        if problem_type == "Classification":
            st.write(f"**Number of Classes:** {characteristics['n_classes']}")
            st.write(f"**Class Balance:** {characteristics['class_balance']:.2f}")
        else:
            st.write(f"**Target Range:** {characteristics['target_range']:.2f}")
            st.write(f"**Target Std:** {characteristics['target_std']:.2f}")
    
    # Get model recommendations
    st.markdown("### 🎯 Recommended Models")
    
    if problem_type == "Classification":
        models = recommend_classification_models(characteristics)
    else:
        models = recommend_regression_models(characteristics)
    
    # Rank models
    ranked_models = rank_models(models, characteristics)
    
    # Create model comparison table
    model_comparison_data = []
    for model in ranked_models:
        model_comparison_data.append({
            'Model': model['name'],
            'Suitability': model['suitability'],
            'Complexity': model['complexity'],
            'Recommendation Score': model['recommendation_score']
        })
    
    comparison_df = pd.DataFrame(model_comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization of recommendation scores
    # Convert to native Python types for Plotly
    model_names = [str(model['name']) for model in ranked_models]
    scores = [float(model['recommendation_score']) for model in ranked_models]
    
    fig = px.bar(
        x=model_names,
        y=scores,
        title='Model Recommendation Scores',
        labels={'x': 'Model', 'y': 'Recommendation Score'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed model information
    st.markdown("### 📋 Detailed Model Information")
    
    selected_model = st.selectbox(
        "Select a model to view detailed information:",
        [model['name'] for model in ranked_models]
    )
    
    if selected_model:
        model_info = next(model for model in ranked_models if model['name'] == selected_model)
        
        # Model overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Suitability", model_info['suitability'])
        with col2:
            st.metric("Complexity", model_info['complexity'])
        with col3:
            st.metric("Score", model_info['recommendation_score'])
        
        # Pros and Cons
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Pros")
            for pro in model_info['pros']:
                st.write(f"• {pro}")
        
        with col2:
            st.markdown("#### ❌ Cons")
            for con in model_info['cons']:
                st.write(f"• {con}")
        
        # Best for and requirements
        st.markdown("#### 🎯 Best For")
        st.write(model_info['best_for'])
        
        st.markdown("#### 📋 Data Requirements")
        st.write(model_info['data_requirements'])
    
    # Top 3 recommendations
    st.markdown("### 🏆 Top 3 Recommended Models")
    
    top_models = ranked_models[:3]
    for i, model in enumerate(top_models, 1):
        with st.expander(f"{i}. {model['name']} (Score: {model['recommendation_score']})", expanded=i==1):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Suitability:** {model['suitability']}")
                st.write(f"**Complexity:** {model['complexity']}")
                st.write(f"**Best For:** {model['best_for']}")
            
            with col2:
                st.write("**Key Advantages:**")
                for pro in model['pros'][:3]:
                    st.write(f"• {pro}")
    
    # ML Problem Type Specific Recommendations
    st.markdown("### 💡 ML Problem Type Recommendations")
    
    if problem_type == "Classification":
        st.info("🎯 **Classification Problem Detected**")
        st.write("• **Binary Classification:** Use Logistic Regression, Random Forest, or SVM for 2-class problems")
        st.write("• **Multi-class Classification:** Random Forest, XGBoost, or Neural Networks work best")
        st.write("• **Imbalanced Classes:** Consider class weights, SMOTE, or F1-score optimization")
        
        if characteristics.get('is_binary', False):
            st.success("✅ Binary classification - simpler models may perform well")
        else:
            st.info(f"ℹ️ Multi-class classification with {characteristics.get('n_classes', 'unknown')} classes detected")
            
    else:  # Regression
        st.info("🎯 **Regression Problem Detected**")
        st.write("• **Linear Relationships:** Start with Linear, Ridge, or Lasso Regression")
        st.write("• **Non-linear Relationships:** Random Forest, XGBoost, or SVR recommended")
        st.write("• **Feature Importance:** Tree-based models provide automatic feature importance")
        
        if characteristics.get('target_range', 0) > 1000:
            st.warning("⚠️ Wide target range - consider log transformation for better performance")
    
    # Dataset-specific ML recommendations
    st.markdown("### 📊 Dataset-Specific ML Recommendations")
    
    if characteristics['n_samples'] < 1000:
        st.warning("⚠️ **Small Dataset:** Use simpler models to avoid overfitting")
        st.write("• Recommended: Logistic/Linear Regression, Decision Tree")
        st.write("• Use cross-validation for reliable evaluation")
    elif characteristics['n_samples'] > 50000:
        st.success("✅ **Large Dataset:** Complex models will perform well")
        st.write("• Recommended: Random Forest, XGBoost, Neural Networks")
        st.write("• Consider feature selection for faster training")
    
    if characteristics['n_features'] > 50:
        st.info("🔧 **High-Dimensional Data:**")
        st.write("• Use regularization (Ridge/Lasso) or feature selection")
        st.write("• Tree-based models handle high dimensions well")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("**Step 1:** Start with the top recommended model")
    st.markdown("**Step 2:** Train the model with default parameters")
    st.markdown("**Step 3:** Evaluate performance using appropriate metrics")
    st.markdown("**Step 4:** If needed, try other top models for comparison")
    st.markdown("**Step 5:** Fine-tune the best performing model")
    
    # Select models for training
    st.markdown("### ✅ Select Models for Training")
    
    selected_models_for_training = st.multiselect(
        "Choose models to train in the next step:",
        [model['name'] for model in ranked_models],
        default=[ranked_models[0]['name']]  # Default to top recommendation
    )
    
    if selected_models_for_training:
        st.session_state.selected_models = selected_models_for_training
        st.success(f"Selected {len(selected_models_for_training)} models for training")
    else:
        st.info("Please select at least one model to proceed with training.")
    
    # Save recommendations to session state
    st.session_state.model_recommendations = ranked_models
    
    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous: Problem Detection", type="secondary"):
            st.session_state.explicit_navigation = "🎯 Problem Type Detection"
            st.rerun()
    
    with col2:
        if st.button("💾 Save Progress", type="primary"):
            st.success("✅ Model recommendations saved!")
            st.info("💡 Your selected models and recommendations have been saved to session.")
    
    with col3:
        if st.button("➡️ Next: Model Training", type="primary"):
            if 'selected_models' in st.session_state and st.session_state.selected_models:
                st.session_state.current_phase = "training"
                st.success("✅ Model recommendation completed! Moving to Model Training...")
                st.session_state.explicit_navigation = "🚀 Model Training"
                st.rerun()
            else:
                st.error("⚠️ Please select at least one model for training first!")
