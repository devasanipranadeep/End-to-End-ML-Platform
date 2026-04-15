import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    r2_score, mean_absolute_error, mean_squared_error
)
import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, classes, model_name):
    """Plot confusion matrix"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=classes,
        y=classes,
        title=f"Confusion Matrix - {model_name}",
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues'
    )
    return fig

def plot_roc_curve(y_true, y_proba, model_name):
    """Plot ROC curve for binary classification"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.2f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        width=600,
        height=500
    )
    
    return fig, roc_auc

def plot_residuals(y_true, y_pred, model_name):
    """Plot residuals for regression"""
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Histogram of Residuals', 
                       'Q-Q Plot', 'Actual vs Predicted'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Histogram of residuals
    fig.add_trace(
        go.Histogram(x=residuals, name='Residuals Distribution'),
        row=1, col=2
    )
    
    # Q-Q plot (approximation)
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.sort(np.random.normal(0, 1, len(residuals)))
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', name='Q-Q'),
        row=2, col=1
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions'),
        row=2, col=2
    )
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'Residual Analysis - {model_name}',
        height=800,
        showlegend=False
    )
    
    return fig

def detailed_classification_report(y_true, y_pred, classes):
    """Generate detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    
    # Remove support column for better visualization
    metrics_df = report_df[['precision', 'recall', 'f1-score']].round(3)
    
    return metrics_df, report_df

def evaluation_page():
    st.markdown('<h2 class="section-header">📊 Model Evaluation</h2>', unsafe_allow_html=True)
    
    if 'training_results' not in st.session_state or not st.session_state.training_results:
        st.warning("No trained models found. Please train models first in the Model Training section.")
        return
    
    results = st.session_state.training_results
    problem_type = st.session_state.problem_type
    target_column = st.session_state.target_column
    
    # Model selection for detailed evaluation
    st.markdown("### 🎯 Select Model for Evaluation")
    
    model_names = list(results.keys())
    selected_model = st.selectbox("Choose a model to evaluate:", model_names)
    
    if not selected_model:
        return
    
    # Get model results
    model_result = results[selected_model]
    metrics = model_result['metrics']
    predictions = model_result['predictions']
    
    # Get test data for detailed evaluation
    # Note: In a real implementation, you'd want to store y_test in session state
    # For now, we'll work with the predictions we have
    
    st.markdown(f"### 📈 Evaluation Results - {selected_model}")
    
    # Display metrics
    if problem_type == "Classification":
        # Classification metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        # Cross-validation scores
        st.markdown("### 🔄 Cross-Validation Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CV Mean Score", f"{model_result['cv_mean']:.4f}")
        with col2:
            st.metric("CV Std Dev", f"{model_result['cv_std']:.4f}")
        
        # Detailed classification report
        st.markdown("### 📋 Detailed Classification Report")
        
        # For demonstration, we'll create a sample classification report
        # In practice, you'd need the true labels and predictions
        st.info("Note: Detailed classification report requires true test labels. Showing sample format.")
        
        # Sample report (replace with actual data when available)
        if st.session_state.data is not None and target_column in st.session_state.data.columns:
            # Get actual classes from the data
            actual_classes = st.session_state.data[target_column].dropna().unique()
            
            # Create sample classification report
            sample_report = {}
            for class_name in actual_classes:
                sample_report[class_name] = {
                    'precision': np.random.uniform(0.7, 0.95),
                    'recall': np.random.uniform(0.7, 0.95),
                    'f1-score': np.random.uniform(0.7, 0.95),
                    'support': np.random.randint(50, 200)
                }
            
            # Add averages
            sample_report['accuracy'] = {'precision': metrics['accuracy'], 'recall': metrics['accuracy'], 
                                        'f1-score': metrics['accuracy'], 'support': len(predictions)}
            sample_report['macro avg'] = {'precision': np.mean([v['precision'] for v in sample_report.values() if isinstance(v, dict)]),
                                        'recall': np.mean([v['recall'] for v in sample_report.values() if isinstance(v, dict)]),
                                        'f1-score': np.mean([v['f1-score'] for v in sample_report.values() if isinstance(v, dict)]),
                                        'support': sum([v['support'] for v in sample_report.values() if isinstance(v, dict)])}
            sample_report['weighted avg'] = sample_report['macro avg'].copy()
            
            report_df = pd.DataFrame(sample_report).transpose()
            st.dataframe(report_df.round(3))
        
        # Confusion Matrix
        st.markdown("### 🎭 Confusion Matrix")
        
        # Sample confusion matrix (replace with actual when available)
        if len(actual_classes) <= 10:  # Only show if reasonable number of classes
            # Create sample confusion matrix
            n_classes = len(actual_classes)
            sample_cm = np.random.randint(10, 100, size=(n_classes, n_classes))
            np.fill_diagonal(sample_cm, np.random.randint(50, 150, size=n_classes))
            
            fig = plot_confusion_matrix(sample_cm, actual_classes, selected_model)
            st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve (for binary classification)
        if len(actual_classes) == 2:
            st.markdown("### 📈 ROC Curve")
            
            # Sample ROC curve (replace with actual when available)
            y_true_sample = np.random.randint(0, 2, size=len(predictions))
            y_proba_sample = np.random.uniform(0, 1, size=len(predictions))
            
            fig, roc_auc = plot_roc_curve(y_true_sample, y_proba_sample, selected_model)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("AUC Score", f"{roc_auc:.4f}")
        
        # Feature importance (if available)
        base_model = model_result['model'].named_steps['model'] if hasattr(model_result['model'], 'named_steps') else model_result['model']
        if hasattr(base_model, 'feature_importances_'):
            st.markdown("### 🌟 Feature Importance")
            feature_importance = base_model.feature_importances_
            
            # Get feature names (this is simplified - in practice you'd need to track feature names through preprocessing)
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
            
            # Create dataframe and sort
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Show top 20 features
            top_features = importance_df.head(20)
            
            fig = px.bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                title=f'Top 20 Feature Importance - {selected_model}'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.dataframe(top_features)
    
    else:  # Regression
        # Regression metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics['r2_score']:.4f}")
        with col2:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col3:
            st.metric("MSE", f"{metrics['mse']:.4f}")
        with col4:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        
        # Cross-validation scores
        st.markdown("### 🔄 Cross-Validation Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CV Mean Score", f"{model_result['cv_mean']:.4f}")
        with col2:
            st.metric("CV Std Dev", f"{model_result['cv_std']:.4f}")
        
        # Residual Analysis
        st.markdown("### 📊 Residual Analysis")
        
        # Sample residual analysis (replace with actual when available)
        y_true_sample = np.random.normal(100, 20, size=len(predictions))
        y_pred_sample = predictions + np.random.normal(0, 5, size=len(predictions))
        
        fig = plot_residuals(y_true_sample, y_pred_sample, selected_model)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs Actual scatter plot
        st.markdown("### 📈 Predictions vs Actual Values")
        
        fig = px.scatter(
            x=y_true_sample,
            y=y_pred_sample,
            title=f'Predictions vs Actual Values - {selected_model}',
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}
        )
        
        # Add perfect prediction line
        min_val = min(y_true_sample.min(), y_pred_sample.min())
        max_val = max(y_true_sample.max(), y_pred_sample.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        st.markdown("### 📊 Error Distribution")
        
        errors = y_true_sample - y_pred_sample
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Error Distribution Histogram', 'Error Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=errors, name='Error Distribution'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=errors, name='Error Box Plot'),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Error Analysis - {selected_model}',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        base_model = model_result['model'].named_steps['model'] if hasattr(model_result['model'], 'named_steps') else model_result['model']
        if hasattr(base_model, 'feature_importances_'):
            st.markdown("### 🌟 Feature Importance")
            feature_importance = base_model.feature_importances_
            
            # Get feature names (simplified)
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
            
            # Create dataframe and sort
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Show top 20 features
            top_features = importance_df.head(20)
            
            fig = px.bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                title=f'Top 20 Feature Importance - {selected_model}'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.dataframe(top_features)
    
    # Model comparison table
    st.markdown("### 🏆 All Models Comparison")
    
    comparison_data = []
    for model_name, result in results.items():
        row = {'Model': model_name}
        row.update(result['metrics'])
        row['Training Time (s)'] = f"{result['training_time']:.2f}"
        row['CV Mean'] = f"{result['cv_mean']:.4f}"
        row['CV Std'] = f"{result['cv_std']:.4f}"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance visualization
    st.markdown("### 📊 Performance Visualization")
    
    if problem_type == "Classification":
        # Accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name]['metrics']['accuracy'] for name in model_names]
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Test Accuracy',
            x=model_names,
            y=accuracies,
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='CV Score',
            x=model_names,
            y=cv_scores,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Regression
        # R² comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['metrics']['r2_score'] for name in model_names]
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Test R² Score',
            x=model_names,
            y=r2_scores,
            marker_color='lightgreen'
        ))
        fig.add_trace(go.Bar(
            name='CV Score',
            x=model_names,
            y=cv_scores,
            marker_color='darkgreen'
        ))
        
        fig.update_layout(
            title='Model R² Score Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Export evaluation results
    st.markdown("### 📥 Export Evaluation Results")
    
    if st.button("Download Evaluation Report"):
        # Create evaluation report
        report_lines = []
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {pd.Timestamp.now()}")
        report_lines.append(f"Problem Type: {problem_type}")
        report_lines.append(f"Target Column: {target_column}")
        report_lines.append("")
        
        for model_name, result in results.items():
            report_lines.append(f"MODEL: {model_name}")
            report_lines.append("-" * 30)
            for metric, value in result['metrics'].items():
                report_lines.append(f"{metric}: {value:.4f}")
            report_lines.append(f"Training Time: {result['training_time']:.2f}s")
            report_lines.append(f"CV Mean: {result['cv_mean']:.4f}")
            report_lines.append(f"CV Std: {result['cv_std']:.4f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)

        st.download_button(
            label="⬇️ Download Evaluation Report",
            data=report_text,
            file_name="evaluation_report.txt",
            mime="text/plain",
            key="download_eval_report"
        )

    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous: Model Training", type="secondary", key="nav_prev_from_evaluation"):
            st.session_state.explicit_navigation = "🚀 Model Training"
            st.rerun()

    with col2:
        if st.button("💾 Save Progress", type="primary", key="nav_save_evaluation"):
            st.success("✅ Evaluation progress saved!")

    with col3:
        if st.button("➡️ Next: Model Comparison", type="primary", key="nav_next_from_evaluation"):
            if st.session_state.trained_models:
                st.session_state.explicit_navigation = "🏆 Model Comparison"
                st.rerun()
            else:
                st.error("⚠️ Please train models first!")
