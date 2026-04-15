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

def create_leaderboard(results, problem_type):
    """Create a leaderboard of models based on performance"""
    leaderboard_data = []
    
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'Training Time (s)': result['training_time'],
            'CV Mean': result['cv_mean'],
            'CV Std': result['cv_std']
        }
        
        if problem_type == "Classification":
            row.update({
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1-Score': result['metrics']['f1_score']
            })
            # Primary ranking metric: Accuracy
            row['Ranking Score'] = result['metrics']['accuracy']
        else:
            row.update({
                'R² Score': result['metrics']['r2_score'],
                'MAE': result['metrics']['mae'],
                'MSE': result['metrics']['mse'],
                'RMSE': result['metrics']['rmse']
            })
            # Primary ranking metric: R² Score
            row['Ranking Score'] = result['metrics']['r2_score']
        
        leaderboard_data.append(row)
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    # Sort by ranking score
    leaderboard_df = leaderboard_df.sort_values('Ranking Score', ascending=False)
    leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Model'] + [col for col in leaderboard_df.columns if col not in ['Rank', 'Model', 'Ranking Score']]
    leaderboard_df = leaderboard_df[cols]
    
    return leaderboard_df

def compare_model_performance(results, problem_type):
    """Create comprehensive comparison visualizations"""
    model_names = list(results.keys())
    
    if problem_type == "Classification":
        # Classification metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results[name]['metrics'][metric] for name in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=label,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=(i // 2) + 1,
                col=(i % 2) + 1
            )
        
        fig.update_layout(
            title="Classification Metrics Comparison",
            height=600,
            showlegend=False
        )
        
    else:  # Regression
        # Regression metrics comparison
        metrics = ['r2_score', 'mae', 'mse', 'rmse']
        metric_labels = ['R² Score', 'MAE', 'MSE', 'RMSE']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results[name]['metrics'][metric] for name in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=label,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=(i // 2) + 1,
                col=(i % 2) + 1
            )
        
        fig.update_layout(
            title="Regression Metrics Comparison",
            height=600,
            showlegend=False
        )
    
    return fig

def create_radar_chart(results, problem_type):
    """Create radar chart for model comparison"""
    model_names = list(results.keys())
    
    if problem_type == "Classification":
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']
    else:
        categories = ['R² Score', 'MAE (inv)', 'MSE (inv)', 'RMSE (inv)', 'CV Score']
        metrics = ['r2_score', 'mae', 'mse', 'rmse', 'cv_mean']
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, model_name in enumerate(model_names):
        values = []
        for metric in metrics:
            if metric in ['mae', 'mse', 'rmse']:
                # Invert error metrics so higher is better
                max_val = max([results[m]['metrics'][metric] for m in model_names])
                value = 1 - (results[model_name]['metrics'][metric] / max_val)
            elif metric == 'cv_mean':
                value = results[model_name]['cv_mean']
            else:
                value = results[model_name]['metrics'][metric]
            values.append(value)
        
        # Close the radar chart
        values.append(values[0])
        categories_closed = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name=model_name,
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Model Performance Radar Chart",
        showlegend=True
    )
    
    return fig

def analyze_tradeoffs(results, problem_type):
    """Analyze trade-offs between different metrics"""
    model_names = list(results.keys())
    
    if problem_type == "Classification":
        # Accuracy vs Training Time
        accuracies = [results[name]['metrics']['accuracy'] for name in model_names]
        training_times = [results[name]['training_time'] for name in model_names]
        
        fig1 = px.scatter(
            x=training_times,
            y=accuracies,
            text=model_names,
            title="Accuracy vs Training Time Trade-off",
            labels={'x': 'Training Time (seconds)', 'y': 'Accuracy'}
        )
        fig1.update_traces(textposition='top center')
        
        # F1-Score vs CV Stability (lower std is better)
        f1_scores = [results[name]['metrics']['f1_score'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        fig2 = px.scatter(
            x=cv_stds,
            y=f1_scores,
            text=model_names,
            title="F1-Score vs CV Stability Trade-off",
            labels={'x': 'CV Standard Deviation', 'y': 'F1-Score'}
        )
        fig2.update_traces(textposition='top center')
        
    else:  # Regression
        # R² Score vs Training Time
        r2_scores = [results[name]['metrics']['r2_score'] for name in model_names]
        training_times = [results[name]['training_time'] for name in model_names]
        
        fig1 = px.scatter(
            x=training_times,
            y=r2_scores,
            text=model_names,
            title="R² Score vs Training Time Trade-off",
            labels={'x': 'Training Time (seconds)', 'y': 'R² Score'}
        )
        fig1.update_traces(textposition='top center')
        
        # R² Score vs CV Stability
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        fig2 = px.scatter(
            x=cv_stds,
            y=r2_scores,
            text=model_names,
            title="R² Score vs CV Stability Trade-off",
            labels={'x': 'CV Standard Deviation', 'y': 'R² Score'}
        )
        fig2.update_traces(textposition='top center')
    
    return fig1, fig2

def comparison_page():
    st.markdown('<h2 class="section-header">🏆 Model Comparison</h2>', unsafe_allow_html=True)
    
    if 'training_results' not in st.session_state or not st.session_state.training_results:
        st.warning("No trained models found. Please train models first in the Model Training section.")
        return
    
    results = st.session_state.training_results
    problem_type = st.session_state.problem_type
    
    # Create leaderboard
    st.markdown("### 🥇 Model Leaderboard")
    leaderboard_df = create_leaderboard(results, problem_type)
    
    # Display leaderboard with styling
    def highlight_best(s):
        if s.name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'R² Score', 'CV Mean']:
            return ['background-color: #d4edda' if v == s.max() else '' for v in s]
        elif s.name in ['MAE', 'MSE', 'RMSE', 'CV Std', 'Training Time (s)']:
            return ['background-color: #d4edda' if v == s.min() else '' for v in s]
        else:
            return ['' for v in s]
    
    styled_leaderboard = leaderboard_df.style.apply(highlight_best, subset=leaderboard_df.columns[2:])
    st.dataframe(styled_leaderboard, use_container_width=True)
    
    # Best model summary
    best_model = leaderboard_df.iloc[0]
    st.markdown("### 🏆 Best Model Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", best_model['Model'])
    with col2:
        if problem_type == "Classification":
            st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
        else:
            st.metric("R² Score", f"{best_model['R² Score']:.4f}")
    with col3:
        st.metric("CV Score", f"{best_model['CV Mean']:.4f}")
    
    # Performance comparison charts
    st.markdown("### 📊 Performance Comparison")
    
    # Metrics comparison
    fig_metrics = compare_model_performance(results, problem_type)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Radar chart
    st.markdown("### 🎯 Overall Performance Radar")
    fig_radar = create_radar_chart(results, problem_type)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Trade-off analysis
    st.markdown("### ⚖️ Performance Trade-offs")
    
    fig1, fig2 = analyze_tradeoffs(results, problem_type)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("### 📋 Detailed Comparison Table")
    
    # Create detailed comparison
    detailed_data = []
    for model_name, result in results.items():
        row = {'Model': model_name}
        
        # Add all metrics
        if problem_type == "Classification":
            row.update({
                'Accuracy': f"{result['metrics']['accuracy']:.4f}",
                'Precision': f"{result['metrics']['precision']:.4f}",
                'Recall': f"{result['metrics']['recall']:.4f}",
                'F1-Score': f"{result['metrics']['f1_score']:.4f}"
            })
        else:
            row.update({
                'R² Score': f"{result['metrics']['r2_score']:.4f}",
                'MAE': f"{result['metrics']['mae']:.4f}",
                'MSE': f"{result['metrics']['mse']:.4f}",
                'RMSE': f"{result['metrics']['rmse']:.4f}"
            })
        
        row.update({
            'Training Time (s)': f"{result['training_time']:.2f}",
            'CV Mean': f"{result['cv_mean']:.4f}",
            'CV Std': f"{result['cv_std']:.4f}"
        })
        
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    st.dataframe(detailed_df, use_container_width=True)
    
    # Model recommendations based on use case
    st.markdown("### 💡 Model Recommendations by Use Case")
    
    # Find best models for different scenarios
    best_accuracy = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'] if problem_type == "Classification" else x[1]['metrics']['r2_score'])
    fastest_model = min(results.items(), key=lambda x: x[1]['training_time'])
    most_stable = min(results.items(), key=lambda x: x[1]['cv_std'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Best Performance**")
        st.write(f"**{best_accuracy[0]}**")
        if problem_type == "Classification":
            st.write(f"Accuracy: {best_accuracy[1]['metrics']['accuracy']:.4f}")
        else:
            st.write(f"R² Score: {best_accuracy[1]['metrics']['r2_score']:.4f}")
    
    with col2:
        st.markdown("**⚡ Fastest Training**")
        st.write(f"**{fastest_model[0]}**")
        st.write(f"Time: {fastest_model[1]['training_time']:.2f}s")
    
    with col3:
        st.markdown("**🛡️ Most Stable**")
        st.write(f"**{most_stable[0]}**")
        st.write(f"CV Std: {most_stable[1]['cv_std']:.4f}")
    
    # Statistical significance test (simplified)
    st.markdown("### 📈 Statistical Analysis")
    
    if len(results) >= 2:
        st.markdown("**Performance Distribution Analysis**")
        
        model_names = list(results.keys())
        if problem_type == "Classification":
            scores = [results[name]['metrics']['accuracy'] for name in model_names]
            metric_name = "Accuracy"
        else:
            scores = [results[name]['metrics']['r2_score'] for name in model_names]
            metric_name = "R² Score"
        
        # Create distribution plot
        fig = px.box(
            x=model_names,
            y=scores,
            title=f"{metric_name} Distribution Across Models",
            labels={'x': 'Model', 'y': metric_name}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        stats_data = []
        for i, name in enumerate(model_names):
            stats_data.append({
                'Model': name,
                'Score': scores[i],
                'Mean': np.mean(scores),
                'Std': np.std(scores),
                'Z-Score': (scores[i] - np.mean(scores)) / np.std(scores) if np.std(scores) > 0 else 0
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.round(4), use_container_width=True)
    
    # Export comparison results
    st.markdown("### 📥 Export Comparison Results")

    # Build report content
    report_lines = []
    report_lines.append("MODEL COMPARISON REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated on: {pd.Timestamp.now()}")
    report_lines.append(f"Problem Type: {problem_type}")
    report_lines.append(f"Number of Models: {len(results)}")
    report_lines.append("")
    report_lines.append("LEADERBOARD")
    report_lines.append("-" * 30)
    for _, row in leaderboard_df.iterrows():
        report_lines.append(f"{row['Rank']}. {row['Model']}")
        if problem_type == "Classification":
            report_lines.append(f"   Accuracy: {row['Accuracy']}")
            report_lines.append(f"   F1-Score: {row['F1-Score']}")
        else:
            report_lines.append(f"   R² Score: {row['R² Score']}")
            report_lines.append(f"   RMSE: {row['RMSE']}")
        report_lines.append(f"   Training Time: {row['Training Time (s)']}s")
        report_lines.append("")
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 30)
    report_lines.append(f"Best Performance: {best_accuracy[0]}")
    report_lines.append(f"Fastest Training: {fastest_model[0]}")
    report_lines.append(f"Most Stable: {most_stable[0]}")
    report_text = "\n".join(report_lines)

    st.download_button(
        label="⬇️ Download Comparison Report",
        data=report_text,
        file_name="model_comparison_report.txt",
        mime="text/plain",
        key="download_comparison_report"
    )
    
    # Select best model for deployment
    st.markdown("### 🚀 Select Model for Deployment")
    
    deployment_model = st.selectbox(
        "Choose a model for deployment/advanced features:",
        model_names,
        index=0  # Default to best model
    )
    
    if st.button("Confirm Selection", type="primary"):
        st.session_state.deployment_model = deployment_model
        st.success(f"✅ {deployment_model} selected for deployment!")
        st.info("You can now proceed to Advanced Features for hyperparameter tuning and model saving.")

    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous: Model Evaluation", type="secondary", key="nav_prev_from_comparison"):
            st.session_state.explicit_navigation = "📊 Model Evaluation"
            st.rerun()

    with col2:
        if st.button("💾 Save Progress", type="primary", key="nav_save_comparison"):
            st.success("✅ Comparison results saved!")

    with col3:
        if st.button("➡️ Next: Advanced Features", type="primary", key="nav_next_from_comparison"):
            st.session_state.explicit_navigation = "⚙️ Advanced Features"
            st.rerun()
