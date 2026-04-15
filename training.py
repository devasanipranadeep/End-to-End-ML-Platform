import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

def get_model(model_name, problem_type):
    """Get model instance based on name and problem type"""
    
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Support Vector Machine (SVM)": SVC(random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(random_state=42, eval_metric='logloss')
    
    else:  # Regression models with better parameters
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),  # Added regularization
            "Lasso Regression": Lasso(alpha=0.1, random_state=42, max_iter=2000),  # Better alpha
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=5),  # Prevent overfitting
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=15, min_samples_split=5),  # Better params
            "Support Vector Regression (SVR)": SVR(kernel='rbf', C=1.0, epsilon=0.1),  # Better kernel
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1)  # Better params
    
    return models.get(model_name)

def prepare_data(df, target_column, problem_type):
    """Prepare data for training"""
    # Remove rows with missing target values
    df_clean = df.dropna(subset=[target_column]).copy()
    
    # Separate features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Data quality checks
    st.write("📊 **Data Quality Check:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df_clean))
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        missing_pct = (df_clean.isnull().sum().sum() / (len(df_clean) * X.shape[1])) * 100
        st.metric("Missing Data %", f"{missing_pct:.1f}%")
    
    # Target variable analysis
    st.write("🎯 **Target Variable Analysis:**")
    col1, col2 = st.columns(2)
    with col1:
        if not pd.api.types.is_numeric_dtype(y):
            st.metric("Target Type", "Categorical")
            st.metric("Unique Values", y.nunique())
        else:
            st.metric("Target Type", "Numerical")
            st.metric("Min Value", f"{y.min():.2f}")
            st.metric("Max Value", f"{y.max():.2f}")
    with col2:
        if not pd.api.types.is_numeric_dtype(y):
            st.warning("⚠️ Categorical target detected! Consider classification models.")
        else:
            # Check if target has outliers or is skewed
            skewness = y.skew()
            st.metric("Skewness", f"{skewness:.3f}")
            if abs(skewness) > 1:
                st.warning("⚠️ Target variable is highly skewed!")
    
    # Identify column types
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    st.write("🔧 **Feature Analysis:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Categorical Features", len(categorical_columns))
    with col2:
        st.metric("Numerical Features", len(numerical_columns))
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_columns)
        ],
        remainder='passthrough'
    )
    
    # Encode target if it's categorical
    if problem_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        return X, y_encoded, preprocessor, label_encoder, categorical_columns, numerical_columns
    else:
        return X, y, preprocessor, None, categorical_columns, numerical_columns

def train_model_with_progress(model, X_train, y_train, model_name):
    """Train model with progress indication"""
    start_time = time.time()
    
    # Show training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"Training {model_name}...")
    progress_bar.progress(25)
    
    # Train the model
    model.fit(X_train, y_train)
    progress_bar.progress(75)
    
    status_text.text(f"Finalizing {model_name}...")
    progress_bar.progress(100)
    
    training_time = time.time() - start_time
    status_text.text(f"✅ {model_name} trained successfully in {training_time:.2f} seconds")
    
    time.sleep(1)  # Brief pause to show completion
    status_text.empty()
    progress_bar.empty()
    
    return model, training_time

def evaluate_model(model, X_test, y_test, problem_type):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    if problem_type == "Classification":
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    else:  # Regression
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return metrics, y_pred

def training_page():
    st.markdown('<h2 class="section-header">🚀 Model Training</h2>', unsafe_allow_html=True)
    
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
    
    # Data preparation info
    st.markdown("### 📋 Data Preparation")
    
    # Check for missing values in target
    missing_target = df[target_column].isnull().sum()
    if missing_target > 0:
        st.warning(f"⚠️ Found {missing_target} missing values in target column. These rows will be removed.")
    
    # Prepare data
    X, y, preprocessor, label_encoder, categorical_columns, numerical_columns = prepare_data(df, target_column, problem_type)
    
    # Show data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", len(X))
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        st.metric("Target Type", problem_type)
    with col4:
        if problem_type == "Classification":
            st.metric("Classes", len(np.unique(y)))
        else:
            st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
    
    # Train-test split configuration
    st.markdown("### ⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%):", min_value=10, max_value=40, value=20, step=5) / 100
    with col2:
        random_state = st.number_input("Random State:", value=42, min_value=0, max_value=9999)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if problem_type == "Classification" else None
    )
    
    # Show split info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    
    # Get available models
    if problem_type == "Classification":
        available_models = [
            "Logistic Regression",
            "Decision Tree", 
            "Random Forest",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)"
        ]
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")
    else:
        available_models = [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression", 
            "Decision Tree",
            "Random Forest",
            "Support Vector Regression (SVR)"
        ]
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")
    
    # Check if models were pre-selected from recommendation page
    default_models = st.session_state.get('selected_models', [available_models[0]])
    
    selected_models = st.multiselect(
        "Select models to train:",
        available_models,
        default=default_models
    )
    
    if not selected_models:
        st.info("Please select at least one model to train.")
        return
    
    # Training section
    st.markdown("### 🎯 Model Training")
    
    # Train button
    if st.button("🚀 Train Selected Models", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return
        
        # Initialize results storage
        if 'training_results' not in st.session_state:
            st.session_state.training_results = {}
        
        # Data quality info will be shown in prepare_data function
        st.info("� Starting model training with proper pipeline approach...")
        
        # Train each selected model
        results = {}
        
        for model_name in selected_models:
            st.markdown(f"#### Training {model_name}")
            
            # Get model instance
            model = get_model(model_name, problem_type)
            if model is None:
                st.error(f"Could not create model: {model_name}")
                continue
            
            # Create full pipeline to avoid data leakage
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train model with progress
            trained_model, training_time = train_model_with_progress(
                full_pipeline, X_train, y_train, model_name
            )
            
            # Evaluate model
            metrics, y_pred = evaluate_model(full_pipeline, X_test, y_test, problem_type)
            
            # Cross-validation for more robust evaluation (using full pipeline!)
            cv_scores = cross_val_score(
                full_pipeline, X_train, y_train, 
                cv=5, 
                scoring='accuracy' if problem_type == "Classification" else 'r2'
            )
            
            # Store results
            results[model_name] = {
                'model': trained_model,
                'metrics': metrics,
                'training_time': training_time,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'preprocessor': preprocessor,
                'label_encoder': label_encoder,
                'feature_columns': X.columns.tolist(),
                'categorical_columns': categorical_columns,
                'numerical_columns': numerical_columns
            }
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                if problem_type == "Classification":
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                else:
                    st.metric("R² Score", f"{metrics['r2_score']:.4f}")
            
            with col2:
                st.metric("CV Mean", f"{cv_scores.mean():.4f}")
            
            with col3:
                st.metric("Training Time", f"{training_time:.2f}s")
            
            st.success(f"✅ {model_name} training completed!")
        
        # Store results in session state
        st.session_state.training_results = results
        st.session_state.trained_models = {name: result['model'] for name, result in results.items()}
        
        st.balloons()
        st.success("🎉 All selected models have been trained successfully!")
    
    # Display training results if available
    if 'training_results' in st.session_state and st.session_state.training_results:
        st.markdown("### 📊 Training Results Summary")
        
        results = st.session_state.training_results
        
        # Create results table
        results_data = []
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'Training Time (s)': f"{result['training_time']:.2f}",
                'CV Mean': f"{result['cv_mean']:.4f}",
                'CV Std': f"{result['cv_std']:.4f}"
            }
            
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
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, width='stretch')
        
        # Best model recommendation
        st.markdown("### 🏆 Best Model Recommendation")
        
        if problem_type == "Classification":
            best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
            metric_name = "Accuracy"
            metric_value = best_model[1]['metrics']['accuracy']
        else:
            best_model = max(results.items(), key=lambda x: x[1]['metrics']['r2_score'])
            metric_name = "R² Score"
            metric_value = best_model[1]['metrics']['r2_score']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", best_model[0])
        with col2:
            st.metric(metric_name, f"{metric_value:.4f}")
        with col3:
            st.metric("CV Score", f"{best_model[1]['cv_mean']:.4f}")
        
        st.session_state.best_model = best_model[0]
        
        # Model comparison visualization
        st.markdown("### 📈 Model Performance Comparison")
        
        model_names = list(results.keys())
        if problem_type == "Classification":
            accuracies = [results[name]['metrics']['accuracy'] for name in model_names]
            cv_scores = [results[name]['cv_mean'] for name in model_names]
            
            # Create bar chart using matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            x_pos = np.arange(len(model_names))
            bars1 = ax.bar(x_pos - 0.2, accuracies, 0.4, label='Test Accuracy', alpha=0.8)
            bars2 = ax.bar(x_pos + 0.2, cv_scores, 0.4, label='CV Score', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison (Classification)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        else:
            r2_scores = [results[name]['metrics']['r2_score'] for name in model_names]
            cv_scores = [results[name]['cv_mean'] for name in model_names]
            
            # Create bar chart using matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            x_pos = np.arange(len(model_names))
            bars1 = ax.bar(x_pos - 0.2, r2_scores, 0.4, label='Test R² Score', alpha=0.8)
            bars2 = ax.bar(x_pos + 0.2, cv_scores, 0.4, label='CV Score', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison (Regression)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
            
            st.pyplot(fig)
            plt.close()
        
        # Download models
        st.markdown("### 💾 Download Trained Models")

        for model_name, result in results.items():
            model_data = {
                'model': result['model'],
                'preprocessor': result['preprocessor'],
                'label_encoder': result['label_encoder'],
                'target_column': target_column,
                'problem_type': problem_type,
                'metrics': result['metrics'],
                'training_date': datetime.now().isoformat()
            }
            model_pickle = pickle.dumps(model_data)
            st.download_button(
                label=f"⬇️ Download {model_name} Model",
                data=model_pickle,
                file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
                key=f"download_{model_name}"
            )
    
    # Quick predictions section
    if 'training_results' in st.session_state and st.session_state.training_results:
        st.markdown("### 🔮 Quick Predictions")
        
        # Select model for predictions
        trained_model_names = list(st.session_state.training_results.keys())
        selected_prediction_model = st.selectbox(
            "Select model for predictions:",
            trained_model_names
        )
        
        if selected_prediction_model:
            # Manual input for prediction
            st.markdown("#### Make a Single Prediction")
            
            # Create input fields for each feature
            input_data = {}
            for col in X.columns:
                # Use pd.to_numeric to safely check if column is truly numeric
                # (Arrow-backed string columns can pass is_numeric_dtype but fail on .mean())
                numeric_series = pd.to_numeric(X[col], errors='coerce')
                col_is_truly_numeric = numeric_series.notna().any() and pd.api.types.is_numeric_dtype(X[col])
                
                if not col_is_truly_numeric:
                    # Categorical / string feature
                    input_data[col] = st.selectbox(f"{col}:", X[col].dropna().unique(), key=f"pred_sel_{col}")
                else:
                    # Numerical feature — safely compute mean via coerced numeric series
                    try:
                        mean_val = float(numeric_series.mean())
                    except Exception:
                        mean_val = 0.0
                    input_data[col] = st.number_input(f"{col}:", value=mean_val, key=f"pred_num_{col}")
            
            # Make prediction
            if st.button("Make Prediction"):
                # Get the trained pipeline (which includes preprocessing)
                result = st.session_state.training_results[selected_prediction_model]
                trained_pipeline = result['model']  # This is actually the full pipeline
                label_encoder = result['label_encoder']
                
                # Create dataframe from input with proper column ordering
                input_df = pd.DataFrame([input_data])
                
                # Ensure columns are in the same order as training data
                training_columns = result['feature_columns'] if 'feature_columns' in result else X.columns.tolist()
                input_df = input_df[training_columns]
                
                # Ensure proper data types - categorical columns should be object/string type
                stored_categorical_columns = result.get('categorical_columns', [])
                for col in training_columns:
                    if col in stored_categorical_columns:
                        input_df[col] = input_df[col].astype('object')
                    else:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # Make prediction using the full pipeline (includes preprocessing)
                prediction = trained_pipeline.predict(input_df)[0]
                
                # Decode if classification with label encoder
                if problem_type == "Classification" and label_encoder:
                    prediction = label_encoder.inverse_transform([prediction])[0]
                
                st.success(f"🎯 Prediction: {prediction}")
    
    else:
        st.info("👆 Train models above to see results and make predictions.")

    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous: Model Recommendation", type="secondary", key="nav_prev_from_training"):
            st.session_state.explicit_navigation = "🤖 Model Recommendation"
            st.rerun()

    with col2:
        if st.button("💾 Save Progress", type="primary", key="nav_save_training"):
            st.success("✅ Training progress saved!")

    with col3:
        if st.button("➡️ Next: Model Evaluation", type="primary", key="nav_next_from_training"):
            if st.session_state.trained_models:
                st.session_state.explicit_navigation = "📊 Model Evaluation"
                st.rerun()
            else:
                st.error("⚠️ Please train at least one model first!")
