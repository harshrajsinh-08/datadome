from flask import session
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,accuracy_score

from pre_processing.main import main
from pre_processing.modules.transformation import transform
from utils import global_store

regression_models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "svm": SVR(),
    "knn": KNeighborsRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
}

classification_models = {
    "logistic_regression": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(),
    "knn": KNeighborsClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
}

def pre_process_data(file_path):
    # Check if clean_user_data.csv already exists (already processed)
    clean_file_path = "output/clean_user_data.csv"
    if os.path.exists(clean_file_path):
        print("Using already processed data from clean_user_data.csv")
        df = pd.read_csv(clean_file_path)
        return df
    else:
        # If not, process the data
        checkbox = global_store.global_data["checkbox"]
        print(f"Processing data with checkbox value: {checkbox}")
        df, _ = main(file_path, gen_syn_data=checkbox)
        return df

def train_predict_regression(data_csv, model_name, target):
    df = pre_process_data(data_csv)
    df, scaler = transform(df, target_column=target, task="prediction")

    df.to_csv("output/after_preprocess.csv")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = regression_models.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found in regression models")
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)
    # **Inverse transform predicted and actual y values**
    y_test = scaler.inverse_transform(
        y_test.to_numpy().reshape(-1, 1)).flatten()
    y_test_pred = scaler.inverse_transform(
        np.array(y_test_pred).reshape(-1, 1)).flatten()

    # Compute metrics
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * \
        100  # Mean Absolute Percentage Error

    # Store results
    results = {
        "Model Coefficients": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Intercept": model.intercept_.tolist() if hasattr(model, "intercept_") else "Not applicable",
        "Expected": y_test.tolist(),
        "Preds": y_test_pred.tolist(),
        "Performance Metrics": {
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae,
            "Mean Absolute Percentage Error": mape,
            "R-squared Score": r2,
            "Explained Variance Score": explained_variance
        }
    }

    return results, y_test, y_test_pred


def train_predict_classification(data_csv, model_name, target):
    df = pre_process_data(data_csv)
    df, label_encoder = transform(df, target_column=target, task="classification")

    df.to_csv("output/clean_user_data_test.csv")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = classification_models.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found in classification models")
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)

    # Decode labels back to original form for better interpretation
    try:
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_test_pred)
    except:
        # If inverse transform fails, use the encoded values
        y_test_decoded = y_test
        y_pred_decoded = y_test_pred
    
    accuracy = accuracy_score(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, output_dict=True)

    metrics = {
        "Accuracy": accuracy,
        "Classification Report": class_report
    }

    num_classes = len(np.unique(y_train))

    results = {
        "Model Type": "Binary Classification" if num_classes == 2 else "Multiclass Classification",
        "Feature Importances": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Predictions on Test Data": y_pred_decoded.tolist() if hasattr(y_pred_decoded, 'tolist') else list(y_pred_decoded),
        "Performance Metrics": metrics,
        "Original Labels": {
            "Test": y_test_decoded.tolist() if hasattr(y_test_decoded, 'tolist') else list(y_test_decoded),
            "Predicted": y_pred_decoded.tolist() if hasattr(y_pred_decoded, 'tolist') else list(y_pred_decoded)
        }
    }

    return results, y_test_decoded.tolist() if hasattr(y_test_decoded, 'tolist') else list(y_test_decoded), y_pred_decoded.tolist() if hasattr(y_pred_decoded, 'tolist') else list(y_pred_decoded)


def regression_standard(data_csv, model_name, target):
    data = pd.read_csv(data_csv)
    
    # Drop rows where target is NaN (can't train on missing targets)
    data = data.dropna(subset=[target])
    
    X = data.drop(columns=[target])
    y = data[target]

    # Fill missing values only for numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with mean
    if len(numeric_columns) > 0:
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
    
    # Fill categorical columns with mode (most frequent value) and encode them
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
            # Encode categorical columns
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = regression_models.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found in regression models")
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)

    # **Inverse transform predicted and actual y values**
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred = y_scaler.inverse_transform(
        y_test_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * \
        100  # Mean Absolute Percentage Error

    # Store results
    results = {
        "Model Coefficients": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Intercept": model.intercept_.tolist() if hasattr(model, "intercept_") else "Not applicable",
        "Expected": y_test.tolist(),
        "Preds": y_test_pred.tolist(),
        "Performance Metrics": {
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae,
            "Mean Absolute Percentage Error": mape,
            "R-squared Score": r2,
            "Explained Variance Score": explained_variance
        }
    }

    return results, y_test, y_test_pred


def classification_standard(data_csv, model_name, target):
    # Load data
    data = pd.read_csv(data_csv)
    
    # Drop rows where target is NaN (can't train on missing targets)
    data = data.dropna(subset=[target])
    
    X = data.drop(columns=[target])
    y = data[target]

    # Fill missing values only for numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with mean
    if len(numeric_columns) > 0:
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
    
    # Fill categorical columns with mode (most frequent value) and encode them
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
            # Encode categorical columns
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = LabelEncoder()
    y_scaled = y_scaler.fit_transform(y.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = classification_models.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found in classification models")
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)
    
    # Decode labels back to original form
    try:
        y_test_decoded = y_scaler.inverse_transform(y_test)
        y_pred_decoded = y_scaler.inverse_transform(y_test_pred)
    except:
        y_test_decoded = y_test
        y_pred_decoded = y_test_pred
   
    accuracy = accuracy_score(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, output_dict=True)

    metrics = {
        "Accuracy": accuracy,
        "Classification Report": class_report
    }

    # Calculate the number of unique classes
    num_classes = len(np.unique(y_train))

    results = {
        "Model Type": "Binary Classification" if num_classes == 2 else "Multiclass Classification",
        "Feature Importances": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Predictions on Test Data": y_pred_decoded.tolist() if hasattr(y_pred_decoded, 'tolist') else list(y_pred_decoded),
        "Performance Metrics": metrics
    }

    return results, y_test_decoded.tolist() if hasattr(y_test_decoded, 'tolist') else list(y_test_decoded), y_pred_decoded.tolist() if hasattr(y_pred_decoded, 'tolist') else list(y_pred_decoded)


def visualize_classification_results(un_results, un_y_test, un_y_pred, pros_results, pros_test, pros_pred, save_path, model_name="Model"):
    """Visualization specifically for classification results"""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Add main title with model name - positioned higher to avoid overlap
    fig.suptitle(f'{model_name} Classification Comparison: Standard vs DataDome Pipeline', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Convert to numpy arrays and ensure they're integers for proper handling
    un_y_test = np.array(un_y_test)
    un_y_pred = np.array(un_y_pred)
    pros_test = np.array(pros_test)
    pros_pred = np.array(pros_pred)
    
    # Get unique labels from all data
    all_labels = np.unique(np.concatenate([un_y_test, un_y_pred, pros_test, pros_pred]))
    
    # Create label mapping for better display (if labels are encoded)
    if len(all_labels) <= 10 and all(isinstance(x, (int, np.integer)) for x in all_labels):
        # If we have integer labels, create a simple mapping
        label_names = [f'Class_{i}' for i in all_labels]
        label_mapping = dict(zip(all_labels, label_names))
    else:
        label_names = [str(x) for x in all_labels]
        label_mapping = dict(zip(all_labels, label_names))
    
    # Confusion matrices
    cm_standard = confusion_matrix(un_y_test, un_y_pred, labels=all_labels)
    cm_processed = confusion_matrix(pros_test, pros_pred, labels=all_labels)
    
    # Plot confusion matrix for standard approach
    sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=label_names, yticklabels=label_names)
    axes[0].set_title('Standard Approach - Confusion Matrix', fontsize=14, pad=20)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    
    # Plot confusion matrix for processed approach
    sns.heatmap(cm_processed, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=label_names, yticklabels=label_names)
    axes[1].set_title('DataDome Pipeline - Confusion Matrix', fontsize=14, pad=20)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    
    # Accuracy comparison bar chart
    std_acc = un_results['Performance Metrics']['Accuracy']
    proc_acc = pros_results['Performance Metrics']['Accuracy']
    
    methods = ['Standard\nApproach', 'DataDome\nPipeline']
    accuracies = [std_acc, proc_acc]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = axes[2].bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[2].set_title('Accuracy Comparison', fontsize=14, pad=20)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis='y', alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Performance metrics comparison
    std_f1 = np.mean([v['f1-score'] for k, v in un_results['Performance Metrics']['Classification Report'].items() 
                      if k not in ['accuracy', 'macro avg', 'weighted avg']])
    proc_f1 = np.mean([v['f1-score'] for k, v in pros_results['Performance Metrics']['Classification Report'].items() 
                       if k not in ['accuracy', 'macro avg', 'weighted avg']])
    
    metrics_names = ['Accuracy', 'F1-Score']
    std_metrics = [std_acc, std_f1]
    proc_metrics = [proc_acc, proc_f1]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[3].bar(x - width/2, std_metrics, width, label='Standard', 
                        color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = axes[3].bar(x + width/2, proc_metrics, width, label='DataDome Pipeline', 
                        color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)
    
    axes[3].set_title('Performance Metrics Comparison', fontsize=14, pad=20)
    axes[3].set_ylabel('Score', fontsize=12)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(metrics_names)
    axes[3].set_ylim(0, 1.1)
    axes[3].legend(loc='upper right')
    axes[3].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Classification visualization saved to: {save_path}")
    return save_path

def visualize_results(un_results, un_y_test, un_y_pred, pros_results, pros_test, pros_pred, save_path, model_name="Model"):
    """Visualization for regression results"""
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    axes = axes.flatten()
    
    # Add main title with model name
    fig.suptitle(f'{model_name} Performance Comparison: Standard vs DataDome Pipeline', 
                 fontsize=24, fontweight='bold', y=0.98)

    for i, (yt, yp, title) in enumerate([
        (un_results["Expected"], un_results["Preds"], "Conventional Results"),
        (pros_results["Expected"], pros_results["Preds"],
         "DataDome Pipeline Results")
    ]):
        # Convert lists to numpy arrays for element-wise subtraction
        yt = np.array(yt).flatten()
        yp = np.array(yp).flatten()
        residuals = yt - yp

        sns.scatterplot(x=yt, y=yp, alpha=0.7, ax=axes[i])
        axes[i].plot([yt.min(), yt.max()], [
                     yt.min(), yt.max()], 'r', linestyle='--')
        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")
        axes[i].set_title(f"{title} - Actual vs Predicted Values")

        sns.histplot(residuals, kde=True, bins=30,
                     color='blue', alpha=0.7, ax=axes[i + 2])
        axes[i + 2].set_xlabel("Residuals")
        axes[i + 2].set_title(f"{title} - Residuals Distribution")

        sns.boxplot(y=residuals, ax=axes[i + 4])
        axes[i + 4].set_title(f"{title} - Boxplot of Residuals")

        sns.lineplot(x=range(len(yt)), y=yt, label='Actual',
                     marker='o', ax=axes[i + 6])
        sns.lineplot(x=range(len(yp)), y=yp, label='Predicted',
                     marker='s', ax=axes[i + 6])
        axes[i + 6].set_xlabel("Sample Index")
        axes[i + 6].set_ylabel("Values")
        axes[i + 6].set_title(f"{title} - Actual vs Predicted Line Plot")
        axes[i + 6].legend()

    plt.tight_layout()

    # Save with high DPI for better quality
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")
    return save_path
