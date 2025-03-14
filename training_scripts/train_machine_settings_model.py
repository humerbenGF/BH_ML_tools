# import libraries
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  
import shap  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



def train_machine_setting_with_ss_li(csv_filename, model_filename="rf_model", scale=False, max_shots_since_lithium_coat=1000, min_sustain=-1, show_output_plots=True):
    # Load dataset
    df = pd.read_csv(csv_filename).dropna()
    df = df[df['ss_li_pot'] < max_shots_since_lithium_coat]
    df = df[df['ss_li_gun'] < max_shots_since_lithium_coat]
    df = df[df['sust_kV'] > min_sustain]
    
    
    if min_sustain > 0:
        model_filename += "_sustain_only"

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    class_labels = label_encoder.classes_  # Store actual class names

    # Split features (X) and target variable (y)
    X = df.drop(columns=['label'])
    y = df['label']

    # Normalize features
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        
        
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    if scale:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model
    joblib.dump(model, model_filename+".pkl")
    print(f"Model saved as {model_filename}.pkl")

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Model Evaluation
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_labels))

    if show_output_plots:
        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.show()

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        if show_output_plots:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importances)
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.show()

    # SHAP Analysis
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled, check_additivity=False).values[:, :, 1]
    
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    if show_output_plots:
        shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    
    # Convert X_test back to DataFrame
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Debug print
    print(f"SHAP values shape: {shap_values.shape}, X_test shape: {X_test_df.shape}")

    if show_output_plots:
        # SHAP dependence plot
        for feature in feature_importances['Feature'][:3]:
            shap.dependence_plot(feature, shap_values, X_test_df, feature_names=X.columns)

    # **Decision Rule Extraction using Decision Tree Surrogate**
    tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)  # Limit depth for interpretability
    tree_model.fit(X_train, y_train)

    rule_text = export_text(tree_model, feature_names=list(X.columns))

    # Save rules to a file
    with open(model_filename+"_rules.txt", "w") as f:
        f.write(rule_text)

    print(f"Decision rules saved in {model_filename}_rules.txt")

    return feature_importances