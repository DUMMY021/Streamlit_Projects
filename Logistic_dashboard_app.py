import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.datasets import load_iris
import seaborn as sns

# Sidebar: data option
data_option = st.sidebar.selectbox(
    "Upload your file or use sample data",
    ["Upload File", "Use Sample Data"]
)

df = None  # placeholder

if data_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

elif data_option == "Use Sample Data":
    iris = load_iris(as_frame=True)
    df = iris.frame
    st.success("✅ Sample Iris dataset loaded successfully!")

# Sidebar: actions
action = st.sidebar.selectbox(
    "Choose Action",
    ["Preview Data", "Logistic Regression", "Predict Data"]
)

st.title("📊 Logistic Regression Dashboard")

# ---- Preview ----
if df is not None and action == "Preview Data":
    st.subheader("🔎 Preview of Data")
    st.dataframe(df.head(), height=200)
    st.header("📊 About Data")

    rows, cols = df.shape
    missing_values = df.isnull().sum().sum()

    st.write(f"**Rows:** {rows}")
    st.write(f"**Columns:** {cols}")
    st.write(f"**Missing Values:** {missing_values}")
    st.write("**Data Types:**")
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
    st.table(dtypes_df)

# ---- Logistic Regression ----
if df is not None and action == "Logistic Regression":
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.error("⚠️ Dataset must have at least 2 numeric columns for Logistic Regression.")
    else:
        st.sidebar.subheader("⚙️ Model Setup")

        features = st.sidebar.multiselect("Select Features (X)", num_cols, default=num_cols[:-1])
        target = st.sidebar.selectbox(
            "Select Target (y)",
            [c for c in num_cols if c not in features],
            key="target_col"
        )

        if st.sidebar.button("Train your model"):
            if len(features) == 0:
                st.error("⚠️ Please select at least one feature.")
            else:
                st.subheader("⚙️ Logistic Regression Setup")

                # Prepare data
                X = df[features].dropna()
                y = df[target].loc[X.index]

                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Train
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Save model
                st.session_state["model"] = model
                st.session_state["features"] = features
                st.session_state["target"] = target

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.subheader("📌 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{acc:.3f}")
                col2.metric("Precision", f"{prec:.3f}")
                col3.metric("Recall", f"{rec:.3f}")
                col4.metric("F1 Score", f"{f1:.3f}")

                # Classification Report
                st.subheader("📑 Classification Report")
                st.text(classification_report(y_test, y_pred, zero_division=0))

                # Confusion Matrix
                st.subheader("📉 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

# ---- Prediction ----
if df is not None and action == "Predict Data":
    if "model" not in st.session_state:
        st.warning("⚠️ Please run 'Logistic Regression' first to train the model.")
    else:
        model = st.session_state["model"]
        features = st.session_state["features"]
        target = st.session_state["target"]

        st.subheader("🔮 Predict Target Class")

        user_inputs = []
        for feat in features:
            val = st.number_input(f"Enter value for {feat}:", value=0.0)
            user_inputs.append(val)

        if st.button("Predict"):
            prediction = model.predict([user_inputs])[0]
            proba = model.predict_proba([user_inputs])[0]

            st.success(f"✅ Predicted {target}: {prediction}")
            st.write("📊 Class Probabilities:")
            st.write(dict(zip(model.classes_, proba.round(3))))
