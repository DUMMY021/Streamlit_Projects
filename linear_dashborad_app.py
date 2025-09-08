import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    ["Preview Data", "Linear Regression","Predict Data"]
)

st.title("📊 Linear Regression Dashboard")

# ---- Preview ----
if df is not None and action == "Preview Data":
    st.subheader("🔎 Preview of Data")
    st.dataframe(df.head(), height=200)
    st.header("📊 About Data")

    # Data Info (instead of describe())
    rows, cols = df.shape
    missing_values = df.isnull().sum().sum()

    st.write(f"**Rows:** {rows}")
    st.write(f"**Columns:** {cols}")
    st.write(f"**Missing Values:** {missing_values}")
    st.write("**Data Types:**")
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
    st.table(dtypes_df)


# ---- Regression ----
if df is not None and action == "Linear Regression":
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.error("⚠️ Dataset must have at least 2 numeric columns for Linear Regression.")
    else:
        st.sidebar.subheader("⚙️ Model Setup")

        # Multi-select for features
        features = st.sidebar.multiselect("Select Features (X)", num_cols, default=num_cols[:-1])

        # Select target
        target = st.sidebar.selectbox(
            "Select Target (y)",
            [c for c in num_cols if c not in features],
            key="target_col"
        )

        # Button to trigger training
        if st.sidebar.button("Train your model"):
            if len(features) == 0:
                st.error("⚠️ Please select at least one feature.")
            else:
                st.subheader("⚙️ Linear Regression Setup")

                # Prepare data
                X = df[features].dropna()
                y = df[target].loc[X.index]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Save model and setup in session_state for later use
                st.session_state["model"] = model
                st.session_state["features"] = features
                st.session_state["target"] = target

                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.subheader("📌 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R²", f"{r2:.3f}")
                col2.metric("MAE", f"{mae:.3f}")
                col3.metric("MSE", f"{mse:.3f}")
                col4.metric("RMSE", f"{rmse:.3f}")

                st.write(f"**Intercept:** {model.intercept_:.3f}")
                st.write(f"**Coefficients:**")
                coef_df = pd.DataFrame({
                    "Feature": features,
                    "Coefficient": model.coef_
                })
                st.table(coef_df)

                # Plot Actual vs Predicted
                st.subheader("📉 Actual vs Predicted")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.7)
                lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                ax.plot([lo, hi], [lo, hi], "--", color="gray")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

                st.subheader("Correlation Heatmap")
                corr = df.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
if df is not None and action == "Predict Data":
    if "model" not in st.session_state:
        st.warning("⚠️ Please run 'Linear Regression' first to train the model.")
    else:
        model = st.session_state["model"]
        features = st.session_state["features"]
        target = st.session_state["target"]

        st.subheader("🔮 Predict Target Value")

        # Create number inputs dynamically for each feature
        user_inputs = []
        for feat in features:
            val = st.number_input(f"Enter value for {feat}:", value=0.0)
            user_inputs.append(val)

        if st.button("Predict"):
            prediction = model.predict([user_inputs])[0]
            st.success(f"✅ Predicted {target}: {prediction:.3f}")
