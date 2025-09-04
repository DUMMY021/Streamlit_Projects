import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.sidebar.header("Upload your file")
st.sidebar.text("Which regression you want to perform")

# --- Sidebar File Upload ---
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
algorithm = st.sidebar.selectbox(
    "Your Algorithm",
    ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest"],
)
st.title(f"Welcome to {algorithm} Dashboard")

if file is not None:
    # Load Data
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("🔎 Preview of Data")
    st.dataframe(df.head(), height=200)

    # Pick numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.error("⚠️ Dataset must have at least 2 numeric columns for Linear Regression.")
    else:
        # Single place where selectboxes are created, with keys to avoid duplicate IDs
        feature = st.sidebar.selectbox("Select Feature (X)", num_cols, key="feature_col")
        target = st.sidebar.selectbox(
            "Select Target (y)",
            [c for c in num_cols if c != feature],
            key="target_col",
        )

        if algorithm == "Linear Regression":
            st.subheader("⚙️ Linear Regression Setup")

            # Prepare data (single-feature regression)
            X = df[[feature]].dropna()
            y = df[target].loc[X.index]  # align with X after dropna

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Performance Metrics (no 'squared' argument)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)               # <-- compatible with all sklearn versions
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("📌 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{r2:.3f}")
            col2.metric("MAE", f"{mae:.3f}")
            col3.metric("MSE", f"{mse:.3f}")
            col4.metric("RMSE", f"{rmse:.3f}")

            st.write(f"**Intercept:** {model.intercept_:.3f}")
            st.write(f"**Coefficient ({feature}):** {model.coef_[0]:.3f}")

            # Visualization: Actual vs Predicted
            st.subheader("📉 Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "--", color="gray")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            # Regression line over full data (for single feature)
            st.subheader("📈 Regression Line (All data)")
            fig2, ax2 = plt.subplots()
            ax2.scatter(X, df[target], alpha=0.4, label="Data")
            # sort X for a clean line plot
            order = X[feature].argsort()
            ax2.plot(
                X.iloc[order][feature],
                model.predict(X.iloc[order]),
                color="red",
                linewidth=2,
                label="Regression Line",
            )
            ax2.set_xlabel(feature)
            ax2.set_ylabel(target)
            ax2.legend()
            st.pyplot(fig2)
