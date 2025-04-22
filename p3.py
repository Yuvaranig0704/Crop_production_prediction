import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Set page config
st.set_page_config(page_title="Crop Production Analyzer", layout="wide")

# Load data
@st.cache_data
def load_data():
    # Sample data - replace with your actual data loading
    # This creates a simple synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    area = np.random.uniform(1, 100, n_samples)
    yield_val = np.random.uniform(1000, 5000, n_samples)
    production = (area * yield_val / 1000) * (1 + np.random.normal(0, 0.1, n_samples))  # Add some noise
    
    df_cp = pd.DataFrame({
        'Area_ha': area,
        'Yield_kg_per_ha': yield_val,
        'Estimated_Production_tons': production
    })
    
    # For trend analysis, create a simple df
    df = pd.DataFrame({
        'Element': np.random.choice(['Area harvested', 'Yield', 'Production'], n_samples),
        'Item': np.random.choice(['Wheat', 'Rice', 'Corn', 'Soybean'], n_samples),
        'Area': np.random.choice(['USA', 'China', 'India', 'Brazil'], n_samples),
        'Year': np.random.randint(2010, 2023, n_samples),
        'Value': np.random.uniform(100, 10000, n_samples)
    })
    
    return df, df_cp

df, df_cp = load_data()

# Train model
@st.cache_data
def train_model():
    X = df_cp[['Area_ha', 'Yield_kg_per_ha']]
    y = df_cp['Estimated_Production_tons']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    # Evaluation Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Using numpy sqrt for consistency

    return model, poly, r2, mse, mae, rmse

model, poly_features, model_score, mse, mae, rmse = train_model()

# App Layout
st.title("🌱 Crop Production Analysis and Prediction")
st.markdown("""
Analyze trends in crop production by region, crop type, and year.  
Predict future yields based on historical data for better agricultural planning.
""")

# Sidebar
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Select Analysis Type", ["Trend Analysis", "Production Prediction"])

# Trend Analysis Section
if analysis_type == "Trend Analysis":
    st.header("📊 Crop Production Trend Analysis")

    trend_option = st.selectbox("Select Trend to Analyze", ["Most Cultivated Crops", "Regional Production", "Yearly Trends"])

    if trend_option == "Most Cultivated Crops":
        st.subheader("Top 10 Most Cultivated Crops")
        top_crops = df[df['Element'] == 'Area harvested']['Item'].value_counts().head(10).reset_index()
        top_crops.columns = ['Item', 'Count']

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Item', data=top_crops, palette='viridis', ax=ax)
        ax.set_title("Top 10 Cultivated Crops")
        ax.set_xlabel("Number of Records")
        ax.set_ylabel("Crop")
        st.pyplot(fig)

    elif trend_option == "Regional Production":
        st.subheader("Top Regions by Production")
        region_prod = df[df["Element"] == 'Production'].groupby("Area")['Value'].sum().sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=region_prod.values, y=region_prod.index, palette='plasma', ax=ax)
        ax.set_title("Top 15 Productive Regions")
        ax.set_xlabel("Total Production")
        ax.set_ylabel("Region")
        st.pyplot(fig)

    elif trend_option == "Yearly Trends":
        st.subheader("Yearly Production Trends")
        trend_data = df[df['Element'].isin(['Area harvested', 'Yield', 'Production'])] \
            .groupby(['Year', 'Element'])['Value'].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=trend_data, x='Year', y='Value', hue='Element', marker='o', ax=ax)
        ax.set_title("Trends Over Years")
        ax.set_ylabel("Total Value")
        ax.set_xlabel("Year")
        ax.grid(True)
        st.pyplot(fig)

# Prediction Section
else:
    st.header("🔮 Crop Production Prediction")
    st.write(f"**Model Performance:**")
    st.write(f"- R² Score: `{model_score:.3f}`")
    st.write(f"- Mean Squared Error (MSE): `{mse:.2f}`")
    st.write(f"- Mean Absolute Error (MAE): `{mae:.2f}`")
    st.write(f"- Root Mean Squared Error (RMSE): `{rmse:.2f}`")

    col1, col2 = st.columns(2)
    with col1:
        area_input = st.number_input("Area Harvested (hectares)", min_value=0.0, value=1000.0)
    with col2:
        yield_input = st.number_input("Yield (kg per hectare)", min_value=0.0, value=3000.0)

    if st.button("Predict Production"):
        input_data = poly_features.transform([[area_input, yield_input]])
        prediction = model.predict(input_data)[0]
        st.success(f"🌾 Estimated Production: **{prediction:.2f} tons**")

    # Show some actual vs predicted values
    st.subheader("Model Performance Visualization")
    X = df_cp[['Area_ha', 'Yield_kg_per_ha']]
    y = df_cp['Estimated_Production_tons']
    X_poly = poly_features.transform(X)
    y_pred = model.predict(X_poly)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, alpha=0.5, ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual Production (tons)")
    ax.set_ylabel("Predicted Production (tons)")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)

# Sidebar Recommendations
st.sidebar.header("🧠 Recommendations")
st.sidebar.markdown("""
- **Resource Allocation**: Focus on high-yield crops in top-performing regions  
- **Strategic Planning**: Consider yearly trends before planting  
- **Diversification**: Explore less-used crops with good yield potential  
""")

# Footer
st.markdown("---")
st.markdown("**📌 Agricultural Insights App** — Empowering smarter farming with data.")