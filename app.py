import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("🛍️ Retail Store Customer Segmentation using K-Means Clustering")

# Description
st.write("""
This app applies the **K-Means Clustering** algorithm to group retail store customers based on their purchase behavior.
You can upload your own CSV file or use a sample dataset provided below.
""")

# Sample dataset
sample_data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Annual Income (k$)": [15, 16, 17, 18, 19, 20, 70, 75, 80, 85],
    "Spending Score (1-100)": [39, 81, 6, 77, 40, 76, 94, 3, 72, 14],
}
df = pd.DataFrame(sample_data)

# File upload option
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

st.subheader("📊 Dataset Preview")
st.dataframe(df)

# Select columns for clustering
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
x_cols = st.multiselect("Select features for clustering:", numeric_cols, default=numeric_cols[1:])

if len(x_cols) >= 2:
    X = df[x_cols]

    # Select number of clusters
    k = st.slider("Select number of clusters (K):", 2, 10, 3)

    # Apply K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    st.subheader("🔍 Clustered Data")
    st.dataframe(df)

    # Plot clusters
    st.subheader("📈 Cluster Visualization")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=X.iloc[:, 0], y=X.iloc[:, 1], hue=df['Cluster'], palette='tab10', s=100
    )
    plt.xlabel(x_cols[0])
    plt.ylabel(x_cols[1])
    plt.title("Customer Clusters")
    st.pyplot(plt)
else:
    st.warning("Please select at least 2 features for clustering.")
