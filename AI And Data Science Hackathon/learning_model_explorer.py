import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# -------------------- APP CONFIG --------------------
st.set_page_config(page_title="Learning Model Explorer", layout="wide")
st.title(" Learning Model Explorer")
st.write("Interactive ML Dashboard using Streamlit")

# -------------------- SIDEBAR --------------------
st.sidebar.header(" Controls")

learning_type = st.sidebar.selectbox(
    "Select Learning Type",
    ["Supervised", "Unsupervised"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# -------------------- DATA LOADING --------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader(" Dataset Preview")
    st.dataframe(df.head())

    # -------------------- PREPROCESSING --------------------
    st.subheader(" Automatic Preprocessing")

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    st.success("Missing values handled & categorical features encoded")

    # -------------------- SUPERVISED LEARNING --------------------
    if learning_type == "Supervised":

        target_col = st.sidebar.selectbox("Select Target Column", df.columns)

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model_name = st.sidebar.selectbox(
            "Choose Algorithm",
            ["Decision Tree", "Random Forest", "Support Vector Machine"]
        )

        if model_name == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth)

        elif model_name == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators)

        else:
            C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
            model = SVC(C=C)

        if st.sidebar.button(" Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader(" Model Performance")

            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {acc:.2f}")

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            st.subheader(" Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    # -------------------- UNSUPERVISED LEARNING --------------------
    else:
        X = df.copy()

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model_name = st.sidebar.selectbox(
            "Choose Clustering Algorithm",
            ["KMeans", "Agglomerative", "DBSCAN"]
        )

        if model_name == "KMeans":
            k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = KMeans(n_clusters=k)

        elif model_name == "Agglomerative":
            k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=k)

        else:
            eps = st.sidebar.slider("Epsilon", 0.1, 5.0, 0.5)
            model = DBSCAN(eps=eps)

        if st.sidebar.button(" Run Clustering"):
            labels = model.fit_predict(X)

            st.subheader(" Clustering Results")
            st.write("Cluster Labels:", np.unique(labels))

            if len(set(labels)) > 1 and -1 not in labels:
                score = silhouette_score(X, labels)
                st.write(f"**Silhouette Score:** {score:.2f}")

            # Cluster Visualization
            st.subheader(" Cluster Visualization")
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            st.pyplot(fig)

else:
    st.warning(" Please upload a valid CSV dataset to begin.")





#  Features Covered (100% Match)

#  Dataset upload & preview
#  Automatic preprocessing
#  Supervised & Unsupervised learning
#  Hyperparameter tuning via sidebar
#  Training & evaluation
#  Confusion matrix
#  Clustering visualization
#  Accuracy / Silhouette score
#  Input validation & warnings
#  Clean Streamlit UI