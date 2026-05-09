# app.py

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Mental Health Burnout Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data

def load_data():
    df = pd.read_csv("mental_health_burnout_tech_2026.csv")
    return df


df = load_data()

# =========================
# TARGET + FEATURES
# =========================
TARGET = "burnout_level"

DROP_COLUMNS = [
    "employee_id",
    "burnout_level"
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]

# =========================
# NUMERIC & CATEGORICAL
# =========================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# =========================
# PREPROCESSING
# =========================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# =========================
# MODEL
# =========================
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    ["Home", "Insights", "Prediction"]
)

# =========================
# HOME PAGE
# =========================
if page == "Home":

    st.title("🧠 Mental Health Burnout Analysis")

    st.markdown("---")

    st.subheader("📌 Project Overview")

    st.write(
        "This project analyzes employee burnout and mental health in the tech industry. "
        "The dashboard provides insights about stress, burnout, work-life balance, "
        "and predicts burnout levels using Machine Learning."
    )

    st.subheader("📂 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Model Accuracy", f"{accuracy:.2%}")

    st.markdown("---")

    st.subheader("📊 Sample Data")
    st.dataframe(df.head())

# =========================
# INSIGHTS PAGE
# =========================
elif page == "Insights":

    st.title("📈 Dashboard Insights")

    st.markdown("---")

    # =========================
    # FILTERS
    # =========================

    st.sidebar.subheader("Filters")

    selected_country = st.sidebar.multiselect(
        "Select Country",
        options=df['country'].unique(),
        default=df['country'].unique()
    )

    selected_work_mode = st.sidebar.multiselect(
        "Select Work Mode",
        options=df['work_mode'].unique(),
        default=df['work_mode'].unique()
    )

    selected_gender = st.sidebar.multiselect(
        "Select Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )

    filtered_df = df[
        (df['country'].isin(selected_country)) &
        (df['work_mode'].isin(selected_work_mode)) &
        (df['gender'].isin(selected_gender))
    ]

    # =========================
    # KPIs
    # =========================

    avg_stress = round(filtered_df['stress_score'].mean(), 2)
    avg_burnout = round(filtered_df['burnout_score'].mean(), 2)
    avg_work_life = round(filtered_df['work_life_balance_score'].mean(), 2)
    avg_sleep = round(filtered_df['sleep_hours_per_night'].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average Stress", avg_stress)
    col2.metric("Average Burnout", avg_burnout)
    col3.metric("Work Life Balance", avg_work_life)
    col4.metric("Sleep Hours", avg_sleep)

    st.markdown("---")

    # =========================
    # CHARTS
    # =========================

    st.subheader("Burnout Level Distribution")

    burnout_chart = px.histogram(
        filtered_df,
        x='burnout_level',
        color='burnout_level'
    )

    st.plotly_chart(burnout_chart, use_container_width=True)

    st.subheader("Stress Score by Work Mode")

    stress_chart = px.box(
        filtered_df,
        x='work_mode',
        y='stress_score',
        color='work_mode'
    )

    st.plotly_chart(stress_chart, use_container_width=True)

    st.subheader("Average Burnout by Country")

    country_chart = px.bar(
        filtered_df.groupby('country', as_index=False)['burnout_score'].mean(),
        x='country',
        y='burnout_score',
        color='country'
    )

    st.plotly_chart(country_chart, use_container_width=True)

    st.subheader("Work Hours vs Burnout Score")

    scatter_chart = px.scatter(
        filtered_df.sample(2000),
        x='work_hours_per_week',
        y='burnout_score',
        color='burnout_level'
    )

    st.plotly_chart(scatter_chart, use_container_width=True)

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":

    st.title("🤖 Burnout Prediction")

    st.markdown("Enter employee information below:")

    user_input = {}

    for col in X.columns:

        if col in numeric_features:

            min_value = float(df[col].min())
            max_value = float(df[col].max())
            mean_value = float(df[col].mean())

            user_input[col] = st.number_input(
                col,
                min_value=min_value,
                max_value=max_value,
                value=mean_value
            )

        else:

            user_input[col] = st.selectbox(
                col,
                options=df[col].dropna().unique()
            )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Burnout"):

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)

        st.success(f"Predicted Burnout Level: {prediction}")

        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame({
            'Burnout Level': model.classes_,
            'Probability': prediction_proba[0]
        })

        prob_chart = px.bar(
            prob_df,
            x='Burnout Level',
            y='Probability',
            color='Burnout Level'
        )

        st.plotly_chart(prob_chart, use_container_width=True)
```

---

# requirements.txt

```txt
streamlit
pandas
scikit-learn
plotly
```

---

# طريقة التشغيل

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# مهم جدا

حطي ملف الداتا:

mental_health_burnout_tech_2026.csv

جنب ملف:

app.py

في نفس الفولدر بالظبط.
