import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Regresie - Insurance Charges",
    layout="wide"
)

st.title("Regresie - Insurance Charges")

st.write("""
Această pagină prezintă problema de regresie pentru setul de date `insurance.csv`.
Obiectivul este estimarea costurilor medicale facturate pe baza unor caracteristici personale și demografice.
""")

st.markdown("""
**Variabila prezisă:**
- `charges` - costurile medicale facturate
""")


@st.cache_data
def load_data():
    data = pd.read_csv("data/insurance.csv")
    data = data.drop_duplicates().reset_index(drop=True)
    return data


@st.cache_resource
def load_models(version="v6"):
    return {
        "XGBoost Regressor": joblib.load("models/regression/xgboost_model.pkl"),
        "CatBoost Regressor": joblib.load("models/regression/catboost_model.pkl"),
        "Explainable Boosting Regressor": joblib.load("models/regression/ebm_model.pkl"),
        "Random Forest Regressor": joblib.load("models/regression/random_forest_model.pkl"),
        "Linear Regression": joblib.load("models/regression/linear_regression_model.pkl")
    }


df = load_data()
models = load_models("v6")
metrics_df = pd.read_csv("models/regression/regression_metrics.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

st.subheader("Setul de date")

col1, col2, col3 = st.columns(3)

col1.metric("Număr de observații", df.shape[0])
col2.metric("Număr caracteristici", df.shape[1] - 1)
col3.metric("Variabila țintă", "charges")

with st.expander("Afișare primele observații"):
    st.dataframe(df.head(), use_container_width=True)

st.subheader("Analiză exploratorie")

col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.hist(df["charges"], bins=25, edgecolor="black")
    ax.set_xlabel("Costuri medicale")
    ax.set_ylabel("Frecvență")
    ax.set_title("Distribuția costurilor medicale", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

with col_b:
    smoker_counts = df["smoker"].value_counts()

    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.bar(
        ["Nefumător", "Fumător"],
        [smoker_counts.get("no", 0), smoker_counts.get("yes", 0)]
    )
    ax.set_xlabel("Statut fumător")
    ax.set_ylabel("Număr observații")
    ax.set_title("Distribuția fumătorilor", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

col_c, col_d = st.columns(2)

with col_c:
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.scatter(df["age"], df["charges"], alpha=0.6, s=18)
    ax.set_xlabel("Vârstă")
    ax.set_ylabel("Costuri medicale")
    ax.set_title("Vârstă vs costuri medicale", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

with col_d:
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.scatter(df["bmi"], df["charges"], alpha=0.6, s=18)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Costuri medicale")
    ax.set_title("BMI vs costuri medicale", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

st.subheader("Compararea modelelor perfecționate")

st.dataframe(metrics_df.round(4), use_container_width=True)

fig, ax = plt.subplots(figsize=(5, 2.6))
ax.barh(metrics_df["Model"], metrics_df["Best R^2 (CV)"])
ax.set_xlabel("R^2")
ax.set_title("Performanța modelelor perfecționate", fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

st.subheader("Testarea modelelor")

selected_model_name = st.selectbox(
    "Alege modelul pentru predicție:",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.markdown("### Introducerea valorilor pentru caracteristici")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Vârstă", min_value=18, max_value=70, value=35)
    sex = st.selectbox(
        "Sex",
        ["female", "male"],
        format_func=lambda x: "Femeie" if x == "female" else "Bărbat"
    )

with col2:
    bmi = st.number_input(
        "Indice de masă corporală (BMI)",
        min_value=10.0,
        max_value=60.0,
        value=28.0,
        step=0.1
    )
    children = st.number_input(
        "Număr de copii",
        min_value=0,
        max_value=5,
        value=1
    )

with col3:
    smoker = st.selectbox(
        "Fumător",
        ["no", "yes"],
        format_func=lambda x: "Nu" if x == "no" else "Da"
    )

    region = st.selectbox(
        "Regiune",
        ["northeast", "northwest", "southeast", "southwest"],
        format_func=lambda x: {
            "northeast": "Nord-est",
            "northwest": "Nord-vest",
            "southeast": "Sud-est",
            "southwest": "Sud-vest"
        }[x]
    )

input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

if st.button("Generează predicția"):
    prediction = selected_model.predict(input_data)[0]

    st.markdown("### Rezultat predicție")
    st.success(f"Costul medical estimat este: {prediction:.2f}")

st.subheader("Metricile modelului selectat")

selected_metrics = metrics_df[metrics_df["Model"] == selected_model_name]

if not selected_metrics.empty:
    st.dataframe(selected_metrics.round(4), use_container_width=True)

st.subheader("Predicții vs valori reale")

y_pred = selected_model.predict(X_test)

fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.scatter(y_test, y_pred, alpha=0.6, s=18)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax.set_xlabel("Valori reale")
ax.set_ylabel("Predicții")
ax.set_title("Valori reale vs predicții", fontsize=10)
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

st.subheader("Distribuția erorilor")

errors = y_test - y_pred

fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.hist(errors, bins=25, edgecolor="black")
ax.set_xlabel("Eroare")
ax.set_ylabel("Frecvență")
ax.set_title("Distribuția erorilor", fontsize=10)
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

st.subheader("Hiperparametrii modelului selectat")

params = pd.DataFrame(
    list(selected_model.get_params().items()),
    columns=["Hiperparametru", "Valoare"]
)

st.dataframe(params, use_container_width=True)