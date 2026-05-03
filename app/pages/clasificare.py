import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Clasificare - Heart Disease",
    layout="wide"
)

st.title("Clasificare - Heart Disease")
st.write("""
         Această pagină prezintă problema de clasificare binară pentru setul de date `heart.csv`.
         Obiectivul este estimarea prezenței bolii cardiace pe baza unor caracteristici clinice și demografice.
         """
)

st.markdown(
    """
    **Clasele problemei:**
    - `0` - absența bolii cardiace
    - `1` - prezența bolii cardiace
    """
)

@st.cache_data
def load_data():
    data = pd.read_csv("data/heart.csv")
    data = data.drop_duplicates().reset_index(drop=True)
    return data

@st.cache_resource
def load_models():
    models = {
        "Explainable Boosting Machine": joblib.load("models/classification/ebm_model.pkl"),
        "Gaussian Naive Bayes": joblib.load("models/classification/gnb_model.pkl"),
        "Logistic Regression": joblib.load("models/classification/logistic_regression_model.pkl"),
        "Categorical Boosting": joblib.load("models/classification/catboost_model.pkl"),
        "Xtreme Gradient Boosting": joblib.load("models/classification/xgboost_model.pkl")
    }
    
    scaler = joblib.load("models/classification/scaler.pkl")
    return models, scaler

df = load_data()
models, scaler = load_models()
metrics_df = pd.read_csv("models/classification/classification_metrics.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

st.subheader("Setul de date")

col1, col2, col3 = st.columns(3)

col1.metric("Număr de observații", df.shape[0])
col2.metric("Număr caracteristici", df.shape[1] - 1)
col3.metric("Clase", df["target"].nunique())

with st.expander("Afișare primele observații"):
    st.dataframe(df.head())

st.subheader("Analiză claselor")

col_a, col_b = st.columns(2)

with col_a:
    target_counts = df["target"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(target_counts.index.astype(str), target_counts.values)
    ax.set_xlabel("Clasa target")
    ax.set_ylabel("Număr observații")
    ax.set_title("Distribuția variabilei target")
    st.pyplot(fig)

with col_b:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["age"], bins=15)
    ax.set_xlabel("Vârstă")
    ax.set_ylabel("Număr observații")
    ax.set_title("Distribuția vârstei")
    st.pyplot(fig)

st.subheader("Compararea modelelor perfecționate")

st.dataframe(metrics_df.round(4), use_container_width=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(metrics_df["Model"], metrics_df["ROC-AUC"])
ax.set_xlabel("ROC-AUC")
ax.set_title("Performanța modelelor perfecționate")
ax.invert_yaxis()
st.pyplot(fig)

st.subheader("Testarea modelelor")

selected_model_name = st.selectbox(
    "Alege modelul pentru predicție:",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.markdown("### Introducerea valorilor pentru caracteristici")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Vârstă", min_value=20, max_value=90, value=55)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Femeie" if x == 0 else "Bărbat")
    cp = st.selectbox(
        "Tip durere toracică",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Tip 0",
            1: "Tip 1",
            2: "Tip 2",
            3: "Tip 3"
        }[x]
    )
    trestbps = st.number_input("Tensiune arterială în repaus", min_value=80, max_value=220, value=130)
    chol = st.number_input("Colesterol", min_value=100, max_value=600, value=240)

with col2:
    fbs = st.selectbox(
        "Glicemie peste 120 mg/dl",
        [0, 1],
        format_func=lambda x: "Nu" if x == 0 else "Da"
    )
    restecg = st.selectbox(
        "Rezultat ECG în repaus",
        [0, 1, 2],
        format_func=lambda x: f"Categoria {x}"
    )
    thalach = st.number_input("Frecvență cardiacă maximă", min_value=60, max_value=220, value=150)
    exang = st.selectbox(
        "Angină indusă de efort",
        [0, 1],
        format_func=lambda x: "Nu" if x == 0 else "Da"
    )

with col3:
    oldpeak = st.number_input("Depresie ST indusă de efort", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Panta segmentului ST",
        [0, 1, 2],
        format_func=lambda x: f"Categoria {x}"
    )
    ca = st.selectbox(
        "Număr vase colorate fluoroscopic",
        [0, 1, 2, 3, 4]
    )
    thal = st.selectbox(
        "Rezultat thalassemia",
        [0, 1, 2, 3],
        format_func=lambda x: f"Categoria {x}"
    )

input_data = pd.DataFrame(
    [{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }]
)


def prepare_input(model_name, data):
    if model_name in ["Gaussian Naive Bayes", "Logistic Regression"]:
        return scaler.transform(data)
    return data
if st.button("Generează predicția"):
    prepared_input = prepare_input(selected_model_name, input_data)

    prediction = selected_model.predict(prepared_input)[0]
    probability = selected_model.predict_proba(prepared_input)[0][1]

    st.markdown("### Rezultat predicție")

    if int(prediction) == 1:
        st.warning(f"Modelul estimează prezența bolii cardiace. Probabilitatea: {probability:.3f}")
    else:
        st.success(f"Modelul estimează absența cardiace. Probabilitatea: {probability:.3f}")

st.subheader("Metricile modelului selectat")

selected_metrics = metrics_df[metrics_df["Model"] == selected_model_name]

if not selected_metrics.empty:
    st.dataframe(selected_metrics.round(4), use_container_width=True)

st.subheader("Matricea de confuzie pentru modelul selectat")

if selected_model_name in ["Gaussian Naive Bayes", "Logistic Regression"]:
    X_test_eval = scaler.transform(X_test)
else:
    X_test_eval = X_test

y_pred = selected_model.predict(X_test_eval)

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(3.5, 3))
im = ax.imshow(cm, cmap="Reds")

ax.set_title(f"Matrice de confuzie - {selected_model_name}", fontsize=10)
ax.set_xlabel("Predicție")
ax.set_ylabel("Valoarea reală")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["0", "1"])
ax.set_yticklabels(["0", "1"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11)

plt.tight_layout()
st.pyplot(fig, use_container_width=False)

st.subheader("Hiperparametrii modelului selectat")

params = selected_model.get_params()
params_df = pd.DataFrame(
    list(params.items()),
    columns=["Hiperparametru", "Valoare"]
)

st.dataframe(params_df, use_container_width=True)