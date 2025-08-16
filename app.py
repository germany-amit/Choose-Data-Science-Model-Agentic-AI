# -----------------------------
# Future AI Models Lab (2025–2050)
# Streamlit app for Fortune 500 demos
# -----------------------------
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

st.set_page_config(
    page_title="Future AI Models Lab (2025–2050)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Helpers
# ============================================================

def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def infer_task(df: pd.DataFrame, target: str | None):
    """
    Return one of: 'classification', 'regression', 'timeseries', 'unsupervised'
    """
    if target is None or target == "— none —":
        # try to spot time column
        ts_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower()]
        return "timeseries" if len(ts_cols) else "unsupervised"
    y = df[target]
    if is_numeric(y):
        # heuristic: lots of unique numeric values => regression
        return "regression" if y.nunique() > max(10, int(0.05 * len(y))) else "classification"
    else:
        return "classification"

def train_test_prepare(df, target, test_size=0.2, max_rows=5000):
    if target is None or target == "— none —":
        X = df.copy()
        return X.head(max_rows), None, None, None, None, None

    df_ = df.dropna(subset=[target]).copy()
    if len(df_) > max_rows:
        df_ = df_.sample(max_rows, random_state=42)

    y = df_[target]
    X = df_.drop(columns=[target])

    num_cols = [c for c in X.columns if is_numeric(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=None if is_numeric(y) else y)
    return X_train, X_test, y_train, y_test, pre, (num_cols, cat_cols)

def regression_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"RMSE": rmse, "R2": r2_score(y_true, y_pred)}

def classification_metrics(y_true, y_pred, proba=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted")
    }
    # ROC-AUC for binary only (safe guard)
    try:
        if proba is not None and np.unique(y_true).shape[0] == 2:
            out["ROC_AUC"] = roc_auc_score(y_true, proba[:, 1] if proba.ndim == 2 else proba)
    except Exception:
        pass
    return out

def show_metric_table(d: dict):
    dfm = pd.DataFrame([d]).T.reset_index()
    dfm.columns = ["Metric", "Value"]
    st.dataframe(dfm, use_container_width=True)

def info_note():
    st.caption("⚠️ Demo-friendly implementations: heavy models (Quantum, World Models, etc.) are previewed via "
               "lightweight surrogates/visuals suitable for Streamlit Free. In production, replace with full backends.")

# ============================================================
# Model 1 — Neuro-Symbolic (Decision Tree + Rule Hints)
# ============================================================
def run_neurosymbolic(df, task, target):
    st.markdown("### 1) Neuro-Symbolic AI (tree + rules)")
    info_note()
    if target is None or target == "— none —":
        st.info("No target column selected → showing pattern mining via DecisionTree on synthetic label.")
        # synthetic label from top principal numeric column threshold
        num_cols = [c for c in df.columns if is_numeric(df[c])]
        if not num_cols:
            st.warning("Need at least one numeric column.")
            return None
        col = num_cols[0]
        y = (df[col] > df[col].median()).astype(int)
        X = df.drop(columns=[c for c in df.columns if c == target])
        # simple preprocessing
        num_cols = [c for c in X.columns if is_numeric(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]
        pre = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ])
        clf = Pipeline([("pre", pre), ("dt", DecisionTreeClassifier(max_depth=4, random_state=42))])
        clf.fit(X, y)
        st.write("Surrogate rules learned (feature importances):")
        try:
            fi = clf.named_steps["dt"].feature_importances_
            st.bar_chart(pd.DataFrame({"importance": fi}))
        except Exception:
            st.info("Rule extraction not available in this preview.")
        return {"score": 0.5, "explainability": 0.9}

    X_train, X_test, y_train, y_test, pre, _ = train_test_prepare(df, target)
    if task == "classification":
        model = Pipeline([("pre", pre), ("dt", DecisionTreeClassifier(max_depth=6, random_state=42))])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None
        m = classification_metrics(y_test, y_pred, proba)
        show_metric_table(m)
        st.plotly_chart(px.bar(x=list(m.keys()), y=list(m.values()), title="Neuro-Symbolic (Decision Tree) Metrics"), use_container_width=True)
        # feature importances (approx)
        try:
            fi = model.named_steps["dt"].feature_importances_
            st.caption("Feature importances (post-preprocessing)")
            st.bar_chart(pd.DataFrame({"importance": fi}))
        except Exception:
            pass
        return {"score": float(m.get("F1", m.get("Accuracy", 0))), "explainability": 0.9}
    else:
        model = Pipeline([("pre", pre), ("dt", DecisionTreeRegressor(max_depth=6, random_state=42))])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = regression_metrics(y_test, y_pred)
        show_metric_table(m)
        st.plotly_chart(px.bar(x=list(m.keys()), y=list(m.values()), title="Neuro-Symbolic (Tree) Metrics"), use_container_width=True)
        return {"score": float(-m["RMSE"]), "explainability": 0.7}

# ============================================================
# Model 2 — Foundation Models 2.0 (multimodal summary surrogate)
# ============================================================
def run_foundation(df, task, target):
    st.markdown("### 2) Foundation Models 2.0 (multimodal summary)")
    info_note()
    # We create a structured summary + simple baseline model metric as a proxy
    summary = {
        "rows": len(df),
        "cols": df.shape[1],
        "numeric_cols": [c for c in df.columns if is_numeric(df[c])],
        "categorical_cols": [c for c in df.columns if not is_numeric(df[c])]
    }
    st.json(summary)
    if target and target != "— none —":
        X_train, X_test, y_train, y_test, pre, _ = train_test_prepare(df, target)
        if task == "classification":
            base = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=200))])
            base.fit(X_train, y_train)
            y_pred = base.predict(X_test)
            try:
                proba = base.predict_proba(X_test)
            except Exception:
                proba = None
            m = classification_metrics(y_test, y_pred, proba)
            show_metric_table(m)
            st.plotly_chart(px.bar(x=list(m.keys()), y=list(m.values()), title="Foundation surrogate metrics"), use_container_width=True)
            score = float(m.get("F1", m.get("Accuracy", 0)))
        else:
            base = Pipeline([("pre", pre), ("lin", LinearRegression())])
            base.fit(X_train, y_train)
            y_pred = base.predict(X_test)
            m = regression_metrics(y_test, y_pred)
            show_metric_table(m)
            st.plotly_chart(px.bar(x=list(m.keys()), y=list(m.values()), title="Foundation surrogate metrics"), use_container_width=True)
            score = float(-m["RMSE"])
    else:
        st.info("No target selected → showing schema intelligence only.")
        score = 0.4
    return {"score": score, "coverage": 1.0}

# ============================================================
# Model 3 — Quantum ML (optimization surrogate)
# ============================================================
def run_quantum(df, task, target):
    st.markdown("### 3) Quantum ML (QUBO-style surrogate)")
    info_note()
    # Surrogate: select a subset of numeric columns to maximize variance under a budget
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if not num_cols:
        st.warning("Need numeric columns to run optimization.")
        return {"score": 0.1}
    variances = df[num_cols].var().fillna(0.0)
    weights = np.clip((variances / (variances.max() + 1e-9)).values, 0, 1)
    budget = max(1, int(len(num_cols) / 2))
    # greedy "annealing-like" pick
    order = np.argsort(-weights)
    pick, total = [], 0
    for i in order:
        if len(pick) < budget:
            pick.append(num_cols[i])
            total += weights[i]
    st.write("Selected features (surrogate quantum pick):", pick)
    score = float(total / max(1, budget))
    st.plotly_chart(px.bar(x=pick, y=list(variances[pick]), title="Selected features variance"), use_container_width=True)
    return {"score": score}

# ============================================================
# Model 4 — Spiking Neural Networks (event spikes surrogate)
# ============================================================
def run_snn(df, task, target):
    st.markdown("### 4) Spiking Neural Networks (event spikes)")
    info_note()
    # pick first numeric series, compute spikes (large derivative)
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if not num_cols:
        st.warning("Need a numeric column.")
        return {"score": 0.2}
    s = df[num_cols[0]].astype(float).fillna(method="ffill").fillna(method="bfill")
    diff = s.diff().abs()
    thresh = diff.mean() + 2 * diff.std()
    spikes = (diff > thresh).astype(int)
    fig = px.line(pd.DataFrame({"value": s.values, "spike": spikes.values}))
    fig.update_layout(title=f"Spikes on {num_cols[0]} (threshold={thresh:.3f})")
    st.plotly_chart(fig, use_container_width=True)
    score = float(spikes.mean())  # fraction of spike events
    return {"score": 1.0 - min(1.0, score)}  # fewer random spikes = better

# ============================================================
# Model 5 — Graph Neural Networks (graph surrogate)
# ============================================================
def run_gnn(df, task, target):
    st.markdown("### 5) Graph Neural Networks (kNN graph surrogate)")
    info_note()
    # Build kNN graph from numeric features and visualize centrality
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns to build a graph.")
        return {"score": 0.2}
    X = df[num_cols].dropna()
    n = min(200, len(X))
    X = X.sample(n, random_state=42) if len(X) > n else X
    try:
        A = kneighbors_graph(X, n_neighbors=min(8, max(2, int(np.sqrt(len(X))//1))), mode="connectivity", include_self=False)
        G = nx.from_scipy_sparse_array(A)
    except Exception:
        # fallback fully connected small graph
        G = nx.gnm_random_graph(min(50, len(X)), min(200, len(X) * 2), seed=42)

    centrality = nx.degree_centrality(G)
    cvals = np.array(list(centrality.values()))
    st.plotly_chart(px.histogram(cvals, nbins=20, title="Node centrality distribution"), use_container_width=True)
    score = float(cvals.mean())
    return {"score": score}

# ============================================================
# Model 6 — PINNs / Energy-Based (constrained fit surrogate)
# ============================================================
def run_pinn(df, task, target):
    st.markdown("### 6) Physics-Informed / Energy-Based (constrained fit)")
    info_note()
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if len(num_cols) < 2:
        st.warning("Need 2+ numeric columns.")
        return {"score": 0.2}
    x = df[num_cols[0]].astype(float).fillna(method="ffill").values
    y = df[num_cols[1]].astype(float).fillna(method="ffill").values
    # simple polynomial fit + smoothness penalty
    z = np.polyfit(x, y, deg=2)
    y_hat = np.polyval(z, x)
    mse = mean_squared_error(y, y_hat)
    smooth_penalty = np.mean(np.diff(y_hat, 2) ** 2)
    energy = mse + 0.1 * smooth_penalty
    st.plotly_chart(px.scatter(x=x, y=y, title="PINN surrogate: data vs. fit").add_scatter(x=x, y=y_hat, mode="lines"), use_container_width=True)
    return {"score": float(-energy)}

# ============================================================
# Model 7 — Generative World Models (rolling forecast surrogate)
# ============================================================
def run_world(df, task, target):
    st.markdown("### 7) Generative World Models (rolling forecast)")
    info_note()
    # pick a numeric series; use lag features with linear regression for rolling forecast
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if not num_cols:
        st.warning("Need a numeric column.")
        return {"score": 0.2}
    s = df[num_cols[0]].astype(float).reset_index(drop=True).fillna(method="ffill").fillna(method="bfill")
    if len(s) < 40:
        st.warning("Need at least 40 rows for a tiny forecast.")
        return {"score": 0.2}
    lag = 5
    X, y = [], []
    for i in range(lag, len(s)):
        X.append(s[i-lag:i].values)
        y.append(s[i])
    X = np.array(X); y = np.array(y)
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    reg = LinearRegression().fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    m = regression_metrics(y_te, y_pred)
    show_metric_table(m)
    idx = np.arange(len(y_te))
    fig = px.line(x=idx, y=y_te, labels={"x": "t", "y": "value"}, title="Rolling Forecast (surrogate)")
    fig.add_scatter(x=idx, y=y_pred, mode="lines", name="forecast")
    st.plotly_chart(fig, use_container_width=True)
    return {"score": float(-m["RMSE"])}

# ============================================================
# Model 8 — Continual / Meta-Learning (online SGD)
# ============================================================
def run_continual(df, task, target):
    st.markdown("### 8) Continual / Meta-Learning (online learning)")
    info_note()
    if target is None or target == "— none —":
        st.info("No target column; continual learning shown on synthetic target.")
        # create simple target: median split on first numeric
        num_cols = [c for c in df.columns if is_numeric(df[c])]
        if not num_cols:
            st.warning("Need numeric data.")
            return {"score": 0.2}
        y = (df[num_cols[0]] > df[num_cols[0]].median()).astype(int)
        X = df.drop(columns=[])
        target_local = "__synthetic__"
        X[target_local] = y
        return run_continual(X, "classification", target_local)

    X_train, X_test, y_train, y_test, pre, _ = train_test_prepare(df, target)
    # stream batches to partial_fit
    batch = max(100, int(0.1 * len(X_train)))
    scores = []
    if task == "classification":
        model = Pipeline([("pre", pre), ("sgd", SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", random_state=42))])
        # need classes for partial_fit
        classes = np.unique(y_train)
        # manual incremental loop
        for i in range(0, len(X_train), batch):
            Xi = X_train.iloc[i:i+batch]
            yi = y_train.iloc[i:i+batch]
            model.named_steps["sgd"].partial_fit(
                model.named_steps["pre"].fit_transform(Xi), yi, classes=classes
            )
            # evaluate on holdout
            y_pred = model.named_steps["sgd"].predict(model.named_steps["pre"].transform(X_test))
            scores.append(accuracy_score(y_test, y_pred))
        st.plotly_chart(px.line(y=scores, title="Continual Learning: Accuracy over increments"), use_container_width=True)
        return {"score": float(scores[-1] if scores else 0.0)}
    else:
        model = Pipeline([("pre", pre), ("sgd", SGDRegressor(max_iter=1, random_state=42))])
        for i in range(0, len(X_train), batch):
            Xi = X_train.iloc[i:i+batch]
            yi = y_train.iloc[i:i+batch]
            # fit preprocessor on first batch; then reuse
            if i == 0:
                Zi = model.named_steps["pre"].fit_transform(Xi)
            else:
                Zi = model.named_steps["pre"].transform(Xi)
            model.named_steps["sgd"].partial_fit(Zi, yi)
            y_pred = model.named_steps["sgd"].predict(model.named_steps["pre"].transform(X_test))
            scores.append(-mean_squared_error(y_test, y_pred, squared=False))
        st.plotly_chart(px.line(y=scores, title="Continual Learning: −RMSE over increments"), use_container_width=True)
        return {"score": float(scores[-1] if scores else 0.0)}

# ============================================================
# Model 9 — Swarm Intelligence (agent clustering surrogate)
# ============================================================
def run_swarm(df, task, target):
    st.markdown("### 9) Swarm Intelligence (agent clustering)")
    info_note()
    # Use KMeans as particle grouping surrogate + silhouette score
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if len(num_cols) < 2:
        st.warning("Need 2+ numeric columns.")
        return {"score": 0.2}
    X = df[num_cols].dropna()
    n = min(1000, len(X))
    X = X.sample(n, random_state=42) if len(X) > n else X
    k = min(6, max(2, int(np.sqrt(len(X))//1)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
    labels = km.labels_
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = 0.0
    st.plotly_chart(px.scatter(
        x=X.iloc[:, 0], y=X.iloc[:, 1], color=labels.astype(str), title=f"Swarm surrogate (K={k})"
    ), use_container_width=True)
    return {"score": float(sil)}

# ============================================================
# Model 10 — Causal AI (DAG from correlations surrogate)
# ============================================================
def run_causal(df, task, target):
    st.markdown("### 10) Causal AI (correlation-driven DAG preview)")
    info_note()
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if len(num_cols) < 2:
        st.warning("Need 2+ numeric columns.")
        return {"score": 0.2, "explainability": 0.8}
    corr = df[num_cols].corr().fillna(0.0)
    edges = []
    for i, a in enumerate(num_cols):
        for j, b in enumerate(num_cols):
            if j <= i:
                continue
            w = corr.loc[a, b]
            if abs(w) >= 0.35:
                edges.append((a, b, float(w)))
    if not edges:
        st.info("No strong relationships found at |corr| ≥ 0.35")
    G = nx.Graph()
    G.add_nodes_from(num_cols)
    for a, b, w in edges:
        G.add_edge(a, b, weight=abs(w))
    pos = nx.spring_layout(G, seed=42)
    # draw with plotly
    xe, ye = [], []
    for e in G.edges():
        xe += [pos[e[0]][0], pos[e[1]][0], None]
        ye += [pos[e[0]][1], pos[e[1]][1], None]
    node_x = [pos[k][0] for k in G.nodes()]
    node_y = [pos[k][1] for k in G.nodes()]
    fig = px.scatter(x=node_x, y=node_y, text=list(G.nodes()))
    fig.update_traces(textposition="top center")
    fig.add_scatter(x=xe, y=ye, mode="lines", name="edges")
    fig.update_layout(title="Causal surrogate: correlation graph (not true causality)")
    st.plotly_chart(fig, use_container_width=True)
    score = float(np.mean([d.get("weight", 0) for _, _, d in G.edges(data=True)]) if G.number_of_edges() else 0.0)
    return {"score": score, "explainability": 0.8}

# ============================================================
# Real-Time Agent — choose the best model
# ============================================================
def agent_recommend(task, metrics_map):
    """
    metrics_map: {model_name: {"score": float, ...}}
    Strategy:
      - Normalize scores
      - Add task priors (e.g., graphs favored for relational data if many categorical columns)
    """
    # Normalize
    vals = [v.get("score", 0.0) for v in metrics_map.values()]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    def norm(x): return 0.5 if hi == lo else (x - lo) / (hi - lo)

    priors = {k: 0.0 for k in metrics_map.keys()}
    # Simple task priors
    if task == "classification":
        for k in ["Neuro-Symbolic", "Foundation 2.0", "Continual", "Causal AI"]:
            if k in priors: priors[k] += 0.05
    elif task == "regression":
        for k in ["PINNs / Energy-Based", "World Models", "Foundation 2.0"]:
            if k in priors: priors[k] += 0.05
    elif task == "timeseries":
        for k in ["SNN (Spiking)", "World Models", "Continual"]:
            if k in priors: priors[k] += 0.07
    else:  # unsupervised
        for k in ["Swarm Intelligence", "GNN", "Causal AI"]:
            if k in priors: priors[k] += 0.07

    scores = {}
    for name, m in metrics_map.items():
        s = norm(m.get("score", 0.0)) + priors.get(name, 0.0)
        scores[name] = s

    best = max(scores, key=scores.get)
    rationale = f"Selected **{best}** based on normalized performance plus task prior for **{task}**."
    return best, scores, rationale

# ============================================================
# Sidebar — Data intake
# ============================================================
st.sidebar.title("📂 Datasets (upload up to 10 CSV)")
uploads = st.sidebar.file_uploader(
    "Upload CSV files", type=["csv"], accept_multiple_files=True, label_visibility="visible"
)

# If none uploaded, generate 10 demo CSVs so the UI always works
demo_sets = {}
if not uploads:
    np.random.seed(7)
    # 1: Retail classification
    demo_sets["retail_orders.csv"] = pd.DataFrame({
        "amount": np.random.gamma(3, 20, 800),
        "items": np.random.poisson(3, 800),
        "country": np.random.choice(["US", "IN", "DE", "BR"], 800),
        "vip": np.random.choice([0, 1], 800, p=[0.8, 0.2])
    })
    # 2: IoT time series
    t = np.arange(1000)
    demo_sets["iot_temperature.csv"] = pd.DataFrame({
        "time": t, "temp": 25 + 0.01*t + np.sin(t/30) + np.random.normal(0, 0.3, len(t))
    })
    # 3: Finance regression
    demo_sets["finance_risk.csv"] = pd.DataFrame({
        "vol": np.random.rand(600)*0.3, "beta": np.random.rand(600)*1.5, "return_next": np.random.normal(0.01, 0.05, 600)
    })
    # 4: Logistics (graphy)
    demo_sets["logistics_nodes.csv"] = pd.DataFrame({
        "hub_load": np.random.randint(10, 100, 500),
        "spoke_load": np.random.randint(5, 80, 500),
        "distance": np.random.randint(1, 200, 500)
    })
    # 5: Healthcare (mixed)
    demo_sets["healthcare.csv"] = pd.DataFrame({
        "age": np.random.randint(18, 90, 700),
        "bmi": np.random.normal(27, 4, 700),
        "sex": np.random.choice(["M", "F"], 700),
        "diabetic": np.random.choice([0, 1], 700, p=[0.7, 0.3])
    })
    # 6: Manufacturing quality
    demo_sets["manufacturing.csv"] = pd.DataFrame({
        "temp": np.random.normal(70, 5, 900),
        "pressure": np.random.normal(5, 0.8, 900),
        "defect": np.random.choice([0, 1], 900, p=[0.9, 0.1])
    })
    # 7: Energy grid
    demo_sets["energy_grid.csv"] = pd.DataFrame({
        "load": np.random.normal(300, 30, 800),
        "supply": np.random.normal(310, 25, 800),
        "price_next": np.random.normal(50, 8, 800)
    })
    # 8: Marketing
    demo_sets["marketing.csv"] = pd.DataFrame({
        "impressions": np.random.randint(1000, 20000, 700),
        "clicks": np.random.binomial(100, 0.05, 700),
        "channel": np.random.choice(["search", "social", "email"], 700),
        "converted": np.random.choice([0, 1], 700, p=[0.85, 0.15])
    })
    # 9: Telecom churn
    demo_sets["telecom_churn.csv"] = pd.DataFrame({
        "tenure": np.random.randint(1, 72, 1000),
        "plan": np.random.choice(["A", "B", "C"], 1000),
        "calls": np.random.poisson(3, 1000),
        "churn": np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })
    # 10: Supply chain costs
    demo_sets["supply_chain.csv"] = pd.DataFrame({
        "distance": np.random.randint(10, 2000, 600),
        "weight": np.random.randint(1, 150, 600),
        "cost": np.random.normal(1200, 300, 600)
    })
    options = list(demo_sets.keys())
else:
    options = [f.name for f in uploads]

st.sidebar.caption("Tip: add your own 10 sector CSVs; the app adapts automatically.")
selected_name = st.sidebar.radio("Choose a dataset", options, index=0)

if uploads:
    # load selected file
    buf = None
    for f in uploads:
        if f.name == selected_name:
            buf = f
            break
    df_raw = pd.read_csv(buf)
else:
    df_raw = demo_sets[selected_name]

st.sidebar.markdown("---")
st.sidebar.write("**Target column (optional)**")
cols_plus_none = ["— none —"] + list(df_raw.columns)
target_col = st.sidebar.selectbox("Supervised learning target:", cols_plus_none, index=0)

task_guess = infer_task(df_raw, None if target_col == "— none —" else target_col)
task = st.sidebar.selectbox("Task type", ["classification", "regression", "timeseries", "unsupervised"], index=["classification","regression","timeseries","unsupervised"].index(task_guess))
st.sidebar.markdown("---")
st.sidebar.write("**Rows to preview**")
preview_n = st.sidebar.slider("Head rows", 5, 50, 10, step=1)

# ============================================================
# Main — Overview
# ============================================================
st.title("🔮 Future AI Models Lab (2025–2050)")
st.write(f"**Dataset:** `{selected_name}` • **Rows:** {len(df_raw)} • **Columns:** {df_raw.shape[1]} • **Task:** `{task}`")

st.subheader("Data preview")
st.dataframe(df_raw.head(preview_n), use_container_width=True)

with st.expander("Basic profiling", expanded=False):
    st.write("Column types:")
    dtypes = pd.DataFrame(df_raw.dtypes, columns=["dtype"])
    st.dataframe(dtypes)
    st.write("Missing values:")
    st.dataframe(df_raw.isna().sum().to_frame("missing"))
    # simple pairwise plot for first 4 numeric cols
    num_cols_preview = [c for c in df_raw.columns if is_numeric(df_raw[c])][:4]
    if len(num_cols_preview) >= 2:
        st.plotly_chart(px.scatter_matrix(df_raw[num_cols_preview], title="Numeric scatter matrix (first 4)"),
                        use_container_width=True)

st.markdown("---")

# ============================================================
# Run all 10 models (lightweight surrogates for demo)
# ============================================================
tabs = st.tabs([
    "1 Neuro-Symbolic", "2 Foundation 2.0", "3 Quantum ML", "4 SNN (Spiking)",
    "5 GNN", "6 PINNs / Energy", "7 World Models", "8 Continual", "9 Swarm", "10 Causal AI"
])

metrics = {}

with tabs[0]:
    res = run_neurosymbolic(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["Neuro-Symbolic"] = res or {"score": 0.0}

with tabs[1]:
    res = run_foundation(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["Foundation 2.0"] = res or {"score": 0.0}

with tabs[2]:
    res = run_quantum(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["Quantum ML"] = res or {"score": 0.0}

with tabs[3]:
    res = run_snn(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["SNN (Spiking)"] = res or {"score": 0.0}

with tabs[4]:
    res = run_gnn(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["GNN"] = res or {"score": 0.0}

with tabs[5]:
    res = run_pinn(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["PINNs / Energy-Based"] = res or {"score": 0.0}

with tabs[6]:
    res = run_world(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["World Models"] = res or {"score": 0.0}

with tabs[7]:
    res = run_continual(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["Continual"] = res or {"score": 0.0}

with tabs[8]:
    res = run_swarm(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["Swarm Intelligence"] = res or {"score": 0.0}

with tabs[9]:
    res = run_causal(df_raw, task, None if target_col == "— none —" else target_col)
    metrics["Causal AI"] = res or {"score": 0.0}

st.markdown("---")

# ============================================================
# Real-Time Agent Recommendation
# ============================================================
st.header("🤖 Real-Time Agent: Best Model for this Dataset")
best, score_map, rationale = agent_recommend(task, metrics)

score_df = pd.DataFrame([score_map]).T.reset_index()
score_df.columns = ["Model", "Agent Score (normalized + priors)"]
st.dataframe(score_df.sort_values(by=score_df.columns[1], ascending=False), use_container_width=True)
st.success(rationale)
st.plotly_chart(px.bar(score_df, x="Model", y="Agent Score (normalized + priors)", title="Agent scores"), use_container_width=True)

with st.expander("Raw model outputs (for auditability)"):
    st.json(metrics)

st.caption("✅ Production note: swap surrogates with your enterprise implementations (neuro-symbolic reasoners, graph DL, causal discovery, PINNs, QML backends). The agent logic is modular and can ingest any metrics.")
