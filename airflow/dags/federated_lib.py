"""Aprendizaje federado (FedAvg) para la prediccion de ACV: simulacion de hospitales.

Cada hospital (cliente) entrena una regresion logistica sobre SUS pacientes y envia solo
parametros (pesos); el servidor los promedia. Los datos clinicos nunca salen del cliente.

Lo unico que se comparte ademas de los pesos son estadisticos agregados (cantidad, suma y
suma de cuadrados de cada variable, y conteo de clases) para estandarizar y balancear de
forma consistente. Esto es lo que habria que proteger con agregacion segura en un despliegue real.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Mismas variables que usa el modelo de produccion (ver train_stroke_model.py)
FEATURES = ["age", "hypertension", "avg_glucose_level", "has_risk_factors", "is_female"]
RECALL_OBJETIVO = 0.90   # misma regla de umbral que el modelo de produccion


# ----------------------------------------------------------------------------- datos
def ingenieria_de_variables(df: pd.DataFrame):
    """Replica el feature engineering de produccion. Devuelve (X con todas las columnas, y)."""
    df = df.drop(columns=["id"]).copy()
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 70, 120],
                             labels=["Joven", "Adulto", "Mayor", "Anciano"]).astype(str)
    df["avg_glucose_level"] = np.clip(df["avg_glucose_level"], 50, 300)
    df["has_risk_factors"] = np.where((df["hypertension"] == 1) | (df["heart_disease"] == 1), 1, 0)
    df["is_female"] = np.where(df["gender"] == "Female", 1, 0)
    df = df.drop(columns=["gender"])
    df["work_type"] = df["work_type"].replace({"Never_worked": "children"})
    y = df["stroke"].to_numpy()
    return df.drop(columns=["stroke"]), y


def separar(X_df, y, seed):
    """train / validation (para elegir el umbral) / test, estratificados."""
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    i_tr, i_va = train_test_split(i_tr, test_size=0.2, stratify=y[i_tr], random_state=seed)
    return i_tr, i_va, i_te


# ----------------------------------------------------------------------------- particiones
def particionar_iid(idx, y, k, rng):
    """Cada hospital recibe una muestra aleatoria de pacientes (misma distribucion)."""
    return [p for p in np.array_split(rng.permutation(idx), k)]


def particionar_por_edad(idx, edades, k):
    """non-IID: cada hospital atiende un tramo de edad (pediatrico ... geriatrico).
    Cambian tanto la distribucion de las variables como la prevalencia de ACV."""
    orden = idx[np.argsort(edades[idx], kind="stable")]
    return [p for p in np.array_split(orden, k)]


# ----------------------------------------------------------------------------- modelo
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def entrenar_local(w, b, X, y, epocas, lr, l2, peso_pos):
    """Descenso de gradiente sobre la perdida logistica ponderada por clase."""
    w, b = w.copy(), float(b)
    sw = np.where(y == 1, peso_pos, 1.0)
    for _ in range(epocas):
        err = sw * (sigmoide(X @ w + b) - y)
        w -= lr * (X.T @ err / sw.sum() + l2 * w)
        b -= lr * err.sum() / sw.sum()
    return w, b


def estadisticos_globales(partes):
    """Media y desvio global a partir de (n, suma, suma de cuadrados) de cada cliente."""
    n = sum(len(Xc) for Xc, _ in partes)
    s = sum(Xc.sum(axis=0) for Xc, _ in partes)
    ss = sum((Xc ** 2).sum(axis=0) for Xc, _ in partes)
    media = s / n
    desvio = np.sqrt(np.maximum(ss / n - media ** 2, 1e-12))
    pos = sum(int(yc.sum()) for _, yc in partes)
    return media, desvio, (n - pos) / max(pos, 1)       # ultimo: peso de la clase positiva


def fedavg(partes, media, desvio, peso_pos, rondas, epocas, lr, l2, eval_fn=None,
           clip=None, ruido=0.0, rng=None):
    """FedAvg: en cada ronda todos los hospitales entrenan localmente y el servidor promedia.

    Si `clip` esta definido se aplica DP a nivel de cliente: cada actualizacion se recorta a
    norma L2 <= clip, se promedian con pesos iguales y se agrega ruido gaussiano de desvio
    ruido*clip/K. No se calcula un epsilon formal (no hay contador de privacidad).
    """
    Xs = [((Xc - media) / desvio, yc) for Xc, yc in partes]
    n_k = np.array([len(yc) for _, yc in Xs], dtype=float)
    w, b = np.zeros(len(media)), 0.0
    hist = []
    for _ in range(rondas):
        deltas = []
        for Xc, yc in Xs:
            wk, bk = entrenar_local(w, b, Xc, yc, epocas, lr, l2, peso_pos)
            d = np.append(wk - w, bk - b)
            if clip is not None:
                d = d * min(1.0, clip / (np.linalg.norm(d) + 1e-12))
            deltas.append(d)
        deltas = np.array(deltas)
        if clip is None:
            prom = (n_k / n_k.sum()) @ deltas                       # promedio ponderado por n_k
        else:
            prom = deltas.mean(axis=0) + rng.normal(0, ruido * clip / len(Xs), size=deltas.shape[1])
        w, b = w + prom[:-1], b + prom[-1]
        if eval_fn:
            hist.append(eval_fn(w, b))
    return w, b, hist


def entrenar_centralizado(partes, media, desvio, peso_pos, epocas, lr, l2):
    """Linea de base: mismos datos reunidos en un solo lugar."""
    X = np.vstack([Xc for Xc, _ in partes])
    y = np.concatenate([yc for _, yc in partes])
    return entrenar_local(np.zeros(X.shape[1]), 0.0, (X - media) / desvio, y, epocas, lr, l2, peso_pos)


# ----------------------------------------------------------------------------- metricas
def elegir_umbral(y_val, p_val, recall_objetivo=RECALL_OBJETIVO):
    """Mayor umbral con recall >= objetivo en validacion (misma regla que produccion)."""
    _, rec, thr = precision_recall_curve(y_val, p_val)
    cand = thr[rec[:-1] >= recall_objetivo]
    return float(cand.max()) if len(cand) else 0.5


def metricas(y, p, umbral):
    pred = (p >= umbral).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def predecir(w, b, X, media, desvio):
    return sigmoide(((X - media) / desvio) @ w + b)


# ----------------------------------------------------------------------------- experimento
CONFIG = dict(k=5, rondas=40, epocas=5, lr=0.5, l2=1e-3, clip=0.5, ruidos=(0.5, 1.0, 2.0),
              sens_epocas=(1, 20, 50, 200, 1000))   # barrido de epocas locales (client drift)


def correr_seed(X_df, y, seed, cfg=CONFIG, con_historia=False):
    """Una repeticion completa del experimento. Devuelve {configuracion: metricas de test}."""
    X = X_df[FEATURES].astype(float).to_numpy()
    edades = X_df["age"].to_numpy(dtype=float)
    i_tr, i_va, i_te = separar(X_df, y, seed)
    rng = np.random.RandomState(seed)

    def con_datos(particion):
        return [(X[p], y[p]) for p in particion]

    def evaluador(media, desvio):
        return lambda w, b: float(roc_auc_score(y[i_va], predecir(w, b, X[i_va], media, desvio)))

    def evaluar_test(w, b, media, desvio):
        thr = elegir_umbral(y[i_va], predecir(w, b, X[i_va], media, desvio))
        m = metricas(y[i_te], predecir(w, b, X[i_te], media, desvio), thr)
        m["umbral"] = thr
        return m

    out, historia, modelos = {}, {}, {}
    escenarios = {
        "federado_iid": con_datos(particionar_iid(i_tr, y, cfg["k"], rng)),
        "federado_non_iid": con_datos(particionar_por_edad(i_tr, edades, cfg["k"])),
    }
    todo_train = [(X[i_tr], y[i_tr])]
    media_c, desvio_c, pp_c = estadisticos_globales(todo_train)
    wc, bc = entrenar_centralizado(todo_train, media_c, desvio_c, pp_c,
                                   cfg["rondas"] * cfg["epocas"], cfg["lr"], cfg["l2"])
    out["centralizado"] = evaluar_test(wc, bc, media_c, desvio_c)
    modelos["centralizado"] = (wc, bc, media_c, desvio_c, out["centralizado"]["umbral"])
    # referencia en VALIDACION (no en test) para compararla con las curvas federadas por ronda
    auc_val_central = float(roc_auc_score(y[i_va], predecir(wc, bc, X[i_va], media_c, desvio_c)))
    historia["_centralizado_val"] = [auc_val_central] * cfg["rondas"]

    for nombre, partes in escenarios.items():
        media, desvio, pp = estadisticos_globales(partes)
        w, b, hist = fedavg(partes, media, desvio, pp, cfg["rondas"], cfg["epocas"], cfg["lr"],
                            cfg["l2"], eval_fn=evaluador(media, desvio))
        out[nombre] = evaluar_test(w, b, media, desvio)
        modelos[nombre] = (w, b, media, desvio, out[nombre]["umbral"])
        historia[nombre] = hist
        # prevalencia por hospital, para documentar la heterogeneidad
        out[nombre]["prevalencia_por_hospital"] = [float(yc.mean()) for _, yc in partes]

    # privacidad: DP a nivel de cliente sobre el escenario non-IID (el realista)
    partes = escenarios["federado_non_iid"]
    media, desvio, pp = estadisticos_globales(partes)
    for z in cfg["ruidos"]:
        w, b, _ = fedavg(partes, media, desvio, pp, cfg["rondas"], cfg["epocas"], cfg["lr"], cfg["l2"],
                         clip=cfg["clip"], ruido=z, rng=rng)
        out[f"federado_non_iid_dp_z{z}"] = evaluar_test(w, b, media, desvio)

    # sensibilidad: mas epocas locales por ronda (con menos rondas, mismo computo total)
    for e in cfg["sens_epocas"]:
        r = max(1, cfg["rondas"] * cfg["epocas"] // e)
        for nombre, partes in escenarios.items():
            media, desvio, pp = estadisticos_globales(partes)
            w, b, _ = fedavg(partes, media, desvio, pp, r, e, cfg["lr"], cfg["l2"])
            out[f"sens_E{e}_" + nombre.replace("federado_", "")] = evaluar_test(w, b, media, desvio)

    return (out, historia, modelos) if con_historia else out


def correr_experimento(df, seeds=5, cfg=CONFIG):
    """Repite el experimento con varias semillas. Devuelve (resumen, historia_seed0, modelos_seed0, X_df, y)."""
    X_df, y = ingenieria_de_variables(df)
    corridas, historia, modelos = [], None, None
    for s in range(seeds):
        if s == 0:
            r, historia, modelos = correr_seed(X_df, y, s, cfg, con_historia=True)
        else:
            r = correr_seed(X_df, y, s, cfg)
        corridas.append(r)

    resumen = {}
    for conf in corridas[0]:
        resumen[conf] = {}
        for m in ("roc_auc", "pr_auc", "recall", "precision", "f1", "umbral"):
            v = np.array([c[conf][m] for c in corridas])
            resumen[conf][m] = {"media": float(v.mean()), "desvio": float(v.std())}
        if "prevalencia_por_hospital" in corridas[0][conf]:
            resumen[conf]["prevalencia_por_hospital"] = corridas[0][conf]["prevalencia_por_hospital"]
    # diferencia pareada (misma particion train/test): AUC IID - AUC non-IID, por semilla y por epocas locales.
    # Las desviaciones entre semillas mezclan la variacion de la particion con el efecto; esto aisla el efecto.
    resumen["_pareadas"] = {
        str(e): [c[f"sens_E{e}_iid"]["roc_auc"] - c[f"sens_E{e}_non_iid"]["roc_auc"] for c in corridas]
        for e in cfg["sens_epocas"]
    }
    return resumen, historia, modelos, X_df, y


# ----------------------------------------------------------------------------- modelo servible
def construir_pipeline_sklearn(X_df, y, w, b, media, desvio):
    """Empaqueta el modelo federado como un Pipeline de sklearn compatible con la API.

    Se ajusta una vez para inicializar la estructura y luego se sobreescriben los parametros
    con los obtenidos por FedAvg (el pipeline no vuelve a ver datos de entrenamiento).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("preprocessor", ColumnTransformer([("num", StandardScaler(), FEATURES)])),
        ("model", LogisticRegression()),
    ])
    pipe.fit(X_df, y)
    sc = pipe.named_steps["preprocessor"].named_transformers_["num"]
    sc.mean_, sc.scale_, sc.var_ = media.copy(), desvio.copy(), desvio ** 2
    lr = pipe.named_steps["model"]
    lr.coef_, lr.intercept_ = w.reshape(1, -1).copy(), np.array([b])

    # control: el pipeline debe dar lo mismo que el calculo numpy
    esperado = predecir(w, b, X_df[FEATURES].astype(float).to_numpy(), media, desvio)
    obtenido = pipe.predict_proba(X_df)[:, 1]
    assert np.allclose(esperado, obtenido, atol=1e-9), "el pipeline empaquetado no coincide con FedAvg"
    return pipe
