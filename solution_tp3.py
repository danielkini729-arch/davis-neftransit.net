"""
TP — Semaine 6 : Prédiction du risque de maladie cardiaque
Solution complète : EDA → Préprocessing → 4 modèles → Métriques → Visualisations → Sauvegarde
Sélection du meilleur modèle basée sur la VALIDATION CROISÉE (5-fold, F1).
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, learning_curve,
    train_test_split, validation_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
RANDOM_STATE = 42

OUT = Path("resultats")
OUT.mkdir(exist_ok=True)

# ============================================================
# ÉTAPE 1 — CHARGEMENT
# ============================================================
print("=" * 60); print("ÉTAPE 1 — Chargement"); print("=" * 60)
df = pd.read_csv("heart_disease.csv")
print(f"Shape : {df.shape}  |  NaN : {df.isna().sum().sum()}")
print(df["target"].value_counts().to_dict())

# ============================================================
# ÉTAPE 2 — EDA — 01_exploration.png
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 2 — EDA"); print("=" * 60)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploration du dataset Heart Disease", fontsize=16, fontweight="bold")

ax = axes[0, 0]
df["target"].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"], ax=ax, edgecolor="black")
ax.set_title("Répartition de la cible"); ax.set_xticklabels(["Sain", "Malade"], rotation=0)

ax = axes[0, 1]
for t, color, lbl in [(0, "#2ecc71", "Sain"), (1, "#e74c3c", "Malade")]:
    ax.hist(df[df["target"] == t]["age"], bins=20, alpha=0.6, color=color, label=lbl, edgecolor="black")
ax.set_title("Distribution de l'âge"); ax.set_xlabel("Âge"); ax.legend()

ax = axes[0, 2]
pd.crosstab(df["sex"], df["target"]).plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black")
ax.set_title("Sexe vs maladie"); ax.set_xticklabels(["Femme", "Homme"], rotation=0)
ax.legend(["Sain", "Malade"])

ax = axes[1, 0]; sns.boxplot(x="target", y="thalach", data=df, ax=ax, palette=["#2ecc71", "#e74c3c"])
ax.set_title("Fréquence cardiaque max"); ax.set_xticklabels(["Sain", "Malade"])

ax = axes[1, 1]; sns.boxplot(x="target", y="chol", data=df, ax=ax, palette=["#2ecc71", "#e74c3c"])
ax.set_title("Cholestérol"); ax.set_xticklabels(["Sain", "Malade"])

ax = axes[1, 2]
sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax, cbar=True, xticklabels=True, yticklabels=True)
ax.set_title("Matrice de corrélation")

plt.tight_layout()
plt.savefig(OUT / "01_exploration.png", bbox_inches="tight"); plt.close()
print("✓ 01_exploration.png")

# ============================================================
# ÉTAPE 3 — PRÉPROCESSING ET SPLIT
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 3 — Préprocessing"); print("=" * 60)
X = df.drop("target", axis=1); y = df["target"]
feature_names = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# ÉTAPE 4 — ENTRAÎNEMENT DES 4 MODÈLES
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 4 — 4 modèles"); print("=" * 60)
models = {
    "Régression Logistique": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
    "Arbre de Décision":     DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "K-NN":                  KNeighborsClassifier(n_neighbors=7),
    "Random Forest":         RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
}
needs_scaling = {"Régression Logistique", "K-NN"}

results = {}
for name, model in models.items():
    Xtr = X_train_scaled if name in needs_scaling else X_train.values
    Xte = X_test_scaled  if name in needs_scaling else X_test.values
    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]
    results[name] = dict(
        model=model, y_pred=y_pred, y_proba=y_proba,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
    )
    print(f"  {name:25s} | Acc={results[name]['accuracy']:.3f} | "
          f"Recall={results[name]['recall']:.3f} | F1={results[name]['f1']:.3f} | AUC={results[name]['roc_auc']:.3f}")

metrics_df = pd.DataFrame({
    n: {k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    for n, r in results.items()
}).T
print("\nRécap :\n", metrics_df.round(3))

# ============================================================
# ÉTAPE 5 — Matrices de confusion (02)
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 5 — Matrices de confusion"); print("=" * 60)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Matrices de confusion — 4 modèles", fontsize=16, fontweight="bold")
for ax, (n, r) in zip(axes.flat, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Sain", "Malade"], yticklabels=["Sain", "Malade"],
                annot_kws={"size": 14})
    ax.set_title(f"{n}\nRecall={r['recall']:.3f} | F1={r['f1']:.3f}")
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Réalité")
plt.tight_layout(); plt.savefig(OUT / "02_matrices_confusion.png", bbox_inches="tight"); plt.close()
print("✓ 02_matrices_confusion.png")

# ============================================================
# ÉTAPE 6 — Courbes ROC (03)
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 6 — Courbes ROC"); print("=" * 60)
plt.figure(figsize=(9, 7))
for n, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    plt.plot(fpr, tpr, lw=2, label=f"{n} (AUC = {r['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire (AUC=0.5)")
plt.xlabel("Taux de Faux Positifs (FPR)"); plt.ylabel("Taux de Vrais Positifs (Recall)")
plt.title("Courbes ROC — comparaison des 4 modèles", fontweight="bold")
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "03_courbes_ROC.png", bbox_inches="tight"); plt.close()
print("✓ 03_courbes_ROC.png")

# ============================================================
# ÉTAPE 7 — VALIDATION CROISÉE (06)  →  SÉLECTION DU MEILLEUR MODÈLE
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 7 — Validation croisée (5-fold F1)"); print("=" * 60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}
for name, model in models.items():
    Xall = StandardScaler().fit_transform(X) if name in needs_scaling else X.values
    scores = cross_val_score(model, Xall, y, cv=cv, scoring="f1")
    cv_scores[name] = scores
    print(f"  {name:25s} | F1 = {scores.mean():.3f} ± {scores.std():.3f}")

plt.figure(figsize=(10, 6))
plt.boxplot(cv_scores.values(), labels=cv_scores.keys(), patch_artist=True,
            boxprops=dict(facecolor="#3498db", alpha=0.6))
plt.ylabel("F1-Score")
plt.title("Stabilité des modèles — Validation croisée 5-fold (F1)", fontweight="bold")
plt.xticks(rotation=15); plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(OUT / "06_validation_croisee.png", bbox_inches="tight"); plt.close()
print("✓ 06_validation_croisee.png")

# ★ SÉLECTION FINALE basée sur la CV
best_name = max(cv_scores, key=lambda n: cv_scores[n].mean())
best = results[best_name]
print(f"\n★ MEILLEUR MODÈLE (CV F1) : {best_name}  "
      f"({cv_scores[best_name].mean():.3f} ± {cv_scores[best_name].std():.3f})")

# ============================================================
# ÉTAPE 8 — Courbe PR du meilleur modèle (04)
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 8 — Courbe Precision-Recall"); print("=" * 60)
prec, rec, _ = precision_recall_curve(y_test, best["y_proba"])
ap = average_precision_score(y_test, best["y_proba"])

plt.figure(figsize=(9, 7))
plt.plot(rec, prec, lw=2, color="#3498db", label=f"{best_name} (AP = {ap:.3f})")
plt.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Baseline = {y_test.mean():.2f}")
plt.xlabel("Recall (Sensibilité)"); plt.ylabel("Precision")
plt.title(f"Courbe Precision-Recall — {best_name}", fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "04_courbe_PR.png", bbox_inches="tight"); plt.close()
print("✓ 04_courbe_PR.png")

# ============================================================
# ÉTAPE 9 — Impact du seuil (05)
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 9 — Impact du seuil"); print("=" * 60)
thresholds = np.linspace(0.05, 0.95, 91)
prec_l, rec_l, f1_l = [], [], []
for t in thresholds:
    pred_t = (best["y_proba"] >= t).astype(int)
    prec_l.append(precision_score(y_test, pred_t, zero_division=0) if pred_t.sum() else 1.0)
    rec_l.append(recall_score(y_test, pred_t, zero_division=0))
    f1_l.append(f1_score(y_test, pred_t, zero_division=0))
best_thr = float(thresholds[int(np.argmax(f1_l))])

plt.figure(figsize=(10, 6))
plt.plot(thresholds, prec_l, label="Precision", color="#2ecc71", lw=2)
plt.plot(thresholds, rec_l,  label="Recall",    color="#e74c3c", lw=2)
plt.plot(thresholds, f1_l,   label="F1-Score",  color="#3498db", lw=2)
plt.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="Seuil défaut = 0.5")
plt.axvline(best_thr, color="purple", linestyle=":", lw=2, label=f"F1-max = {best_thr:.2f}")
plt.xlabel("Seuil de décision"); plt.ylabel("Score")
plt.title(f"Impact du seuil sur Precision/Recall/F1 — {best_name}", fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "05_impact_seuil.png", bbox_inches="tight"); plt.close()
print(f"✓ 05_impact_seuil.png (seuil optimal F1 = {best_thr:.2f})")

# ============================================================
# ÉTAPE 10 — Courbe de validation (07) — sur Random Forest n_estimators
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 10 — Courbe de validation"); print("=" * 60)
param_range = [10, 25, 50, 100, 150, 200, 300, 500]
tr_sc, va_sc = validation_curve(
    RandomForestClassifier(max_depth=6, random_state=RANDOM_STATE),
    X.values, y, param_name="n_estimators", param_range=param_range,
    cv=cv, scoring="f1", n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, tr_sc.mean(axis=1), "o-", color="#2ecc71", label="Entraînement")
plt.fill_between(param_range, tr_sc.mean(axis=1) - tr_sc.std(axis=1),
                 tr_sc.mean(axis=1) + tr_sc.std(axis=1), alpha=0.2, color="#2ecc71")
plt.plot(param_range, va_sc.mean(axis=1), "o-", color="#e74c3c", label="Validation")
plt.fill_between(param_range, va_sc.mean(axis=1) - va_sc.std(axis=1),
                 va_sc.mean(axis=1) + va_sc.std(axis=1), alpha=0.2, color="#e74c3c")
plt.xlabel("n_estimators (Random Forest)"); plt.ylabel("F1-Score")
plt.title("Courbe de validation — Random Forest", fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "07_courbe_validation.png", bbox_inches="tight"); plt.close()
print("✓ 07_courbe_validation.png")

# ============================================================
# ÉTAPE 11 — Courbe d'apprentissage (08) — sur le meilleur modèle
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 11 — Courbe d'apprentissage"); print("=" * 60)
Xall_lc = StandardScaler().fit_transform(X) if best_name in needs_scaling else X.values
best_clone = type(best["model"])(**best["model"].get_params())
ts, tr_lc, va_lc = learning_curve(
    best_clone, Xall_lc, y, cv=cv, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1, random_state=RANDOM_STATE)

plt.figure(figsize=(10, 6))
plt.plot(ts, tr_lc.mean(axis=1), "o-", color="#2ecc71", label="Entraînement")
plt.fill_between(ts, tr_lc.mean(axis=1) - tr_lc.std(axis=1),
                 tr_lc.mean(axis=1) + tr_lc.std(axis=1), alpha=0.2, color="#2ecc71")
plt.plot(ts, va_lc.mean(axis=1), "o-", color="#e74c3c", label="Validation")
plt.fill_between(ts, va_lc.mean(axis=1) - va_lc.std(axis=1),
                 va_lc.mean(axis=1) + va_lc.std(axis=1), alpha=0.2, color="#e74c3c")
plt.xlabel("Taille de l'ensemble d'entraînement"); plt.ylabel("F1-Score")
plt.title(f"Courbe d'apprentissage — {best_name}", fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "08_courbe_apprentissage.png", bbox_inches="tight"); plt.close()
print("✓ 08_courbe_apprentissage.png")

# ============================================================
# ÉTAPE 12 — Importance des features (09)
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 12 — Importance des variables"); print("=" * 60)
rf_imp = results["Random Forest"]["model"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": rf_imp})\
          .sort_values("importance", ascending=True)
lr_coefs = pd.DataFrame({"feature": feature_names,
                         "coef": results["Régression Logistique"]["model"].coef_[0]})\
             .sort_values("coef")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].barh(fi_df["feature"], fi_df["importance"], color="#3498db", edgecolor="black")
axes[0].set_title("Random Forest — Importance des variables", fontweight="bold")
axes[0].set_xlabel("Importance")

colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in lr_coefs["coef"]]
axes[1].barh(lr_coefs["feature"], lr_coefs["coef"], color=colors, edgecolor="black")
axes[1].set_title("Régression Logistique — Coefficients", fontweight="bold")
axes[1].set_xlabel("Coefficient (+ ⇒ ↑ risque, − ⇒ ↓ risque)")
axes[1].axvline(0, color="black", lw=0.8)
plt.tight_layout(); plt.savefig(OUT / "09_importance_features.png", bbox_inches="tight"); plt.close()
print("✓ 09_importance_features.png")
print("Top 5 features (RF):")
print(fi_df.sort_values("importance", ascending=False).head().to_string(index=False))

# ============================================================
# ÉTAPE 13 — SAUVEGARDE (modèle, scaler, rapport JSON)
# ============================================================
print("\n" + "=" * 60); print("ÉTAPE 13 — Sauvegarde"); print("=" * 60)

# Ré-entraîne le meilleur modèle sur TOUT le dataset pour la production
final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X)
final_model = type(best["model"])(**best["model"].get_params())
Xfit = X_full_scaled if best_name in needs_scaling else X.values
final_model.fit(Xfit, y)

joblib.dump(final_model,  OUT / "meilleur_modele.pkl")
joblib.dump(final_scaler, OUT / "scaler.pkl")
print(f"✓ meilleur_modele.pkl  ({best_name})")
print("✓ scaler.pkl")

rapport = {
    "date_entrainement": datetime.now().isoformat(timespec="seconds"),
    "meilleur_modele": best_name,
    "modele_classe": type(final_model).__name__,
    "selection_critere": "F1-Score moyen — validation croisée stratifiée 5-fold",
    "hyperparametres": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                        for k, v in final_model.get_params().items()},
    "necessite_scaling": best_name in needs_scaling,
    "metriques_test_set": {
        "accuracy":  round(results[best_name]["accuracy"], 4),
        "precision": round(results[best_name]["precision"], 4),
        "recall":    round(results[best_name]["recall"], 4),
        "f1":        round(results[best_name]["f1"], 4),
        "roc_auc":   round(results[best_name]["roc_auc"], 4),
    },
    "validation_croisee_F1": {
        n: {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
        for n, s in cv_scores.items()
    },
    "seuil_optimal_F1": round(best_thr, 3),
    "comparaison_tous_modeles_test": {
        n: {k: round(r[k], 4) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
        for n, r in results.items()
    },
    "top_features": fi_df.sort_values("importance", ascending=False).head(5)
                         .to_dict(orient="records"),
    "recommandation": (
        f"Le modèle '{best_name}' est recommandé sur la base de la VALIDATION CROISÉE 5-fold "
        "(critère plus robuste qu'un simple test set sur 303 patients). "
        "Dans un contexte médical, le RECALL (sensibilité) est la métrique prioritaire : "
        "il faut minimiser les FAUX NÉGATIFS — patients réellement malades classés à tort comme sains, "
        "ce qui les priverait de la coronarographie. L'Accuracy est trompeuse car elle traite à parts "
        "égales les deux types d'erreur, alors qu'un FN est ici beaucoup plus grave qu'un FP. "
        f"En production, on peut abaisser le seuil de décision à ~{best_thr:.2f} pour augmenter le recall."
    ),
}
with open(OUT / "rapport_modele.json", "w", encoding="utf-8") as f:
    json.dump(rapport, f, indent=2, ensure_ascii=False)
print("✓ rapport_modele.json")

print("\n" + "=" * 60)
print("✅ TP TERMINÉ — dossier 'resultats/'")
print("=" * 60)
print(f"Meilleur modèle      : {best_name}")
print(f"CV F1 (5-fold)       : {cv_scores[best_name].mean():.3f} ± {cv_scores[best_name].std():.3f}")
print(f"Test — Recall        : {results[best_name]['recall']:.3f}")
print(f"Test — F1            : {results[best_name]['f1']:.3f}")
print(f"Test — ROC AUC       : {results[best_name]['roc_auc']:.3f}")
print(f"Seuil optimal (F1)   : {best_thr:.2f}")
