# HPC_CML_DNN.py — DNN: Δ-learning + Direct DFT, ALL & Q9 variants, PDF parity plots (final)

import os, json, warnings, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import optuna as opt
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow / Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TF info logs
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras import backend as K

warnings.filterwarnings("ignore")

# -----------------------------
# Matplotlib "paper-like" defaults for PDFs
# -----------------------------
plt.rcParams.update({
    "pdf.fonttype": 42,  # embed TrueType
    "ps.fonttype": 42,
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
})

def parity_plot_pdf(y_true, y_pred, out_pdf, xlabel, ylabel, annotate=None, title=None):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # ~single-column figure
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    ax.scatter(y_pred, y_true, s=14, alpha=0.7, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.25)  # y=x
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, pad=6)
    if annotate:
        ax.text(0.05, 0.95, annotate, transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--data_dir",   required=True, type=str)

parser.add_argument("--model", default="dnn")                     # for CLI consistency
parser.add_argument("--target_col", default="ae_delta")           # 'ae_delta' or 'dh_delta'
parser.add_argument("--tuner", default="optuna")                  # for CLI consistency
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--n_jobs", type=int, default=6)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--cv_folds", type=int, default=5)
parser.add_argument("--cv_repeats", type=int, default=2)
parser.add_argument("--feature_selection", default="corr90")      # Xabier-style (train-only corr drop)
parser.add_argument("--pca_components", default="none")           # not used for DNN (kept for CLI parity)
parser.add_argument("--feature_set", nargs="*", default=None)     # optional explicit subset
parser.add_argument("--run_variants", default="all", choices=["all","q9","both"],
                    help="Run on full feature set, Q9 subset, or both in one job.")
args = parser.parse_args()

# -----------------------------
# Setup / seeds / threads
# -----------------------------
os.makedirs(args.output_dir, exist_ok=True)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[var] = str(args.n_jobs)
try:
    tf.config.threading.set_intra_op_parallelism_threads(args.n_jobs)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass
K.set_floatx("float32")

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)

# -----------------------------
# Data
# -----------------------------
train_csv = Path(args.data_dir) / "train_df_new.csv"
test_csv  = Path(args.data_dir) / "test_df_new.csv"
train_df  = pd.read_csv(train_csv, index_col=0)
test_df   = pd.read_csv(test_csv,  index_col=0)

# Base features/targets
X_train_base = train_df.drop(columns=["dh_delta","ae_delta","dh_dft","ae_dft"])
X_test_base  = test_df.drop(columns=["dh_delta","ae_delta","dh_dft","ae_dft"])
y_train_df   = train_df[["dh_delta","ae_delta"]]
y_test_df    = test_df[["dh_delta","ae_delta"]]

# Variant setup
Q9_LIST = ["exp_mopac","AE_mopac","Par_n_Pople","Mul","ch_f","DH_Mopac","ZPE_TS_R","Freq","ZPE_P_R"]

if args.feature_set:
    # If user passed explicit list, use only that in ALL variant
    feat_subset = [c for c in args.feature_set if c in X_train_base.columns]
    if feat_subset:
        X_train_base = X_train_base[feat_subset].copy()
        X_test_base  = X_test_base[feat_subset].copy()
        log(f"Using explicit feature_set ({len(feat_subset)} features).")
    else:
        log("WARNING: none of the requested feature_set names found; using all features.")

VARIANTS = []
if args.run_variants in ("all","both"):
    VARIANTS.append(("ALL", X_train_base.columns.tolist()))
if args.run_variants in ("q9","both"):
    q9_cols = [c for c in Q9_LIST if c in X_train_base.columns]
    if not q9_cols:
        raise ValueError("Q9 feature names not found in your data columns.")
    VARIANTS.append(("Q9", q9_cols))

# -----------------------------
# Utility: model builder
# -----------------------------
def build_model(n_layers: int, units: int, dropout: float, lr: float, input_dim: int) -> tf.keras.Model:
    model = Sequential()
    model.add(Dense(units, activation="relu", input_shape=(input_dim,)))
    for _ in range(n_layers - 1):
        model.add(Dense(units, activation="relu"))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1))  # regression head
    model.compile(optimizer=Adam(learning_rate=lr), loss="mae")
    return model

# -----------------------------
# One variant runner (Δ + Direct DFT)
# -----------------------------
def run_variant(variant_name: str, feat_cols: list) -> dict:
    var_dir = Path(args.output_dir) / variant_name
    var_dir.mkdir(parents=True, exist_ok=True)

    # Slice features
    Xtr = X_train_base[feat_cols].copy()
    Xte = X_test_base[feat_cols].copy()

    # TRAIN-ONLY correlation pruning (|r| > 0.90)
    if args.feature_selection.lower() == "corr90" and Xtr.shape[1] > 1:
        corr = Xtr.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > 0.90).any()]
        Xtr_red = Xtr.drop(columns=to_drop)
        Xte_red = Xte.drop(columns=to_drop, errors="ignore")
        with open(var_dir / "dropped_corr90.txt", "w") as f:
            for c in to_drop: f.write(c + "\n")
        log(f"[{variant_name}] Corr-pruned: {Xtr.shape[1]} -> {Xtr_red.shape[1]} (dropped {len(to_drop)})")
    else:
        Xtr_red, Xte_red = Xtr, Xte
        log(f"[{variant_name}] Corr pruning skipped or not applicable.")

    n_features = Xtr_red.shape[1]

    # ---- Column mapping for Δ reconstruction ----
    tcol = args.target_col  # 'ae_delta' or 'dh_delta'
    if tcol not in y_train_df.columns:
        raise KeyError(f"target_col '{tcol}' not in y_train columns: {list(y_train_df.columns)}")

    if tcol == "ae_delta":
        dft_col  = "ae_dft"
        pm7_col  = "AE_mopac"
        delta_lbl = "ΔAE (kcal/mol)"
        dft_lbl   = "AE (kcal/mol)"
    else:
        dft_col  = "dh_dft"
        pm7_col  = "DH_Mopac"
        delta_lbl = "ΔH‡ (kcal/mol)"
        dft_lbl   = "H‡ (kcal/mol)"

    if pm7_col not in test_df.columns:
        fallbacks = [c for c in test_df.columns if "mopac" in c.lower() or "pm7" in c.lower()]
        if fallbacks:
            pm7_col = fallbacks[0]
            log(f"[{variant_name}] PM7 column not found; using fallback '{pm7_col}'")
        else:
            raise KeyError(f"PM7 base column '{pm7_col}' not found in test_df.")

    # Align targets to feature indices
    y_tr_delta = y_train_df[tcol].values.astype(np.float32)
    y_te_delta = y_test_df[tcol].values.astype(np.float32)
    y_tr_dft   = train_df.loc[Xtr_red.index, dft_col].values.astype(np.float32)
    y_te_dft   = test_df .loc[Xte_red.index, dft_col].values.astype(np.float32)

    Xtr_np = Xtr_red.values.astype(np.float32)
    Xte_np = Xte_red.values.astype(np.float32)

    # -----------------------------
    # Optuna objective (generic over y)
    # -----------------------------
    def make_objective(y_vector: np.ndarray):
        def objective(trial: opt.trial.Trial) -> float:
            n_layers      = trial.suggest_int("n_layers", 1, 4)
            units         = trial.suggest_int("units", 64, 512, step=64)
            dropout_rate  = trial.suggest_float("dropout_rate", 0.0, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128])
            epochs        = trial.suggest_int("epochs", 60, 200)

            rkf = RepeatedKFold(n_splits=args.cv_folds, n_repeats=args.cv_repeats, random_state=args.seed)

            fold_maes = []
            for fold_idx, (tr_idx, va_idx) in enumerate(rkf.split(Xtr_np), start=1):
                X_A, X_B = Xtr_np[tr_idx], Xtr_np[va_idx]
                y_A, y_B = y_vector[tr_idx], y_vector[va_idx]

                scaler = StandardScaler()
                XA_s = scaler.fit_transform(X_A)
                XB_s = scaler.transform(X_B)

                model = build_model(n_layers, units, dropout_rate, learning_rate, input_dim=n_features)
                es    = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
                rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0)

                model.fit(XA_s, y_A, validation_data=(XB_s, y_B),
                          epochs=epochs, batch_size=batch_size, verbose=0,
                          callbacks=[es, rlrop])

                preds = model.predict(XB_s, verbose=0).reshape(-1)
                fold_maes.append(mean_absolute_error(y_B, preds))

                # pruning
                trial.report(float(np.mean(fold_maes)), step=fold_idx)
                if trial.should_prune():
                    K.clear_session()
                    raise opt.TrialPruned()
                K.clear_session()

            return float(np.mean(fold_maes))
        return objective

    # -----------------------------
    # Tuning helpers
    # -----------------------------
    def tune_and_train(label: str, y_vector: np.ndarray, out_prefix: str):
        log(f"[{variant_name} | {label}] Optuna: trials={args.n_trials}, folds={args.cv_folds}x, repeats={args.cv_repeats}x")
        sampler = TPESampler(seed=args.seed, multivariate=True)
        pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=1)
        study   = opt.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(make_objective(y_vector), n_trials=args.n_trials)

        # Save study artifacts
        trials_df = study.trials_dataframe()
        trials_df.to_csv(var_dir / f"{out_prefix}_optuna_trials.csv", index=False)
        with open(var_dir / f"{out_prefix}_best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

        log(f"[{variant_name} | {label}] Best CV MAE: {study.best_value:.6f}")
        log(f"[{variant_name} | {label}] Best params: {study.best_params}")

        # Final 80/20 train/val from ALL train rows
        bp = study.best_params
        X_trA, X_vaA, y_trA, y_vaA = train_test_split(
            Xtr_np, y_vector, test_size=0.20, random_state=args.seed
        )

        scaler_final = StandardScaler()
        X_trA_s = scaler_final.fit_transform(X_trA)
        X_vaA_s = scaler_final.transform(X_vaA)
        X_te_s  = scaler_final.transform(Xte_np)

        model = build_model(
            n_layers=bp["n_layers"], units=bp["units"],
            dropout=bp["dropout_rate"], lr=bp["learning_rate"],
            input_dim=n_features
        )

        ckpt_path   = var_dir / f"{out_prefix}_best_weights.h5"
        csvlog_path = var_dir / f"{out_prefix}_train_log.csv"
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=1),
            ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
            CSVLogger(str(csvlog_path))
        ]

        model.fit(X_trA_s, y_trA, validation_data=(X_vaA_s, y_vaA),
                  epochs=bp["epochs"], batch_size=bp["batch_size"],
                  verbose=1, callbacks=callbacks)

        if ckpt_path.exists():
            model.load_weights(str(ckpt_path))

        return study.best_value, bp, model, scaler_final

    # ============================================================
    # A) Δ-learning: tune on Δ, predict Δ, reconstruct DFT = PM7 + Δ
    # ============================================================
    cv_mae_delta, bp_delta, model_delta, scaler_delta = tune_and_train("Δ-learning", y_tr_delta, out_prefix="dnn_delta")

    X_te_scaled = scaler_delta.transform(Xte_np)
    pred_delta  = model_delta.predict(X_te_scaled, verbose=0).reshape(-1)
    pm7_base    = test_df.loc[Xte_red.index, pm7_col].values.astype(np.float32)
    pred_dft_from_delta = pred_delta + pm7_base

    mae_delta = mean_absolute_error(y_te_delta, pred_delta)
    mae_dft_from_delta = mean_absolute_error(y_te_dft, pred_dft_from_delta)
    r2_dft_from_delta  = r2_score(y_te_dft, pred_dft_from_delta)

    # Save Δ artifacts
    pd.DataFrame({
        "true_delta": y_te_delta,
        "pred_delta": pred_delta,
        "true_dft":   y_te_dft,
        "pred_dft":   pred_dft_from_delta,
    }, index=Xte_red.index).to_csv(var_dir / "dnn_predictions_delta.csv", index=False)

    # Parity PDFs
    parity_plot_pdf(
        y_true=y_te_delta, y_pred=pred_delta,
        out_pdf=var_dir / "dnn_parity_delta.pdf",
        xlabel=f"Predicted {delta_lbl}", ylabel=f"True {delta_lbl}",
        annotate=f"MAE = {mae_delta:.2f} kcal/mol"
    )
    parity_plot_pdf(
        y_true=y_te_dft, y_pred=pred_dft_from_delta,
        out_pdf=var_dir / "dnn_parity_dft_from_delta.pdf",
        xlabel=f"Predicted {dft_lbl}", ylabel=f"True {dft_lbl}",
        annotate=f"MAE = {mae_dft_from_delta:.2f} kcal/mol\nR² = {r2_dft_from_delta:.2f}"
    )

    # ============================================================
    # B) Direct DFT (“normal ML”): tune on DFT, predict DFT directly
    # ============================================================
    cv_mae_direct, bp_direct, model_direct, scaler_direct = tune_and_train("Direct DFT", y_tr_dft, out_prefix="dnn_direct_dft")

    X_te_scaled2   = scaler_direct.transform(Xte_np)
    pred_dft_direct = model_direct.predict(X_te_scaled2, verbose=0).reshape(-1)
    mae_dft_direct  = mean_absolute_error(y_te_dft, pred_dft_direct)
    r2_dft_direct   = r2_score(y_te_dft, pred_dft_direct)

    pd.DataFrame({
        "true_dft": y_te_dft,
        "pred_dft": pred_dft_direct,
    }, index=Xte_red.index).to_csv(var_dir / "dnn_predictions_direct_dft.csv", index=False)

    parity_plot_pdf(
        y_true=y_te_dft, y_pred=pred_dft_direct,
        out_pdf=var_dir / "dnn_parity_direct_dft.pdf",
        xlabel=f"Predicted {dft_lbl}", ylabel=f"True {dft_lbl}",
        annotate=f"MAE = {mae_dft_direct:.2f} kcal/mol\nR² = {r2_dft_direct:.2f}"
    )

    # Save feature list
    with open(var_dir / "features_after_prune.txt", "w") as f:
        for c in Xtr_red.columns: f.write(c + "\n")

    # Save minimal artifacts for reload (weights + scalers + params)
    # Note: saving full SavedModel twice can be heavy; keep light by weights + JSON.
    arch_json_delta  = model_delta.to_json()
    arch_json_direct = model_direct.to_json()
    (var_dir / "artifacts").mkdir(exist_ok=True)
    with open(var_dir / "artifacts/dnn_delta_arch.json", "w") as f:  f.write(arch_json_delta)
    with open(var_dir / "artifacts/dnn_direct_arch.json", "w") as f: f.write(arch_json_direct)
    model_delta.save_weights(str(var_dir / "artifacts/dnn_delta_weights.h5"))
    model_direct.save_weights(str(var_dir / "artifacts/dnn_direct_weights.h5"))
    # Save scalers
    import joblib
    joblib.dump(scaler_delta,  var_dir / "artifacts/dnn_delta_scaler.joblib")
    joblib.dump(scaler_direct, var_dir / "artifacts/dnn_direct_scaler.joblib")

    # Combined metrics for this variant
    metrics = {
        "variant": variant_name,
        "target_col": tcol,
        "n_features_after_prune": int(n_features),

        "delta_cv_mae": float(cv_mae_delta),
        "delta_test_mae_delta": float(mae_delta),
        "delta_test_mae_dft": float(mae_dft_from_delta),
        "delta_test_r2_dft": float(r2_dft_from_delta),

        "direct_cv_mae": float(cv_mae_direct),
        "direct_test_mae_dft": float(mae_dft_direct),
        "direct_test_r2_dft": float(r2_dft_direct),
    }
    with open(var_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"[{variant_name}] Δ: MAEΔ={mae_delta:.3f}, DFT(Δ) MAE={mae_dft_from_delta:.3f}, R²={r2_dft_from_delta:.3f}")
    log(f"[{variant_name}] Direct DFT: MAE={mae_dft_direct:.3f}, R²={r2_dft_direct:.3f}")

    return metrics

# -----------------------------
# Run requested variants
# -----------------------------
all_metrics = []
for vname, vcols in VARIANTS:
    all_metrics.append(run_variant(vname, vcols))

# Write top-level summary
summary_path = Path(args.output_dir) / "DNN_summary.json"
with open(summary_path, "w") as f:
    json.dump({"variants": all_metrics}, f, indent=2)

log(f"Saved summary to {summary_path}")
