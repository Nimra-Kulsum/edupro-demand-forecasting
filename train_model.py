"""
train_model.py  -  EduPro Predictive Modeling pipeline
Run:  python train_model.py            (auto-finds the .xlsx inside /data)
      python train_model.py path/to/file.xlsx

Steps: load -> clean -> merge -> time-based target construction -> feature engineering ->
       correlation filter -> 6 models x 2 targets x 2 feature sets -> evaluation -> importance -> save artifacts
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (CAT_BASE, GROUPS, HIST, NUM_BASE, engineer, fit_bands, norm, rename_cols, to_num)

warnings.filterwarnings("ignore")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)
HIST_QUANTILE = 0.70   # first 70% of the timeline = "history", last 30% = "future" to be predicted
SEED = 42

# ------------------------------------------------------------------ 1. LOAD
def find_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    files = sorted(Path("data").glob("*.xlsx"))
    if not files:
        raise SystemExit("❌ Put your Excel file inside the /data folder (or pass the path as an argument).")
    return files[0]

def get_sheet(sheets, keyword):
    for k, v in sheets.items():
        if keyword in k:
            return v
    raise SystemExit(f"❌ No sheet containing '{keyword}' found. Sheets: {list(sheets)}")

path = find_file()
print(f"📂 Loading {path}")
raw = pd.read_excel(path, sheet_name=None)
sheets = {norm(k): v for k, v in raw.items()}
courses = rename_cols(get_sheet(sheets, "course"))
teachers = rename_cols(get_sheet(sheets, "teacher"))
tx = rename_cols(get_sheet(sheets, "transaction"))

for name, frame, need in [("Courses", courses, ["CourseID"]), ("Transactions", tx, ["CourseID", "TransactionDate", "Amount"])]:
    miss = [c for c in need if c not in frame.columns]
    if miss:
        raise SystemExit(f"❌ {name} sheet is missing {miss}. Columns found: {list(frame.columns)}")

# ------------------------------------------------------------------ 2. CLEAN + MERGE
quality = {"raw_transactions": int(len(tx)), "raw_courses": int(len(courses))}
courses["CourseID"] = courses["CourseID"].astype(str).str.strip()
tx["CourseID"] = tx["CourseID"].astype(str).str.strip()
tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
tx["Amount"] = to_num(tx["Amount"])
if "TransactionID" in tx:
    tx = tx.drop_duplicates("TransactionID")
tx = tx.dropna(subset=["TransactionDate", "Amount"])
tx = tx[tx["CourseID"].isin(courses["CourseID"])]
courses = courses.drop_duplicates("CourseID")
quality["clean_transactions"] = int(len(tx))
quality["clean_courses"] = int(len(courses))

if "TeacherID" in courses and "TeacherID" in teachers:
    tcols = [c for c in ["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"] if c in teachers]
    teachers["TeacherID"] = teachers["TeacherID"].astype(str).str.strip()
    courses["TeacherID"] = courses["TeacherID"].astype(str).str.strip()
    df = courses.merge(teachers[tcols].drop_duplicates("TeacherID"), on="TeacherID", how="left", suffixes=("", "_t"))
else:
    print("⚠️ Could not link Courses to Teachers (no TeacherID in Courses). Instructor features will be imputed.")
    df = courses.copy()

# ------------------------------------------------------------------ 3. TARGETS (time-based, no leakage)
cutoff = tx["TransactionDate"].quantile(HIST_QUANTILE)
hist, fut = tx[tx["TransactionDate"] <= cutoff], tx[tx["TransactionDate"] > cutoff]
months_hist = max((cutoff - tx["TransactionDate"].min()).days / 30.44, 1)
horizon_days = int((tx["TransactionDate"].max() - cutoff).days)

h = hist.groupby("CourseID").agg(PastEnrollments=("Amount", "size"), PastRevenue=("Amount", "sum"))
f = fut.groupby("CourseID").agg(FutureEnrollments=("Amount", "size"), FutureRevenue=("Amount", "sum"))
a = tx.groupby("CourseID").agg(TotalEnrollments=("Amount", "size"), TotalRevenue=("Amount", "sum"))
df = df.merge(h, on="CourseID", how="left").merge(f, on="CourseID", how="left").merge(a, on="CourseID", how="left")
for c in ["PastEnrollments", "PastRevenue", "FutureEnrollments", "FutureRevenue", "TotalEnrollments", "TotalRevenue"]:
    df[c] = df[c].fillna(0)
df["PastAvgMonthlyRevenue"] = df["PastRevenue"] / months_hist
df["RevenuePerEnrollment"] = np.where(df["PastEnrollments"] > 0, df["PastRevenue"] / df["PastEnrollments"].replace(0, np.nan), np.nan)

# ------------------------------------------------------------------ 4. FEATURE ENGINEERING
for c in ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating"]:
    if c in df:
        df[c] = to_num(df[c])
for c in ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating"]:
    if c not in df:
        df[c] = np.nan
bands = fit_bands(df)
df = engineer(df, bands)
quality["missing_values"] = {c: int(df[c].isna().sum()) for c in ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating"]}

def drop_correlated(frame, cols, thr=0.9):
    num = frame[cols].apply(pd.to_numeric, errors="coerce")
    corr, keep, dropped = num.corr().abs(), [], []
    for c in cols:
        if num[c].nunique() <= 1 or any(corr.loc[c, k] > thr for k in keep):
            dropped.append(c)
        else:
            keep.append(c)
    return keep, dropped

launch_num, d1 = drop_correlated(df, NUM_BASE)
full_num, d2 = drop_correlated(df, NUM_BASE + HIST)
FEATURESETS = {"launch": (launch_num, CAT_BASE), "full": (full_num, CAT_BASE)}
print(f"🧹 Redundant features removed -> launch: {d1}  | full: {d2}")

def make_pre(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])

MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=5.0),
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=20000),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED),
}

def grouped_importance(est, X, y, groups, n=40):
    rng = np.random.default_rng(SEED)
    base, rows = r2_score(y, est.predict(X)), []
    for g, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n):
            Xp = X.copy()
            Xp[cols] = X[cols].iloc[rng.permutation(len(X))].values
            drops.append(base - r2_score(y, est.predict(Xp)))
        rows.append((g, float(np.mean(drops)), float(np.std(drops))))
    return pd.DataFrame(rows, columns=["Feature", "Importance", "Std"]).sort_values("Importance", ascending=False)

# ------------------------------------------------------------------ 5. MODELING
results, best_names = [], {}
n_splits = int(min(5, max(2, len(df) // 5)))
for target in ["FutureEnrollments", "FutureRevenue"]:
    y = df[target]
    for fs, (num, cat) in FEATURESETS.items():
        X = df[num + cat]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
        scores = {}
        for mname, mk in MODELS.items():
            pipe = Pipeline([("pre", make_pre(num, cat)), ("m", mk())])
            pipe.fit(Xtr, ytr)
            p = np.clip(pipe.predict(Xte), 0, None)
            cv = cross_val_score(pipe, X, y, cv=KFold(n_splits, shuffle=True, random_state=SEED), scoring="r2")
            results.append(dict(Target=target, FeatureSet=fs, Model=mname, MAE=mean_absolute_error(yte, p),
                                RMSE=float(np.sqrt(mean_squared_error(yte, p))), R2=r2_score(yte, p),
                                CV_R2=float(cv.mean()), CV_R2_std=float(cv.std())))
            scores[mname] = cv.mean()
        best = max(scores, key=scores.get)
        best_names[f"{target}|{fs}"] = best
        print(f"🏆 {target:18s} [{fs:6s}] best model = {best}  (CV R² = {scores[best]:.3f})")

        # importance (on held-out data) + final refit on ALL data for deployment
        est = Pipeline([("pre", make_pre(num, cat)), ("m", MODELS[best]())]).fit(Xtr, ytr)
        grouped_importance(est, Xte, yte, GROUPS).to_csv(ART / f"importance_{target}_{fs}.csv", index=False)
        final = Pipeline([("pre", make_pre(num, cat)), ("m", MODELS[best]())]).fit(X, y)
        joblib.dump(final, ART / f"model_{target}_{fs}.joblib")
        if fs == "full":   # honest out-of-fold predictions for the dashboard
            oof = cross_val_predict(clone(final), X, y, cv=KFold(n_splits, shuffle=True, random_state=SEED))
            df[f"Pred_{target}"] = np.clip(oof, 0, None)

pd.DataFrame(results).round(4).to_csv(ART / "model_results.csv", index=False)
df.to_csv(ART / "course_table.csv", index=False)

# ------------------------------------------------------------------ 6. MONTHLY SERIES + METADATA
cat_map = df.set_index("CourseID")["CourseCategory"]
m = tx.assign(Month=tx["TransactionDate"].dt.to_period("M").dt.to_timestamp(), CourseCategory=tx["CourseID"].map(cat_map))
m.groupby(["Month", "CourseCategory"]).agg(Revenue=("Amount", "sum"), Enrollments=("Amount", "size")).reset_index().to_csv(ART / "monthly.csv", index=False)

def rng_of(c):
    s = df[c].dropna()
    return [float(s.min()), float(s.max()), float(s.median())] if len(s) else [0.0, 1.0, 0.5]

meta = {
    "bands": bands, "cutoff": str(cutoff.date()), "horizon_days": horizon_days,
    "date_min": str(tx["TransactionDate"].min().date()), "date_max": str(tx["TransactionDate"].max().date()),
    "featuresets": {k: {"num": v[0], "cat": v[1]} for k, v in FEATURESETS.items()},
    "best_models": best_names, "dropped": {"launch": d1, "full": d2}, "quality": quality,
    "options": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in ["CourseCategory", "CourseType", "CourseLevel", "Expertise"]},
    "ranges": {c: rng_of(c) for c in ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating"]},
}
json.dump(meta, open(ART / "meta.json", "w"), indent=2)
print("\n✅ Done. Artifacts saved in /artifacts  ->  now run:  streamlit run app.py")
