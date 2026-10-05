"""
features.py  -  shared feature-engineering logic (used by train_model.py AND app.py)
Keeping this in ONE place guarantees the dashboard builds features exactly the way the models were trained.
"""
import re
import numpy as np
import pandas as pd

# ---------- column-name tolerance (handles 'Course Price', 'course_price', etc.) ----------
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

CANONICAL = ["CourseID", "CourseName", "CourseCategory", "CourseType", "CourseLevel", "CoursePrice",
             "CourseDuration", "CourseRating", "TeacherID", "Expertise", "YearsOfExperience",
             "TeacherRating", "TransactionID", "TransactionDate", "Amount", "UserID"]
ALIASES = {norm(c): c for c in CANONICAL}

def rename_cols(df):
    return df.rename(columns={c: ALIASES.get(norm(c), c) for c in df.columns})

def to_num(s):
    """'12 hours' -> 12.0 ,  '$1,200' -> 1200.0"""
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")

# ---------- feature groups ----------
RAW_NUM = ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating"]
RAW_CAT = ["CourseCategory", "CourseType", "CourseLevel", "Expertise"]
NUM_BASE = RAW_NUM + ["LevelEnc", "ExpertiseMatch"]
CAT_BASE = ["CourseCategory", "CourseType", "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket"]
HIST = ["PastEnrollments", "PastAvgMonthlyRevenue", "PastRevenue", "RevenuePerEnrollment"]

LEVEL_MAP = {"beginner": 0, "intermediate": 1, "advanced": 2}

# business-friendly groups for importance analysis
GROUPS = {
    "💲 Course Price": ["CoursePrice", "PriceBand"],
    "⏱️ Course Duration": ["CourseDuration", "DurationBucket"],
    "⭐ Course Rating": ["CourseRating", "RatingTier"],
    "🎚️ Course Level": ["LevelEnc"],
    "🗂️ Category": ["CourseCategory"],
    "📦 Course Type": ["CourseType"],
    "👨‍🏫 Instructor Experience": ["YearsOfExperience", "ExperienceBucket"],
    "🏅 Instructor Rating": ["TeacherRating"],
    "🎯 Expertise-Category Match": ["ExpertiseMatch"],
    "📈 Historical Performance": HIST,
}

def fit_bands(courses):
    """Quantile cut-points for low/medium/high style buckets (learned on training data)."""
    bands = {}
    for col, key in [("CoursePrice", "price"), ("CourseDuration", "duration"), ("CourseRating", "rating")]:
        s = courses[col].dropna()
        e1, e2 = (float(s.quantile(1 / 3)), float(s.quantile(2 / 3))) if len(s) else (0.0, 1.0)
        if e2 <= e1:
            e2 = e1 + 1e-6
        bands[key] = [e1, e2]
    return bands

def _cut(s, edges, labels):
    out = pd.cut(s, [-np.inf] + list(edges) + [np.inf], labels=labels).astype(object)
    return out.where(s.notna(), "Unknown")

def _match(expertise, category):
    e, c = norm(expertise), norm(category)
    if not e or not c or e == "nan" or c == "nan":
        return 0.0
    if e == c:
        return 1.0
    if e in c or c in e:
        return 0.7
    et = set(re.findall(r"[a-z]{3,}", str(expertise).lower()))
    ct = set(re.findall(r"[a-z]{3,}", str(category).lower()))
    return 0.5 if et & ct else 0.0

def _level(x):
    x = str(x).lower()
    for k, v in LEVEL_MAP.items():
        if k in x:
            return float(v)
    return np.nan

def engineer(df, bands):
    d = df.copy()
    for c in RAW_NUM:
        d[c] = to_num(d[c]) if c in d else np.nan
    for c in RAW_CAT:
        d[c] = d[c].astype(object).fillna("Unknown").astype(str) if c in d else "Unknown"
    d["PriceBand"] = _cut(d["CoursePrice"], bands["price"], ["Low", "Medium", "High"])
    d["DurationBucket"] = _cut(d["CourseDuration"], bands["duration"], ["Short", "Medium", "Long"])
    d["RatingTier"] = _cut(d["CourseRating"], bands["rating"], ["Standard", "Good", "Top"])
    d["ExperienceBucket"] = _cut(d["YearsOfExperience"], [2, 5, 10],
                                 ["Junior (0-2y)", "Mid (3-5y)", "Senior (6-10y)", "Expert (10y+)"])
    d["LevelEnc"] = d["CourseLevel"].map(_level)
    d["ExpertiseMatch"] = [_match(e, c) for e, c in zip(d["Expertise"], d["CourseCategory"])]
    return d
