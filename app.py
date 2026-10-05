"""
app.py  -  EduPro Predictive Intelligence Dashboard
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import engineer

st.set_page_config(page_title="EduPro Predictive Intelligence", page_icon="🎓", layout="wide")
ART = Path(".")
if not (ART / "meta.json").exists():
    st.error("⚠️ No trained artifacts found. Run `python train_model.py` first.")
    st.stop()

# ============================================================ THEME
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 10%,rgba(99,102,241,.25),transparent 40%),
 radial-gradient(circle at 85% 20%,rgba(236,72,153,.18),transparent 40%),
 linear-gradient(135deg,#0b1020 0%,#151a33 50%,#0d1226 100%);}
section[data-testid="stSidebar"]{background:rgba(15,20,45,.92);border-right:1px solid rgba(255,255,255,.08);}
.hero{padding:28px 32px;border-radius:22px;margin-bottom:18px;
 background:linear-gradient(120deg,rgba(99,102,241,.35),rgba(236,72,153,.25));border:1px solid rgba(255,255,255,.15);}
.hero h1{margin:0;font-size:2.1rem;} .hero p{margin:6px 0 0;opacity:.85;}
.kpi{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:16px 18px;height:100%;}
.kpi .ico{font-size:1.7rem;} .kpi .lbl{opacity:.7;font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;}
.kpi .val{font-size:1.7rem;font-weight:700;} .kpi .sub{opacity:.6;font-size:.78rem;}
.insight{background:rgba(16,185,129,.12);border-left:4px solid #10b981;padding:12px 16px;border-radius:10px;margin:8px 0;}
.warn{background:rgba(245,158,11,.12);border-left:4px solid #f59e0b;padding:12px 16px;border-radius:10px;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

PALETTE = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#06b6d4", "#a855f7", "#ef4444", "#84cc16"]

def style(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=50, b=10), colorway=PALETTE)
    return fig

def hero(icon, title, sub):
    st.markdown(f'<div class="hero"><h1>{icon} {title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)

def kpi(icon, label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="ico">{icon}</div><div class="lbl">{label}</div>'
                f'<div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

# ============================================================ DATA / MODELS
@st.cache_data
def load():
    return (json.load(open(ART / "meta.json")), pd.read_csv(ART / "course_table.csv"),
            pd.read_csv(ART / "monthly.csv", parse_dates=["Month"]), pd.read_csv(ART / "model_results.csv"))

@st.cache_resource
def load_models():
    return {(t, f): joblib.load(ART / f"model_{t}_{f}.joblib")
            for t in ["FutureEnrollments", "FutureRevenue"] for f in ["launch", "full"]}

meta, courses, monthly, results = load()
models = load_models()
HZ = meta["horizon_days"]
R = meta["ranges"]

def predict_launch(inp):
    row = engineer(pd.DataFrame([inp]), meta["bands"])
    fs = meta["featuresets"]["launch"]
    cols = fs["num"] + fs["cat"]
    e = max(0.0, float(models[("FutureEnrollments", "launch")].predict(row[cols])[0]))
    r = max(0.0, float(models[("FutureRevenue", "launch")].predict(row[cols])[0]))
    return e, r

def opts(c):
    return meta["options"].get(c) or ["Unknown"]

# ============================================================ SIDEBAR
st.sidebar.markdown("## 🎓 EduPro\n**Predictive Intelligence**")
page = st.sidebar.radio("Navigate", [
    "🏠 Executive Overview", "🔮 Demand Predictor", "💰 Revenue Forecast", "🧠 Feature Importance",
    "📊 Category Comparison", "🧪 Model Lab", "🔍 EDA Explorer", "📄 Executive Summary"])
st.sidebar.markdown("---")
st.sidebar.caption(f"📅 Data: {meta['date_min']} → {meta['date_max']}\n\n🔭 Forecast window: **{HZ} days** after {meta['cutoff']}")

# ============================================================ PAGES
if page.startswith("🏠"):
    hero("🏠", "Executive Overview", "From reactive reporting to proactive planning: where will EduPro's demand and revenue come from next?")
    c = st.columns(4)
    with c[0]: kpi("📚", "Courses", f"{len(courses):,}", f"{courses['CourseCategory'].nunique()} categories")
    with c[1]: kpi("🧾", "Total Enrollments", f"{int(courses['TotalEnrollments'].sum()):,}", "all history")
    with c[2]: kpi("💰", "Total Revenue", f"{courses['TotalRevenue'].sum():,.0f}", "all history")
    with c[3]: kpi("🔮", f"Predicted Revenue ({HZ}d)", f"{courses['Pred_FutureRevenue'].sum():,.0f}", "model, out-of-fold")
    st.write("")
    tot = monthly.groupby("Month", as_index=False)[["Revenue", "Enrollments"]].sum()
    fig = go.Figure()
    fig.add_bar(x=tot["Month"], y=tot["Enrollments"], name="Enrollments", marker_color="#6366f1", opacity=.55, yaxis="y2")
    fig.add_scatter(x=tot["Month"], y=tot["Revenue"], name="Revenue", line=dict(color="#ec4899", width=3))
    fig.update_layout(title="📈 Monthly Revenue & Enrollments", yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(style(fig), width="stretch")
    a, b = st.columns(2)
    top = courses.nlargest(10, "Pred_FutureRevenue")
    label = "CourseName" if "CourseName" in top else "CourseID"
    a.plotly_chart(style(px.bar(top, x="Pred_FutureRevenue", y=label, orientation="h", color="CourseCategory",
                                title="🚀 Top 10 courses by predicted revenue")).update_yaxes(autorange="reversed"), width="stretch")
    cat = courses.groupby("CourseCategory", as_index=False)["TotalRevenue"].sum()
    b.plotly_chart(style(px.pie(cat, names="CourseCategory", values="TotalRevenue", hole=.55, title="🗂️ Revenue share by category")), width="stretch")

elif page.startswith("🔮"):
    hero("🔮", "Course Demand Predictor", "Design a hypothetical course and see its forecast enrollments & revenue before launching it.")
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("🗂️ Category", opts("CourseCategory"))
    ctype = c1.selectbox("📦 Course type", opts("CourseType"))
    level = c1.selectbox("🎚️ Level", opts("CourseLevel"))
    price = c2.slider("💲 Course price", float(R["CoursePrice"][0]), float(R["CoursePrice"][1]), float(R["CoursePrice"][2]))
    dur = c2.slider("⏱️ Duration", float(R["CourseDuration"][0]), float(R["CourseDuration"][1]), float(R["CourseDuration"][2]))
    crate = c2.slider("⭐ Expected course rating", float(R["CourseRating"][0]), float(R["CourseRating"][1]), float(R["CourseRating"][2]))
    exp = c3.slider("👨‍🏫 Instructor experience (years)", float(R["YearsOfExperience"][0]), float(R["YearsOfExperience"][1]), float(R["YearsOfExperience"][2]))
    trate = c3.slider("🏅 Instructor rating", float(R["TeacherRating"][0]), float(R["TeacherRating"][1]), float(R["TeacherRating"][2]))
    expertise = c3.selectbox("🎯 Instructor expertise", opts("Expertise"), index=opts("Expertise").index(cat) if cat in opts("Expertise") else 0)
    inp = dict(CourseCategory=cat, CourseType=ctype, CourseLevel=level, CoursePrice=price, CourseDuration=dur,
               CourseRating=crate, Expertise=expertise, YearsOfExperience=exp, TeacherRating=trate)
    e, r = predict_launch(inp)
    st.write("")
    k = st.columns(3)
    with k[0]: kpi("🧾", f"Predicted enrollments ({HZ}d)", f"{e:,.1f}")
    with k[1]: kpi("💰", f"Predicted revenue ({HZ}d)", f"{r:,.0f}")
    with k[2]: kpi("🏷️", "Revenue per enrollment", f"{(r / e if e > 0 else 0):,.0f}")

    st.markdown("### 💲 Price sensitivity curve")
    grid = np.linspace(R["CoursePrice"][0], R["CoursePrice"][1], 25)
    sens = pd.DataFrame([dict(Price=p, **dict(zip(["Enrollments", "Revenue"], predict_launch({**inp, "CoursePrice": p})))) for p in grid])
    fig = go.Figure()
    fig.add_scatter(x=sens["Price"], y=sens["Enrollments"], name="Enrollments", line=dict(color="#6366f1", width=3))
    fig.add_scatter(x=sens["Price"], y=sens["Revenue"], name="Revenue", line=dict(color="#ec4899", width=3), yaxis="y2")
    fig.add_vline(x=price, line_dash="dot", line_color="#f59e0b")
    fig.update_layout(xaxis_title="Course price", yaxis_title="Enrollments", yaxis2=dict(title="Revenue", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(style(fig), width="stretch")
    best = sens.loc[sens["Revenue"].idxmax()]
    st.markdown(f'<div class="insight">💡 With all other inputs fixed, the model projects the highest revenue near a price of '
                f'<b>{best["Price"]:,.0f}</b>. Treat this as a directional hint, not a guarantee.</div>', unsafe_allow_html=True)

elif page.startswith("💰"):
    hero("💰", "Revenue Forecast", "Course-level and category-level revenue projections.")
    t1, t2 = st.tabs(["📅 Time-series trend forecast", "🎯 Model-based course/category forecast"])
    with t1:
        c1, c2 = st.columns(2)
        sel = c1.selectbox("Category", ["All categories"] + sorted(monthly["CourseCategory"].dropna().unique()))
        hz = c2.slider("Forecast horizon (months)", 3, 12, 6)
        s = monthly if sel == "All categories" else monthly[monthly["CourseCategory"] == sel]
        s = s.groupby("Month", as_index=False)["Revenue"].sum().sort_values("Month")
        s = s.iloc[:-1] if len(s) > 4 else s     # last month is usually partial
        if len(s) >= 4:
            tail = s.tail(12).reset_index(drop=True)
            x = np.arange(len(tail)); coef = np.polyfit(x, tail["Revenue"], 1)
            sd = np.std(tail["Revenue"] - np.polyval(coef, x))
            fx = np.arange(len(tail), len(tail) + hz)
            fdates = pd.date_range(tail["Month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=hz, freq="MS")
            fc = np.clip(np.polyval(coef, fx), 0, None)
            fig = go.Figure()
            fig.add_scatter(x=s["Month"], y=s["Revenue"], name="Actual", line=dict(color="#6366f1", width=3))
            fig.add_scatter(x=fdates, y=fc, name="Forecast", line=dict(color="#ec4899", width=3, dash="dash"))
            fig.add_scatter(x=list(fdates) + list(fdates[::-1]), y=list(fc + 1.96 * sd) + list((fc - 1.96 * sd)[::-1]),
                            fill="toself", fillcolor="rgba(236,72,153,.15)", line=dict(width=0), name="95% band")
            fig.update_layout(title=f"📈 {sel}: next {hz} months")
            st.plotly_chart(style(fig), width="stretch")
            st.markdown(f'<div class="warn">⚠️ Linear-trend baseline on the last {len(tail)} months. Use as a sanity check next to the ML forecasts.</div>', unsafe_allow_html=True)
        else:
            st.info("Not enough monthly history for a trend forecast.")
    with t2:
        catf = courses.groupby("CourseCategory", as_index=False)[["FutureRevenue", "Pred_FutureRevenue"]].sum()
        m = catf.melt("CourseCategory", var_name="Type", value_name="Revenue").replace({"FutureRevenue": "Actual", "Pred_FutureRevenue": "Predicted"})
        st.plotly_chart(style(px.bar(m, x="CourseCategory", y="Revenue", color="Type", barmode="group",
                                     title=f"🗂️ Category revenue, last {HZ} days: actual vs predicted (out-of-fold)")), width="stretch")
        label = "CourseName" if "CourseName" in courses else "CourseID"
        show = courses[[label, "CourseCategory", "CoursePrice", "FutureRevenue", "Pred_FutureRevenue"]].sort_values("Pred_FutureRevenue", ascending=False)
        st.dataframe(show.round(1), width="stretch", hide_index=True)

elif page.startswith("🧠"):
    hero("🧠", "Feature Importance Explorer", "Which factors actually drive demand and revenue? (grouped permutation importance on held-out courses)")
    c1, c2 = st.columns(2)
    tgt = c1.radio("🎯 Target", ["FutureEnrollments", "FutureRevenue"], horizontal=True)
    fs = c2.radio("🧩 Feature set", ["launch", "full"], horizontal=True, format_func=lambda x: "Launch (new course, no history)" if x == "launch" else "Full (incl. past performance)")
    imp = pd.read_csv(ART / f"importance_{tgt}_{fs}.csv")
    fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", error_x="Std",
                 title=f"Drop in R² when the factor is shuffled  |  model: {meta['best_models'][f'{tgt}|{fs}']}")
    st.plotly_chart(style(fig, 460), width="stretch")
    pos = imp[imp["Importance"] > 0.01].head(3)
    if len(pos):
        st.markdown('<div class="insight">💡 <b>Top demand drivers:</b> ' + ", ".join(pos["Feature"]) +
                    ". Prioritise these levers when planning launches and pricing.</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn">⚠️ No factor shows a clear signal on held-out data, so demand is only weakly explained by these attributes. Report this honestly; it is a finding, not a failure.</div>', unsafe_allow_html=True)
    st.caption("Values near 0 or negative mean the factor adds nothing beyond noise.")

elif page.startswith("📊"):
    hero("📊", "Category-Level Demand Comparison", "Where is demand strongest, and where is the model most/least certain?")
    g = courses.groupby("CourseCategory").agg(Courses=("CourseID", "count"), AvgPrice=("CoursePrice", "mean"), AvgRating=("CourseRating", "mean"),
                                               ActualEnroll=("FutureEnrollments", "sum"), PredEnroll=("Pred_FutureEnrollments", "sum"),
                                               ActualRev=("FutureRevenue", "sum"), PredRev=("Pred_FutureRevenue", "sum")).reset_index()
    m = g.melt("CourseCategory", ["ActualEnroll", "PredEnroll"], "Type", "Enrollments")
    st.plotly_chart(style(px.bar(m, x="CourseCategory", y="Enrollments", color="Type", barmode="group", title="🧾 Enrollments: actual vs predicted")), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(style(px.scatter(g, x="AvgPrice", y="PredRev", size="Courses", color="CourseCategory", title="💲 Avg price vs predicted revenue")), width="stretch")
    heat = courses.pivot_table(index="CourseCategory", columns="PriceBand", values="Pred_FutureEnrollments", aggfunc="mean")
    heat = heat.reindex(columns=[c for c in ["Low", "Medium", "High"] if c in heat.columns])
    b.plotly_chart(style(px.imshow(heat, text_auto=".1f", color_continuous_scale="Purples", title="🔥 Avg predicted enrollments: category × price band")), width="stretch")
    st.dataframe(g.round(1), width="stretch", hide_index=True)

elif page.startswith("🧪"):
    hero("🧪", "Model Lab", "Baseline vs advanced models: MAE, RMSE, R² (held-out) and 5-fold cross-validated R².")
    tgt = st.radio("🎯 Target", ["FutureEnrollments", "FutureRevenue"], horizontal=True)
    r = results[results["Target"] == tgt]
    st.plotly_chart(style(px.bar(r, x="Model", y="CV_R2", color="FeatureSet", barmode="group", error_y="CV_R2_std", title="Cross-validated R² (higher is better)")), width="stretch")
    st.dataframe(r.sort_values("CV_R2", ascending=False).round(3), width="stretch", hide_index=True)
    pc = f"Pred_{tgt}"
    st.plotly_chart(style(px.scatter(courses, x=tgt, y=pc, color="CourseCategory", trendline="ols", title="🎯 Actual vs predicted (out-of-fold, full model)")), width="stretch")
    st.caption("Tip: if R² is modest, say so in your paper. Explain what the model can and cannot capture.")

elif page.startswith("🔍"):
    hero("🔍", "EDA Explorer", "Data quality, distributions and correlations behind the models.")
    q = meta["quality"]
    c = st.columns(4)
    with c[0]: kpi("🧾", "Raw transactions", f"{q['raw_transactions']:,}")
    with c[1]: kpi("🧹", "After cleaning", f"{q['clean_transactions']:,}")
    with c[2]: kpi("📚", "Courses", f"{q['clean_courses']:,}")
    with c[3]: kpi("❓", "Missing values", f"{sum(q['missing_values'].values()):,}", "price/duration/ratings/experience")
    st.write("")
    num = ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating", "TotalEnrollments", "TotalRevenue"]
    num = [c for c in num if c in courses and courses[c].notna().any()]
    a, b = st.columns(2)
    v = a.selectbox("Distribution of", num)
    a.plotly_chart(style(px.histogram(courses, x=v, nbins=30, marginal="box", title=f"📊 {v}")), width="stretch")
    b.plotly_chart(style(px.imshow(courses[num].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="🔗 Correlation heatmap")), width="stretch")
    x = st.selectbox("Relationship: TotalEnrollments vs", [c for c in num if c != "TotalEnrollments"])
    st.plotly_chart(style(px.scatter(courses, x=x, y="TotalEnrollments", color="CourseCategory", trendline="ols")), width="stretch")

else:
    hero("📄", "Executive Summary", "Auto-generated briefing for stakeholders. Download and attach to your submission.")
    bestrev = results[results["Target"] == "FutureRevenue"].sort_values("CV_R2", ascending=False).iloc[0]
    bestenr = results[results["Target"] == "FutureEnrollments"].sort_values("CV_R2", ascending=False).iloc[0]
    catr = courses.groupby("CourseCategory")["Pred_FutureRevenue"].sum().sort_values(ascending=False)
    imp = pd.read_csv(ART / "importance_FutureEnrollments_launch.csv")
    drivers = ", ".join(imp[imp["Importance"] > 0.01]["Feature"].head(3)) or "no single dominant factor (weak signal)"
    md = f"""# EduPro: Course Demand & Revenue Forecasting - Executive Summary

**Data window:** {meta['date_min']} to {meta['date_max']}  |  **Forecast window:** {HZ} days after {meta['cutoff']}
**Scope:** {len(courses)} courses, {courses['CourseCategory'].nunique()} categories, {int(courses['TotalEnrollments'].sum()):,} enrollments.

## Key findings
1. **Demand model:** best = {bestenr['Model']} (CV R² {bestenr['CV_R2']:.2f}, MAE {bestenr['MAE']:.1f} enrollments).
2. **Revenue model:** best = {bestrev['Model']} (CV R² {bestrev['CV_R2']:.2f}, MAE {bestrev['MAE']:,.0f}).
3. **Top predicted revenue category:** {catr.index[0]} ({catr.iloc[0]:,.0f}); lowest: {catr.index[-1]} ({catr.iloc[-1]:,.0f}).
4. **Main demand drivers for new courses:** {drivers}.

## Recommendations
- Use the Demand Predictor to screen proposed launches (price, level, instructor profile) before committing budget.
- Prioritise investment in the highest-forecast categories; review the lowest for repositioning.
- Re-train quarterly as new transactions arrive, and track forecast vs actual.

## Limitations
- Forecasts are statistical estimates from historical patterns; they do not account for marketing, seasonality or competitors.
- Accuracy is bounded by the signal in the available attributes (see Model Lab).
"""
    st.markdown(md)
    st.download_button("⬇️ Download summary (.md)", md, file_name="EduPro_Executive_Summary.md")
