import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet
from datetime import datetime
import plotly.express as px
import logging

# Silence prophet logs
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

# ----------------------------
# 1. ARXIV TAXONOMY MAPPING
# ----------------------------
arxiv_categories = {
    # … your full dict from cs.*, math.*, physics.*, stat.* etc. …
}

# ----------------------------
# 2. LOAD DATA
# ----------------------------
# raw dataframe (columns are codes like 'cs.LG', 'math.NA', …)
df_raw = pd.read_excel("arxiv_monthly_publications.xlsx", index_col=0)
df_raw.index = pd.to_datetime(df_raw.index)

# human-readable dataframe
df = df_raw.rename(columns=arxiv_categories)

# ----------------------------
# 3. PAGE NAVIGATION
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📖 Documentation", "🛠 Functionalities"])

# ----------------------------
# 4. DOCUMENTATION PAGE
# ----------------------------
if page == "📖 Documentation":
    st.title("ArXiv Tracker 📖 Documentation")

    st.header("Motivation")
    st.markdown(
        """
        - **Hyperconnected environment**: Traditional metrics (patents, journals) arrive too slowly.  
        - **Need**: A **high-frequency**, **high-granularity** proxy to detect shocks or breakthroughs as they happen.  
        - **Solution**: Use arXiv preprint volumes on a monthly basis as a real-time research thermometer.
        """
    )

    st.header("Suggested Use Cases")
    st.markdown(
        """
        1. **Assessing the impact of funding cuts**  
           Track in real time how changes in R&D budgets ripple through global outputs.  
        2. **Studying synergies across fields**  
           Spot interdisciplinary “hot spots,” e.g. quantum physics × AI.  
        3. **Forecasting technology shocks**  
           Early warning of breakthroughs (robotics, novel materials) to inform policy & investment.
        """
    )

    st.header("Methodology")
    st.markdown(
        """
        - **Web-scraping**: Monthly category listings from arXiv.org, extracted with Python.  
        - **No heavy cleaning**: We simply count new preprints per category (1991–present).  
        - **Output**: 129 real-time indicators (40 CS, 51 physics, 38 math/stats).
        """
    )

    st.header("Validation")
    st.markdown(
        """
        We benchmark our arXiv indicators against established sources:  
        - **Computer Science** vs. **Dimensions**: near-perfect tracking since 2012.  
        - **AI subfield** vs. **Stanford AI Index**: arXiv often leads by a few months, capturing the GenAI boom early.
        """
    )

    st.subheader("▶️ Example: CS / Physics / Mathematics Volume")
    # regenerate the “CS vs Physics vs Math” chart here for illustration
    cs_codes = [k for k in arxiv_categories if k.startswith("cs.")]
    math_codes = [k for k in arxiv_categories if k.startswith("math.") or k.startswith("stat.")]
    phys_codes = [k for k in df_raw.columns if k not in cs_codes + math_codes]
    grp = pd.DataFrame({
        "Computer Science": df_raw[cs_codes].sum(axis=1),
        "Math & Stats":     df_raw[math_codes].sum(axis=1),
        "Physics":          df_raw[phys_codes].sum(axis=1),
    })
    fig_doc = px.line(
        grp.reset_index(),
        x="index",
        y=grp.columns,
        labels={"index":"Date","value":"Monthly Preprints","variable":"Field"},
        title="Monthly Volume by Major Field (1996–2025)"
    )
    st.plotly_chart(fig_doc, use_container_width=True)

    st.header("Limitations")
    st.markdown(
        """
        - **Quality variance**: No peer review → mix of minor & revolutionary findings.  
        - **Category overlap**: Papers can live in up to 3 categories → slight double counting.  
        - **Adoption bias**: CS only really took off on arXiv post-2012.
        """
    )

    st.header("Future Work")
    st.markdown(
        """
        - **Incorporate** SSRN, Research Square, Preprints.org for other fields.  
        - **Analyze** citation counts for impact, not just volume.  
        - **Extend** to bioRxiv, medRxiv, and domain-specific repositories.
        """
    )

# ----------------------------
# 5. FUNCTIONALITIES PAGE
# ----------------------------
else:
    st.title("ArXiv Tracker 🛠 Dashboard")

    # -- Sidebar: Date Filter --
    st.sidebar.header("📅 Filter Data by Date")
    min_date, max_date = df.index.min(), df.index.max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    if len(date_range) == 2:
        start, end = date_range
        df_filt     = df.loc[start:end]
        df_raw_filt = df_raw.loc[start:end]
    else:
        df_filt, df_raw_filt = df, df_raw

    # ==== 1) FORECASTING ====
    st.header("1️⃣ Forecasting")
    cat_fc = st.selectbox("Select category to forecast", list(arxiv_categories.values()))
    horizon = st.slider("Forecast horizon (months)", 3, 48, 12)
    smooth_hist = st.checkbox("Apply historical smoothing", value=False)
    if smooth_hist:
        win = st.slider("Smoothing window (months)", 1, 12, 3)

    # prepare Prophet data
    df_prop = df_filt[[cat_fc]].reset_index().rename(columns={"index":"ds", cat_fc:"y"})
    if smooth_hist:
        df_prop["y"] = df_prop["y"].rolling(win).mean()
        df_prop = df_prop.dropna()

    m = Prophet(weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prop)
    future = m.make_future_dataframe(periods=horizon, freq="MS")
    forecast = m.predict(future)

    fig1 = px.line(
        forecast,
        x="ds", y="yhat",
        labels={"ds":"Date","yhat":"Publications"},
        title=f"Forecast for {cat_fc}"
    )
    fig1.add_scatter(
        x=df_prop["ds"], y=df_prop["y"],
        mode="markers", name="Actual"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.divider()

    # ==== 2) TIME SERIES DECOMPOSITION ====
    st.header("2️⃣ Time Series Decomposition")
    cat_dc = st.selectbox("Select category to decompose", list(arxiv_categories.values()))
    try:
        decomp = sm.tsa.seasonal_decompose(df_filt[cat_dc], model="additive", period=12)
        fig2 = decomp.plot()
        fig2.set_size_inches(10, 8)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")
    st.divider()

    # ==== 3) CUSTOM INDEX BUILDER ====
    st.header("3️⃣ Build Your Own Index")
    idx_cats = st.multiselect(
        "Select categories to aggregate",
        list(arxiv_categories.values()),
        default=list(arxiv_categories.values())[:3]
    )
    method = st.selectbox("Aggregation method", ["Sum","Average"])
    apply_smooth_idx = st.checkbox("Apply smoothing to index")
    if apply_smooth_idx:
        ws = st.slider("Index smoothing window", 1, 12, 3)
    std_idx = st.checkbox("Standardize index")

    if idx_cats:
        df_idx = df_filt[idx_cats].copy()
        series = df_idx.sum(axis=1) if method=="Sum" else df_idx.mean(axis=1)
        if apply_smooth_idx:
            series = series.rolling(ws).mean().dropna()
        if std_idx:
            series = (series-series.mean())/series.std()

        fig3 = px.line(
            x=series.index, y=series.values,
            labels={"x":"Date","y":"Index Value"},
            title="Custom Aggregated Index"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Pick at least one category to build your index.")
    st.divider()

    # ==== 4) GROUP COMPARISON ====
    st.header("4️⃣ Major-Field Comparison")
    grp = pd.DataFrame({
        "Computer Science":      df_raw_filt[cs_codes].sum(axis=1),
        "Math & Statistics":     df_raw_filt[math_codes].sum(axis=1),
        "Physics & Related":     df_raw_filt[phys_codes].sum(axis=1),
    })
    fig4 = px.line(
        grp.reset_index(),
        x="index", y=grp.columns,
        labels={"index":"Date","value":"Monthly Preprints","variable":"Field"},
        title="Monthly Volume: CS vs Math/Stats vs Physics"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.divider()

    # ==== 5) YoY GROWTH ANALYSIS ====
    st.header("5️⃣ Year-over-Year Growth")
    yoy_cats = st.multiselect(
        "Select categories for YoY growth",
        list(arxiv_categories.values()),
        default=[list(arxiv_categories.values())[0]]
    )
    if yoy_cats:
        yoy = df_filt[yoy_cats].pct_change(12) * 100
        fig5 = px.line(
            yoy.reset_index(),
            x="index", y=yoy_cats,
            labels={"index":"Date","value":"YoY Growth (%)"},
            title="Year-over-Year Growth (%)"
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Select one or more categories above to see YoY growth.")
    st.divider()

    # ==== 6) DATA EXPORT ====
    st.header("💾 Data Export")
    csv = df_filt.to_csv().encode("utf-8")
    st.download_button("Download filtered data as CSV", csv, "arxiv_filtered.csv", "text/csv")

'''
# Package imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from prophet import Prophet
from datetime import datetime, timedelta
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

# arXiv category taxonomy (see definitions at https://arxiv.org/category_taxonomy)
arxiv_categories = {
    'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
    'astro-ph.EP': 'Earth and Planetary Astrophysics',
    'astro-ph.GA': 'Astrophysics of Galaxies',
    'astro-ph.HE': 'High Energy Astrophysical Phenomena',
    'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
    'astro-ph.SR': 'Solar and Stellar Astrophysics',
    'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
    'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
    'cond-mat.mtrl-sci': 'Materials Science',
    'cond-mat.other': 'Other Condensed Matter',
    'cond-mat.quant-gas': 'Quantum Gases',
    'cond-mat.soft': 'Soft Condensed Matter',
    'cond-mat.stat-mech': 'Statistical Mechanics',
    'cond-mat.str-el': 'Strongly Correlated Electrons',
    'cond-mat.supr-con': 'Superconductivity',
    'cs.AI': 'Artificial Intelligence',
    'cs.AR': 'Hardware Architecture',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.CL': 'Computation and Language',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.DB': 'Databases',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LG': 'Machine Learning',
    'cs.LO': 'Logic in Computer Science',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.MS': 'Mathematical Software',
    'cs.NA': 'Numerical Analysis',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.OH': 'Other Computer Science',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SC': 'Symbolic Computation',
    'cs.SD': 'Sound',
    'cs.SE': 'Software Engineering',
    'cs.SI': 'Social and Information Networks',
    'cs.SY': 'Systems and Control',
    'gr-qc': 'General Relativity and Quantum Cosmology',
    'hep-ex': 'High Energy Physics - Experiment',
    'hep-lat': 'High Energy Physics - Lattice',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-th': 'High Energy Physics - Theory',
    'math-ph': 'Mathematical Physics',
    "math.AC": "Commutative Algebra",
    'math.AG': 'Algebraic Geometry',
    'math.AP': 'Analysis of PDEs',
    'math.AT': 'Algebraic Topology',
    'math.CA': 'Classical Analysis and ODEs',
    'math.CO': 'Combinatorics',
    'math.CT': 'Category Theory',
    'math.CV': 'Complex Variables',
    'math.DG': 'Differential Geometry',
    'math.DS': 'Dynamical Systems',
    'math.FA': 'Functional Analysis',
    'math.GM': 'General Mathematics',
    'math.GN': 'General Topology',
    'math.GR': 'Group Theory',
    'math.GT': 'Geometric Topology',
    'math.HO': 'History and Overview',
    'math.IT': 'Information Theory',
    'math.KT': 'K-Theory and Homology',
    'math.LO': 'Logic',
    'math.MG': 'Metric Geometry',
    "math.MP": "Mathematical Physics",
    'math.NA': 'Numerical Analysis',
    'math.NT': 'Number Theory',
    'math.OA': 'Operator Algebras',
    'math.OC': 'Optimization and Control',
    'math.PR': 'Probability',
    'math.QA': 'Quantum Algebra',
    'math.RA': 'Rings and Algebras',
    'math.RT': 'Representation Theory',
    'math.SG': 'Symplectic Geometry',
    'math.SP': 'Spectral Theory',
    'math.ST': 'Statistics Theory',
    'nlin.AO': 'Adaptation and Self-Organizing Systems',
    'nlin.CD': 'Chaotic Dynamics',
    'nlin.CG': 'Cellular Automata and Lattice Gases',
    'nlin.PS': 'Pattern Formation and Solitons',
    'nlin.SI': 'Exactly Solvable and Integrable Systems',
    'nucl-ex': 'Nuclear Experiment',
    'nucl-th': 'Nuclear Theory',
    'physics.acc-ph': 'Accelerator Physics',
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    'physics.app-ph': 'Applied Physics',
    "physics.atm-clus": "Atomic and Molecular Clusters",
    'physics.atom-ph': 'Atomic Physics',
    'physics.bio-ph': 'Biological Physics',
    'physics.chem-ph': 'Chemical Physics',
    "physics.class-ph": "Classical Physics",
    'physics.comp-ph': 'Computational Physics',
    'physics.data-an': 'Data Analysis, Statistics and Probability',
    'physics.ed-ph': 'Physics Education',
    'physics.flu-dyn': 'Fluid Dynamics',
    'physics.gen-ph': 'General Physics',
    'physics.geo-ph': 'Geophysics',
    'physics.hist-ph': 'History and Philosophy of Physics',
    'physics.ins-det': 'Instrumentation and Detectors',
    'physics.med-ph': 'Medical Physics',
    'physics.optics': 'Optics',
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    'physics.soc-ph': 'Physics and Society',
    'physics.space-ph': 'Space Physics',
    'quant-ph': 'Quantum Physics',
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory"
}

# ======================
# 1. Load & Prepare Data
# ======================
st.title("ArXiv Tracker")
st.markdown("This app leverages data extracted from arXiv to analyze the **evolution of science worldwide**. In particular, with this app you can:")
st.markdown("(1) **Forecast** the future evolution of a certain category.")
st.markdown("(2) **Numerically and visually analyze** different arXiv categories.")
st.markdown("(3) **Decompose** a certain category to analyze its time components.")
st.markdown("(4) **Build your own index** by aggregating categories, smoothing and standardizing.")
st.markdown("**[arXiv](%s)** is an **open-access research paper repository** covering a wide range of disciplines, including physics, mathematics, computer science, biology, statistics, quantitative finance, economics and electronic engineering. As one of the most important platforms for the early dissemination of research results, arXiv is considered a **reliable indicator of global scientific progress**.")
st.divider()

# Load the Excel file; assume the date column is the index.
df = pd.read_excel("arxiv_monthly_publications.xlsx", index_col=0)
df.index = pd.to_datetime(df.index)
df.rename(columns=arxiv_categories, inplace=True)

# ======================
# Sidebar Controls
# ======================
# ====================== (1) Forecasting Parameters ======================
with st.sidebar:
    st.subheader("📈 (1) Forecasting Settings")
    
    # Select a category for forecasting
    all_categories = list(arxiv_categories.values())
    selected_forecast = st.selectbox("Select Category for Forecasting", all_categories)

    # Number of months to forecast
    future_months = st.slider("Months to Forecast", min_value=3, max_value=48, value=12)
    
    # ====================== (2) Category Comparison ======================
    st.header("📊 (2) Statistics Settings")
    
    # -- Category Selection --
    selected_categories = st.multiselect("Select Categories for Examination",
                                                   all_categories, default=all_categories[:3])
    
    # -- Date Range Filter --
    min_date = df.index.min()
    max_date = df.index.max()
    date_range = st.date_input("Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        df_filtered = df.loc[pd.to_datetime(date_range[0]): pd.to_datetime(date_range[1])]
    else:
        df_filtered = df.copy()
    
    # -- Standardization Toggle --
    standardize = st.checkbox("Standardize Data", value=False)
    if standardize:
        # Standardize each selected series
        df_std = (df_filtered[selected_categories] - df_filtered[selected_categories].mean()) / df_filtered[selected_categories].std()
    else:
        df_std = df_filtered[selected_categories]
    
    # ====================== (3) Time Series Decomposition ======================
    st.header("⏱️ (2) Decomposition Settings")
    # Pick one category for time series decomposition (from the selected list)
    selected_decomp = st.selectbox("Select Category for Decomposition", all_categories)

    # ====================== (4) Custom Index ======================
    st.header("🛠️ (4) Build Your Own Index")
    # Select categories to aggregate for the index
    selected_index_categories = st.multiselect(
        "Select Categories to Aggregate for Index",
        all_categories, 
        default=all_categories[:3]
    )
    # Aggregation method: Sum or Average
    agg_method = st.selectbox("Aggregation Method", ["Sum", "Average"])
    # Option to apply smoothing (moving average)
    apply_smoothing = st.checkbox("Apply Smoothing (Moving Average)", value=False)
    if apply_smoothing:
        ma_window = st.slider("Moving Average Window (months)", min_value=1, max_value=12, value=3)
    # Option to standardize the aggregated index
    standardize_index = st.checkbox("Standardize Index", value=False)


# ======================
#       Dashboard
# ======================
# ====================== (1) Forecasting ======================
st.subheader("📈 (1) Forecasting")
url = "https://facebook.github.io/prophet/"
st.write("Here you can **forecast the evolution** of your category with Meta's [Prophet](%s) model." % url)
if selected_forecast:
    # Prepare data for Prophet: reset index and rename columns
    df_prophet = df_filtered[[selected_forecast]].reset_index()
    df_prophet.columns = ["ds", "y"]
    
    # Fit Prophet model with adjustable changepoint prior scale
    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    try:
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=future_months, freq='MS')
        forecast = model.predict(future)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_prophet["ds"], df_prophet["y"], label="Actual Data", marker='o')
        ax1.plot(forecast["ds"], forecast["yhat"], label="Predicted Data", linestyle='dashed')
        ax1.set_title(f"Predicted Monthly Publications for {selected_forecast}")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("ArXiv Monthly Publications")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
    except Exception as e:
        st.error(f"Forecasting error: {e}")
else:
    st.write("Select a category for forecasting.")

st.divider()

# ====================== (2) Stats ======================
st.subheader("📊 (2) Statistics")

# -- Visualization
st.markdown("Visualize how your selected categories **perform over time**.")
fig2, ax2 = plt.subplots(figsize=(10, 6))
for col in df_std.columns:
    ax2.plot(df_std.index, df_std[col], label=col)
ax2.set_title("Monthly Publications Comparison")
ax2.set_xlabel("Date")
ax2.set_ylabel("Standardized Value" if standardize else "Publications")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# -- Summary stats
st.markdown("**Basic statistics** of your selected categories: ")
if selected_categories:
    st.write(df_filtered[selected_categories].describe())
else:
    st.write("Please select at least one category.")

# -- Correlation heatmap
st.markdown("Examine how your selected categories relate to each other with the **correlation heatmap** below.")
if selected_categories:
    corr = df_filtered[selected_categories].corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title("Correlation between Categories")
    st.pyplot(fig3)
else:
    st.write("Please select categories to view the correlation heatmap.")

st.divider()

# ======================
# Time Series Decomposition
# ======================
st.subheader("⏱️ (3) Time Series Decomposition")
st.markdown("Here you can decompose your selected category into the **trend**, **seasonality** and **residuals** for better understanding of its performance over time.")
url2 = "https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html"
st.markdown("See the **decomposition methodology** [here](%s)." % url2)
if selected_decomp:
    # Using an additive model and period=12 (monthly data)
    try:
        decomp_result = sm.tsa.seasonal_decompose(df_filtered[selected_decomp], model='additive', period=12)
        fig4 = decomp_result.plot()
        fig4.set_size_inches(10, 8)
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Decomposition error: {e}")
else:
    st.write("Select a category for decomposition.")

st.divider()

# ====================== (4) Custom Index ======================
st.subheader("🛠️ (4) Build Your Own Index")
if selected_index_categories:
    # Extract the selected categories from the filtered DataFrame.
    df_index = df_filtered[selected_index_categories].copy()
    
    # Aggregate data according to the chosen method.
    if agg_method == "Sum":
        index_series = df_index.sum(axis=1)
    else:  # "Average"
        index_series = df_index.mean(axis=1)
    
    # Apply smoothing if selected (using a rolling window).
    if apply_smoothing:
        index_series = index_series.rolling(window=ma_window).sum()
    
    # Standardize if selected.
    if standardize_index:
        index_series = (index_series - index_series.mean()) / index_series.std()
    
    # Plot the custom index.
    fig_index, ax_index = plt.subplots(figsize=(10, 5))
    ax_index.plot(index_series.index, index_series, marker='o')
    ax_index.set_title("Custom Aggregated Index")
    ax_index.set_xlabel("Date")
    ax_index.set_ylabel("Index Value")
    ax_index.grid(True)
    st.pyplot(fig_index)
else:
    st.write("Please select at least one category to build the index.")

st.divider()
    
# ======================
# Data Export Options
# ======================
st.subheader("💾 Data Export")
# CSV Export of filtered data for selected categories
if not df_filtered[selected_categories].empty:
    csv = df_filtered[selected_categories].to_csv().encode('utf-8')
    st.download_button("Export data as CSV", csv, "arxiv_data.csv", "text/csv")
else:
    st.write("No data available for export.")
'''
