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
    'eess.AS': 'Audio and Speech Processing',
    'eess.IV': 'Image and Video Processing',
    'eess.SP': 'Signal Processing',
    "eess.SY": "Systems and Control",
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
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",
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
st.title("ArXiv Publications Dashboard")
st.write("Explore and forecast trends in monthly arXiv publications.")

# Load the Excel file; assume the date column is the index.
df = pd.read_excel("arxiv_monthly_publications.xlsx", index_col=0)
df.index = pd.to_datetime(df.index)
df.rename(columns=arxiv_categories, inplace=True)

# ======================
# Sidebar Controls
# ======================
# ====================== (1) Forecasting Parameters ======================
st.sidebar.subheader("Forecast Model Settings")
st.write("Forecasts are calculated with Meta's Prophet model.")
# Select a category for forecasting
all_categories = list(arxiv_categories.values())
selected_forecast = st.sidebar.selectbox("Select Category for Forecasting", all_categories)

# Adjust Prophet’s changepoint prior scale
cp_scale = st.sidebar.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, step=0.001)
st.write("The `changepoint prior scale` is a regularization term that controls how much the model is allowed to change its trend.")
st.write("If you suspect the variable is affected by many external events or regime shifts, a higher value might capture those dynamics better. On the other hand, if you think the data is more stable and changes slowly, a lower value might be appropriate.")
# Number of months to forecast
future_months = st.sidebar.slider("Months to Forecast", min_value=3, max_value=48, value=12)

# ====================== (2) Category Comparison ======================
st.sidebar.header("Stats Settings")

# -- Category Selection --
selected_categories = st.sidebar.multiselect("Select Categories for Examination",
                                               all_categories, default=all_categories[:3])

# -- Date Range Filter --
min_date = df.index.min()
max_date = df.index.max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
if len(date_range) == 2:
    df_filtered = df.loc[pd.to_datetime(date_range[0]): pd.to_datetime(date_range[1])]
else:
    df_filtered = df.copy()

# -- Standardization Toggle --
standardize = st.sidebar.checkbox("Standardize Data", value=False)
if standardize:
    # Standardize each selected series
    df_std = (df_filtered[selected_categories] - df_filtered[selected_categories].mean()) / df_filtered[selected_categories].std()
else:
    df_std = df_filtered[selected_categories]

# ====================== (3) Time Series Decomposition ======================
st.sidebar.header("Decomposition Settings")
# Pick one category for time series decomposition (from the selected list)
selected_decomp = st.sidebar.selectbox("Select Category for Decomposition", all_categories)

# ======================
#       Dashboard
# ======================
# ====================== (1) Forecasting ======================
st.subheader("Forecasting")
if selected_forecast:
    # Prepare data for Prophet: reset index and rename columns
    df_prophet = df_filtered[[selected_forecast]].reset_index()
    df_prophet.columns = ["ds", "y"]
    
    # Fit Prophet model with adjustable changepoint prior scale
    model = Prophet(changepoint_prior_scale=cp_scale, weekly_seasonality=False, daily_seasonality=False)
    try:
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=future_months, freq='MS')
        forecast = model.predict(future)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_prophet["ds"], df_prophet["y"], label="Actual Data", marker='o')
        ax1.plot(forecast["ds"], forecast["yhat"], label="Predicted Data", linestyle='dashed')
        ax1.set_title(f"Predicted Monthly Publications for {field_label}")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("ArXiv Monthly Publications")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
    except Exception as e:
        st.error(f"Forecasting error: {e}")
else:
    st.write("Select a category for forecasting.")

# ====================== (2) Stats ======================
st.subheader("Statistics")

# -- Visualization
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
if selected_categories:
    st.write(df_filtered[selected_categories].describe())
else:
    st.write("Please select at least one category.")

# -- Correlation heatmap
if selected_categories:
    corr = df_filtered[selected_categories].corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title("Correlation between Categories")
    st.pyplot(fig3)
else:
    st.write("Please select categories to view the correlation heatmap.")

# ======================
# Time Series Decomposition
# ======================
st.subheader("Time Series Decomposition")
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
    
# ======================
# Data Export Options
# ======================
st.subheader("Export Data")
# CSV Export of filtered data for selected categories
if not df_filtered[selected_categories].empty:
    csv = df_filtered[selected_categories].to_csv().encode('utf-8')
    st.download_button("Export Data as CSV", csv, "arxiv_data.csv", "text/csv")
else:
    st.write("No data available for export.")
