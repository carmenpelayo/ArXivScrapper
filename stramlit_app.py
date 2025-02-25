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
inverted_dict = {v: k for k, v in arxiv_categories.items()}

# ======================
# 1. Load & Prepare Data
# ======================
st.title("ArXiv Publications Dashboard")
st.write("Explore and forecast trends in monthly arXiv publications.")

# Load the Excel file; assume the date column is the index.
df = pd.read_excel("arxiv_monthly_publications.xlsx", index_col=0)
df.index = pd.to_datetime(df.index)

# ======================
# Sidebar Controls
# ======================
st.sidebar.header("Filter and Settings")

# -- Date Range Filter --
min_date = df.index.min()
max_date = df.index.max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
if len(date_range) == 2:
    df_filtered = df.loc[pd.to_datetime(date_range[0]): pd.to_datetime(date_range[1])]
else:
    df_filtered = df.copy()

# -- Category Selection --
all_categories = list(arxiv_categories.values())
selected_categories = st.sidebar.multiselect("Select Categories for Comparison",
                                               all_categories, default=all_categories[:3])
selected_categories = [inverted_dict[label] for label in selected_categories]

# -- Normalization Toggle --
normalize = st.sidebar.checkbox("Normalize Data", value=False)
if normalize:
    # Normalize each selected series to the [0,1] range
    df_norm = (df_filtered[selected_categories] - df_filtered[selected_categories].min()) / (
                df_filtered[selected_categories].max() - df_filtered[selected_categories].min())
else:
    df_norm = df_filtered[selected_categories]

# -- Forecasting Parameters --
st.sidebar.subheader("Forecast Model Settings")
# Adjust Prophet’s changepoint prior scale
cp_scale = st.sidebar.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, step=0.001)
# Select a category for forecasting (from the ones selected above)
if selected_categories:
    selected_forecast = st.sidebar.selectbox("Select Category for Forecasting", selected_categories)
else:
    selected_forecast = None
# Number of months to forecast
future_months = st.sidebar.slider("Months to Forecast", min_value=3, max_value=48, value=12)

# -- Decomposition --
# Pick one category for time series decomposition (from the selected list)
if selected_categories:
    selected_decomp = st.sidebar.selectbox("Select Category for Decomposition", selected_categories)
else:
    selected_decomp = None

# ======================
# 2. Summary Statistics
# ======================
st.subheader("Summary Statistics")
if selected_categories:
    st.write(df_filtered[selected_categories].describe())
else:
    st.write("Please select at least one category.")

# ======================
# 3. Interactive Comparison Plot
# ======================
st.subheader("Category Comparison")
fig1, ax1 = plt.subplots(figsize=(10, 6))
for col in df_norm.columns:
    ax1.plot(df_norm.index, df_norm[col], label=col)
ax1.set_title("Monthly Publications Comparison")
ax1.set_xlabel("Date")
ax1.set_ylabel("Normalized Value" if normalize else "Publications")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# ======================
# 4. Correlation Heatmap
# ======================
st.subheader("Correlation Heatmap")
if selected_categories:
    corr = df_filtered[selected_categories].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title("Correlation between Categories")
    st.pyplot(fig2)
else:
    st.write("Please select categories to view the correlation heatmap.")

# ======================
# 5. Time Series Decomposition
# ======================
st.subheader("Time Series Decomposition")
if selected_decomp:
    # Using an additive model and period=12 (monthly data)
    try:
        decomp_result = sm.tsa.seasonal_decompose(df_filtered[selected_decomp], model='additive', period=12)
        fig3 = decomp_result.plot()
        fig3.set_size_inches(10, 8)
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Decomposition error: {e}")
else:
    st.write("Select a category for decomposition.")

# ======================
# 6. Forecasting with Prophet
# ======================
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
        fig4 = model.plot(forecast)
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Forecasting error: {e}")
else:
    st.write("Select at least one category for forecasting.")

# ======================
# 7. Data Export Options
# ======================
st.subheader("Export Data and Plots")
# CSV Export of filtered data for selected categories
if not df_filtered[selected_categories].empty:
    csv = df_filtered[selected_categories].to_csv().encode('utf-8')
    st.download_button("Export Data as CSV", csv, "arxiv_data.csv", "text/csv")
else:
    st.write("No data available for export.")

'''
---------------------------------
st.title("Tracking the Evolution of Science with arXiv Publications")
st.write("""TBD""")

# Data processing
df = pd.read_excel("arxiv_monthly_publications.xlsx", sheet_name="raw", index_col=0)
df.index = pd.to_datetime(df.index)
df = df[df.index <= "2025-01"]

# Interactive function for forecasting
def forecast_trends(field_label, future_months):
    """
    field_label: El nombre descriptivo elegido por el usuario (p.ej. "Cosmology and Nongalactic Astrophysics")
    future_months: Entero con el número de meses a predecir
    """
    # Obtener la clave real del DataFrame a partir del nombre legible
    real_key = inverted_dict[field_label]

    # Preparar datos para Prophet
    df_prophet = df[[real_key]].reset_index()  # asumiendo que df tiene un DatetimeIndex
    df_prophet.columns = ["ds", "y"]

    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=future_months, freq='MS')  # 'MS' = start of month
    forecast = model.predict(future)

    # Crear la figura con matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prophet["ds"], df_prophet["y"], label="Actual Data", marker='o')
    ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Data", linestyle='dashed')
    ax.set_title(f"Future Predictions for {field_label}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Publications")
    ax.legend()
    ax.grid(True)

    # Devolver la figura para que Streamlit la muestre
    return fig

# 1) Selectbox con los nombres de las categorías
selected_field_label = st.selectbox(
    "Which arXiv category would you like to visualize?",
    list(arxiv_categories.values())  # Muestra los valores legibles
)

# 2) Slider para elegir meses a predecir
future_months = st.slider("How many months to predict into the future?", 
                          min_value=3, max_value=48, value=12)

# 3) Botón o acción directa para generar la predicción
if st.button("Generate Forecast"):
    fig = forecast_trends(selected_field_label, future_months)
    st.pyplot(fig)

'''
