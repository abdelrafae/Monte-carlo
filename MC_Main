import pathlib, textwrap, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numexpr as ne
from typing import Dict, List

st.set_page_config(page_title="Monte Carlo Studio – Bounds-Only", layout="wide", page_icon="🎛️")

# Force light theme
cfg_dir = pathlib.Path(".streamlit")
cfg_dir.mkdir(exist_ok=True)
(cfg_dir / "config.toml").write_text(textwrap.dedent("""
[theme]
base = "light"
primaryColor = "#2563eb"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f6f8ff"
textColor = "#111827"
"""), encoding="utf-8")

# ============================== HELPERS ======================================

def lhs(n, k, rng):
   """Latin Hypercube Sampling: n samples, k variables, U(0,1)."""
   cut = np.linspace(0, 1, n + 1)
   u = rng.uniform(size=(n, k))
   a = cut[:n]
   b = cut[1:n+1]
   rdpoints = u * (b - a)[:, None] + a[:, None]
   H = np.zeros_like(rdpoints)
   for j in range(k):
       order = rng.permutation(n)
       H[:, j] = rdpoints[order, 0]
   return H

def safe_eval(expr: str, local_vars: Dict[str, np.ndarray]) -> np.ndarray:
   """Use numexpr first; if it fails, fall back to very restricted eval on numpy."""
   try:
       return ne.evaluate(expr, local_dict=local_vars)
   except Exception:
       safe_globals = {"__builtins__": {}}
       safe_locals = {
           **local_vars,
           'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'exp': np.exp, 'log': np.log,
           'sqrt': np.sqrt, 'abs': np.abs, 'minimum': np.minimum, 'maximum': np.maximum,
           'where': np.where, 'clip': np.clip, 'pi': np.pi
       }
       return eval(expr, safe_globals, safe_locals)

def percentile(a, q):
   try:
       return float(np.percentile(a, q, method="linear"))
   except TypeError:
       return float(np.percentile(a, q))

# ---- Parameter inference from (min, max) bounds -----------------------------

def infer_normal_from_bounds(xmin, xmax, z):
   """Assume xmin/xmax are central-coverage quantiles at ±z (e.g., z=1.645 for 90%).
   Return mean, std."""
   mean = (xmax + xmin) / 2.0
   std  = (xmax - xmin) / (2.0 * z) if z > 0 else 0.0
   return mean, max(std, 1e-16)

def infer_lognormal_from_bounds(xmin, xmax, z):
   """xmin, xmax > 0. Treat them as central coverage quantiles at ±z for ln(X)."""
   xmin = max(xmin, 1e-16)
   xmax = max(xmax, 1e-16)
   ln_min, ln_max = np.log(xmin), np.log(xmax)
   mu = (ln_max + ln_min) / 2.0
   sigma = (ln_max - ln_min) / (2.0 * z) if z > 0 else 0.0
   return mu, max(sigma, 1e-16)

def to_distribution_samples_from_bounds(dist: str, xmin: float, xmax: float, u: np.ndarray, z: float):
   """
   Map U(0,1) -> target distribution using only (min, max) entered by the user.
   Assumptions:
     - uniform: exact bounds
     - triangular: symmetric, mode = (min + max)/2
     - normal: min/max are central-coverage quantiles at ±z -> derive mean/std
     - lognormal: same idea in log-space -> derive μ,σ
     - PERT: symmetric (min, mode=(min+max)/2, max) with λ=4
     - fixed: value = (min + max)/2
   """
   dist = dist.lower()
   if xmin > xmax:
       xmin, xmax = xmax, xmin  # auto-fix swapped bounds

   if dist == "fixed":
       val = 0.5*(xmin + xmax)
       return np.full_like(u, val, dtype=float)

   if dist == "uniform":
       return xmin + (xmax - xmin) * u

   if dist == "triangular":
       a, c, b = xmin, 0.5*(xmin+xmax), xmax
       if not (a <= c <= b):
           c = min(max(c, a), b)
       Fc = (c - a) / (b - a) if b > a else 0.5
       out = np.where(u < Fc,
                      a + np.sqrt(u * (b - a) * (c - a)),
                      b - np.sqrt((1 - u) * (b - a) * (b - c)))
       return out

   if dist == "pert":
       # Modified Beta-PERT: symmetric by default; lambda=4
       a, m, b = xmin, 0.5*(xmin+xmax), xmax
       lam = 4.0
       from scipy.stats import beta as betad
       alpha = 1 + lam * (m - a) / (b - a) if b > a else 1.0
       beta  = 1 + lam * (b - m) / (b - a) if b > a else 1.0
       x = betad.ppf(u, alpha, beta)
       return a + x * (b - a)

   if dist == "normal":
       mean, std = infer_normal_from_bounds(xmin, xmax, z)
       from scipy.stats import norm
       return norm.ppf(u, loc=mean, scale=std)

   if dist == "lognormal":
       mu, sigma = infer_lognormal_from_bounds(max(xmin, 1e-16), max(xmax, 1e-16), z)
       from scipy.stats import lognorm
       s = sigma; scale = np.exp(mu)
       return lognorm.ppf(u, s=s, scale=scale)

   raise ValueError(f"Unsupported distribution: {dist}")

# ============================== SIDEBAR =======================================

with st.sidebar:
   st.title("Settings")
   n = st.number_input("Iterations", 1000, 5_000_000, value=200000, step=10000)
   seed = st.number_input("Random Seed", value=42)
   use_lhs = st.toggle("Use Latin Hypercube (LHS)", value=True)
   corr_on = st.toggle("Impose Correlations", value=False)

   st.markdown("---")
   st.caption("Bounds interpretation for Normal/Lognormal")
   coverage = st.slider("Central coverage % for (min, max)", min_value=50, max_value=99, value=90, step=1,
                        help="For Normal/Lognormal: min/max are treated as the central coverage limits.")
   # convert central coverage percentage to z-score (two-tailed)
   # central = 2*Phi(z) - 1 -> Phi(z) = (1+central)/2
   from math import erf, sqrt
   # Approx inverse-CDF for normal using scipy if available; otherwise quick approx
   try:
       from scipy.stats import norm
       z_from_cov = float(norm.ppf(0.5 + coverage/200.0))
   except Exception:
       # Winitzki approximation via erfinv:
       def erfinv(x):
           a = 0.147
           ln = np.log(1 - x**2)
           term = 2/(np.pi*a) + ln/2
           return np.sign(x) * np.sqrt(np.sqrt(term**2 - ln/a) - term)
       z_from_cov = float(np.sqrt(2) * erfinv(coverage/100.0))

z = max(z_from_cov, 1e-6)

st.title("Monte Carlo Studio – Bounds-Only (5-decimal inputs)")
st.caption("For EACH variable: choose a distribution and enter only **min** and **max** (5-decimal precision). Parameters are inferred internally.")

# ============================== DYNAMIC VARIABLES UI ==========================

DIST_LIST = ["fixed", "uniform", "triangular", "normal", "lognormal", "PERT"]

k = st.number_input("Number of variables", min_value=1, max_value=100, value=4, step=1)
vars_specs: List[Dict] = []

default_names = [f"X{i+1}" for i in range(int(k))]

for i in range(int(k)):
   with st.expander(f"Variable {i+1}", expanded=(i < 6)):
       col1, col2, col3 = st.columns([1.2, 1, 1])
       with col1:
           name = st.text_input(f"Name (v{i+1})", value=default_names[i], key=f"name_{i}")
       with col2:
           dist = st.selectbox(f"Distribution (v{i+1})", DIST_LIST, index=3 if i==1 else 1, key=f"dist_{i}")
       with col3:
           st.caption("Bounds format (min/max)")
       c1, c2 = st.columns(2)
       xmin = c1.number_input(f"min (v{i+1})", value=0.00000, format="%.5f", key=f"min_{i}")
       xmax = c2.number_input(f"max (v{i+1})", value=1.00000, format="%.5f", key=f"max_{i}")

       if dist.lower() == "lognormal" and xmin <= 0:
           st.warning(f"v{i+1}: Lognormal requires min>0; using 1e-6 internally.")
       vars_specs.append({"name": name.strip(), "dist": dist, "min": float(xmin), "max": float(xmax)})

# Check unique names
names = [s["name"] for s in vars_specs]
if len(set(names)) != len(names):
   st.error("Variable names must be unique.")
   st.stop()

# ============================== CORRELATION MATRIX ============================

corr_matrix = None
if corr_on and len(vars_specs) > 1:
   st.subheader("Correlation Matrix")
   st.caption("Edit the matrix (1.0 on diagonal). Values ~[-0.95, 0.95].")
   default = np.eye(len(vars_specs))
   corr_df = pd.DataFrame(default, index=names, columns=names)
   corr_df = st.data_editor(corr_df, use_container_width=True, key="corr_edit")
   corr_matrix = corr_df.to_numpy(dtype=float)
   if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-6):
       st.error("Correlation matrix must be symmetric.")
       st.stop()
   try:
       np.linalg.cholesky(corr_matrix + 1e-12*np.eye(corr_matrix.shape[0]))
   except Exception as e:
       st.error(f"Correlation matrix must be positive definite: {e}")
       st.stop()

# ============================== EXPRESSION ====================================

st.subheader("Model Expression")
st.caption("Use your variable names. Example: `NPV = Qi*365 - CAPEX` → **write only the right-hand side**: `Qi*365 - CAPEX`. The result is **OUTPUT**.")
expr = st.text_input("Expression", value="+".join(names))

st.markdown("---")

# ============================== RUN ===========================================
run = st.button("Run Simulation", type="primary")

if run:
   rng = np.random.default_rng(int(seed))
   k = len(vars_specs)
   # Base U(0,1)
   U = lhs(int(n), k, rng) if use_lhs else rng.uniform(size=(int(n), k))

   # Correlate U via Gaussian copula if needed
   if corr_on and corr_matrix is not None:
       from scipy.stats import norm
       Z = norm.ppf(U)
       L = np.linalg.cholesky(corr_matrix)
       Zc = Z @ L.T
       U = norm.cdf(Zc)

   # Map marginals using bounds-only spec
   samples = {}
   for j, spec in enumerate(vars_specs):
       samples[spec["name"]] = to_distribution_samples_from_bounds(
           spec["dist"], spec["min"], spec["max"], U[:, j], z
       )

   # Evaluate expression
   try:
       y = safe_eval(expr, samples)
       y = np.asarray(y, dtype=float).reshape(-1)
   except Exception as e:
       st.error(f"Expression error: {e}")
       st.stop()

   df = pd.DataFrame(samples)
   df["OUTPUT"] = y

   # Summary
   st.subheader("Results")
   colA, colB = st.columns([1, 2])
   with colA:
       mean = float(np.mean(y)); std = float(np.std(y, ddof=1))
       p10 = percentile(y, 10); p50 = percentile(y, 50); p90 = percentile(y, 90)
       st.metric("Mean", f"{mean:,.5f}")
       st.metric("Std Dev", f"{std:,.5f}")
       st.metric("P10", f"{p10:,.5f}")
       st.metric("P50 (Median)", f"{p50:,.5f}")
       st.metric("P90", f"{p90:,.5f}")
   with colB:
       st.write("Preview (first 10 rows):")
       st.dataframe(df.head(10), use_container_width=True)

   # Charts
   st.subheader("Charts")
   fig1, ax1 = plt.subplots()
   ax1.hist(y, bins=50)
   ax1.set_title("Output Distribution")
   ax1.set_xlabel("OUTPUT"); ax1.set_ylabel("Frequency")
   st.pyplot(fig1)

   fig2, ax2 = plt.subplots()
   ys = np.sort(y); xs = np.linspace(0, 1, len(ys))
   ax2.plot(ys, xs)
   ax2.set_title("Empirical CDF")
   ax2.set_xlabel("OUTPUT"); ax2.set_ylabel("Cumulative Probability")
   st.pyplot(fig2)

   # Sensitivity
   st.subheader("Sensitivity (Tornado)")
   cors = []
   for nm in names:
       try:
           r = np.corrcoef(df[nm], y)[0,1]
       except Exception:
           r = np.nan
       cors.append((nm, r))
   sens = pd.DataFrame(cors, columns=["Variable", "Correlation"]).dropna().sort_values(
       "Correlation", key=lambda s: s.abs(), ascending=True
   )
   fig3, ax3 = plt.subplots()
   ax3.barh(sens["Variable"], sens["Correlation"])
   ax3.set_title("Pearson Correlation with OUTPUT")
   ax3.set_xlabel("Correlation")
   st.pyplot(fig3)

   # Downloads
   st.subheader("Downloads")
   st.download_button("Download Simulated Data (CSV)", df.to_csv(index=False).encode("utf-8"),
                      file_name="mc_bounds_results.csv")
   summary = pd.DataFrame({"metric": ["mean","std","p10","p50","p90"],
                           "value": [mean, std, p10, p50, p90]})
   st.download_button("Download Summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                      file_name="mc_bounds_summary.csv")

else:
   st.info("Set the number of variables, give each a name, pick a distribution, enter only min/max (5-decimals), write your expression, then Run.")
