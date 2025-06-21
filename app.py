import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
st.title("Asset Allocation Monte Carlo Dashboard")

st.markdown(
    """This interactive tool lets you:
* Allocate capital across **Stocks, Government Bonds, Inflation‑Linked Bonds,
  Corporate Bonds, Commodities, and Gold**
* Apply a **portfolio‑level leverage factor**
* Edit the **expected return, volatility** and **correlation matrix**
* Run a Monte‑Carlo engine that simulates thousands of paths and shows
  annualised return, volatility and maximum drawdown distributions
  together with key summary statistics.
""")

# ----------------------------
# Asset universe & default parameters
assets = [
    "Stocks",
    "Government Bonds",
    "Inflation‑Linked Bonds",
    "Corporate Bonds",
    "Commodities",
    "Gold"
]

default_returns = {
    "Stocks": 0.07,
    "Government Bonds": 0.03,
    "Inflation‑Linked Bonds": 0.03,
    "Corporate Bonds": 0.04,
    "Commodities": 0.04,
    "Gold": 0.03,
}

default_vols = {
    "Stocks": 0.15,
    "Government Bonds": 0.05,
    "Inflation‑Linked Bonds": 0.06,
    "Corporate Bonds": 0.07,
    "Commodities": 0.12,
    "Gold": 0.14,
}

default_corr = pd.DataFrame(
    [
        [1.00, 0.20, 0.15, 0.35, 0.30, 0.10],
        [0.20, 1.00, 0.60, 0.50, 0.10, 0.00],
        [0.15, 0.60, 1.00, 0.40, 0.15, 0.05],
        [0.35, 0.50, 0.40, 1.00, 0.25, 0.05],
        [0.30, 0.10, 0.15, 0.25, 1.00, 0.20],
        [0.10, 0.00, 0.05, 0.05, 0.20, 1.00],
    ],
    index=assets,
    columns=assets,
)

# ----------------------------
st.sidebar.header("1️⃣ Portfolio Weights")

def weight_slider(asset):
    return st.sidebar.slider(f"Weight: {asset}", 0.0, 1.0, 0.2, 0.01)

weights = {asset: weight_slider(asset) for asset in assets}
total_weight = sum(weights.values())
st.sidebar.write(f"**Total weight:** {total_weight:.2f}")

leverage = st.sidebar.slider("2️⃣ Leverage factor", 0.0, 3.0, 1.0, 0.05)

# Warn if weights don't sum to 1
if abs(total_weight - 1) > 1e-6:
    st.sidebar.warning("Weights do not sum to 1. They will be normalised before simulation.")

# ----------------------------
st.sidebar.header("3️⃣ Asset Assumptions")

exp_returns = {}
volatilities = {}
for asset in assets:
    exp_returns[asset] = st.sidebar.number_input(
        f"{asset} – expected annual return",
        value=default_returns[asset],
        format="%.2f",
        step=0.01,
    )
    volatilities[asset] = st.sidebar.number_input(
        f"{asset} – annual volatility",
        value=default_vols[asset],
        format="%.2f",
        step=0.01,
    )

st.sidebar.header("4️⃣ Correlation Matrix")
corr_edit = st.sidebar.data_editor(
    default_corr.copy(),
    num_rows="fixed",
    use_container_width=True,
    key="corr_editor",
)

# Force symmetry & ones on diagonal
for i in range(len(assets)):
    corr_edit.iloc[i, i] = 1.0
    for j in range(i + 1, len(assets)):
        corr_edit.iloc[j, i] = corr_edit.iloc[i, j]

# ----------------------------
st.sidebar.header("5️⃣ Simulation Parameters")
years = st.sidebar.number_input("Years to simulate", 1, 50, 10, 1)
n_paths = st.sidebar.number_input("Number of Monte‑Carlo paths", 1_000, 50_000, 10_000, 1_000)
seed = st.sidebar.number_input("Random seed (optional, 0 = random)", 0, 2**32 - 1, 0)

if st.sidebar.button("🚀 Run Simulation"):
    # ------------------------
    st.subheader("Simulation in progress…")
    np.random.seed(None if seed == 0 else int(seed))

    # Vectorise inputs
    mu = np.array([exp_returns[a] for a in assets])
    sigma = np.array([volatilities[a] for a in assets])
    weights_vec = np.array([weights[a] for a in assets])
    if total_weight != 0:
        weights_vec = weights_vec / total_weight
    weights_vec = weights_vec * leverage

    # Convert to daily
    trading_days = 252
    mu_d = mu / trading_days
    sigma_d = sigma / np.sqrt(trading_days)

    # Covariance matrix
    cov = np.outer(sigma_d, sigma_d) * corr_edit.values

    # Cholesky factor
    chol = np.linalg.cholesky(cov)

    # Pre‑allocate results
    rets = np.empty(n_paths)
    vols = np.empty(n_paths)
    mdds = np.empty(n_paths)

    horizon = int(years * trading_days)

    progress = st.empty()
    batch = max(1, n_paths // 100)  # update progress 100 times

    for k in range(n_paths):
        z = np.random.normal(size=(horizon, len(assets)))
        paths = mu_d + z @ chol.T
        port_daily = paths @ weights_vec
        cum = np.cumprod(1 + port_daily)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        mdds[k] = dd.min()
        rets[k] = cum[-1] ** (1 / years) - 1
        vols[k] = np.std(port_daily) * np.sqrt(trading_days)

        if (k + 1) % batch == 0 or k == n_paths - 1:
            progress.progress((k + 1) / n_paths, text=f"{k + 1:,}/{n_paths:,} paths")

    progress.empty()  # clear progress bar

    res = pd.DataFrame(
        {
            "Annual Return": rets,
            "Annual Volatility": vols,
            "Max Drawdown": mdds,
        }
    )

    st.subheader("Results Summary")
    st.write(res.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T)

    # Histogram of returns
    st.subheader("Distribution of Annualised Returns")
    fig, ax = plt.subplots()
    ax.hist(rets, bins=50)
    ax.set_xlabel("Annual Return")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Scatter plot Return vs Vol
    st.subheader("Risk–Return Scatter")
    fig2, ax2 = plt.subplots()
    ax2.scatter(vols, rets, alpha=0.3, s=5)
    ax2.set_xlabel("Annual Volatility")
    ax2.set_ylabel("Annual Return")
    ax2.set_title("Each point = one Monte‑Carlo path")
    st.pyplot(fig2)

    # Drawdown plot example of one path
    st.subheader("Example Path Drawdown")
    example = 0
    z = np.random.normal(size=(horizon, len(assets)))
    path = mu_d + z @ chol.T
    port_daily = path @ weights_vec
    cum = np.cumprod(1 + port_daily)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    fig3, ax3 = plt.subplots()
    ax3.plot(dd)
    ax3.set_ylabel("Drawdown")
    ax3.set_xlabel("Days")
    st.pyplot(fig3)

    st.success("Simulation complete!")

else:
    st.info("Adjust parameters in the sidebar and click **Run Simulation**.")
