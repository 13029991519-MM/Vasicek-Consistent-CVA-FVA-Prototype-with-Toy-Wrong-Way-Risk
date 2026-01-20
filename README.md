# Vasicek-Consistent CVA/FVA Prototype with Toy Wrong-Way Risk (WWR)

A compact, research-oriented CCR/XVA prototype for an interest-rate swap under a **one-factor Vasicek short-rate model**, with:
- **Curve-consistent discounting** via analytic Vasicek ZCB pricing (no `exp(-r·τ)` shortcut),
- Baseline **unilateral CVA** (constant hazard) and a **simple FVA proxy**,
- A **toy WWR/RWR mechanism** by making hazard intensity state-dependent (rate-linked and/or exposure-linked),
- **Monte Carlo standard errors** for EE and CVA, plus CSV exports for paper-ready plots.

This repo is meant to read like a mini “numerical finance note”: clear model choices, reproducible runs, and diagnostics.

---

## Why “Vasicek-consistent” matters

A common shortcut for discounting along simulated short-rate paths is `exp(-r_t * tau)`. Under Vasicek, the model-implied ZCB has a closed form, so you can price/discount consistently using:
\[
P(t,T) = A(\tau)\exp(-B(\tau)\,r_t).
\]
This code uses the analytic ZCB consistently for:
- swap MTM,
- discount factors \(P(0,t)\),
- CVA/FVA aggregation.

---

## What’s implemented

### 1) Market model (Vasicek)
- Exact discretization for the Vasicek OU process.
- Analytic ZCB functions \(A(\tau), B(\tau)\), and ZCB pricing.

### 2) Swap valuation and exposures
- Receiver swap MTM using ZCB representation.
- Exposure profiles:
  - \(E^+(t)=\max(MTM(t),0)\),
  - EE = mean of \(E^+(t)\), ENE = mean of negative MTM.

### 3) XVA
- **Unilateral CVA (baseline)** under constant hazard with survival increments and analytic \(P(0,t)\). :contentReference[oaicite:0]{index=0}  
- **Simple FVA proxy**: \( \sum P(0,t_i)\, s_f \, EE(t_i)\, \Delta t \). :contentReference[oaicite:1]{index=1}  

### 4) Toy WWR/RWR (state-dependent intensity)
- **Log-link intensity** (strictly positive):
  \[
  \lambda(t)=\lambda_0\exp\big(\beta(r_t-r_{\text{center}}) + \gamma \cdot E^+(t)/N\big),
  \]
  with an optional cap to prevent numerical blow-ups.
- **Pathwise survival + pathwise CVA** estimator (captures dependence via \(E^+\times \Delta PD\)). :contentReference[oaicite:2]{index=2}  

### 5) Monte Carlo error + exports
- EE with standard error: `EE ± SE` per time bucket.
- CVA pathwise SE for both baseline and WWR experiments.
- Exports:
  - `ee_curve.csv` with columns `t,EE,SE`,
  - `cva_vs_beta.csv` with `beta,CVA,SE,uplift,lam_* diagnostics`.
  :contentReference[oaicite:3]{index=3}  

---

## Repository layout (recommended)

> Your current script filename is `import numpy as np.py` (Windows will allow it, but it’s not repo-friendly). Rename to something like:

