import numpy as np

# ============================================================
# 1) Vasicek: exact simulation + analytic ZCB pricing
# ============================================================

def simulate_vasicek_exact(
    r0: float,
    a: float,
    b: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
):
    """
    Exact discretization under Vasicek:
    r_{t+dt} = b + (r_t - b) e^{-a dt} + sigma * sqrt((1-e^{-2a dt})/(2a)) * Z
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    r = np.zeros((n_paths, n_steps + 1), dtype=float)
    r[:, 0] = r0

    exp_adt = np.exp(-a * dt)
    vol = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))

    z = rng.standard_normal((n_paths, n_steps))
    for i in range(n_steps):
        r[:, i + 1] = b + (r[:, i] - b) * exp_adt + vol * z[:, i]

    return times, r


def vasicek_B(a: float, tau: np.ndarray):
    return (1.0 - np.exp(-a * tau)) / a


def vasicek_A(a: float, b: float, sigma: float, tau: np.ndarray):
    """
    A(t,T) depends only on tau in time-homogeneous Vasicek:
    A(t,T)=exp( (b - sigma^2/(2a^2)) (B - tau) - (sigma^2/(4a)) B^2 )
    """
    B = vasicek_B(a, tau)
    term1 = (b - (sigma**2) / (2.0 * a**2)) * (B - tau)
    term2 = (sigma**2) / (4.0 * a) * (B**2)
    return np.exp(term1 - term2)


def zcb_price_vasicek(r_t: np.ndarray, a: float, b: float, sigma: float, tau: np.ndarray):
    """
    P(t, t+tau) = A(tau) * exp(-B(tau) * r_t)
    r_t: shape (n_paths,) or scalar
    tau: scalar or array
    Returns:
      if tau scalar -> shape like r_t
      if tau array  -> broadcasted
    """
    tau = np.asarray(tau)
    A = vasicek_A(a, b, sigma, tau)
    B = vasicek_B(a, tau)
    return A * np.exp(-B * r_t[..., None] if tau.ndim > 0 else -B * r_t)


def zcb_0t_curve(r0: float, a: float, b: float, sigma: float, times: np.ndarray):
    """
    Deterministic discount curve from time 0 under Vasicek:
    P(0,t) = A(t) * exp(-B(t) r0)
    """
    tau = np.asarray(times)
    A = vasicek_A(a, b, sigma, tau)
    B = vasicek_B(a, tau)
    return A * np.exp(-B * r0)


# ============================================================
# 2) Swap PV (model-consistent) and exposures
# ============================================================

def swap_mtm_vasicek_receiver(
    r_t: np.ndarray,
    t: float,
    T_maturity: float,
    pay_times: np.ndarray,
    fixed_rate: float,
    a: float,
    b: float,
    sigma: float,
    notional: float = 1.0,
):
    """
    Receiver swap MTM at time t with model-consistent ZCBs.
    Floating PV approx: 1 - P(t, T)
    Fixed PV: K * sum_i alpha_i P(t, T_i)
    """
    future = pay_times[pay_times > t]
    if future.size == 0 or t >= T_maturity:
        return np.zeros_like(r_t)

    # year fractions: alpha_1 = T1 - t, alpha_i = Ti - T_{i-1}
    all_times = np.concatenate([[t], future])
    alpha = np.diff(all_times)  # shape (n_payments,)

    tau_i = future - t
    P_i = zcb_price_vasicek(r_t, a, b, sigma, tau_i)  # shape (n_paths, n_payments)

    P_T = zcb_price_vasicek(r_t, a, b, sigma, np.array([T_maturity - t]))[:, 0]

    pv_float = 1.0 - P_T
    pv_fixed = fixed_rate * (P_i @ alpha)  # sum alpha_i P_i

    mtm = notional * (pv_float - pv_fixed)  # receiver
    return mtm


def compute_exposure_profiles(
    times: np.ndarray,
    r_paths: np.ndarray,
    T_maturity: float,
    pay_times: np.ndarray,
    fixed_rate: float,
    a: float,
    b: float,
    sigma: float,
    notional: float = 1.0,
):
    n_paths, n_times = r_paths.shape
    MTM = np.zeros((n_paths, n_times), dtype=float)

    for i, t in enumerate(times):
        r_t = r_paths[:, i]
        if t >= T_maturity:
            MTM[:, i] = 0.0
        else:
            MTM[:, i] = swap_mtm_vasicek_receiver(
                r_t=r_t,
                t=t,
                T_maturity=T_maturity,
                pay_times=pay_times,
                fixed_rate=fixed_rate,
                a=a, b=b, sigma=sigma,
                notional=notional,
            )

    EE = np.mean(np.maximum(MTM, 0.0), axis=0)
    ENE = np.mean(np.minimum(MTM, 0.0), axis=0)
    return MTM, EE, ENE


# ============================================================
# 3) Unilateral CVA and simple FVA proxy with analytic P(0,t)
# ============================================================

def unilateral_cva(
    times: np.ndarray,
    EE: np.ndarray,
    P0t: np.ndarray,
    hazard_rate: float,
    recovery: float,
):
    """
    CVA ≈ (1-R) * sum_{i=1..n} P(0,t_i) * EE(t_i) * (S(t_{i-1}) - S(t_i))
    with S(t)=exp(-lambda t).
    """
    S = np.exp(-hazard_rate * times)
    dPD = S[:-1] - S[1:]
    return (1.0 - recovery) * np.sum(P0t[1:] * EE[1:] * dPD)


def simple_fva_proxy(
    times: np.ndarray,
    EE: np.ndarray,
    P0t: np.ndarray,
    funding_spread: float,
):
    """
    Simple proxy: FVA ≈ sum_{i=1..n} P(0,t_i) * funding_spread * EE(t_i) * dt
    """
    dt = np.diff(times)
    return np.sum(P0t[1:] * funding_spread * EE[1:] * dt)


# ============================================================
# 4) Run
# ============================================================

if __name__ == "__main__":
    # Contract
    T = 5.0
    pay_freq = 0.5
    pay_times = np.arange(pay_freq, T + 1e-12, pay_freq)
    K = 0.03
    N = 1_000_000

    # Vasicek params (toy)
    r0, a, b, sigma = 0.03, 0.6, 0.03, 0.02

    # Simulation
    n_steps = 200
    n_paths = 20000
    times, r_paths = simulate_vasicek_exact(r0, a, b, sigma, T, n_steps, n_paths, seed=7)

    # Analytic discount curve P(0,t)
    P0t = zcb_0t_curve(r0, a, b, sigma, times)

    # Exposure
    MTM, EE, ENE = compute_exposure_profiles(
        times, r_paths, T, pay_times, K, a, b, sigma, notional=N
    )

    # CVA / FVA
    hazard = 0.02
    recovery = 0.4
    cva = unilateral_cva(times, EE, P0t, hazard, recovery)

    funding_spread = 0.01
    fva = simple_fva_proxy(times, EE, P0t, funding_spread)

    print(f"Unilateral CVA (Vasicek-consistent) = {cva:,.2f}")
    print(f"Simple FVA proxy (Vasicek-consistent) = {fva:,.2f}")
    print(f"EE summary: min={EE.min():.2f}, max={EE.max():.2f}, last={EE[-1]:.2f}")

import numpy as np

def cva_independent_constant_hazard(times, MTM, P0t, hazard_rate, recovery):
    """
    Baseline: constant hazard independent of exposure.
    Uses EE(t) * dPD(t).
    """
    EE = np.mean(np.maximum(MTM, 0.0), axis=0)
    S = np.exp(-hazard_rate * times)
    dPD = S[:-1] - S[1:]
    cva = (1.0 - recovery) * np.sum(P0t[1:] * EE[1:] * dPD)
    return cva


def cva_wrong_way_pathwise(
    times,
    MTM,
    r_paths,
    P0t,
    recovery,
    lambda0=0.02,
    beta_r=0.0,
    gamma_mtm=0.0,
    r_center=None,
    notional=1.0,
    eps=1e-6,
):
    """
    Wrong-way risk toy model:
      lambda_p(t) = max(eps, lambda0 + beta_r*(r_t - r_center) + gamma_mtm*(max(MTM,0)/notional))
    Pathwise survival:
      S_p(t_i) = exp(-sum_{k<i} lambda_p(t_k) dt)
    Pathwise default increment:
      dPD_{p,i} = S_p(t_{i-1}) - S_p(t_i)
    CVA:
      (1-R) * sum_i P(0,t_i) * E[ max(MTM_{p,i},0) * dPD_{p,i} ]
    """
    times = np.asarray(times)
    dt = np.diff(times)  # length n_steps
    n_paths, n_times = MTM.shape

    if r_center is None:
        # A sensible center is long-run mean b; but if unknown, use sample mean at t=0 or global mean.
        r_center = np.mean(r_paths[:, 0])

    # Positive exposure per path/time
    Epos = np.maximum(MTM, 0.0)

    # Build pathwise lambda at discrete times t_0..t_{n-1} (use left-point for intensity over [t_i, t_{i+1}])
    r_left = r_paths[:, :-1]          # (n_paths, n_steps)
    Epos_left = Epos[:, :-1]          # (n_paths, n_steps)


    x = beta_r * (r_left - r_center) + gamma_mtm * (Epos_left / notional)
    lam = lambda0 * np.exp(x)
    lam = np.minimum(lam, 0.2)  # cap at 50% annual intensity, toy but prevents blow-ups
    print("lambda mean:", lam.mean(), "p5:", np.quantile(lam,0.05), "p95:", np.quantile(lam,0.95), "max:", lam.max())
    print("share at eps:", np.mean(lam <= eps*1.0001))


    # Pathwise integrated intensity: cum sum lam * dt
    # Need dt to broadcast: (n_steps,) -> (1, n_steps)
    int_lam = np.cumsum(lam * dt[None, :], axis=1)  # (n_paths, n_steps)

    # Survival at t_i for i>=1 corresponds to exp(-int up to i-1)
    # Build S at all grid points t_0..t_n:
    S = np.ones((n_paths, n_times), dtype=float)
    S[:, 1:] = np.exp(-int_lam)

    # Pathwise default increments dPD_i = S(t_{i-1}) - S(t_i), for i=1..n
    dPD = S[:, :-1] - S[:, 1:]  # (n_paths, n_steps)

    # CVA sum over i=1..n with exposure at t_i and dPD over (t_{i-1}, t_i)
    # Align: use Epos[:, 1:] with dPD (which is interval ending at t_i)
    contrib = Epos[:, 1:] * dPD  # (n_paths, n_steps)
    cva = (1.0 - recovery) * np.sum(P0t[1:] * np.mean(contrib, axis=0))
    return cva

cva_base = cva_independent_constant_hazard(times, MTM, P0t, hazard_rate=0.02, recovery=0.4)
print("Baseline CVA =", cva_base)
for beta in [-5, 0.0, 2.0, 5.0, 10.0]:
    cva_wwr_r = cva_wrong_way_pathwise(
        times, MTM, r_paths, P0t,
        recovery=0.4,
        lambda0=0.02,
        beta_r=beta,
        gamma_mtm=0.0,
        r_center=0.03,      # 用 b 或 r0
        notional=1_000_000
    )
    print(beta, cva_wwr_r, "uplift =", (cva_wwr_r / cva_base - 1.0))
for gamma in [0.0, 0.5, 1.0, 2.0]:
    cva_wwr_e = cva_wrong_way_pathwise(
        times, MTM, r_paths, P0t,
        recovery=0.4,
        lambda0=0.02,
        beta_r=0.0,
        gamma_mtm=gamma,
        r_center=0.03,
        notional=1_000_000
    )
    print(gamma, cva_wwr_e, "uplift =", (cva_wwr_e / cva_base - 1.0))
import numpy as np
import csv

def ee_with_se(MTM: np.ndarray):
    Epos = np.maximum(MTM, 0.0)
    EE = Epos.mean(axis=0)
    SE = Epos.std(axis=0, ddof=1) / np.sqrt(Epos.shape[0])
    return EE, SE

def cva_pathwise_constant_hazard(times, MTM, P0t, hazard_rate, recovery):
    """
    Returns: (CVA_mean, CVA_se, CVA_pathwise)
    """
    Epos = np.maximum(MTM, 0.0)
    S = np.exp(-hazard_rate * times)
    dPD = S[:-1] - S[1:]  # length n_steps
    # pathwise CVA_p
    # align: interval (t_{i-1},t_i] uses Epos at t_i => Epos[:,1:]
    contrib = Epos[:, 1:] * dPD[None, :]  # (n_paths, n_steps)
    cva_p = (1.0 - recovery) * np.sum(P0t[1:][None, :] * contrib, axis=1)
    cva_mean = cva_p.mean()
    cva_se = cva_p.std(ddof=1) / np.sqrt(cva_p.shape[0])
    return cva_mean, cva_se, cva_p

def cva_pathwise_wwr(
    times, MTM, r_paths, P0t, recovery,
    lambda0=0.02, beta_r=0.0, gamma_mtm=0.0,
    r_center=0.03, notional=1.0, cap=0.2
):
    """
    log-link intensity:
      lam = lambda0 * exp(beta_r*(r - r_center) + gamma_mtm*(Epos/notional))
    with cap as safeguard.
    Returns: (CVA_mean, CVA_se, CVA_pathwise, lambda_diagnostics_dict)
    """
    times = np.asarray(times)
    dt = np.diff(times)
    n_paths, n_times = MTM.shape

    Epos = np.maximum(MTM, 0.0)
    r_left = r_paths[:, :-1]
    Epos_left = Epos[:, :-1]

    x = beta_r * (r_left - r_center) + gamma_mtm * (Epos_left / notional)
    lam = lambda0 * np.exp(x)
    lam = np.minimum(lam, cap)  # safeguard

    # survival
    int_lam = np.cumsum(lam * dt[None, :], axis=1)
    S = np.ones((n_paths, n_times), dtype=float)
    S[:, 1:] = np.exp(-int_lam)

    dPD = S[:, :-1] - S[:, 1:]  # pathwise default increment for each interval

    contrib = Epos[:, 1:] * dPD
    cva_p = (1.0 - recovery) * np.sum(P0t[1:][None, :] * contrib, axis=1)
    cva_mean = cva_p.mean()
    cva_se = cva_p.std(ddof=1) / np.sqrt(n_paths)

    # diagnostics for paper
    lam_flat = lam.reshape(-1)
    diag = {
        "lambda_mean": float(lam_flat.mean()),
        "lambda_p5": float(np.quantile(lam_flat, 0.05)),
        "lambda_p95": float(np.quantile(lam_flat, 0.95)),
        "lambda_max": float(lam_flat.max()),
        "cap_hit_rate": float(np.mean(lam_flat >= cap - 1e-15)),
    }
    return cva_mean, cva_se, cva_p, diag

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---- Example usage after you compute MTM, times, P0t, r_paths ----

# 1) EE curve (with SE)
EE, EE_se = ee_with_se(MTM)
ee_rows = [(float(t), float(m), float(s)) for t, m, s in zip(times, EE, EE_se)]
write_csv("ee_curve.csv", ["t", "EE", "SE"], ee_rows)

# 2) CVA baseline (with SE)
hazard = 0.02
cva0, cva0_se, _ = cva_pathwise_constant_hazard(times, MTM, P0t, hazard, recovery=0.4)

# 3) CVA vs beta (WWR rate-linked)
betas = [-5, 0, 2, 5, 10]  # use the ones you tested
cva_beta_rows = []
for b in betas:
    cva_b, cva_b_se, _, diag = cva_pathwise_wwr(
        times, MTM, r_paths, P0t, recovery=0.4,
        lambda0=0.02, beta_r=b, gamma_mtm=0.0,
        r_center=0.03, notional=1_000_000, cap=0.2
    )
    uplift = cva_b / cva0 - 1.0
    cva_beta_rows.append((
        float(b),
        float(cva_b), float(cva_b_se),
        float(uplift),
        diag["lambda_mean"], diag["lambda_p5"], diag["lambda_p95"], diag["lambda_max"],
        diag["cap_hit_rate"]
    ))

write_csv(
    "cva_vs_beta.csv",
    ["beta", "CVA", "SE", "uplift", "lam_mean", "lam_p5", "lam_p95", "lam_max", "cap_hit_rate"],
    cva_beta_rows
)

print("Wrote ee_curve.csv and cva_vs_beta.csv")
print("Baseline CVA:", cva0, "SE:", cva0_se)
