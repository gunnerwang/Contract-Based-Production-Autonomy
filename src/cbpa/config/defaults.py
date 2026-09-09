"""Default coefficients and thresholds calibrated to Table 3.

Also contains factory-level constants for multi-cell operation.
"""

# --- Noise model: noise_db = base + k1*r1_speed + k2*r2_speed ---
# Note: in analytical mode, coefficients tuned for best overall fit.
# Deterministic mode uses exact Table 3 values.
NOISE_BASE_DB = 62.5
NOISE_K1 = 6.0
NOISE_K2 = 17.0

# --- Energy: base + c_robot*(r1^2 + r2^2)*hours + c_human*h_rate*hours ---
# Calibrated: all positive, exact match for S1/S2/S3
ENERGY_BASE_KWH = 92.75
ENERGY_ROBOT_COEFF = 2.5014
ENERGY_HUMAN_COEFF = 36.1985

# --- Fatigue: alpha * effective_load^beta ---
# effective_load = h_rate * (0.5 + 0.3*(r1_speed + r2_speed))
FATIGUE_ALPHA = 0.4171
FATIGUE_BETA = 1.7534

# --- Defect: base * (1 + alpha*(avg_speed - 0.5)) * (1 + beta*fatigue) ---
DEFECT_BASE_RATE = 0.006
DEFECT_SPEED_ALPHA = 0.5
DEFECT_FATIGUE_BETA = 0.6

# --- V/R normalization baselines ---
VR_THROUGHPUT_MAX = 60.0
VR_ENERGY_MAX = 550.0
VR_DOWNTIME_MAX = 0.15
VR_LABOR_LOAD_MAX = 1.0
VR_COST_MAX = 1.0

# --- Constraint defaults ---
DEFAULT_FATIGUE_LIMIT = 0.4
DEFAULT_NOISE_LIMIT_DB = 80.0
DEFAULT_CYBER_RISK_LIMIT = 2.0

# --- Table 3 exact target values (for deterministic mode) ---
TABLE3_TARGETS = {
    "S1": {
        "throughput_uph": 44.2,
        "defect_rate": 0.008,
        "noise_db": 76.5,
        "fatigue_index": 0.33,
        "energy_kwh": 398.0,
        "deadline_gap_pct": 15.8,
    },
    "S2": {
        "throughput_uph": 53.6,
        "defect_rate": 0.010,
        "noise_db": 85.0,
        "fatigue_index": 0.60,
        "energy_kwh": 462.0,
        "deadline_gap_pct": 0.0,
    },
    "S3": {
        "throughput_uph": 46.7,
        "defect_rate": 0.009,
        "noise_db": 79.1,
        "fatigue_index": 0.38,
        "energy_kwh": 417.0,
        "deadline_gap_pct": 14.9,
    },
}

# ---------------------------------------------------------------------------
# Factory-level constants (multi-cell extension)
# ---------------------------------------------------------------------------

# Variant-specific cycle-time multipliers relative to nominal.
# V_A = standard product, V_B = more assembly work, V_C = more fasteners.
VARIANT_CYCLE_FACTORS: dict[str, dict[str, float]] = {
    "V_A": {"r1": 1.0, "r2": 1.0, "human": 1.0},
    "V_B": {"r1": 1.15, "r2": 0.9, "human": 1.1},
    "V_C": {"r1": 0.8, "r2": 1.3, "human": 1.25},
}

# AGV model parameters
AGV_BASE_TRANSFER_S = 45.0  # seconds, empty corridor
AGV_QUEUE_FACTOR = 8.0  # extra seconds per buffered unit
AGV_CAPACITY_UPH = 80.0  # max units/hour the AGV can transfer

# Factory noise: combined via logarithmic addition
# factory_noise = 10 * log10(10^(noiseA/10) + 10^(noiseB/10))
DEFAULT_FACTORY_NOISE_LIMIT_DB = 82.0

# Cell balance: max acceptable throughput imbalance
DEFAULT_CELL_BALANCE_LOSS_MAX_PCT = 25.0
