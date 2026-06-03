import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. TEST & POPULATION PARAMETERS
# ==========================================
N_CANDIDATES = 2400000
N_ITEMS = 180
N_BIO, N_CHEM, N_PHY = 90, 45, 45
BATCH_SIZE = 400000

MARKS_CORRECT = 4
MARKS_INCORRECT = -1
D = 1.702

np.random.seed(42)

# ==========================================
# 2. CALIBRATED EXAM DIFFICULTY (4PL)
# ==========================================
a_params = np.random.lognormal(mean=0.15, sigma=0.2, size=N_ITEMS)

# 4-Tier Difficulty explicitly matching the NEET exam structure
b_very_easy = np.random.normal(loc=-1.5, scale=0.3, size=30)
b_easy = np.random.normal(loc=-0.2, scale=0.4, size=60)
b_medium = np.random.normal(loc=0.45, scale=0.3, size=60)
b_hard = np.random.normal(loc=2.3, scale=0.3, size=30)

b_params = np.concatenate([b_very_easy, b_easy, b_medium, b_hard])
np.random.shuffle(b_params)

c_params = np.random.uniform(0.12, 0.16, size=N_ITEMS)
d_params = np.random.uniform(0.97, 0.99, size=N_ITEMS)

# ==========================================
# 3. RUN FULL POPULATION SIMULATION
# ==========================================
start_time = time.time()
data = {'theta_g': [], 'c_bio': [], 'c_chem': [], 'i_tot': [], 'score': []}

for batch_start in range(0, N_CANDIDATES, BATCH_SIZE):
    batch_size = min(BATCH_SIZE, N_CANDIDATES - batch_start)

    # 5-COMPONENT EMPIRICAL GMM
    rands = np.random.rand(batch_size, 1)
    c1 = np.random.normal(loc=-1.1, scale=0.65, size=(batch_size, 1))
    c2 = np.random.normal(loc=-0.02, scale=0.50, size=(batch_size, 1))
    c3 = np.random.normal(loc=0.80, scale=0.30, size=(batch_size, 1))
    c4 = np.random.normal(loc=1.50, scale=0.20, size=(batch_size, 1))
    c5 = np.random.normal(loc=2.34, scale=0.08, size=(batch_size, 1))

    theta_g = np.where(rands < 0.58, c1,
                np.where(rands < 0.895, c2,
                  np.where(rands < 0.992, c3,
                    np.where(rands < 0.99968, c4, c5))))

    theta_bio = theta_g + np.random.normal(loc=0.1, scale=0.25, size=(batch_size, 1))
    theta_chem = theta_g + np.random.normal(loc=0.0, scale=0.30, size=(batch_size, 1))
    theta_phy = theta_g + np.random.normal(loc=-0.15, scale=0.35, size=(batch_size, 1))

    theta_matrix = np.hstack([np.repeat(theta_bio, N_BIO, axis=1), np.repeat(theta_chem, N_CHEM, axis=1), np.repeat(theta_phy, N_PHY, axis=1)])

    exp_term = np.exp(-D * a_params * (theta_matrix - b_params))
    p_know = 1.0 / (1.0 + exp_term)
    p_success = c_params + (d_params - c_params) * p_know
    adaptive_risk = np.clip(0.10 + (theta_g * 0.05), 0.05, 0.20)
    p_attempt = np.clip(p_know + adaptive_risk, 0, 1)

    attempted = np.random.rand(batch_size, N_ITEMS) < p_attempt
    correct_guess = np.random.rand(batch_size, N_ITEMS) < p_success

    correct = attempted & correct_guess
    incorrect = attempted & ~correct_guess

    c_bio_batch = np.sum(correct[:, :N_BIO], axis=1)
    c_chem_batch = np.sum(correct[:, N_BIO:N_BIO+N_CHEM], axis=1)
    i_tot_batch = np.sum(incorrect, axis=1)
    scores_batch = (np.sum(correct, axis=1) * MARKS_CORRECT) + (i_tot_batch * MARKS_INCORRECT)

    data['theta_g'].append(theta_g.flatten())
    data['c_bio'].append(c_bio_batch)
    data['c_chem'].append(c_chem_batch)
    data['i_tot'].append(i_tot_batch)
    data['score'].append(scores_batch)

theta_g_full = np.concatenate(data['theta_g'])
bio_full = np.concatenate(data['c_bio'])
chem_full = np.concatenate(data['c_chem'])
itot_full = np.concatenate(data['i_tot'])
scores_full = np.concatenate(data['score'])

# ==========================================
# 4. RANK ASSIGNMENT & DISPLACEMENT ANALYSIS
# ==========================================
true_ranks = stats.rankdata(-theta_g_full, method='min')
sort_keys = (itot_full, -chem_full, -bio_full, -scores_full)
sort_idx = np.lexsort(sort_keys)
off_ranks = np.empty_like(sort_idx)
off_ranks[sort_idx] = np.arange(1, N_CANDIDATES + 1)

print("\n=== ADVANCED COUNSELING ZONE ANALYSIS (TOP 1 LAKH) ===")
print("\n[+] For students whose TRUE ABILITY is in the Top 1,000,000 (Counseling Eligible):")
mask_1M = true_ranks <= 1000000
disp = np.abs(true_ranks[mask_1M] - off_ranks[mask_1M])
print(f"    -> Median Rank Displacement: {np.median(disp):.0f} positions")
print(f"    -> 10% of them were displaced by more than: {np.percentile(disp, 90):.0f} positions")
print(f"    -> 1% of them (worst-case) were displaced by more than: {np.percentile(disp, 99):.0f} positions")
print(f"    -> Maximum single displacement observed: {np.max(disp):.0f} positions")

print("\n[+] Exclusion Probabilities by Institution Tier:")
t_100 = true_ranks <= 100
print(f"    -> AIIMS Delhi Tier (Top 100): {np.sum(t_100 & (off_ranks > 100)) / np.sum(t_100) * 100:.1f}% lost their Tier")

t_10k = true_ranks <= 10000
print(f"    -> Elite Govt Tier (Top 10k): {np.sum(t_10k & (off_ranks > 10000)) / np.sum(t_10k) * 100:.1f}% lost their Tier")
print(f"    -> Elite Govt Tier (Top 10k): {np.sum(t_10k & (off_ranks > 50000)) / np.sum(t_10k) * 100:.1f}% were pushed out of Top 50k")

t_50k = true_ranks <= 50000
print(f"    -> Standard GMC Tier (Top 50k): {np.sum(t_50k & (off_ranks > 50000)) / np.sum(t_50k) * 100:.1f}% missed their Tier")

t_1L = true_ranks <= 100000
print(f"    -> Overall Counseling (Top 1 Lakh): {np.sum(t_1L & (off_ranks > 100000)) / np.sum(t_1L) * 100:.1f}% were pushed out of Counseling")

print("\n[+] Seat Infiltration by 'Lucky' Candidates:")
off_50k = off_ranks <= 50000
print(f"    -> {np.sum(off_50k & (true_ranks > 50000)) / np.sum(off_50k) * 100:.1f}% of students holding an Official Top 50k rank don't belong there")

print(f"\nSimulation Complete in {time.time() - start_time:.2f} seconds.")

# ==========================================
# 5. EMPIRICAL VALIDATION TABLES
# ==========================================
nta_bins = [144, 201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 721]
nta_obs = [303040, 198346, 157952, 126935, 105578, 88239, 69503, 39521, 10658, 1259, 73]
sim_obs, _ = np.histogram(scores_full, bins=nta_bins)

print("\n\n=== EMPIRICAL VALIDATION AGAINST OFFICIAL NTA DATA ===")
print("\n[+] Table (l) Validation: Score Distribution Match")
print(f"{'Marks Range':<15} | {'NTA Official':<15} | {'Simulated':<15} | {'Error (%)':<15}")
print("-" * 65)
for i in range(len(nta_bins)-1):
    rng = f"{nta_bins[i]}-{nta_bins[i+1]-1 if i < len(nta_bins)-2 else 686}"
    err = ((sim_obs[i] - nta_obs[i]) / nta_obs[i]) * 100
    print(f"{rng:<15} | {nta_obs[i]:<15} | {sim_obs[i]:<15} | {err:>10.1f}%")

print("\n[+] Table (m) Validation: Rank vs. Marks Curve Match")
print(f"{'Rank':<15} | {'NTA Marks':<15} | {'Sim Marks':<15} | {'Diff (Marks)':<15}")
print("-" * 65)
ranks_m = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
nta_marks_m = [502, 464, 433, 405, 380, 357, 336, 316, 298, 281]
sim_marks_sorted = np.sort(scores_full)[::-1]
for r, n_m in zip(ranks_m, nta_marks_m):
    s_m = sim_marks_sorted[r - 1]
    diff = np.abs(s_m - n_m)
    print(f"{r:<15} | {n_m:<15} | {s_m:<15.0f} | {diff:>10.0f}")


# ==========================================
# 6. MATHEMATICAL CALCULATIONS FOR CSEM
# ==========================================
print("\nCalculating Test Information and CSEM Curves...")
theta_range = np.linspace(-4, 4, 2000) # Increased resolution for Section 8 gradient accuracy
info_total = np.zeros_like(theta_range)
T_theta = np.zeros_like(theta_range)

for i in range(N_ITEMS):
    z = np.exp(-D * a_params[i] * (theta_range - b_params[i]))
    P = c_params[i] + (d_params[i] - c_params[i]) / (1 + z)
    P_prime = (d_params[i] - c_params[i]) * D * a_params[i] * z / (1 + z)**2
    I_i = (P_prime**2) / (P * (1 - P) + 1e-10)
    info_total += I_i

    # Also calculate Expected True Score (TCC) for Section 8 simultaneously
    a_r = np.clip(0.10 + (theta_range * 0.05), 0.05, 0.20)
    p_a = np.clip(P + a_r, 0, 1)
    exp_item_score = p_a * (5 * P - 1)
    T_theta += exp_item_score

csem = 1.0 / np.sqrt(info_total)


# ==========================================
# 7. MATPLOTLIB VISUALIZATIONS (GridSpec)
# ==========================================
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3)

# --- Top Plot: Information vs CSEM (Dual Axis) ---
ax1 = fig.add_subplot(gs[0, :])
ax1_twin = ax1.twinx()

l1 = ax1.plot(theta_range, info_total, color='tab:blue', linewidth=2.5, label='Test Information')
l2 = ax1_twin.plot(theta_range, csem, color='tab:red', linestyle='--', linewidth=2.5, label='CSEM')
ax1_twin.axvspan(2.0, 4.2, color='lightcoral', alpha=0.2, label='Top Rankers (High Error)')

ax1.set_xlabel('Candidate Ability (Theta)', fontsize=12)
ax1.set_ylabel('Test Information', color='tab:blue', fontsize=12)
ax1_twin.set_ylabel('CSEM (Error Margin)', color='tab:red', fontsize=12)
ax1.set_title('NEET 4PL Psychometric Profile: Information vs. CSEM', fontsize=14)
ax1.set_xlim(-4.2, 4.2)
ax1.grid(True, alpha=0.3)

lines = l1 + l2 + [plt.Rectangle((0,0),1,1, color='lightcoral', alpha=0.2)]
labels = ['Test Information', 'CSEM', 'Top Rankers (High Error)']
ax1.legend(lines, labels, loc='upper right')

# --- Bottom Left Plot: Bar Chart ---
ax2 = fig.add_subplot(gs[1, 0])
x = np.arange(len(nta_bins)-1)
width = 0.35

ax2.bar(x - width/2, nta_obs, width, label='NTA Official', color='navy')
ax2.bar(x + width/2, sim_obs, width, label='Simulated', color='darkorange')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{nta_bins[i]}-{nta_bins[i+1]-1 if i < len(nta_bins)-2 else 686}" for i in range(len(nta_bins)-1)], rotation=45)
ax2.set_ylabel('Number of Candidates')
ax2.set_title('Goodness of Fit: Candidate Score Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Bottom Right Plot: Line Chart ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(ranks_m, nta_marks_m, 'bo-', linewidth=2, label='NTA Official')
ax3.plot(ranks_m, [sim_marks_sorted[r-1] for r in ranks_m], 'r^--', linewidth=2, label='Simulated')

ax3.set_xlabel('Rank')
ax3.set_ylabel('Marks Obtained')
ax3.set_title('Validation: Rank vs. Marks Density Curve')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.90, hspace=0.4, wspace=0.3)
plt.show(block=False) # block=False allows the script to continue to Section 8

# ==========================================
# 8. THE CEILING EFFECT & ILLUSION OF PRECISION
# ==========================================
print("\n\n=== PSYCHOMETRIC CRISIS: COMPRESSION & TIE-BREAKING ===")
print("Evaluating how much true intelligence (Standard Deviations of Theta) ")
print("is hidden within a single 4-mark scoring band.\n")

target_marks = [715, 700, 650, 600, 500, 400, 300, 200]

T_prime = np.gradient(T_theta, theta_range)

print(f"{'NTA Score':<10} | {'Median θ (SD)':<15} | {'Latent Width (Δθ) per 4 Marks':<33} | {'Policy/Ranking Validity'}")
print("-" * 115)

for target in target_marks:
    margin = 2
    while True:
        mask = np.abs(scores_full - target) <= margin
        if np.sum(mask) > 20 or margin > 20:
            break
        margin += 1

    if np.sum(mask) == 0:
        continue

    median_th = np.median(theta_g_full[mask])
    idx = np.argmin(np.abs(theta_range - median_th))
    slope = T_prime[idx]

    if slope <= 0.5:
        delta_theta = 9.99
    else:
        delta_theta = 4.0 / slope

    print(f"{target:<10} | {median_th:<15.2f} | {delta_theta:<33.2f}")
  
