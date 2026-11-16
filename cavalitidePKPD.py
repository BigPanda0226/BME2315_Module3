
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Pharmacokinetic (PK) parameters
ke = 0.0039
ka = 0.006
V = 10.8
MW = 4113.6

Ffree = 0.01
Favailable = 0.01

D = 20.79
interval = 24
n_doses = 100

# Pharmacodynamic (PD) parameters
Kd_GLP1 = 0.17
Kd_Cavalitide = 0.077
GLP1 = 0.015

# PHARMACOKINETIC MODEL

def ODEfunc(t, y):

    Asc, Cc = y

    dAsc = -ka * Asc
    dCc = (ka * Asc / V) - (ke * Cc)

    return [dAsc, dCc]


def simulate_PK(dose):

    # Initial conditions
    y = [0, 0]

    time_values = []
    concentration_values = []

    # Simulate 100 daily doses
    for i in range(n_doses):

        # Add daily dose to absorption compartment
        y[0] += Favailable * dose

        t_start = i * interval
        t_end = (i + 1) * interval

        t_eval = np.linspace(t_start, t_end, 241)

        # Solve differential equations
        solution = solve_ivp(
            ODEfunc,
            [t_start, t_end],
            y,
            t_eval=t_eval
        )

        # Store results
        if i == 0:
            time_values.extend(solution.t)
            concentration_values.extend(solution.y[1])
        else:
            time_values.extend(solution.t[1:])
            concentration_values.extend(solution.y[1][1:])

        # Carry final values into next dosing interval
        y = [
            solution.y[0][-1],
            solution.y[1][-1]
        ]

    # Convert concentration from mg/L to nM
    Cc = np.array(concentration_values)
    Cc_nM = Cc * 1e6 / MW

    # Calculate free Cavalitide concentration
    Cfree = Ffree * Cc_nM

    return np.array(time_values), Cfree

# PHARMACODYNAMIC MODEL

def Fact(GLP1, Cavalitide, Kd1, Kd2):

    numerator = (Kd2 * GLP1) + (Kd1 * Cavalitide)

    denominator = (
        (Kd1 * Kd2)
        + (Kd2 * GLP1)
        + (Kd1 * Cavalitide)
    )

    return numerator / denominator

# COMBINE PK AND PD MODELS

def simulate_PKPD(dose):

    t, Cfree = simulate_PK(dose)

    activation = Fact(
        GLP1,
        Cfree,
        Kd_GLP1,
        Kd_Cavalitide
    )

    return t, Cfree, activation

# TASK 3A: SIMULATE RECEPTOR ACTIVATION USING ORIGINAL DOSE

# Use original daily dose of 20.79 mg
t, Cfree, activation = simulate_PKPD(D)

# Examine the final 24 hours
final_activation = activation[-241:] * 100

# Print Task 3A results
print("\nTASK 3A RESULTS")
print("---------------------------------------")
print(f"Original dose: {D:.2f} mg/day")
print(f"Minimum activation: {final_activation.min():.2f}%")
print(f"Maximum activation: {final_activation.max():.2f}%")
print(f"Average activation: {final_activation.mean():.2f}%")


# Plot Task 3A
plt.figure(figsize=(10, 5))

plt.plot(
    t / 24,
    activation * 100,
    linewidth=2,
    label="Original Cavalitide Dose"
)

plt.xlabel("Time (days)")
plt.ylabel("Fraction of Active GLP1 Receptors (%)")
plt.title("Task 3A: Cavalitide GLP1 Receptor Activation")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("cavalitide_PKPD.png", dpi=200)
plt.show()

# TASK 3B: OPTIMIZE THE DAILY CAVALITIDE DOSE

# Calculate average receptor activation for a given dose
def activation_at_dose(dose):

    t_test, Cfree_test, activation_test = simulate_PKPD(dose)

    # Select final 24 hours
    final_interval = activation_test[-241:]
    final_time = t_test[-241:]

    # Calculate time-weighted average activation
    average_activation = (
        np.trapezoid(final_interval, final_time)
        / (final_time[-1] - final_time[0])
    ) * 100

    return average_activation


# Find dose that produces 90% average receptor activation
optimized_dose = brentq(
    lambda dose: activation_at_dose(dose) - 90,
    20,
    50
)


# Print Task 3B optimization results
print("\nTASK 3B RESULTS")
print("---------------------------------------")
print(f"Original dose: {D:.2f} mg/day")
print(f"Optimized dose: {optimized_dose:.2f} mg/day")
print(
    f"Average receptor activation: "
    f"{activation_at_dose(optimized_dose):.2f}%"
)
# SIMULATE PKPD MODEL USING OPTIMIZED DOSE

t_opt, Cfree_opt, activation_opt = simulate_PKPD(optimized_dose)

# Examine final 24 hours
final_activation_opt = activation_opt[-241:] * 100


# Print optimized final-day results
print("\nOPTIMIZED FINAL 24-HOUR RESULTS")
print("---------------------------------------")
print(f"Minimum activation: {final_activation_opt.min():.2f}%")
print(f"Maximum activation: {final_activation_opt.max():.2f}%")
print(f"Average activation: {final_activation_opt.mean():.2f}%")

# PLOT TASK 3B: OPTIMIZED RECEPTOR ACTIVATION


plt.figure(figsize=(10, 5))

plt.plot(
    t_opt / 24,
    activation_opt * 100,
    linewidth=2,
    label="Optimized Cavalitide Dose"
)

plt.axhline(
    90,
    linestyle="--",
    label="90% Activation Target"
)

plt.xlabel("Time (days)")
plt.ylabel("Fraction of Active GLP1 Receptors (%)")
plt.title("Task 3B: Optimized Cavalitide GLP1 Receptor Activation")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("cavalitide_PKPD_optimized.png", dpi=200)
plt.show()


# TASK 3C: Additional Analyses - Dose-Response and Sensitivity Analysis

# PART 1: DOSE-RESPONSE ANALYSIS

# Test different daily Cavalitide doses
dose_range = np.arange(10, 46, 2)

activation_range = []

for dose in dose_range:

    average_activation = activation_at_dose(dose)

    activation_range.append(average_activation)


# Plot dose-response relationship
plt.figure(figsize=(10, 5))

plt.plot(
    dose_range,
    activation_range,
    marker="o",
    linewidth=2,
    label="Average GLP1R Activation"
)

plt.axhline(
    90,
    linestyle="--",
    label="90% Activation Target"
)

plt.axvline(
    optimized_dose,
    linestyle=":",
    label=f"Optimized Dose: {optimized_dose:.2f} mg/day"
)

plt.xlabel("Daily Cavalitide Dose (mg/day)")
plt.ylabel("Average GLP1 Receptor Activation (%)")

plt.title("Task 3C: Cavalitide Dose-Response Analysis")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("cavalitide_dose_response.png", dpi=200)
plt.show()

# PART 2: SENSITIVITY ANALYSIS

# Investigate how changes in model parameters
# affect the optimized Cavalitide dose.


# Save original parameter values
original_ka = ka
original_ke = ke
original_Kd = Kd_Cavalitide


# Store baseline optimized dose
baseline_dose = optimized_dose


# Define parameters to investigate
parameters = {
    "Absorption rate (ka)": "ka",
    "Elimination rate (ke)": "ke",
    "Binding affinity (Kd)": "Kd_Cavalitide"
}


# Store results
sensitivity_results = []


# Change each parameter by -20% and +20%
for parameter_name, parameter_variable in parameters.items():

    original_value = globals()[parameter_variable]

    for change in [-0.20, 0.20]:

        # Apply parameter change
        globals()[parameter_variable] = original_value * (1 + change)

        # Find new dose required for 90% activation
        new_dose = brentq(
            lambda dose: activation_at_dose(dose) - 90,
            5,
            80
        )

        # Calculate percentage difference from baseline dose
        dose_difference = (
            (new_dose - baseline_dose) / baseline_dose
        ) * 100

        sensitivity_results.append(
            (parameter_name, change * 100, new_dose, dose_difference)
        )


# Restore original parameters
ka = original_ka
ke = original_ke
Kd_Cavalitide = original_Kd


print("\nTASK 3C: SENSITIVITY ANALYSIS")
print("---------------------------------------")

print(f"Baseline optimized dose: {baseline_dose:.2f} mg/day")

for parameter, change, new_dose, difference in sensitivity_results:

    print(
        f"{parameter} ({change:+.0f}%): "
        f"Optimized dose = {new_dose:.2f} mg/day | "
        f"Change = {difference:+.2f}%"
    )


