import matplotlib.pyplot as plt
import matplotlib

# Set the font family globally
matplotlib.rcParams['font.family'] = 'Times New Roman'


# Updated data with Efficacy Accuracy - Hard Cases converted to percentage
iterations = [1, 2, 3, 4, 5]

efficacy_all_cases = [99.00, 100.00, 100.00, 100.00, 100.00]
efficacy_hard_cases = [0 * 100, 0.83 * 100, 0.75 * 100, 0.83 * 100, 0.83 * 100]
approx_error_all_cases = [3.90, 0.02, 0.05, 0.01, None]
approx_error_hard_cases = [69.4, 0.55, 0.54, 0.28, None]

# Create the figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Plot for efficacy accuracy (subplot 1)
axs[0].plot(iterations, efficacy_all_cases, label="Efficacy Accuracy - All", marker='o')
axs[0].plot(iterations, efficacy_hard_cases, label="Efficacy Accuracy - UnderEdit", marker='o', linestyle='--')
axs[0].set_ylabel("Efficacy Accuracy (%)", fontsize=14)
axs[0].legend(loc="lower right", fontsize=14)
axs[0].set_title("Efficacy Accuracy Across Iterations", fontsize=14)
axs[0].set_xticks(iterations)
axs[0].tick_params(axis='both', which='major', labelsize=14)

# Plot for approximation error (subplot 2)
axs[1].plot(iterations, approx_error_all_cases, label="Approx. Error - All", marker='s')
axs[1].plot(iterations, approx_error_hard_cases, label="Approx. Error - UnderEdit", marker='s', linestyle='--')
axs[1].set_ylabel("Approximation Error", fontsize=14)
axs[1].set_xlabel("Iterations", fontsize=14)
axs[1].legend(loc="upper right", fontsize=14)
axs[1].set_xticks(iterations)
axs[1].tick_params(axis='both', which='major', labelsize=14)

# Remove grid and extra borders, leaving only x and y axes
for ax in axs:
    ax.grid(False)  # Remove grid
    ax.spines['top'].set_visible(False)  # Remove top border
    ax.spines['right'].set_visible(False)  # Remove right border

# Adjust layout and save
plt.tight_layout()
plt.savefig('hard-case.pdf')