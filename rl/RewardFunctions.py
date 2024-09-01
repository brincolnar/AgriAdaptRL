
import numpy as np
import matplotlib.pyplot as plt

# 1.0 factor
# def linear_factor(battery):
#     return battery / 100.0

# 5.0 factor 
# def linear_factor(battery):
#     return (battery / 100.0) * 5.0 

# 10.0 factor
def linear_factor(battery):
    return (battery / 100.0) * 10.0 

# 1.0 factor
# def logarithmic_factor_60(battery):
#     if battery > 60:
#         return 1.0
#     elif battery > 0:
#         # Adjusting the base of the logarithm and the scaling to flatten and shift the curve
#         return np.log1p(battery / 10.0) / np.log1p(6.0)  # log1p ensures that log(0) is avoided naturally
#     else:
#         return 0

# 5.0 factor 
# def logarithmic_factor_60(battery):
#     if battery > 60:
#         return 5.0  
#     elif battery > 0:
#         return (np.log1p(battery / 10.0) / np.log1p(6.0)) * 5.0
#     else:
#         return 0

# 10.0 factor
def logarithmic_factor_60(battery):
    if battery > 60:
        return 10.0  
    elif battery > 0:
        return (np.log1p(battery / 10.0) / np.log1p(6.0)) * 10.0
    else:
        return 0

# 1.0 factor
# def logarithmic_factor_90(battery):
#     if battery > 90:
#         return 1.0
#     elif battery > 0:
#         return np.log1p(battery / 10.0) / np.log1p(9.0)  
#     else:
#         return 0

# 5.0 factor
# def logarithmic_factor_90(battery):
#     if battery > 90:
#         return 5.0  
#     elif battery > 0:
#         return (np.log1p(battery / 10.0) / np.log1p(9.0)) * 5.0
#     else:
#         return 0

# 10.0 factor
def logarithmic_factor_90(battery):
    if battery > 90:
        return 10.0  
    elif battery > 0:
        return (np.log1p(battery / 10.0) / np.log1p(9.0)) * 10.0
    else:
        return 0

# 1.0 factor
# def logarithmic_factor_40(battery):
#     if battery > 40:
#         return 1.0
#     elif battery > 0:
#         return np.log1p(battery / 10.0) / np.log1p(4.0) 
#     else:
#         return 0

# 5.0 factor
# def logarithmic_factor_40(battery):
#     if battery > 40:
#         return 5.0  
#     elif battery > 0:
#         return (np.log1p(battery / 10.0) / np.log1p(4.0)) * 5.0
#     else:
#         return 0

# 10.0 factor
def logarithmic_factor_40(battery):
    if battery > 40:
        return 10.0  
    elif battery > 0:
        return (np.log1p(battery / 10.0) / np.log1p(4.0)) * 10.0
    else:
        return 0


def plot_factors(battery_levels, linear_factors, log_60_factors, log_90_factors, log_40_factors):
    plt.figure(figsize=(12, 6))

    plt.plot(battery_levels, linear_factors, label='Linear Factor', color='blue', linestyle='-', marker='')
    plt.plot(battery_levels, log_90_factors, label='Logarithmic Factor (90)', color='green', linestyle='-', marker='')
    plt.plot(battery_levels, log_60_factors, label='Logarithmic Factor (60)', color='red', linestyle='-', marker='')
    plt.plot(battery_levels, log_40_factors, label='Logarithmic Factor (40)', color='purple', linestyle='-', marker='')

    plt.title('Reward Scaling Factor vs. Battery Level')
    plt.xlabel('Battery Level (%)')
    plt.ylabel('Factor')
    plt.legend()
    plt.grid(True)

    plt.savefig('./factor_plots.png')
    plt.show()

battery_levels = np.arange(0, 101, 1)  # From 0% to 100%, in 1% increments
linear_factors = [linear_factor(battery) for battery in battery_levels]
log_90_factors = [logarithmic_factor_90(battery) for battery in battery_levels]
log_60_factors = [logarithmic_factor_60(battery) for battery in battery_levels]
log_40_factors = [logarithmic_factor_40(battery) for battery in battery_levels]

plot_factors(battery_levels, linear_factors, log_90_factors, log_60_factors, log_40_factors)