#     \mathcal{L}_R(r_\phi) & = -\mathbb{E}_{(x_p,y_w,y_l) \sim \mathcal{D}_p}\left[\log \sigma(0.5 - 0.8)\right] \\ &= -\log \sigma(-0.3) = -\log \frac{1}{1 + e^{-(-0.3)}} \approx 0.37
import numpy as np

pi_theta_yw = 0.2
pi_theta_yl = 0.3
pi_sft_yw = 0.2
pi_sft_yl = 0.4
beta = 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(pi_theta_yw, pi_theta_yl, pi_sft_yw, pi_sft_yl):
    return -np.log(sigmoid(beta * np.log(pi_theta_yw / pi_sft_yw) - beta * np.log(pi_theta_yl / pi_sft_yl)))

print(dpo_loss(pi_theta_yw, pi_theta_yl, pi_sft_yw, pi_sft_yl))

r_phi_yw = 0.5
r_phi_yl = 0.8

def rlhf_loss(r_phi_yw, r_phi_yl):
    return -np.log(sigmoid(r_phi_yw - r_phi_yl))

print(rlhf_loss(r_phi_yw, r_phi_yl))