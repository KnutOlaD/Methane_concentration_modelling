import numpy as np
import matplotlib.pyplot as plt


def calc_loss_profile(k_ox, range=100):
    output = np.arange(0, range, 1)
    output = np.exp(-k_ox * output)
    return output


def calc_turnover_time(k_ox, percent=0.01):
    return -np.log(percent) / k_ox


def calc_turnover_from_k_in_seconds(k_ox, percent=0.01):
    return calc_turnover_time(k_ox, percent=percent) / 86400


k_ox = 0.00735
k_ox = (2.15 * 10**-7) * 3600 * 24

loss_profile = 100 * calc_loss_profile(k_ox, range=100000)

t_50 = calc_turnover_time(k_ox, percent=0.5)
t_90 = calc_turnover_time(k_ox, percent=0.1)
t_99 = calc_turnover_time(k_ox, percent=0.01)


print(t_50)

plotit = True
if plotit == True:
    plt.figure()
    plt.plot(loss_profile)
    # plt.axvline(x=t_50, color='k', linestyle='--')  # Add vertical line
    # plt.text(t_50, 70, '50% turnover time', rotation=270, verticalalignment='center')  # Add textbox
    # plt.axvline(x=t_90, color='k', linestyle='--')  # Add vertical line
    # plt.text(t_90, 70, '90% turnover time', rotation=270, verticalalignment='center')  # Add textbox
    # plt.axvline(x=t_99, color='k', linestyle='--')  # Add vertical line
    # plt.text(t_99, 70, '99% turnover time', rotation=270, verticalalignment='center')  # Add textbox
    plt.ylabel("Percent left")
    plt.xlabel("Days")
    plt.title("k_{ox} = k_ox")
    plt.xlim(0, 30)
    plt.show()
