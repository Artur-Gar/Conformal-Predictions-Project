from . import utils_SPCI as utils
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import pacf
import numpy as np
from . import data
import pandas as pd
import warnings
import torch
import pickle
import pdb
from sklearn.ensemble import RandomForestRegressor
import SPCI_class as SPCI
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
titlesize = 22
plt.rcParams.update({'axes.labelsize': titlesize, 'axes.titlesize': titlesize,
                    'legend.fontsize': titlesize, 'xtick.labelsize': titlesize, 'ytick.labelsize': titlesize})


def detach_torch(input):
    return input.cpu().detach().numpy()


def CI_on_Ytest(results_EnbPI_SPCI, Ytest, train_size, mtd='SPCI', dataname='solar'):
    if mtd == 'SPCI':
        PIs = results_EnbPI_SPCI.PIs_SPCI
    else:
        PIs = results_EnbPI_SPCI.PIs_EnbPI
    stride = results_EnbPI_SPCI.stride
    stride_save = '' if stride == 1 else f'_{stride}'
    fig, ax = plt.subplots(figsize=(10, 3))
    xaxes = range(train_size, train_size + len(Ytest))
    ax.scatter(xaxes, Ytest, color='black', s=3)
    ax.fill_between(xaxes, PIs['upper'],
                    PIs['lower'], alpha=0.25, color='blue')
    width = (np.array(PIs['upper']) - np.array(PIs['lower'])).mean()
    cov = ((np.array(PIs['lower']) <= Ytest) & (
        np.array(PIs['upper']) >= Ytest)).mean()
    ax.set_xlabel('Prediction Time Index')
    ax.set_title(mtd + r' $C_{\alpha}(X_t)$ around $Y$'
                 + f', coverage {cov:.2f}, width {width:.2f}')
    fig.savefig(f'{mtd}_Interval_on_Ytest{stride_save}_{dataname}.png', dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    return fig


def plot_burn_in(PIs_ls, Ytest, window_size, savename, use_NeuralProphet=False):
    window_size_dict = {'electric': 100, 'solar': 100, 'window': 50}
    window_size = window_size_dict['solar']
    PIs_EnbPI, PIs_SPCI, PIs_AdaptiveCI, PI_nexCP_WLS = PIs_ls[:4]
    if use_NeuralProphet:
        PIs_SPCINeuralProphet = PIs_ls[-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    first = (np.array(PIs_EnbPI['upper'])
             - np.array(PIs_EnbPI['lower']))[:window_size]
    first_cov = ((np.array(PIs_EnbPI['lower']) <= Ytest)
                 & (np.array(PIs_EnbPI['upper']) >= Ytest))[:window_size]
    ax.plot(
        first, label=f'EnbPI: {first.mean():.2f} & {first_cov.mean():.2f}', color='black')
    second = (np.array(PIs_SPCI['upper'])
              - np.array(PIs_SPCI['lower']))[:window_size]
    second_cov = ((np.array(PIs_SPCI['lower']) <= Ytest)
                  & (np.array(PIs_SPCI['upper']) >= Ytest))[:window_size]
    ax.plot(
        second, label=f'SPCI: {second.mean():.2f} & {second_cov.mean():.2f}', color='orange')
    if use_NeuralProphet:
        second_NP = (np.array(PIs_SPCINeuralProphet['upper'])
                     - np.array(PIs_SPCINeuralProphet['lower']))[:window_size]
        second_NP_cov = ((np.array(PIs_SPCINeuralProphet['lower']) <= Ytest)
                         & (np.array(PIs_SPCINeuralProphet['upper']) >= Ytest))[:window_size]
        ax.plot(
            second_NP, label=f'SPCI-NeuralProphet: {second_NP.mean():.2f} & {second_NP_cov.mean():.2f}', color='yellow')
    third = (np.array(PIs_AdaptiveCI['upper'])
             - np.array(PIs_AdaptiveCI['lower']))[:window_size]
    third_cov = ((np.array(PIs_AdaptiveCI['lower']) <= Ytest)
                 & (np.array(PIs_AdaptiveCI['upper']) >= Ytest))[:window_size]
    ax.plot(
        third, label=f'AdaptiveCI: {third.mean():.2f} & {third_cov.mean():.2f}', color='gray', linewidth=0.75)
    fourth = (np.array(PI_nexCP_WLS[:, 1])
              - np.array(PI_nexCP_WLS[:, 0]))[:window_size]
    fourth_cov = ((PI_nexCP_WLS[:, 0] <= Ytest)
                  & (PI_nexCP_WLS[:, 1] >= Ytest))[:window_size]
    ax.plot(
        fourth, label=f'Nex-CP WLS: {fourth.mean():.2f} & {fourth_cov.mean():.2f}', color='magenta')
    ax.set_xlabel('Burn-in Period')
    ax.set_ylabel('Width')
    # ax.legend(title='Method: Ave Width in burn-in', title_fontsize=17,
    #           loc='upper center', ncol=1, bbox_to_anchor=(1.4, 0.45))
    ax.legend(title='Method: Ave Width & Coverage in burn-in', title_fontsize=22,
              loc='lower center', ncol=2, bbox_to_anchor=(0.475, -0.63))
    plt.savefig(f'Brun_in_plot_{savename}.png', dpi=300,
                bbox_inches='tight',
                pad_inches=0)


wind_loc = 0


def plot_rolling(alpha, train_frac, non_stat_solar=True, dsets=['electric']):
    if 'simulate' in dsets[0]:
        make_plot = False
        methods = ['SPCI', 'EnbPI']
    else:
        make_plot = True
        methods = ['SPCI', 'EnbPI', 'AdaptiveCI', 'NEXCP']
        colors = ['black', 'orange', 'blue', 'magenta']
    window_size_dict = {'electric': 100, 'solar': 100, 'wind': 50}
    full_cov_width_table = np.zeros(
        (len(methods), len(dsets) * 2 * 2), dtype=object)
    for i, data_name in enumerate(dsets):
        if make_plot:
            window_size = window_size_dict[data_name]
            fig, ax = plt.subplots(1, 2, figsize=(20, 4), sharex=True)
        for j, name in enumerate(methods):
            print(f'{name} on {data_name}')
            if make_plot:
                dloader = data.real_data_loader()
            else:
                simul_name_dict = {1: 'simulation_state_space',
                                   2: 'simulate_nonstationary', 3: 'simulate_heteroskedastic'}
                simul_type = 2+i
                data_name = simul_name_dict[simul_type]
                simul_loader = data.simulate_data_loader()
                Data_dict = simul_loader.get_simul_data(simul_type)
                X_full, Y_full = Data_dict['X'].to(
                    device), Data_dict['Y'].to(device)
                X_full, Y_full = detach_torch(X_full), detach_torch(Y_full)
            N = len(Y_full)
            N0 = int(train_frac * N)
            Y_test = Y_full[N0:]
            with open(f'{name}_{data_name}_train_frac_{np.round(train_frac,2)}_alpha_{alpha}.p', 'rb') as fp:
                dict_rolling = pickle.load(fp)
            num_trials = len(dict_rolling.keys())
            cov_ls, width_ls = [], []
            for itrial in range(num_trials):
                PI = dict_rolling[f'Itrial{itrial}']
                cov_stat = ((np.array(PI['lower']) <= Y_test)
                            & (np.array(PI['upper']) >= Y_test))
                width_stat = ((np.array(PI['upper']) - np.array(PI['lower'])))
                cov_ls.append(cov_stat)
                width_ls.append(width_stat)
            covs = [np.mean(c) for c in cov_ls]
            widths = [np.mean(w) for w in width_ls]
            full_cov_width_table[j, i * 4] = f'{np.mean(covs):.2f}'
            full_cov_width_table[j, i * 4 + 1] = f'{np.std(covs):.2e}'
            full_cov_width_table[j, i * 4 + 2] = f'{np.mean(widths):.2f}'
            full_cov_width_table[j, i * 4 + 3] = f'{np.std(widths):.2e}'
            if make_plot:
                cov_rolling = [utils.rolling_avg(
                    cov, window=window_size) for cov in cov_ls]
                cov_rolling_mean, cov_rolling_std = np.mean(
                    cov_rolling, 0), np.std(cov_rolling, 0)
                width_rolling = [utils.rolling_avg(
                    width, window=window_size) for width in width_ls]
                width_rolling_mean, width_rolling_std = np.mean(
                    width_rolling, 0), np.std(width_rolling, 0)
                # Plot
                if j == 0:
                    ax[0].axhline(y=1 - alpha, linestyle='--', color='gray')
                xaxis = np.arange(N0 + window_size, N)
                ax[0].plot(xaxis, cov_rolling_mean,
                           color=colors[j], label=name)
                ax[0].fill_between(xaxis, cov_rolling_mean - cov_rolling_std,
                                   cov_rolling_mean + cov_rolling_std, color=colors[j], alpha=0.3)
                ax[1].plot(xaxis, width_rolling_mean, color=colors[j])
                ax[1].fill_between(xaxis, width_rolling_mean - width_rolling_std,
                                   width_rolling_mean + width_rolling_std, color=colors[j], alpha=0.3)
        if make_plot:
            ax[0].set_xlabel('Data index')
            ax[0].set_ylim([1 - 4 * alpha, 1])
            ax[0].set_ylabel('Rolling coverage')
            ax[0].legend(ncol=2, loc='lower center')
            ax[1].set_ylabel('Rolling width')
            ax[1].set_xlabel('Data index')
            fig.tight_layout()
            plt.savefig(f'Rolling_comparison_{data_name}.png', dpi=300,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.show()
            plt.close()
    dsets = np.array([[f'{dname} cov mean', f'{dname} cov std',
                       f'{dname} width mean', f'{dname} width std']
                      for dname in dsets]).flatten()
    full_cov_width_table = pd.DataFrame(
        full_cov_width_table, index=methods, columns=dsets)
    return full_cov_width_table





def plot_resid_and_pacf(EnbPI):
    # Plot residual and pacf given trained model
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    N = len(EnbPI.X_train)
    resid_rest = EnbPI.Ensemble_online_resid[:N]
    low, up = np.percentile(
        resid_rest, 4), np.percentile(resid_rest, 95)
    resid_rest = resid_rest[(resid_rest >= low) & (
        resid_rest < up)]
    sns.histplot(resid_rest, bins=15, kde=True, ax=ax[0])
    ax[0].set_xticks([int(resid_rest.min()), 0, int(resid_rest.max())])
    ax[0].set_title(
        r'Histogram of $\{\hat{\epsilon}_t\}_{t=1}^T$', fontsize=24)
    ax[0].set_ylabel('')
    ax[0].yaxis.set_ticks([])
    ax[1].plot(pacf(EnbPI.Ensemble_online_resid),
               marker='o', markersize=4)
    ax[1].set_title("PACF", fontsize=24)
    ax[1].grid()
    plt.savefig('Resid_histogram_and_PACF.png', dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    plt.show()
    plt.close()

##################
