import matplotlib.pyplot as plt
import numpy as np
import os

#from python import Fixed_step_size_NUTS_simulation as fv


def plot_funnel_spectral_1D(N, log_sig_range, h, destination):
    # Get funnel model

    # funnel_model = fv.create_model_stan_and_json("funnel", "funnel")
    # Create gridpoints and initial theta

    if not os.path.exists(destination):
        os.makedirs(destination)

    log_sigs = np.linspace(-log_sig_range, log_sig_range, N)
    #
    # theta_0 = np.zeros(funnel_model.param_unc_num())

    '''
    spectral_radii = np.zeros(N)

    #Calculate spectral radii
    for j in range(N):
        theta_0[0] = log_sigs[j]
        _,_,H = funnel_model.log_density_hessian(theta_0)
        eigvals, eigvects = np.linalg.eigh(-H)
        print(eigvals)
        spectral_radii[j] = np.max(np.abs(eigvals))
    '''
    one_over_sqrt_L = np.minimum(3, np.exp(log_sigs / 2))
    # Plot spectral radii
    # print(f"1 / sqrt(L) = {one_over_sqrt_L}")

    # Finding intersection points

    # intersection_points = log_sigs[np.isclose(h/2, one_over_sqrt_L, atol=0.0005)]
    # intersection_h = np.full(intersection_points.shape, h)
    intersection_points = -2 * np.log(2 / h)
    intersection_h = h

    fig, ax = plt.subplots()
    ax.plot(intersection_points, intersection_h / 2, 'go')

    x_point = -2 * np.log(2 / h)

    ax.axvline(x=x_point, color='gray', linestyle=':')
    ax.text(x_point, 1.0, '$\omega$' + f' = {x_point:.2f}', horizontalalignment='center', verticalalignment='top')
    ax.set_xlabel('$\omega$')
    # ax.set_ylabel('$1/\sqrt{ \\rho(H)}$')
    ax.plot(log_sigs, one_over_sqrt_L, label="$1/\sqrt{ \\rho(D^2 U)}$")
    ax.axhline(y=h / 2, color='r', linestyle='--', label="$h*\sqrt{\\rho(D^2 U)} = 2$")
    ax.legend()
    # ax.title.set_text(f'Predicted Bottleneck in $\log(\sigma)$ for h = {h}')
    plt.savefig(f"{destination}/funnel_spectral_cutoff_{N}_{log_sig_range}_{h}.png")
    plt.show()
    # Create Plots

    # Save Plots


def plot_funnel_trace_1D(N, log_sig_range, h):
    # Get funnel model
    funnel_model = fv.create_model_stan_and_json("funnel", "funnel")
    # Create gridpoints and initial theta
    log_sigs = np.linspace(-log_sig_range, log_sig_range, N)
    theta_0 = np.zeros(funnel_model.param_unc_num())
    spectral_radii = np.zeros(N)

    # Calculate spectral radii
    for j in range(N):
        theta_0[0] = log_sigs[j]
        _, _, H = funnel_model.log_density_hessian(theta_0)
        eigvals, eigvects = np.linalg.eigh(-H)
        # print(eigvals)
        spectral_radii[j] = np.sqrt(np.mean(eigvals ** 2))

    one_over_sqrt_L = (1 / np.sqrt(spectral_radii))
    # Plot spectral radii
    # print(f"1 / sqrt(L) = {one_over_sqrt_L}")

    # Finding intersection points

    intersection_points = log_sigs[np.isclose(h, one_over_sqrt_L, atol=0.001)]
    intersection_h = np.full(intersection_points.shape, h)

    fig, ax = plt.subplots()
    ax.plot(intersection_points, intersection_h / 2, 'go')
    x_point = intersection_points[0]
    ax.axvline(x=x_point, color='gray', linestyle=':')
    ax.text(x_point, 1.0, f'$\omega$ = {x_point:.2f}', horizontalalignment='center', verticalalignment='top')

    ax.plot(log_sigs, one_over_sqrt_L, label="1/sqrt(L)")
    ax.set_ylabel('$1/\sqrt{Tr(H^2)/dim}$')
    ax.set_xlabel('$\omega$')
    ax.axhline(y=h, color='r', linestyle='--')
    ax.title.set_text(f'Predicted Bottleneck in $\omega$ for h = {h}')
    plt.savefig(
        f'/Users/milostevenmarsden/Documents/Simulations/June2024/Trace_Cutoff/funnel_spectral_cutoff_{N}_{log_sig_range}_{h}.png')
    plt.show()
    # Create Plots


if __name__ == "__main__":
    destination = '/Users/milostevenmarsden/Documents/Simulations/August2024/Exact_Spectral_Radius_Cutoff'
    for h in [1 / 32, 1 / 16, 1 / 8, 1 / 4]:
        # plot_funnel_trace_1D(1000, 9, h)
        plot_funnel_spectral_1D(1000, 9, h,destination )
