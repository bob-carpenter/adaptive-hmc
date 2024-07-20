import traceback

import bridgestan as bs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

import step_size_fixed_NUTS_diagnostic_version as dn


def create_model_stan_and_json(stan_name, json_name):
    program_path = "../stan/" + stan_name + ".stan"
    data_path = "../stan/" + json_name + ".json"
    model_bs = bs.StanModel(model_lib=program_path, data=data_path)
    return model_bs


def single_draw_plot(theta_0, sampler_constructor, destination_path, title, funnel):
    model = create_model_stan_and_json("funnel", funnel)
    D = model.param_unc_num()
    draw_contours = (D == 2)
    rng = np.random.default_rng(2007070719960628)
    sampler = sampler_constructor(model, rng, theta_0)
    sampler.draw_diagnostic()
    plot_extensions(sampler, f"{destination_path}/{title}_extensions.png")
    print(sampler._forward_backward_choices)


def get_dark_viridis(darkness_factor=0.6):
    # Load the Viridis colormap
    viridis = plt.cm.get_cmap('viridis', 256)
    # Extract all colors from the Viridis colormap
    colors = viridis(np.linspace(0, 1, 256))

    # Darken the colors
    # Increase the 0.6 factor to reduce darkening if needed
    darkened_colors = colors * darkness_factor
    darkened_colors[:, 3] = 1  # Ensure the alpha channel remains at 1

    # Create a new colormap from the modified colors
    darkened_viridis = LinearSegmentedColormap.from_list('darkened_viridis', darkened_colors)

    # Use this new colormap for your plot
    cmap_lists = darkened_viridis
    return cmap_lists


def plot_extensions(sampler, filename):
    plt.figure(figsize=(12, 10))
    one_dim_model = create_model_stan_and_json("funnel", "one_dimensional_funnel")
    x_lim = 9
    y_lim = 10
    x = np.linspace(-x_lim, x_lim, 1000)
    y = np.linspace(-y_lim, y_lim, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = one_dim_model.log_density(np.array([X[i, j], Y[i, j]]))
    contour_levels = np.percentile(Z, np.arange(0, 101, 18))
    print(f"Contour levels: {contour_levels}")
    plt.contour(X, Y, Z,
                levels=contour_levels,
                colors='black',
                linestyles='dotted',
                linewidths=0.7,
                alpha=1,
                zorder=1)

    position_extensions = sampler.get_position_extensions()
    velocity_extensions = sampler.get_velocity_extensions()
    for extension in position_extensions:
        print(len(extension))
    for extension in velocity_extensions:
        print(len(extension))

    initial_point = position_extensions[0][0]
    initial_velocity = velocity_extensions[0][0]

    position_extensions = [[array.reshape(1, -1) for array in sublist] for sublist in position_extensions]
    velocity_extensions = [[array.reshape(1, -1) for array in sublist] for sublist in velocity_extensions]

    print(f"Length of position extensions: {len(position_extensions)}")
    print(f"Length of velocity extensions: {len(velocity_extensions)}")

    flattened_position_extensions = [array for sublist in position_extensions for array in sublist]
    flattened_velocity_extensions = [array for sublist in velocity_extensions for array in sublist]
    # Concatenate all points
    all_positions = np.vstack(flattened_position_extensions)
    all_velocities = np.vstack(flattened_velocity_extensions)

    print(f"All positions shape: {all_positions.shape}")
    print(f"All velocities shape: {all_velocities.shape}")

    print(f"All positions: {all_positions}")
    print(f"All velocities: {all_velocities}")
    # Create an array of colors for each list
    cmap_lists = get_dark_viridis(1)
    norm = Normalize(vmin=0, vmax=len(position_extensions) - 1)

    colors = []
    list_index = 0
    for sublist in position_extensions:
        color = cmap_lists(norm(list_index))
        # Darken the color by multiplying the RGB values by 0.6
        for array in sublist:
            colors.extend([color] * array.shape[0])
        list_index += 1
    # Convert colors list to a numpy array
    colors = np.array(colors)
    sizes = np.array([sampler.log_joint(pos, vel) for pos, vel in zip(all_positions, all_velocities)])
    print(f'The sizes are {sizes}')
    sizes = 100 * (sizes - np.min(sizes)) / (np.max(sizes) - np.min(sizes)) + 20
    # Calculate the size of the points based on the Hamiltonian

    # Plotting
    start_idx = 0
    plt.scatter(initial_point[0], initial_point[1], color='#75bbfd', s=1.5 * sizes[0], marker="D", zorder=3)
    for i, (pos_sublist, vel_sublist) in enumerate(zip(position_extensions, velocity_extensions)):
        if i == 0:
            pass
        for pos_array, vel_array in zip(pos_sublist, vel_sublist):
            end_idx = start_idx + pos_array.shape[0]
            (plt.scatter(all_positions[start_idx:end_idx, 0],
                         all_positions[start_idx:end_idx, 1],
                         # color=colors[start_idx:end_idx],
                         color='black',
                         s=1.5 * sizes[start_idx:end_idx],
                         zorder=2
                         )
             )

            # label=f'Extension {i + 1}')
            '''
            for j in range(pos_array.shape[0]):
                plt.arrow(pos_array[j, 0],
                          pos_array[j, 1],
                          velocity_scale*vel_array[j, 0],
                          velocity_scale*vel_array[j, 1],
                          head_width=0.01,
                          head_length=0.01,
                          fc=colors[start_idx],
                          ec=colors[start_idx]
                          )
            '''
            start_idx = end_idx
    # plt.title(f'Draw from NUTS with initial point {initial_point} \n and velocity {initial_velocity} \n h = {sampler._stepsize} \n NUTS max = {sampler._max_nuts_search_depth}')
    plt.xlabel('$\log(\sigma))$')
    plt.ylabel('First X coodinate')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust rect to leave space for the legend
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def trajectory_diagnosis(log_sig_initial, sigma_component, x_component, step_size_denom, NUTS_depth):
    rho_0 = np.empty(11)
    theta_0 = np.empty(11)
    theta_0[0] = log_sig_initial
    theta_0[1:] = 0

    rho_0[0] = sigma_component
    rho_0[1:] = x_component
    rho_0 = (rho_0 / np.linalg.norm(rho_0)) * np.sqrt(11)

    sampler_constructor = lambda model, rng, theta: dn.FixedStepSizeNUTSDiagnostic(model, rng, theta, rho_0, 1 / step_size_denom,
                                                                                   NUTS_depth)
    single_draw_plot(theta_0,
                     sampler_constructor,
                     f"/Users/milostevenmarsden/Documents/Simulations/June2024/Basic_NUTS_Evolution_Overleaf_h=1_{step_size_denom}",
                     f"trajectory_diagnosis_with_velocity_initial_log_sigma_{log_sig_initial}_initial_velocity_{sigma_component, x_component}_step_size_{step_size_denom}_NUTS_depth_{NUTS_depth}",
                     'funnel', )


if __name__ == '__main__':

    for start in [-2.0]:  # np.linspace(-6, 6, 13):
        for step_size_denom in [4]:
            try:
                print(f"Starting at {start} with step size {1 / step_size_denom}")
                trajectory_diagnosis(start, 2.0, -1.0, step_size_denom, 10)
            except Exception as e:
                traceback.print_exc()
                continue
