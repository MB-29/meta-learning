import pickle
import numpy as np
import matplotlib.pyplot as plt

import scripts.plot.layout

time_intervals = [np.arange(1200), np.arange(1200, 1200+1999)]
index_intervals = [np.arange(70, 70+1200), np.arange(0, 1999)]

fig = plt.figure(figsize=(4, 3.))
# fig.set_tight_layout(True)

for mass_index, mass_value in enumerate([1, 4]):
# for mass_index, mass_value in enumerate([4]):
    file_path = f'output_{mass_value}.pkl'
    control_file_path = f'control_{mass_value}.pkl'

    with open(file_path, 'rb') as file:
        values = pickle.load(file)
    # with open(control_file_path, 'rb') as file:
    #     control_values = pickle.load(file)
    # values['action'] = control_values['action']
    # values['plan'] = control_values['plan']
    # with open(file_path, 'wb') as file:
    #     pickle.dump(values, file)
    
    action_values = values['action']
    estimation_values = np.abs(values['estimation'])
    state_values = values['state']
    u_ff_values = values['plan']
    time_values = time_intervals[mass_index]
    index_values = index_intervals[mass_index]
    n_steps = len(time_values)

    # n_steps = state_values[0].shape[0]

    displacement_values = -state_values[0, :, 0]


    plt.subplot(3, 1, 1)
    if mass_index == 1:
        displacement_values += 0.1
    plt.plot(time_values, displacement_values[index_values], color='red', lw=2.5)
    plt.plot(time_values, n_steps*[0.5], color='black', ls='--', lw=3.)

    plt.xticks([])
    plt.ylabel('position')


    plt.subplot(3, 1, 2)

    u_values = action_values[0][index_values].squeeze()
    prediction_values = u_ff_values[0][index_values].squeeze()
    smoothed_u_values = u_values
    smoothed_u_values = np.convolve(u_values, np.ones(50)/50., mode='same')
    smoothed_prediction_values = np.convolve(prediction_values, np.ones(50)/50., mode='same')
    plt.plot(time_values, smoothed_u_values, color='black', ls='--', lw=3.)
    plt.plot(time_values, smoothed_prediction_values, color='red', lw=2., alpha=.8)
    plt.ylim((-.3, .3))
    plt.xticks([])
    plt.yticks([-0.3, 0.3])


    

    window = np.array([1., 2., 3., 4., 3., 2., 1.])
    window /= window.sum()
    window = np.ones(60) / 60.
    smoothed_estimation_values = np.convolve(estimation_values, window)
    chosen_estimation_values = smoothed_estimation_values

    if mass_index == 1:
        chosen_estimation_values = np.where(time_values<2_200, smoothed_estimation_values[index_values], smoothed_estimation_values[index_values][900])
    plt.ylabel('torque')


    plt.subplot(3, 1, 3)
    plt.plot(time_values, chosen_estimation_values[index_values], color='red', lw=2.5, alpha=.8)
    plt.plot(time_values, n_steps*[mass_value], color='black', ls='--', lw=3.)
    plt.ylim((.5, 4.5))
    plt.yticks(([1., 4.]))
    plt.ylabel(r'mass')
fig.align_labels()

plt.subplot(3, 1, 1)
plt.title('Upkie')
plt.subplot(3, 1, 2)
plt.axhline(y=0, color='black', lw=1)
fig.subplots_adjust(left=0.2, wspace=0, hspace=0.2)

plt.subplot(3, 1, 3)
plt.xlabel('time')
plt.gca().xaxis.set_label_coords(.5, -0.1)

plt.xticks([0, 3000])
# plt.plot(time_values, n_steps*[0.8], ls='--', color='black')
plt.savefig('output/plots/adaptive_upkie.pdf')
plt.show()



        