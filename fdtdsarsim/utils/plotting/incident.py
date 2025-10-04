import matplotlib.pyplot as plt
import numpy as np
import h5py

import fdtdsarsim.utils.log as log


def plot_pulse(inc_beam):
    """ Plot the pulse used to create the incident beam.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True,
                                        figsize=(6, 6))

    # Axis 1: Time domain pulse
    ax1.plot(inc_beam.time_vals, np.real(inc_beam.pulse))
    ax1.set(xlabel='Time (s)', ylabel='Amplitude',
            title='Incident beam excitation pulse')

    # Axis 2: Frequency domain of pulse
    freq = np.fft.fftshift(inc_beam.pulse_freqs)
    fft_mag = 20 * np.log10(np.abs(np.fft.fftshift(inc_beam.pulse_fft)))
    ax2.plot(freq, fft_mag)
    ax2.set(xlabel='Frequency (Hz)', ylabel='Amplitude (dB)',
            title='Excitation pulse spectrum')

    # Axis 3: Zoom in on the spectrum
    maxval_idx = np.argmax(fft_mag)
    disp_min_idx = int(0.9 * maxval_idx)
    disp_max_idx = int(1.1 * maxval_idx)
    maxval_freq = freq[maxval_idx]
    ax3.plot(freq[disp_min_idx:disp_max_idx],
             fft_mag[disp_min_idx:disp_max_idx])
    ax3.axvline(maxval_freq, linestyle='--', color='r')
    xlims = ax3.get_xlim()
    ylims = ax3.get_ylim()
    text_loc = (maxval_freq + 0.15 * (xlims[1] - xlims[0]),
                ylims[0] + 0.8 * (ylims[1] - ylims[0]))
    ax3.text(text_loc[0], text_loc[1], f'f = {maxval_freq:.4e}', color='r')
    ax3.set(xlabel='Frequency (Hz)', ylabel='Amplitude (dB)',
            title='Excitation pulse spectrum (cropped to relevant area)')


def incident_excitation_movie(pulse_ex, pulse_ey, disp_step):
    """ Plot the excitation to the incident beam simulation.

    This plots the 3D array created by the 'create_inc_excitation' function.
    """

    logger = log.getLog(__name__)
    logger.debug('Creating incident excitation movie')
    nt_inc = pulse_ex.shape[0]

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True,
                                   figsize=(10, 5))
    view_max = np.max(np.real(pulse_ey))
    view_min = np.min(np.real(pulse_ey))

    ax1.set_title('Real($E_x$)')
    ax1.set_xlabel('X (cells)')
    ax1.set_ylabel('Y (cells)')

    ax2.set_title('Real($E_y$)')
    ax2.set_xlabel('X (cells)')
    ax2.set_ylabel('Y (cells)')

    re_ex_img = ax1.imshow(np.real(pulse_ex[0, :, :]), vmin=view_min,
                           vmax=view_max)
    re_ey_img = ax2.imshow(np.real(pulse_ey[0, :, :]), vmin=view_min,
                           vmax=view_max)
    plt.show()

    for i in range(nt_inc // disp_step):
        re_ex_img.set_data(np.real(pulse_ex[i * disp_step, :, :]))
        re_ey_img.set_data(np.real(pulse_ey[i * disp_step, :, :]))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)


def incident_beam_movie(inc_filename: str, disp_lim: float, disp_step: int):
    """ Plot the incident beam in the specified HDF5 file.
    """
    z0 = 376.730313668  # [Ohms] Impedance of free space
    with h5py.File(inc_filename, 'r') as inc_file:
        ex_inc = inc_file['ex']
        ey_inc = inc_file['ey']
        hx_inc = inc_file['hx']
        hy_inc = inc_file['hy']

        plt.ion()
        fig, ((ex_ax, ey_ax), (hx_ax, hy_ax)) = \
            plt.subplots(2, 2, constrained_layout=True, sharey=True,
                         figsize=(8, 8))

        # Initialize plots
        ex_ax.set_title('$E_x$')
        ex_ax.set_xlabel('X (cells)')
        ex_ax.set_ylabel('Y (cells)')

        ey_ax.set_title('$E_y$')
        ey_ax.set_xlabel('X (cells)')
        ey_ax.set_ylabel('Y (cells)')

        hx_ax.set_title('$H_x$')
        hx_ax.set_xlabel('X (cells)')
        hx_ax.set_ylabel('Y (cells)')

        hy_ax.set_title('$H_y$')
        hy_ax.set_xlabel('X (cells)')
        hy_ax.set_ylabel('Y (cells)')

        view_min = -disp_lim
        view_max = disp_lim
        nt_inc = ex_inc.shape[0]

        ex_img = ex_ax.imshow(ex_inc[0, :, :], vmin=view_min, vmax=view_max,
                              cmap='RdBu')
        ey_img = ey_ax.imshow(ey_inc[0, :, :], vmin=view_min, vmax=view_max,
                              cmap='RdBu')
        hx_img = hx_ax.imshow(hx_inc[0, :, :], vmin=view_min, vmax=view_max,
                              cmap='RdBu')
        hy_img = hy_ax.imshow(hy_inc[0, :, :], vmin=view_min, vmax=view_max,
                              cmap='RdBu')

        for i in range(nt_inc // disp_step):
            ex_img.set_data(ex_inc[i * disp_step, :, :])
            ey_img.set_data(ey_inc[i * disp_step, :, :])
            hx_img.set_data(hx_inc[i * disp_step, :, :] * z0)
            hy_img.set_data(hy_inc[i * disp_step, :, :] * z0)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)
