from luminet.blackhole import BlackHole
from luminet.isoradial import Isoradial
from luminet.isoredshift import Isoredshift
import numpy as np

M = 1.0

############ The Isoradial class ############
# Calculate single isoradial

params = {
    "isoradial_angular_parameters": {
        "start_angle": 0,
        "end_angle": np.pi,
        "angular_precision": 100,
        "mirror": True,
    },
    "isoradial_solver_parameters": {
        "initial_guesses": 12,
        "midpoint_iterations": 20,
        "plot_inbetween": False,
        "min_periastron": 3.001,
        "use_ellipse": True,
    },
    "isoredshift_solver_parameters": {
        "initial_guesses": 12,
        "midpoint_iterations": 12,
        "times_inbetween": 2,
        "retry_angular_precision": 15,
        "min_periastron": 3.01,
        "use_ellipse": True,
        "retry_tip": 50,
        "initial_radial_precision": 15,
        "plot_inbetween": False,
    },
    "plot_params": {
        "plot_isoredshifts_inbetween": True,
        "save_plot": True,
        "plot_ellipse": True,
        "plot_core": True,
        "redshift": True,
        "linestyle": "dashed",
        "linewidth": 1.0,
        "key": "",
        "face_color": "black",
        "line_color": "white",
        "text_color": "white",
        "alpha": 1.0,
        "show_grid": False,
        "legend": True,
        "orig_background": False,
        "plot_disk_edges": True,
        "ax_lim": [-100, 100],
    },
}

isoradial = Isoradial(
    radius=30 * M, incl=80 * np.pi / 180, bh_mass=M, order=0, params=params
)

isoradial.calculate()

# isoradial.plot_redshift()  # plot its redshifts along the line ERROR M IS NOT DEFINE

############ The BlackHole class ############
blackhole = BlackHole(inclination=85, mass=M)

## Plot isoradial lines. Plotting 1 Isoradial is equivalent to the above method
blackhole.plot_isoradials(np.arange(5, 101, 1).tolist(), np.arange(5, 101, 5).tolist())

## Plot Isoredshift lines
blackhole.plot_isoredshifts(
    redshifts=[-0.5, -0.35, -0.15, 0.0, 0.15, 0.25, 0.5, 0.75, 1.0]
)
# This method fails for extreme inclinations i.e. edge-on and top-down

## Sample points on the accretion disk and plot them
blackhole.sample_points(n_points=2000)

blackhole.plot_points()

# Plot isoredshift lines from the sampled points (useful for edge-on or top-down view, where the other method fails)
blackhole.plot_isoredshifts_from_points()
