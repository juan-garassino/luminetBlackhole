from black_hole import *
import os
import datetime
import numpy as np
from colorama import Fore, Style
from params import params


def generate_inclinations(incl, num_inclinations=None, random=False, step=False):
    if num_inclinations is None and not random and not step:
        return [incl]  # Only one inclination is given

    inclinations = []

    if random:
        inclinations.extend(np.random.randint(45, 136, num_inclinations))

    if step:
        num_steps = num_inclinations or 1
        step_size = 180 / (num_steps + 1)
        inclinations.extend(np.arange(step_size, 181, step_size))

    return inclinations

incl = None

M = 1.0

num_inclinations = 5

inclinations = generate_inclinations(incl, num_inclinations, random=True, step=False)

print(f"\n{Fore.YELLOW}Inclinations: {inclinations}{Style.RESET_ALL}")

# Create folders for SVG and PNG files
svg_folder = "svg_files"

os.makedirs(svg_folder, exist_ok=True)

png_folder = "png_files"

os.makedirs(png_folder, exist_ok=True)

isoradials = True

isoredshifts = True

sample_points = False

isoredshifts_from_points = True

isoradial_redshift = True

# Define the facecolor variable
facecolor = "#1a1a1a"  # Change the value to the desired facecolor

for incl in inclinations:

    incl = int(incl)

    print(f'\nWorking on inclination {incl}')

    blackhole = BlackHole(inclination=incl, mass=M)

    blackhole.disk_outer_edge = 100 * M

    blackhole.iz_solver_params["radial_precision"] = 30

    blackhole.angular_properties["angular_precision"] = 200

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if isoradials:

        # Generate a random step size
        step_size = np.random.randint(1, 10)

        print(f'\nWorking on step size {step_size}')

        fig, ax = blackhole.plot_isoradials(
            np.arange(5, 101, step_size).tolist(), np.arange(5, 101, step_size).tolist()
        )

        for file_type in ["svg", "png"]:
            file_name = f"isoradials_{incl}_{timestamp}.{file_type}"
            file_path = os.path.join(
                svg_folder if file_type == "svg" else png_folder, file_name
            )
            print(f"\n{Fore.CYAN}Plotting {file_name} @ {file_path}{Style.RESET_ALL}")
            fig.savefig(file_path, format=file_type, facecolor=facecolor)
            plt.close(fig)

    if isoredshifts:

        M = 1.0

        incl = 76

        bh = BlackHole(inclination=incl, mass=M)

        bh.disk_outer_edge = 100 * M

        bh.iz_solver_params["radial_precision"] = 30

        bh.angular_properties["angular_precision"] = 200

        sample_size = np.random.randint(10, 25)

        print(f'\nWorking on sample size {sample_size}')

        lower_bound = np.random.uniform(-0.99, -0.69, 1)

        upper_bound = np.random.uniform(0.69, 0.99, 1)

        values = np.linspace(lower_bound, upper_bound, sample_size)

        rounded_values = np.round(values, decimals=2).flatten().tolist()

        print(f'\nWorking on isoredshifts {rounded_values}')

        fig, ax = bh.plot_isoredshifts(
            redshifts=rounded_values,
            plot_core=True,
        )

        # fig.savefig(f"isoredshifts_{incl}_2.svg", facecolor="#F0EAD6")

        # n_isoradials = np.random.randint(2, 10)

        # radii = np.linspace(3.01, 60, n_isoradials)

        # blackhole.calc_isoradials(radii, radii)

        # fig, ax = blackhole.plot_isoredshifts(
        #     redshifts=[-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75],
        #     plot_core=True,
        # )

        for file_type in ["svg", "png"]:
            file_name = f"isoredshifts_{incl}_{timestamp}.{file_type}"
            file_path = os.path.join(
                svg_folder if file_type == "svg" else png_folder, file_name
            )
            print(f"\n{Fore.CYAN}Plotting {file_name} @ {file_path}{Style.RESET_ALL}")
            fig.savefig(file_path, format=file_type, facecolor=facecolor)
            plt.close(fig)

    if sample_points:
        blackhole.sample_points(n_points=7000)
        fig, ax = blackhole.plot_points()
        print(
            f"\n{Fore.CYAN}Plotting accretion disk points @ {file_path}{Style.RESET_ALL}"
        )
        file_name = f"disk_points_{incl}_{timestamp}.png"
        file_path = os.path.join(png_folder, file_name)
        fig.savefig(file_path, format="png", facecolor='#000000')
        plt.close(fig)

    if isoredshifts_from_points:
        fig, ax = blackhole.plot_isoredshifts_from_points()
        print(
            f"\n{Fore.CYAN}Plotting isoredshifts from points @ {file_path}{Style.RESET_ALL}"
        )
        file_name = f"isoredfrompoints_{incl}_{timestamp}.svg"
        file_path = os.path.join(svg_folder, file_name)
        fig.savefig(file_path, format="svg", facecolor="#F0EAD6")
        plt.close(fig)

    if isoradial_redshift:
        isoradial = Isoradial(
            radius=30 * M, incl=80 * np.pi / 180, bh_mass=M, order=0, params=params
        )
        isoradial.calculate()
        fig, ax = isoradial.plot_redshift()
        print(
            f"\n{Fore.CYAN}Plotting isoradial redshift @ {file_path}{Style.RESET_ALL}"
        )

        for file_type in ["svg", "png"]:
            file_name = f"isoradial_redshift_{incl}_{timestamp}.{file_type}"
            file_path = os.path.join(
                svg_folder if file_type == "svg" else png_folder, file_name
            )
            print(f"\n{Fore.CYAN}Plotting {file_name} @ {file_path}{Style.RESET_ALL}")
            fig.savefig(file_path, format=file_type, facecolor=facecolor)
            plt.close(fig)


# n_isoradials = np.random.randint(2, 10)


# radii = np.linspace(3.01, 60, n_isoradials)

# bh.calc_isoradials(radii, radii)

##


# # Sample points on the accretion disk and plot them

# bh.sample_points(n_points=2000)

# bh.plot_points()

# # Plot isoredshift lines from the sampled points (useful for edge-on or top-down view, where the other method fails)

# fig, ax = bh.plot_isoredshifts_from_points()

# print(f'isoredfrompoints_{incl}_2.svg')

# fig.savefig(f"isoredfrompoints_{incl}_2.svg", facecolor="#F0EAD6")


# isoradial = Isoradial(radius=30*M, incl=80 * np.pi / 180, bh_mass=M, order=0, params=params)

# isoradial.calculate()

# isoradial.plot_redshift()  # plot its redshifts along the line ERROR M IS NOT DEFINE


# ###

# bh.sample_points(n_points=10000)

# bh.calc_isoradials([10, 20], [])

# bh.plot_isoradials([10, 20], [10, 20], show=True)

# ####
