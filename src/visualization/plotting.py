import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

matplotlib.rcParams["text.usetex"] = True


def plot_IRF(tspan, irf, setting_time, experiment_name, output_dir):
    linewidth = 0.75
    label_font = {"fontname": "Times New Roman", "fontsize": 9}
    cm = 1 / 2.54  # centimeters in inches
    title_font_dict = {"size": 10, "weight": "bold", "family": "Times New Roman"}
    plt.figure(figsize=(9 * cm, 6 * cm), dpi=300)
    plt.subplots_adjust(bottom=0.20, left=0.20)
    plt.title("Impulse response function", fontdict=title_font_dict)
    plt.plot(tspan, irf, label="IRF", linewidth=linewidth)
    plt.axvline(
        x=setting_time,
        color="r",
        linestyle="--",
        label="setting time",
        linewidth=linewidth,
    )
    plt.xlabel("Time/s", **label_font)
    plt.ylabel("Amplitude", **label_font)
    plt.xticks(**label_font)
    plt.legend(fontsize=9, loc="upper right")
    plt.savefig(f"{output_dir}/{experiment_name}_IRF.png")


def plot_comparison_auto(tspan, x, k, c, setting_time, experiment_name, output_dir):
    linewidth = 0.75
    label_font = {"fontname": "Times New Roman", "size": 9}
    cm = 1 / 2.54  # centimeters in inches
    title_font_dict = {"size": 10, "weight": "bold", "family": "Times New Roman"}
    plt.figure(figsize=(9 * cm, 6 * cm), dpi=300)
    plt.subplots_adjust(bottom=0.20, left=0.20)
    plt.title("Input and responses", fontdict=title_font_dict)
    plt.plot(tspan, k * x[:, 0], label="k*x1", linewidth=linewidth)
    plt.plot(tspan, c * x[:, 1], label="c*x2", linewidth=linewidth)
    plt.axvline(
        x=setting_time,
        color="r",
        linestyle="--",
        label="setting time",
        linewidth=linewidth,
    )
    plt.xlabel("Time/s", **label_font)
    plt.ylabel("Amplitude", **label_font)
    plt.xticks(**label_font)
    plt.legend(fontsize=9, loc="upper right")
    plt.savefig(f"{output_dir}/{experiment_name}_input_and_responses.png")


def plot_comparison(tspan, u, x, k, c, setting_time, experiment_name, output_dir):
    linewidth = 0.75
    label_font = {"fontname": "Times New Roman", "size": 9}
    cm = 1 / 2.54  # centimeters in inches
    title_font_dict = {"size": 10, "weight": "bold", "family": "Times New Roman"}
    plt.figure(figsize=(9 * cm, 6 * cm), dpi=300)
    plt.subplots_adjust(bottom=0.20, left=0.20)
    plt.title("Input and responses", fontdict=title_font_dict)
    plt.plot(tspan, u, label="u(t)", linewidth=linewidth)
    plt.plot(tspan, k * x[:, 0], label="k*x1", linewidth=linewidth)
    plt.plot(tspan, c * x[:, 1], label="c*x2", linewidth=linewidth)
    plt.axvline(
        x=setting_time,
        color="r",
        linestyle="--",
        label="setting time",
        linewidth=linewidth,
    )
    plt.xlabel("Time/s", **label_font)
    plt.ylabel("Amplitude", **label_font)
    plt.xticks(**label_font)
    plt.legend(fontsize=9, loc="upper right")
    plt.savefig(f"{output_dir}/{experiment_name}_input_and_responses.png")


def plot_displacement_prediction(
    tspan,
    displacement_pred,
    displacement_true,
    setting_time,
    experiment_name,
    output_dir,
):
    # make sure that the inputs have the same shape
    if displacement_pred.shape != displacement_true.shape:
        displacement_pred = displacement_pred.T
    error = displacement_pred - displacement_true
    rms_ratio = (error**2).mean() / (displacement_true**2).mean()
    linewidth = 0.75
    label_font = {"fontname": "Times New Roman", "size": 9}
    cm = 1 / 2.54  # centimeters in inches
    title_font_dict = {"size": 10, "weight": "bold", "family": "Times New Roman"}
    plt.figure(figsize=(9 * cm, 6 * cm), dpi=300)
    plt.subplots_adjust(bottom=0.20, left=0.20)
    plt.title(f"Displacement - RMS {rms_ratio:.3f}", fontdict=title_font_dict)
    plt.plot(tspan, displacement_pred, label="Prediction", linewidth=linewidth)
    plt.plot(tspan, displacement_true, label="True", linewidth=linewidth)
    plt.plot(tspan, error, label="Error", linewidth=linewidth)
    plt.axvline(
        x=setting_time,
        color="r",
        linestyle="--",
        label="setting time",
        linewidth=linewidth,
    )
    plt.xlabel("Time/s", **label_font)
    plt.ylabel("Amplitude", **label_font)
    plt.xticks(**label_font)
    plt.legend(fontsize=9, loc="upper right")
    plt.savefig(f"{output_dir}/{experiment_name}_displacement_prediction.png")


def plot_velocity_prediction(
    tspan, velocity_pred, velocity_true, setting_time, experiment_name, output_dir
):
    if velocity_pred.shape != velocity_true.shape:
        velocity_true = velocity_true.T
    error = velocity_pred - velocity_true
    rms_ratio = (error**2).mean() / (velocity_true**2).mean()
    label_font = {"fontname": "Times New Roman", "size": 9}
    cm = 1 / 2.54  # centimeters in inches
    linewidth = 0.75
    title_font_dict = {"size": 10, "weight": "bold", "family": "Times New Roman"}
    plt.figure(figsize=(9 * cm, 6 * cm), dpi=300)
    plt.subplots_adjust(bottom=0.20, left=0.20)
    plt.title(f"Velocity - RMS {rms_ratio:.3f}", fontdict=title_font_dict)
    plt.plot(tspan, velocity_pred, label="Prediction", linewidth=linewidth)
    plt.plot(tspan, velocity_true, label="True", linewidth=linewidth)
    plt.plot(tspan, error, label="Error", linewidth=linewidth)
    plt.axvline(
        x=setting_time,
        color="r",
        linestyle="--",
        label="setting time",
        linewidth=linewidth,
    )
    plt.xlabel("Time/s", **label_font)
    plt.ylabel("Amplitude", **label_font)
    plt.xticks(**label_font)
    plt.legend(fontsize=9, loc="upper right")
    plt.savefig(f"{output_dir}/{experiment_name}_velocity_prediction.png")


def plot_losses(Loss_data, Loss_physics, experiment_name, output_dir):
    linewidth = 0.75
    label_font = {"fontname": "Times New Roman", "size": 9}
    cm = 1 / 2.54  # centimeters in inches
    title_font_dict = {"size": 10, "weight": "bold", "family": "Times New Roman"}
    label_font = {"size": 10, "family": "Times New Roman"}

    fig, axs = plt.subplots(
        2, 1, figsize=(9 * cm, 12 * cm), dpi=300
    )  # Create 2 subplots vertically
    fig.subplots_adjust(
        bottom=0.20, left=0.20, hspace=0.5
    )  # Adjust the space between subplots

    # Formatter for scientific notation
    formatter = FuncFormatter(lambda y, _: "{:.1e}".format(y))

    # Plot Loss_data on the first subplot
    axs[0].set_title("Data loss", fontdict=title_font_dict)
    axs[0].plot(Loss_data, label="Data loss", linewidth=linewidth)
    axs[0].set_xlabel("Epochs", **label_font)
    axs[0].set_ylabel("Loss", **label_font)
    axs[0].legend(fontsize=9, loc="upper right")
    axs[0].set_yscale("log")
    axs[0].yaxis.set_major_formatter(formatter)  # Set y-ticks to scientific notation

    # Plot Loss_physics on the second subplot
    axs[1].set_title("Physics loss", fontdict=title_font_dict)
    axs[1].plot(Loss_physics, label="Physics loss", linewidth=linewidth)
    axs[1].set_xlabel("Epochs", **label_font)
    axs[1].set_ylabel("Loss", **label_font)
    axs[1].legend(fontsize=9, loc="upper right")
    axs[1].set_yscale("log")
    axs[1].yaxis.set_major_formatter(formatter)  # Set y-ticks to scientific notation

    plt.savefig(f"{output_dir}/{experiment_name}_losses.png")


def plot_mck(m, c, k, experiment_name, output_dir):
    linewidth = 0.75
    label_font = {"fontname": "Times New Roman", "size": 9}
    cm = 1 / 2.54  # centimeters in inches
    title_font_dict = {"size": 10, "weight": "bold", "family": "Times New Roman"}
    label_font = {"size": 10, "family": "Times New Roman"}

    fig, axs = plt.subplots(
        3, 1, figsize=(9 * cm, 12 * cm), dpi=300
    )  # Create 2 subplots vertically
    fig.subplots_adjust(
        bottom=0.20, left=0.20, hspace=0.5
    )  # Adjust the space between subplots

    # Plot Loss_data on the first subplot
    axs[0].set_title("Mass", fontdict=title_font_dict)
    axs[0].plot(m, label="Mass", linewidth=linewidth)
    axs[0].set_xlabel("Epochs", **label_font)
    axs[0].set_ylabel("Mass", **label_font)
    axs[0].legend(fontsize=9, loc="upper right")

    # Plot Loss_physics on the second subplot
    axs[1].set_title("Damping", fontdict=title_font_dict)
    axs[1].plot(c, label="Damping", linewidth=linewidth)
    axs[1].set_xlabel("Epochs", **label_font)
    axs[1].set_ylabel("Damping", **label_font)
    axs[1].legend(fontsize=9, loc="upper right")

    # Plot Loss_physics on the second subplot
    axs[2].set_title("Stiffness", fontdict=title_font_dict)
    axs[2].plot(k, label="Stiffness", linewidth=linewidth)
    axs[2].set_xlabel("Epochs", **label_font)
    axs[2].set_ylabel("Stiffness", **label_font)
    axs[2].legend(fontsize=9, loc="upper right")

    plt.savefig(f"{output_dir}/{experiment_name}_mck.png")
