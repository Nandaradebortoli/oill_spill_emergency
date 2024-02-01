import matplotlib.pyplot as plt
from pandas import Series


def plot_viscosity(
    time: Series,
    visc_min: Series,
    visc_max: Series,
    visc_mean: Series,
    output_file: str,
):
    plt.style.use("ggplot")

    fig, axs = plt.subplots(1, figsize=(8, 5))
    axs.plot(time, visc_min, linewidth=3, label="min")
    axs.plot(time, visc_max, linewidth=3, label="max")
    axs.plot(time, visc_mean, linewidth=3, label="mean")
    legend = axs.legend(bbox_to_anchor=(0.7, -0.12), loc=1, fontsize="small", ncol=3)
    legend.get_frame().set_facecolor("grey")
    # axs.axvline(x=16.4, color='k')
    axs.tick_params(labelsize=8)
    fig.text(
        0.04,
        0.5,
        "Viscosity (cP)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=11,
    )
    axs.set_xlabel("Time in days", fontsize=10)
    plt.savefig(output_file, dpi=200, transparent=False, bbox_inches="tight")
    plt.close()
