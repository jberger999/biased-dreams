import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import numpy as np
from typing import Optional
from collections import defaultdict

    
def plot_funcs_shared(means: list,
                      stds: list,
                      labels: list,
                      colors: list,
                      linestyles: list,
                      linewidths: list,
                      vline: Optional[int] = None,
                      path: str = "plot",
                      y_label: str = None,
                      title: str = None,
                      y_lim: Optional[list] = None,
                      y_lim2: Optional[list] = None,
                      y_axis_2: Optional[int] = None,
                      y_label_1: Optional[str] = None,
                      y_label_2: Optional[str] = None):
    """
    Plot two functions, with their respective mean and std.
    """
    traj_len = means[0].shape[0]
    x = np.arange(0, traj_len, step=1)
    
    ax = plt.gca()
    
    for i in range(len(means)):
        if i != y_axis_2:
            plt.plot(means[i][:traj_len], color=colors[i], label=labels[i], linestyle=linestyles[i], linewidth=linewidths[i])
    for i in range(len(means)):
        if i != y_axis_2:
            plt.fill_between(x, means[i][:traj_len] - stds[i][:traj_len], means[i][:traj_len] + stds[i][:traj_len], facecolor=colors[i], alpha=0.4)

    if y_axis_2 is not None:
        # Plot index corresponding to y_axis_2 on separate y-axis
        ax2 = ax.twinx()
        ax2.plot(means[y_axis_2], color=colors[y_axis_2], label=labels[y_axis_2], linestyle=linestyles[y_axis_2], linewidth=linewidths[y_axis_2])
        ax2.fill_between(x, means[y_axis_2] - stds[y_axis_2], means[y_axis_2] + stds[y_axis_2], facecolor=colors[y_axis_2], alpha=0.4)
        if y_label_1 is not None and y_label_2 is not None:
            ax.set_ylabel(y_label_1)
            ax2.set_ylabel(y_label_2, rotation=270)
        ax.legend(loc="upper left")
        
        if y_lim2 is not None:
            ax2.set_ylim(y_lim2[0], y_lim2[1])
            
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    plt.xlabel("Time Step $t$")
    if y_label is not None:
        plt.ylabel(y_label)
    
    plt.legend(loc="upper right")
    
    if title is not None:
        plt.title(title)
        
    plt.margins(x=0)
        
    if vline is not None:
        plt.axvline(x=vline, ymin=0, ymax=100, color="gray", alpha=0.6, linestyle="--", zorder=-100)
    
    plt.tight_layout()
    
    plt.savefig(path+".png", bbox_inches='tight', format='png', dpi=300)
    plt.close()


def plot_vector_field_with_lines(
    pts,
    lines,
    lines_cmaps,
    path,
    figname,
    colorbar=True,
    n_bins=25,
    min_bin_count=10,
    num_plots=3,
):
    """
    Plot averaged latent transition vector field for a (shape [num_seq, seq_len, 2]).
    Optionally overlay reference trajectories b.

    Arrows correspond to averaged transitions z_t -> z_{t+1} within PCA-space bins.
    """

    n_sequences, seq_len = pts.shape[:2]

    fig, axs = plt.subplots(1, num_plots, sharex=True, sharey=True, subplot_kw={'aspect':'equal'})
    
    # Color map for lines, under the assumption that all sequences have the same length
    cmaps = {
        "viridis": cm.viridis,
        "plasma": cm.plasma,
        "cool": cm.cool,
        "Greys": cm.Greys
    }
    lines_cmaps = [cmaps[line_cmap] for line_cmap in lines_cmaps]
    color_norm = colors.Normalize(vmin=0, vmax=lines[0].shape[-2]-1)

    # Axis limits
    xmin, xmax = pts[..., 0].min() - 0.25, pts[..., 0].max() + 0.25
    ymin, ymax = pts[..., 1].min() - 0.25, pts[..., 1].max() + 0.25

    # Precompute transitions (p_t -> p_{t+1})
    starts, ends = pts[:, :-1].reshape(-1, 2), pts[:, 1:].reshape(-1, 2)
    deltas = ends - starts
    steps = np.repeat(np.arange(seq_len - 1), n_sequences)

    # Bin transitions
    x_edges, y_edges = np.linspace(xmin, xmax, n_bins + 1), np.linspace(ymin, ymax, n_bins + 1)
    x_idx, y_idx = np.digitize(starts[:, 0], x_edges) - 1, np.digitize(starts[:, 1], y_edges) - 1
    bins = defaultdict(list)
    for xi, yi, s, d, st in zip(x_idx, y_idx, starts, deltas, steps):
        bins[(xi, yi)].append((s, d, st))

    # Compute averaged bin transitions
    bin_data = []
    for (_, _), items in bins.items():
        if len(items) < min_bin_count:
            continue

        starts_b = np.array([it[0] for it in items])
        deltas_b = np.array([it[1] for it in items])
        steps_b = np.array([it[2] for it in items])

        mean_start = starts_b.mean(axis=0)
        mean_delta = deltas_b.mean(axis=0)
        mean_step = steps_b.mean()

        bin_data.append((mean_start, mean_delta, mean_step))

    # ----------- RENDER -----------
    for i, ax in enumerate(axs):
        ax.clear()
        
        if i < len(lines):
            # Plot vector field (same for every subplot)
            for start, delta, step in bin_data:
                dx, dy = delta
                norm_vec = np.hypot(dx, dy)
                if norm_vec == 0:
                    continue
                ux = dx / norm_vec / 7.0
                uy = dy / norm_vec / 7.0
                ax.quiver(
                    start[0],
                    start[1],
                    ux,
                    uy,
                    angles="xy",
                    scale_units="xy",
                    linewidth=1.4,
                    scale=1.0,
                    alpha=0.4,
                    width=0.007,
                    zorder=300,
                )
            
            # Optional line
            for step in range(lines[i].shape[-2] - 1):
                p0, p1 = lines[i][step], lines[i][step+1]
                seg = np.array([[[p0[0], p0[1]], [p1[0], p1[1]]]])
                lc_b = LineCollection(
                    seg,
                    cmap=lines_cmaps[i],
                    norm=color_norm,
                    linewidth=2.0,
                    capstyle="round",
                    joinstyle="round",
                    alpha=1.0
                )
                lc_b.set_array(np.array([step]))
                ax.add_collection(lc_b)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            #ax.set_aspect("equal", adjustable="datalim")
        else:
            ax.axis("off")
            ax.annotate("N/A", (0.5, 0.5), xycoords="axes fraction", va="center")
            
        # Colorbar only for further right plot.
        if colorbar and i == len(axs) - 1:
            for j in range(len(lines_cmaps)):
                sm = cm.ScalarMappable(cmap=lines_cmaps[j], norm=color_norm)
                sm.set_array([])
                # [left, bottom, width, height] in figure coordinates
                cax = fig.add_axes([ax.get_position().x1+0.03+j*0.03,
                                    ax.get_position().y0,
                                    0.02,
                                    ax.get_position().height])
                if j == len(lines_cmaps) - 1:
                    fig.colorbar(sm, cax=cax, label="Sequence Step")
                else:
                    cbar = fig.colorbar(sm, cax=cax)
                    cbar.set_ticklabels([])

    plt.subplots_adjust(wspace=0.1)
    plt.savefig(
        path + "/" + figname + ".pdf",
        bbox_inches="tight",
        format="pdf",
        dpi=300,
    )
    plt.close()