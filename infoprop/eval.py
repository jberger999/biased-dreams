import argparse
import torch
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from collections import defaultdict
from sklearn.decomposition import PCA


def plot_vector_field_with_lines(
    pts,                 # now expected as (obs, next_obs)
    lines,
    lines_cmaps,
    path,
    figname,
    n_bins=25,
    min_bin_count=10,
    num_plots=3,
):
    """
    Plot averaged transition vector field for flat transitions
    obs -> next_obs, both of shape [N, 2].

    Arrows correspond to averaged transitions within 2D bins.
    """
    obs, next_obs = pts  # unpack
    assert obs.shape == next_obs.shape
    assert obs.shape[-1] == 2

    n_samples = obs.shape[0]

    fig, axs = plt.subplots(
        1,
        num_plots,
        sharex=True,
        sharey=True,
        subplot_kw={'aspect': 'equal'}
    )

    if num_plots == 1:
        axs = [axs]

    # Color map for lines (UNCHANGED)
    cmaps = {
        "viridis": cm.viridis,
        "plasma": cm.plasma,
        "cool": cm.cool,
        "Greys": cm.Greys
    }
    lines_cmaps = [cmaps[line_cmap] for line_cmap in lines_cmaps]
    color_norm = colors.Normalize(vmin=0, vmax=lines[0].shape[-2] - 1) if len(lines) > 0 else None

    # Axis limits
    xmin, xmax = obs[:, 0].min() - 0.25, obs[:, 0].max() + 0.25
    ymin, ymax = obs[:, 1].min() - 0.25, obs[:, 1].max() + 0.25

    # ----------- TRANSITIONS -----------
    starts = obs.reshape(-1, 2)
    ends = next_obs.reshape(-1, 2)
    deltas = ends - starts

    # Dummy step index (since no sequence structure anymore)
    steps = np.zeros(n_samples)

    # ----------- BINNING -----------
    x_edges = np.linspace(xmin, xmax, n_bins + 1)
    y_edges = np.linspace(ymin, ymax, n_bins + 1)

    x_idx = np.digitize(starts[:, 0], x_edges) - 1
    y_idx = np.digitize(starts[:, 1], y_edges) - 1

    bins = defaultdict(list)
    for xi, yi, s, d, st in zip(x_idx, y_idx, starts, deltas, steps):
        bins[(xi, yi)].append((s, d, st))

    # ----------- AVERAGE BIN TRANSITIONS -----------
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

        if i < len(lines) or len(lines) == 0:

            # Vector field
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

            # ----------- LINE CODE (UNTOUCHED) -----------
            if i < len(lines):
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

        else:
            ax.axis("off")
            ax.annotate("N/A", (0.5, 0.5),
                        xycoords="axes fraction",
                        va="center")

        # Colorbar
        if i == len(axs) - 1 and len(lines) > 0:
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


def save_to_csv(data, header_names, path, file_name):
    seq_len = len(data[0]) # assuming all data items have the same length
    with open(path + "/" + file_name, "w", newline="") as f:
        writer = csv.writer(f)
        
        header = ["step"] + header_names
        writer.writerow(header)
        
        for step in range(seq_len):
            row = [step] + [d[step].item() for d in data]
            writer.writerow(row)


def _create_env_and_set_state(state, task="halfcheetah"):
    """
    Create environment and set simulator state.
    """
    if task == "halfcheetah":
        assert state.shape == (17,), "State must be 17-dimensional"
    else:
        raise NotImplementedError

    if task == "halfcheetah":
        env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    else:
        raise NotImplementedError

    env.reset()

    # Split into qpos and qvel
    if task == "halfcheetah":
        qpos = np.zeros(9)
        qpos[0] = 0.0
        qpos[1:] = state[:8]
        qvel = state[8:]
    else:
        raise NotImplementedError

    env.unwrapped.set_state(qpos, qvel)
    
    return env


def set_state_and_render(state, task="halfcheetah"):
    """
    Set simulator state and render corresponding image.
    """
    env = _create_env_and_set_state(state=state,
                                    task=task)
    img = env.render()
    env.close()
    return img


def simulate_acts(start_state, acts, task="halfcheetah"):
    """
    Set simulator to start physical state, perform actions to get simulator rollout.
    """
    env = _create_env_and_set_state(state=start_state,
                                    task=task)
    
    # Simulate actions
    collected_obs = []
    for i in range(acts.shape[0]):
        obs, _, _, _, _ = env.step(acts[i])
        collected_obs.append(obs)
    collected_obs = np.stack(collected_obs, 0)
    
    return collected_obs


def vis_discr(env, load_path, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    with np.load(load_path + "/infos.npz") as data:
        obs = data["obs_trajectory"]
        uncertainties = data["uncertainties"]
        acts = data["act_trajectory"]
        
    gt_obs = simulate_acts(start_state=obs[0],
                           acts=acts)

    # Compute discrepancy between positions only
    if env == "halfcheetah":
        stepwise_obs_discr = abs(gt_obs[:, :8] - obs[:, :8]).mean(-1)
    else:
        raise NotImplementedError
    cum_uncertainties = uncertainties.cumsum(0)
    
    plt.plot(np.arange(0, len(stepwise_obs_discr)), stepwise_obs_discr, "-", color="tab:orange", label="Phys. Discrepancy")
    plt.plot(np.arange(0, len(stepwise_obs_discr)), cum_uncertainties, "-", color="forestgreen", label="Uncertainty")
    plt.yscale("log")
    plt.xlabel("Rollout Time Step")
    plt.ylim(0.1, 1100)
    plt.xlim(0, len(stepwise_obs_discr) - 1)
    plt.legend()
    plt.savefig(save_path + "/discr.png", bbox_inches='tight', format='png', dpi=300)
    plt.close()

    save_to_csv(data=[stepwise_obs_discr, cum_uncertainties],
                header_names=["obs_discr", "uncertainty"],
                path=save_path,
                file_name="rollout_specs.csv")
    

def vis_rollout(load_path, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    with np.load(load_path + "/infos.npz") as data:
        obs = data["obs_trajectory"]

    imgs = []
    for i in range(obs.shape[0]):
        img = set_state_and_render(obs[i], task="halfcheetah")
        
        img = np.array(img)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        # Save image
        cv2.imwrite(save_path + "/" + str(i) + ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        imgs.append(Image.fromarray(img, "RGB"))

    # Save corresponding gif
    gif_path = os.path.join(save_path + "/", 'trajectory.gif')
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], loop=0)


def visualize_trajectories(env, load_path, save_path):
    """
    Visualize most certain and most uncertain trajectories extracted during eval phase of Infoprop.
    """
    # -------- MOST CERTAIN TRAJECTORY -------- #
    most_c_load_path = load_path + "/most_certain_trajectory"
    most_c_save_path = save_path + "/most_certain_trajectory"
    vis_discr(
        env=env,
        load_path=most_c_load_path,
        save_path=most_c_save_path)
    vis_rollout(
        load_path=most_c_load_path,
        save_path=most_c_save_path + "/imgs")
    
    # -------- MOST UNCERTAIN TRAJECTORY -------- #
    most_unc_load_path = load_path + "/most_uncertain_trajectory"
    most_unc_save_path = save_path + "/most_uncertain_trajectory"
    if not os.path.isdir(most_unc_save_path):
        os.makedirs(most_unc_save_path)
    vis_discr(
        env=env,
        load_path=most_unc_load_path,
        save_path=most_unc_save_path)
    vis_rollout(
        load_path=most_unc_load_path,
        save_path=most_unc_save_path + "/imgs")
    
    
def analyze_attractor(rollout_length, load_path, save_path):
    rollout_length = args.rollout_length
    
    with np.load(load_path + "/sac_buffer_dump/replay_buffer.npz") as data:
        obs, next_obs = data["obs"], data["next_obs"]
    with np.load(load_path + "/most_certain_trajectory/infos.npz") as data:
        id_obs = data["obs_trajectory"] # next obs already included
    with np.load(load_path + "/most_uncertain_trajectory/infos.npz") as data:
        ood_obs = data["obs_trajectory"] # next obs already included
        
    seqs = [torch.from_numpy(id_obs).cpu(), torch.from_numpy(ood_obs).cpu()]
    seqs = [seq[:rollout_length] for seq in seqs]
    
    obs = torch.from_numpy(obs).cpu()
    next_obs = torch.from_numpy(next_obs).cpu()
    
    # Normalize.
    combined_obs = torch.cat((obs, next_obs), dim=0)
    max_per_dim = torch.max(combined_obs, dim=0, keepdim=True).values
    min_per_dim = torch.min(combined_obs, dim=0, keepdim=True).values
    
    for o in [*seqs, obs, next_obs]:
        o_max_per_dim = max_per_dim.broadcast_to(o.shape)
        o_min_per_dim = min_per_dim.broadcast_to(o.shape)
        o = (o - o_min_per_dim) /  (o_max_per_dim - o_min_per_dim)
    
    # Perfom PCA.
    proj = PCA(n_components=2)
    proj = proj.fit(combined_obs)
    
    # Transform.
    transform_seqs = [proj.transform(seq) for seq in seqs]
    transform_obs = torch.from_numpy(proj.transform(obs))
    transform_next_obs = torch.from_numpy(proj.transform(next_obs))
    transform_combined_obs = torch.cat((transform_obs, transform_next_obs), dim=0)
    
    # Normalize between -1 and 1 for better visualization.
    x_min, x_max = torch.min(transform_combined_obs[..., 0]), torch.max(transform_combined_obs[..., 0])
    y_min, y_max = torch.min(transform_combined_obs[..., 1]), torch.max(transform_combined_obs[..., 1])
    
    min_per_dim = torch.tensor([x_min, y_min])[None].broadcast_to(transform_obs.shape)
    max_per_dim = torch.tensor([x_max, y_max])[None].broadcast_to(transform_obs.shape)
    transform_obs = 2 * (transform_obs - min_per_dim) / (max_per_dim - min_per_dim) - 1
    transform_next_obs = 2 * (transform_next_obs - min_per_dim) / (max_per_dim - min_per_dim) - 1
    
    transform_seqs_to_plot = []
    for seq in transform_seqs:  
        min_per_dim = torch.tensor([x_min, y_min])[None].broadcast_to(seq.shape)
        max_per_dim = torch.tensor([x_max, y_max])[None].broadcast_to(seq.shape)
        transform_seqs_to_plot.append(2 * (torch.from_numpy(seq) - min_per_dim) / (max_per_dim - min_per_dim) - 1)
    
    # Plot all lines + (same) vector field in separate subplots
    plot_vector_field_with_lines(pts=(transform_obs, transform_next_obs),
                                 lines=transform_seqs_to_plot,
                                 lines_cmaps=["viridis", "plasma"],
                                 n_bins=20,
                                 min_bin_count=1,
                                 num_plots=2,
                                 path=save_path,
                                 figname="pca_plot")
        
        
def analyze(args):
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    print("[EVAL] Visualizing extracted trajectories...")
    visualize_trajectories(env=args.env, load_path=args.load_path, save_path=args.save_path)
    
    print("[EVAL] Performing attractor analysis...")
    analyze_attractor(rollout_length=args.rollout_length, load_path=args.load_path, save_path=args.save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah")
    parser.add_argument("--load_path", default="exp/halfcheetah_300k")
    parser.add_argument("--save_path", default="eval/halfcheetah_300k")
    parser.add_argument("--rollout_length", default=50, type=int)
    
    args = parser.parse_args()
    
    analyze(args)