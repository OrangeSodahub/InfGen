import math
import os
import torch
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy.typing as npt
import fnmatch
import io
import seaborn as sns
import matplotlib.axes as Axes
import matplotlib.transforms as mtransforms
from PIL import Image
from functools import wraps
from typing import Sequence, Union, Optional
from tqdm import tqdm
from typing import List, Literal
from argparse import ArgumentParser
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle
from matplotlib.collections import LineCollection
from torch_geometric.data import HeteroData, Dataset
from waymo_open_dataset.protos import scenario_pb2

from infgen.utils.func import CONSOLE
from infgen.modules.attr_tokenizer import Attr_Tokenizer
from infgen.datasets.preprocess import TokenProcessor, cal_polygon_contour, AGENT_TYPE
from infgen.datasets.scalable_dataset import WaymoTargetBuilder


__all__ = ['plot_occ_grid', 'plot_interact_edge', 'plot_map_edge', 'plot_insert_grid', 'plot_binary_map',
           'plot_map_token', 'plot_prob_seed', 'plot_scenario', 'get_heatmap', 'draw_heatmap', 'plot_val', 'plot_tokenize']


def safe_run(func):

    @wraps(func)
    def wrapper1(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return

    @wraps(func)
    def wrapper2(*args, **kwargs):
        return func(*args, **kwargs)

    if int(os.getenv('DEBUG', 0)):
        return wrapper2
    else:
        return wrapper1


@safe_run
def plot_occ_grid(scenario_id, occ, gt_occ=None, save_path='', mode='agent', prefix=''):

    def generate_box_edges(matrix, find_value=1):
        y, x = np.where(matrix == find_value)
        edges = []

        for xi, yi in zip(x, y):
            edges.append([(xi - 0.5, yi - 0.5), (xi + 0.5, yi - 0.5)])
            edges.append([(xi + 0.5, yi - 0.5), (xi + 0.5, yi + 0.5)])
            edges.append([(xi + 0.5, yi + 0.5), (xi - 0.5, yi + 0.5)])
            edges.append([(xi - 0.5, yi + 0.5), (xi - 0.5, yi - 0.5)])
    
        return edges

    os.makedirs(save_path, exist_ok=True)
    n = int(math.sqrt(occ.shape[-1]))

    plot_n = 3
    plot_t = 5

    occ_list = []
    for i in range(plot_n):
        for j in range(plot_t):
            occ_list.append(occ[i, j].reshape(n, n))

    occ_gt_list = []
    if gt_occ is not None:
        for i in range(plot_n):
            for j in range(plot_t):
                occ_gt_list.append(gt_occ[i, j].reshape(n, n))

    row_labels = [f'n={n}' for n in range(plot_n)]
    col_labels = [f't={t}' for t in range(plot_t)]

    fig, axes = plt.subplots(plot_n, plot_t, figsize=(9, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, ax in enumerate(axes.flat):
        # NOTE: do not set vmin and vamx!
        ax.imshow(occ_list[i], cmap='viridis', interpolation='nearest')
        ax.axis('off')

        if occ_gt_list:
            gt_edges = generate_box_edges(occ_gt_list[i])
            gts = LineCollection(gt_edges, colors='blue', linewidths=0.5)
            ax.add_collection(gts)
            insert_edges = generate_box_edges(occ_gt_list[i], find_value=-1)
            inserts = LineCollection(insert_edges, colors='red', linewidths=0.5)
            ax.add_collection(inserts)

        ax.add_patch(plt.Rectangle((-0.5, -0.5), occ_list[i].shape[1], occ_list[i].shape[0],
                                linewidth=2, edgecolor='black', facecolor='none'))

    for i, ax in enumerate(axes[:, 0]):
        ax.annotate(row_labels[i], xy=(-0.1, 0.5), xycoords="axes fraction",
                    fontsize=12, ha="right", va="center", rotation=0)

    for j, ax in enumerate(axes[0, :]):
        ax.annotate(col_labels[j], xy=(0.5, 1.05), xycoords="axes fraction",
                    fontsize=12, ha="center", va="bottom")

    plt.savefig(os.path.join(save_path, f'{prefix}{scenario_id}_occ_{mode}.png'), dpi=500, bbox_inches='tight')
    plt.close()


@safe_run
def plot_interact_edge(edge_index, scenario_ids, batch_sizes, num_seed, num_step, save_path='interact_edge_map',
                        **kwargs):

    num_batch = len(scenario_ids)
    batches = torch.cat([
        torch.arange(num_batch).repeat_interleave(repeats=batch_sizes, dim=0),
        torch.arange(num_batch).repeat_interleave(repeats=num_seed, dim=0),
    ], dim=0).repeat(num_step).numpy()

    num_agent = batch_sizes.sum() + num_seed * num_batch
    batch_sizes = torch.nn.functional.pad(batch_sizes, (1, 0), mode='constant', value=0)
    ptr = torch.cumsum(batch_sizes, dim=0)
    # assume difference scenarios and different timestep have the same number of seed agents
    ptr_seed = torch.tensor(np.array([0] + [num_seed] * num_batch), device=ptr.device)

    all_av_index = None
    if 'av_index' in kwargs:
        all_av_index = kwargs.pop('av_index').cpu() - ptr[:-1]

    is_bos = np.zeros((batch_sizes.sum(), num_step)).astype(np.bool_)
    if 'is_bos' in kwargs:
        is_bos = kwargs.pop('is_bos').cpu().numpy()

    src_index = torch.unique(edge_index[1])
    for idx, src in enumerate(tqdm(src_index)):

        src_batch = batches[src]

        src_row = src % num_agent
        if src_row // batch_sizes.sum() > 0:
            seed_row = src_row % batch_sizes.sum() - ptr_seed[src_batch]
            src_row = batch_sizes[src_batch + 1] + seed_row
        else:
            src_row = src_row - ptr[src_batch]

        src_col = src // (num_agent)
        src_mask = np.zeros((batch_sizes[src_batch + 1] + num_seed, num_step))
        src_mask[src_row, src_col] = 1

        tgt_mask = np.zeros((src_mask.shape[0], num_step))
        tgt_index = edge_index[0, edge_index[1] == src]
        for tgt in tgt_index:

            tgt_batch = batches[tgt]

            tgt_row = tgt % num_agent
            if tgt_row // batch_sizes.sum() > 0:
                seed_row = tgt_row % batch_sizes.sum() - ptr_seed[tgt_batch]
                tgt_row = batch_sizes[tgt_batch + 1] + seed_row
            else:
                tgt_row = tgt_row - ptr[tgt_batch]

            tgt_col = tgt // num_agent
            tgt_mask[tgt_row, tgt_col] = 1
            assert tgt_batch == src_batch

        selected_step = tgt_mask.sum(axis=0) > 0
        if selected_step.sum() > 1:
            print(f"\nidx={idx}", src.item(), src_row.item(), src_col.item())
            print(selected_step)
            print(edge_index[:, edge_index[1] == src].tolist())

        if all_av_index is not None:
            kwargs['av_index'] = int(all_av_index[src_batch])

        t = kwargs.get('t', src_col)
        n = kwargs.get('n', 0)
        is_bos_batch = is_bos[ptr[src_batch] : ptr[src_batch + 1]]
        plot_binary_map(src_mask, tgt_mask, save_path, suffix=f'_{scenario_ids[src_batch]}_{t:02d}_{n:02d}_{idx:04d}',
                        is_bos=is_bos_batch, **kwargs)


@safe_run
def plot_map_edge(edge_index, pos_a, data, save_path='map_edge_map'):

    map_points = data['map_point']['position'][:, :2].cpu().numpy()
    token_pos = data['pt_token']['position'][:, :2].cpu().numpy()
    token_heading = data['pt_token']['orientation'].cpu().numpy()
    num_pt = token_pos.shape[0]
    
    agent_index = torch.unique(edge_index[1])
    for i in tqdm(agent_index):
        xy = pos_a[i].cpu().numpy()
        pt_index = edge_index[0, edge_index[1] == i].cpu().numpy()
        pt_index = pt_index % num_pt

        plt.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)
        _, ax = plt.subplots()
        ax.set_axis_off()

        plot_map_token(ax, map_points, token_pos[pt_index], token_heading[pt_index], colors='blue')

        ax.scatter(xy[0], xy[1], s=0.5, c='red', edgecolors='none')

        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'map_{i}.png'), dpi=600, bbox_inches='tight')
        plt.close()


def get_heatmap(x, y, prob, s=3, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=prob, density=True)

    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


@safe_run
def draw_heatmap(vector, vector_prob, gt_idx):
    fig, ax = plt.subplots(figsize=(10, 10))
    vector_prob = vector_prob.cpu().numpy()

    for j in range(vector.shape[0]):
        if j in gt_idx:
            color = (0, 0, 1)
        else:
            grey_scale = max(0, 0.9 - vector_prob[j])
            color = (0.9, grey_scale, grey_scale)

        # if lane[j, k, -1] == 0: continue
        x0, y0, x1, y1, = vector[j, :4]
        ax.plot((x0, x1), (y0, y1), color=color, linewidth=2)

    return plt


@safe_run
def plot_insert_grid(scenario_id, prob, grid, ego_pos, map, save_path='', prefix='', inference=False, indices=None,
                     all_t_in_one=False):

    """
        prob: float array of shape (num_step, num_grid)
        grid: float array of shape (num_grid, 2)
    """

    os.makedirs(save_path, exist_ok=True)

    n = int(math.sqrt(prob.shape[1]))

    # grid = grid[:, np.newaxis] + ego_pos[np.newaxis, ...]
    for t in range(ego_pos.shape[0]):

        plt.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)
        _, ax = plt.subplots()

        # plot probability
        prob_t = prob[t].reshape(n, n)
        plt.imshow(prob_t, cmap='viridis', interpolation='nearest')

        if indices is not None:
            indice = indices[t]

            if isinstance(indice, (int, float, np.int_)):
                indice = [indice]

            for _indice in indice:
                if _indice == -1: continue

                row = _indice // n
                col = _indice % n

                rect = Rectangle((col - 0.5, row - 0.5), 1, 1, edgecolor='red', facecolor='none', lw=2)
                ax.add_patch(rect)

        ax.grid(False)
        ax.set_aspect('equal', adjustable='box')

        plt.title('Prob of Rel Position Grid')
        plt.savefig(os.path.join(save_path, f'{prefix}{scenario_id}_heat_map_{t}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        if all_t_in_one:
            break


@safe_run
def plot_insert_grid(scenario_id, prob, indices=None, save_path='', prefix='', inference=False):

    """
        prob: float array of shape (num_seed, num_step, num_grid)
        grid: float array of shape (num_grid, 2)
    """

    os.makedirs(save_path, exist_ok=True)

    n = int(math.sqrt(prob.shape[-1]))

    plot_n = 3
    plot_t = 5

    prob_list = []
    for i in range(plot_n):
        for j in range(plot_t):
            prob_list.append(prob[i, j].reshape(n, n))

    indice_list = []
    if indices is not None:
        for i in range(plot_n):
            for j in range(plot_t):
                indice_list.append(indices[i, j])

    row_labels = [f'n={n}' for n in range(plot_n)]
    col_labels = [f't={t}' for t in range(plot_t)]

    fig, axes = plt.subplots(plot_n, plot_t, figsize=(9, 6))
    fig.suptitle('Prob of Insert Position Grid')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, ax in enumerate(axes.flat):
        ax.imshow(prob_list[i], cmap='viridis', interpolation='nearest')
        ax.axis('off')

        if indice_list:
            row = indice_list[i] // n
            col = indice_list[i] % n
            rect = Rectangle((col - .5, row - .5), 1, 1, edgecolor='red', facecolor='none', lw=2)
            ax.add_patch(rect)

        ax.add_patch(plt.Rectangle((-0.5, -0.5), prob_list[i].shape[1], prob_list[i].shape[0],
                                linewidth=2, edgecolor='black', facecolor='none'))

    for i, ax in enumerate(axes[:, 0]):
        ax.annotate(row_labels[i], xy=(-0.1, 0.5), xycoords="axes fraction",
                    fontsize=12, ha="right", va="center", rotation=0)

    for j, ax in enumerate(axes[0, :]):
        ax.annotate(col_labels[j], xy=(0.5, 1.05), xycoords="axes fraction",
                    fontsize=12, ha="center", va="bottom")

    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(save_path, f'{prefix}{scenario_id}_insert_map.png'), dpi=500, bbox_inches='tight')
    plt.close()


@safe_run
def plot_binary_map(src_mask, tgt_mask, save_path='', suffix='', av_index=None, is_bos=None, **kwargs):

    from matplotlib.colors import ListedColormap
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    title = []
    if kwargs.get('t', None) is not None:
        t = kwargs['t']
        title.append(f't={t}')

    if kwargs.get('n', None) is not None:
        n = kwargs['n']
        title.append(f'n={n}')

    plt.title(' '.join(title))

    cmap = ListedColormap(['white', 'green'])
    axes[0].imshow(src_mask, cmap=cmap, interpolation='nearest')

    cmap = ListedColormap(['white', 'orange'])
    axes[1].imshow(tgt_mask, cmap=cmap, interpolation='nearest')

    if av_index is not None:
        rect = Rectangle((-0.5, av_index - 0.5), src_mask.shape[1], 1, edgecolor='red', facecolor='none', lw=2)
        axes[0].add_patch(rect)
        rect = Rectangle((-0.5, av_index - 0.5), tgt_mask.shape[1], 1, edgecolor='red', facecolor='none', lw=2)
        axes[1].add_patch(rect)

    if is_bos is not None:
        rows, cols = np.where(is_bos)
        for row, col in zip(rows, cols):
            rect = Rectangle((col - 0.5, row - 0.5), 1, 1, edgecolor='blue', facecolor='none', lw=1)
            axes[0].add_patch(rect)
            rect = Rectangle((col - 0.5, row - 0.5), 1, 1, edgecolor='blue', facecolor='none', lw=1)
            axes[1].add_patch(rect)

    for ax in axes:
        ax.set_xticks(range(src_mask.shape[1] + 1), minor=False)
        ax.set_yticks(range(src_mask.shape[0] + 1), minor=False)
        ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5)

    plt.savefig(os.path.join(save_path, f'map{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()


@safe_run
def plot_prob_seed(scenario_id, prob, save_path, prefix='', indices=None):

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.imshow(prob, cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.title('Seed Probability')

    if indices is not None:

        for col in range(indices.shape[1]):
            for row in indices[:, col]:

                if row == -1: continue

                rect = Rectangle((col - 0.5, row - 0.5), 1, 1, edgecolor='red', facecolor='none', lw=2)
                plt.gca().add_patch(rect)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{prefix}{scenario_id}_prob_seed.png'), dpi=300, bbox_inches='tight')
    plt.close()


@safe_run
def plot_raw():
    plt.figure(figsize=(30, 30))
    plt.rcParams['axes.facecolor']='white'

    data_path = 'data/waymo/scenario/training'
    os.makedirs("data/vis/raw/0/", exist_ok=True)
    file_list = os.listdir(data_path)

    for cnt_file, file in enumerate(file_list):
        file_path = os.path.join(data_path, file)
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for scenario_idx, data in enumerate(dataset):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytearray(data.numpy()))
            tqdm.write(f"scenario id: {scenario.scenario_id}")

            # draw maps
            for i in range(len(scenario.map_features)):

                # draw lanes
                if str(scenario.map_features[i].lane) != '':
                    line_x = [z.x for z in scenario.map_features[i].lane.polyline]
                    line_y = [z.y for z in scenario.map_features[i].lane.polyline]
                    plt.scatter(line_x, line_y, c='g', s=5)
                    plt.text(line_x[0], line_y[0], str(scenario.map_features[i].id), fontdict={'family': 'serif', 'size': 20, 'color': 'green'})

                # draw road_edge
                if str(scenario.map_features[i].road_edge) != '':
                    road_edge_x = [polyline.x for polyline in scenario.map_features[i].road_edge.polyline]
                    road_edge_y = [polyline.y for polyline in scenario.map_features[i].road_edge.polyline]
                    plt.scatter(road_edge_x, road_edge_y)
                    plt.text(road_edge_x[0], road_edge_y[0], scenario.map_features[i].road_edge.type, fontdict={'family': 'serif', 'size': 20, 'color': 'black'})
                    if scenario.map_features[i].road_edge.type == 2:
                        plt.scatter(road_edge_x, road_edge_y, c='k')
                    elif scenario.map_features[i].road_edge.type == 3:
                        plt.scatter(road_edge_x, road_edge_y, c='purple')
                        print(scenario.map_features[i].road_edge)
                    else:
                        plt.scatter(road_edge_x, road_edge_y, c='k')

                # draw road_line
                if str(scenario.map_features[i].road_line) != '':
                    road_line_x = [j.x for j in scenario.map_features[i].road_line.polyline]
                    road_line_y = [j.y for j in scenario.map_features[i].road_line.polyline]
                    if scenario.map_features[i].road_line.type == 7:
                        plt.plot(road_line_x, road_line_y, c='y')
                    elif scenario.map_features[i].road_line.type == 8:
                        plt.plot(road_line_x, road_line_y, c='y') 
                    elif scenario.map_features[i].road_line.type == 6:
                        plt.plot(road_line_x, road_line_y, c='y')
                    elif scenario.map_features[i].road_line.type == 1:
                        for i in range(int(len(road_line_x) / 7)):
                            plt.plot(road_line_x[i * 7 : 5 + i * 7], road_line_y[i * 7 : 5 + i * 7], color='w')
                    elif scenario.map_features[i].road_line.type == 2:
                        plt.plot(road_line_x, road_line_y, c='w')
                    else:
                        plt.plot(road_line_x, road_line_y, c='w')
            
            # draw tracks
            scenario_has_invalid_tracks = False
            for i in range(len(scenario.tracks)):
                traj_x = [center.center_x for center in scenario.tracks[i].states]
                traj_y = [center.center_y for center in scenario.tracks[i].states]
                head = [center.heading for center in scenario.tracks[i].states]
                valid = [center.valid for center in scenario.tracks[i].states]
                print(valid)
                if i == scenario.sdc_track_index:
                    plt.scatter(traj_x[0], traj_y[0], s=140, c='r', marker='s')
                    plt.scatter([x for x, v in zip(traj_x, valid) if v],
                                [y for y, v in zip(traj_y, valid) if v], s=14, c='r')
                    plt.scatter([x for x, v in zip(traj_x, valid) if not v],
                                [y for y, v in zip(traj_y, valid) if not v], s=14, c='m')
                else:
                    plt.scatter(traj_x[0], traj_y[0], s=140, c='k', marker='s')
                    plt.scatter([x for x, v in zip(traj_x, valid) if v],
                                [y for y, v in zip(traj_y, valid) if v], s=14, c='b')
                    plt.scatter([x for x, v in zip(traj_x, valid) if not v],
                                [y for y, v in zip(traj_y, valid) if not v], s=14, c='m')
                if valid.count(False) > 0:
                    scenario_has_invalid_tracks = True
            if scenario_has_invalid_tracks:
                plt.savefig(f"scenario_{scenario_idx}_{scenario.scenario_id}.png")
                plt.clf()
                breakpoint()
        break


colors = [
    ('#1f77b4', '#1a5a8a'),  # blue
    ('#2ca02c', '#217721'),  # green
    ('#ff7f0e', '#cc660b'),  # orange
    ('#9467bd', '#6f4a91'),  # purple
    ('#d62728', '#a31d1d'),  # red
    ('#000000', '#000000'),  # black
]

@safe_run
def plot_gif():
    data_path = "data/waymo_processed/training"
    os.makedirs("data/vis/processed/0/gif", exist_ok=True)
    file_list = os.listdir(data_path)

    for scenario_idx, file in tqdm(enumerate(file_list), leave=False, desc="Scenario"):

        fig, ax = plt.subplots()
        ax.set_axis_off()

        file_path = os.path.join(data_path, file)
        data = pickle.load(open(file_path, "rb"))
        scenario_id = data['scenario_id']

        save_path = os.path.join("data/vis/processed/0/gif",
                                 f"scenario_{scenario_idx}_{scenario_id}.gif")
        if os.path.exists(save_path):
            tqdm.write(f"Skipped {save_path}.")
            continue

        # draw maps
        ax.scatter(data['map_point']['position'][:, 0],
                   data['map_point']['position'][:, 1], s=0.2, c='black', edgecolors='none')

        # draw agents
        agent_data = data['agent']
        av_index = agent_data['av_index']
        position = agent_data['position'] # (num_agent, 91, 3)
        heading = agent_data['heading'] # (num_agent, 91)
        shape = agent_data['shape'] # (num_agent, 91, 3)
        category = agent_data['category'] # (num_agent,)
        valid_mask = (position[..., 0] != 0) & (position[..., 1] != 0) # (num_agent, 91)

        num_agent = valid_mask.shape[0]
        num_timestep = position.shape[1]
        is_av = np.arange(num_agent) == int(av_index)

        is_blue = valid_mask.sum(axis=1) == num_timestep
        is_green = ~valid_mask[:, 0] & valid_mask[:, -1]
        is_orange = valid_mask[:, 0] & ~valid_mask[:, -1]
        is_purple = (valid_mask.sum(axis=1) != num_timestep
                    ) & (~is_green) & (~is_orange)
        agent_colors = np.zeros((num_agent,))
        agent_colors[is_blue] = 1
        agent_colors[is_green] = 2
        agent_colors[is_orange] = 3
        agent_colors[is_purple] = 4
        agent_colors[is_av] = 5

        veh_mask = category == 1
        ped_mask = category == 2
        cyc_mask = category == 3
        shape[veh_mask, :, 1] = 1.8
        shape[veh_mask, :, 0] = 1.8
        shape[ped_mask, :, 1] = 0.5
        shape[ped_mask, :, 0] = 0.5
        shape[cyc_mask, :, 1] = 1.0
        shape[cyc_mask, :, 0] = 1.0

        fig_paths = []
        for tid in tqdm(range(num_timestep), leave=False, desc="Timestep"):
            current_valid_mask = valid_mask[:, tid]
            xs = position[current_valid_mask, tid, 0]
            ys = position[current_valid_mask, tid, 1]
            widths = shape[current_valid_mask, tid, 1]
            lengths = shape[current_valid_mask, tid, 0]
            angles = heading[current_valid_mask, tid]
            current_agent_colors = agent_colors[current_valid_mask]

            drawn_agents = []
            contours = cal_polygon_contour(xs, ys, angles, widths, lengths) # (num_agent, 4, 2)
            contours = np.concatenate([contours, contours[:, 0:1]], axis=1) # (num_agent, 5, 2)
            for x, y, width, length, angle, color_type in zip(
                xs, ys, widths, lengths, angles, current_agent_colors):
                agent = plt.Rectangle((x, y), width, length, angle=((angle + np.pi / 2) / np.pi * 360) % 360,
                                      linewidth=0.2,
                                      facecolor=colors[int(color_type) - 1][0],
                                      edgecolor=colors[int(color_type) - 1][1])
                ax.add_patch(agent)
                drawn_agents.append(agent)
            plt.gca().set_aspect('equal', adjustable='box')
            # for contour, color_type in zip(contours, agent_colors):
            #     drawn_agent = ax.plot(contour[:, 0], contour[:, 1])
            #     drawn_agents.append(drawn_agent)

            fig_path = os.path.join("data/vis/processed/0/",
                                    f"scenario_{scenario_idx}_{scenario_id}_{tid}.png")
            plt.savefig(fig_path, dpi=600)
            fig_paths.append(fig_path)

            for drawn_agent in drawn_agents:
                drawn_agent.remove()

        plt.close()

        # generate gif
        import imageio.v2 as imageio
        images = []
        for fig_path in tqdm(fig_paths, leave=False, desc="Generate gif ..."):
            images.append(imageio.imread(fig_path))
        imageio.mimsave(save_path, images, duration=0.1)


@safe_run
def plot_map_token(ax: Axes, map_points: npt.NDArray, token_pos: npt.NDArray, token_heading: npt.NDArray, colors: Union[str, npt.NDArray]=None):

    plot_map(ax, map_points)

    x, y = token_pos[:, 0], token_pos[:, 1]
    u = np.cos(token_heading)
    v = np.sin(token_heading)

    if colors is None:
        colors = np.random.rand(x.shape[0], 3)
    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.2, color=colors, width=0.005,
               headwidth=0.2, headlength=2)
    ax.scatter(x, y, color='blue', s=0.2, edgecolors='none')
    ax.axis("equal")


@safe_run
def plot_map(ax: Axes, map_points: npt.NDArray, color='black'):
    ax.scatter(map_points[:, 0], map_points[:, 1], s=0.5, c=color, edgecolors='none')

    xmin = np.min(map_points[:, 0])
    xmax = np.max(map_points[:, 0])
    ymin = np.min(map_points[:, 1])
    ymax = np.max(map_points[:, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


@safe_run
def plot_agent(ax: Axes, xy: Sequence[float], heading: float, type: str, state, is_av: bool=False,
               pl2seed_radius: float=25., attr_tokenizer: Attr_Tokenizer=None, enter_index: list=[], **kwargs):

    if type == 'veh':
        length = 4.3
        width = 1.8
        size = 1.0
    elif type == 'ped':
        length = 0.5
        width = 0.5
        size = 0.1
    elif type == 'cyc':
        length = 1.9
        width = 0.5
        size = 0.3
    else:
        raise ValueError(f"Unsupported agent type {type}")

    if kwargs.get('label', None) is not None:
        ax.text(
            xy[0] + 1.5, xy[1] + 1.5,
            kwargs['label'], fontsize=2, color="darkred", ha="center", va="center"
        )

    patch = FancyBboxPatch([-length / 2, -width / 2], length, width, linewidth=.2, **kwargs)
    transform = (
        mtransforms.Affine2D().rotate(heading).translate(xy[0], xy[1])
        + ax.transData
    )
    patch.set_transform(transform)

    kwargs['label'] = None
    angles = [0, 2 * np.pi / 3, np.pi, 4 * np.pi / 3]
    pts = np.stack([size * np.cos(angles), size * np.sin(angles)], axis=-1)
    center_patch = Polygon(pts, zorder=10., linewidth=.2, **kwargs)
    center_patch.set_transform(transform)

    ax.add_patch(patch)
    ax.add_patch(center_patch)

    if is_av:

        if attr_tokenizer is not None:

            circle_patch = Circle(
                (xy[0], xy[1]), pl2seed_radius, linewidth=0.5, edgecolor='gray', linestyle='--', facecolor='none'
            )
            ax.add_patch(circle_patch)

            grid = attr_tokenizer.get_grid(torch.tensor(np.array(xy)).float(),
                                           torch.tensor(np.array([heading])).float()).numpy()[0] # (num_grid, 2)
            ax.scatter(grid[:, 0], grid[:, 1], s=0.3, c='blue', edgecolors='none')
            ax.text(grid[0, 0], grid[0, 1], 'Front', fontsize=2, color='darkred', ha='center', va='center')
            ax.text(grid[-1, 0], grid[-1, 1], 'Back', fontsize=2, color='darkred', ha='center', va='center')

        if enter_index:
            for i in enter_index:
                ax.plot(grid[int(i), 0], grid[int(i), 1], marker='x', color='red', markersize=1)

    return patch, center_patch


@safe_run
def plot_all(map, xs, ys, angles, types, colors, is_avs, pl2seed_radius: float=25.,
             attr_tokenizer: Attr_Tokenizer=None, enter_index: list=[], labels: list=[], **kwargs):

    plt.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)
    _, ax = plt.subplots()
    ax.set_axis_off()

    plot_map(ax, map)

    if not labels:
        labels = [None] * xs.shape[0]

    for x, y, angle, type, color, label, is_av in zip(xs, ys, angles, types, colors, labels, is_avs):
        assert type in ('veh', 'ped', 'cyc'), f"Unsupported type {type}."
        plot_agent(ax, [x, y], angle.item(), type, None, is_av, facecolor=color, edgecolor='k', label=label,
                   pl2seed_radius=pl2seed_radius, attr_tokenizer=attr_tokenizer, enter_index=enter_index)

    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')

    # ! set plot limit if need
    if kwargs.get('limit_size', None):
        cx = float(xs[is_avs])
        cy = float(ys[is_avs])

        lx, ly = kwargs['limit_size']
        xmin, xmax = cx - lx, cx + lx
        ymin, ymax = cy - ly, cy + ly

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # ax.legend(loc='best', frameon=True)

    pil_image = None
    if kwargs.get('save_path', None):
        plt.savefig(kwargs['save_path'], dpi=600, bbox_inches="tight")

    else:
        # ！convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf).convert('RGB')

    plt.close()

    return pil_image


@safe_run
def plot_file(gt_folder: str,
              folder: Optional[str] = None,
              files: Optional[str] = None,
              save_gif: bool = True,
              batch_idx: Optional[int] = None,
              time_idx: Optional[List[int]] = None,
              limit_size: Optional[List[int]] = None,
              **kwargs,
    ) -> List[Image.Image]:

    from infgen.metrics.compute_metrics import _unbatch

    shift = 5

    if files is None:
        assert os.path.exists(folder), f'Path {folder} does not exist.'
        files = list(fnmatch.filter(os.listdir(folder), 'idx_*_rollouts.pkl'))
        CONSOLE.log(f'Found {len(files)} rollouts files from {folder}.')

    if folder is None:
        assert os.path.exists(files), f'Path {files} does not exist.'
        folder = os.path.dirname(files)
        files = [files]

    plotted_scenarios = []
    parent, folder_name = os.path.split(folder.rstrip(os.sep))
    if save_gif:
        save_path = os.path.join(parent, f'{folder_name}_plots')
        os.makedirs(save_path, exist_ok=True)
        plotted_scenarios = list(fnmatch.filter(os.listdir(save_path), '*.gif'))
        plotted_scenarios = list(map(lambda fn: fn.split('_')[0], plotted_scenarios))
    else:
        save_path = None

    file_outs = []
    for file in (pbar := tqdm(files, leave=False, desc='Plotting files ...')):
        pbar.set_postfix(file=file)

        with open(os.path.join(folder, file), 'rb') as f:
            preds = pickle.load(f)

        scenario_ids = preds['_scenario_id']
        agent_batch = preds['agent_batch']
        agent_id = _unbatch(preds['agent_id'], agent_batch)
        preds_traj = _unbatch(preds['pred_traj'], agent_batch)
        preds_head = _unbatch(preds['pred_head'], agent_batch)
        preds_type = _unbatch(preds['pred_type'], agent_batch)
        if 'pred_state' in preds:
            preds_state = _unbatch(preds['pred_state'], agent_batch)
        else:
            preds_state = tuple([torch.ones((*traj.shape[:2], traj.shape[2] // shift)) for traj in preds_traj])  # [n_agent, n_rollout, n_step2Hz]
        preds_valid = _unbatch(preds['pred_valid'], agent_batch)

        # ! fetch certain scenario
        if batch_idx is not None:
            scenario_ids = scenario_ids[batch_idx : batch_idx + 1]
            agent_id = (agent_id[batch_idx],)
            preds_traj = (preds_traj[batch_idx],)
            preds_head = (preds_head[batch_idx],)
            preds_type = (preds_type[batch_idx],)
            preds_state = (preds_state[batch_idx],)
            preds_valid = (preds_valid[batch_idx],)

        scenario_outs = []
        for i, scenario_id in enumerate(scenario_ids):
            n_agent, n_rollouts = preds_traj[0].shape[:2]

            rollout_outs = []
            for j in range(n_rollouts):  # 1
                pred = dict(scenario_id=[scenario_id],
                            pred_traj=preds_traj[i][:, j],
                            pred_head=preds_head[i][:, j],
                            pred_state=(
                                torch.cat([torch.zeros(n_agent, 1), preds_state[i][:, j].repeat_interleave(repeats=shift, dim=-1)],
                                          dim=1)
                            ),
                            pred_type=preds_type[i][:, j],
                    )

                # NOTE: hard code!!!
                if 'av_id' in preds:
                    av_index = agent_id[i][:, 0].tolist().index(preds['av_id'])
                else:
                    av_index = n_agent - 1

                # ! load logged data
                data_path = os.path.join(gt_folder, 'validation', f'{scenario_id}.pkl')
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)

                rollout_outs.append(
                    plot_val(data, pred,
                             av_index=av_index,
                             save_path=save_path,
                             save_gif=save_gif,
                             time_idx=time_idx,
                             limit_size=limit_size,
                             **kwargs
                    )
                )

            scenario_outs.append(rollout_outs)
        file_outs.append(scenario_outs)

    return file_outs


@safe_run
def plot_val(data: Union[dict, str], pred: dict, av_index: int, save_path: str, suffix: str='',
             pl2seed_radius: float=75., attr_tokenizer=None, **kwargs):

    if isinstance(data, str):
        assert data.endswith('.pkl'), f'Got invalid data path {data}.'
        assert os.path.exists(data), f'Path {data} does not exist.'
        with open(data, 'rb') as f:
            data = pickle.load(f)

    map_point = data['map_point']['position'].cpu().numpy()

    scenario_id = pred['scenario_id'][0]
    pred_traj = pred['pred_traj'].cpu().numpy() # (num_agent, num_future_step, 2)
    pred_type = list(map(lambda i: AGENT_TYPE[i], pred['pred_type'].tolist()))
    pred_state = pred['pred_state'].cpu().numpy()
    pred_head = pred['pred_head'].cpu().numpy()
    ids = np.arange(pred_traj.shape[0])

    if 'agent_labels' in pred:
        kwargs.update(agent_labels=pred['agent_labels'])

    return plot_scenario(scenario_id, map_point, pred_traj, pred_head, pred_state, pred_type,
                         av_index=av_index, ids=ids, save_path=save_path, suffix=suffix,
                         pl2seed_radius=pl2seed_radius, attr_tokenizer=attr_tokenizer, **kwargs)


@safe_run
def plot_scenario(scenario_id: str,
                  map_data: npt.NDArray,
                  traj: npt.NDArray,
                  heading: npt.NDArray,
                  state: npt.NDArray,
                  types: List[str],
                  av_index: int,
                  color_type: Literal['state', 'type', 'seed', 'insert']='seed',
                  state_type: List[str]=['invalid', 'valid', 'enter', 'exit'],
                  plot_enter: bool=False,
                  suffix: str='',
                  pl2seed_radius: float=25.,
                  attr_tokenizer: Attr_Tokenizer=None,
                  enter_index: List[list] = [],
                  save_gif: bool=True,
                  tokenized: bool=False,
                  agent_labels: List[List[Optional[str]]] = [],
                  **kwargs):

    num_historical_steps = 11
    shift = 5
    num_agent, num_timestep = traj.shape[:2]

    if tokenized:
        num_historical_steps = 2
        shift = 1

    if (
        'save_path' in kwargs
        and kwargs['save_path'] != ''
        and kwargs['save_path'] != None
    ):
        os.makedirs(kwargs['save_path'], exist_ok=True)
        save_id = int(max([0] + list(map(lambda fname: int(fname.split("_")[-1]),
                                        filter(lambda fname: fname.startswith(scenario_id)
                                               and os.path.isdir(os.path.join(kwargs['save_path'], fname)),
                                        os.listdir(kwargs['save_path'])))))) + 1
        os.makedirs(f"{kwargs['save_path']}/{scenario_id}_{str(save_id).zfill(3)}", exist_ok=True)

        if save_id > 1:
            try:
                import shutil
                shutil.rmtree(f"{kwargs['save_path']}/{scenario_id}_{str(save_id - 1).zfill(3)}")
            except:
                pass

    visible_mask = state != state_type.index('invalid')
    if not plot_enter:
        visible_mask &= (state != state_type.index('enter'))

    last_valid_step = visible_mask.shape[1] - 1 - torch.argmax(torch.Tensor(visible_mask).flip(dims=[1]).long(), dim=1)
    ids = None
    if 'ids' in kwargs:
        ids = kwargs['ids']
        last_valid_step = {int(ids[i]): int(last_valid_step[i]) for i in range(len(ids))}

    # agent colors
    agent_colors = np.zeros((num_agent, num_timestep, 3))

    agent_palette = sns.color_palette('husl', n_colors=7)
    state_colors = {state: np.array(agent_palette[i]) for i, state in enumerate(state_type)}
    seed_colors = {seed: np.array(agent_palette[i]) for i, seed in enumerate(['existing', 'entered', 'exited'])}

    if color_type == 'state':
        for t in range(state.shape[1]):
            agent_colors[state[:, t] == state_type.index('invalid'), t * shift : (t + 1) * shift] = state_colors['invalid']
            agent_colors[state[:, t] == state_type.index('valid'), t * shift : (t + 1) * shift] = state_colors['valid']
            agent_colors[state[:, t] == state_type.index('enter'), t * shift : (t + 1) * shift] = state_colors['enter']
            agent_colors[state[:, t] == state_type.index('exit'), t * shift : (t + 1) * shift] = state_colors['exit']

    if color_type == 'seed':
        agent_colors[:, :] = seed_colors['existing']
        is_exited = np.any(state[:, num_historical_steps - 1:] == state_type.index('exit'), axis=-1)
        is_entered = np.any(state[:, num_historical_steps - 1:] == state_type.index('enter'), axis=-1)
        is_entered[av_index + 1:] = True  # NOTE: hard code, need improvment
        agent_colors[is_exited, :] = seed_colors['exited']
        agent_colors[is_entered, :] = seed_colors['entered']

    if color_type == 'insert':
        agent_colors[:, :] = seed_colors['exited']
        agent_colors[av_index + 1:] = seed_colors['existing']

    agent_colors[av_index, :] = np.array(agent_palette[-1])
    is_av = np.zeros_like(state[:, 0]).astype(np.bool_)
    is_av[av_index] = True

    # ! get timesteps to plot
    timesteps = list(range(num_timestep))
    if kwargs.get('time_idx', None) is not None:
        time_idx = kwargs['time_idx']
        assert set(time_idx).issubset(set(timesteps)), f'Got invalid time_idx: {time_idx=} v.s. {timesteps=}'
        timesteps = sorted(time_idx)

    # ! get plot limits
    limit_size = kwargs.get('limit_size', None)
    if limit_size is not None:
        assert len(limit_size) == 2, f'Got invalid `limit_size`: {limit_size=}'

    # ! plot all
    pil_images = []
    fig_paths = []
    for tid in tqdm(timesteps, leave=False, desc="Plot ..."):
        mask_t = visible_mask[:, tid]
        xs = traj[mask_t, tid, 0]
        ys = traj[mask_t, tid, 1]
        angles = heading[mask_t, tid]
        colors = agent_colors[mask_t, tid]
        types_t = [types[i] for i, mask in enumerate(mask_t) if mask]
        if ids is not None:
            ids_t = ids[mask_t]
        is_av_t = is_av[mask_t]
        enter_index_t = enter_index[tid] if enter_index else None
        labels = []
        if agent_labels:
            labels = [agent_labels[i][tid // shift] for i in range(len(agent_labels)) if mask_t[i]]

        fig_path = None
        if kwargs.get('save_path', None) is not None:
            save_path = kwargs['save_path']
            fig_path = os.path.join(f"{save_path}/{scenario_id}_{str(save_id).zfill(3)}", f"{tid}.png")
            fig_paths.append(fig_path)

        pil_images.append(
            plot_all(map_data, xs, ys, angles, types_t,
                     colors=colors,
                     save_path=fig_path,
                     is_avs=is_av_t,
                     pl2seed_radius=pl2seed_radius,
                     attr_tokenizer=attr_tokenizer,
                     enter_index=enter_index_t,
                     labels=labels,
                     limit_size=limit_size,
            )
        )

    # generate gif
    if fig_paths and save_gif:
        os.makedirs(os.path.join(save_path, 'gifs'), exist_ok=True)
        images = []
        gif_path = f"{save_path}/gifs/{scenario_id}_{str(save_id).zfill(3)}.gif"
        for fig_path in tqdm(fig_paths, leave=False, desc="Generate gif ..."):
            images.append(Image.open(fig_path))
        try:
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
            tqdm.write(f"Saved gif at {gif_path}")
            try:
                import shutil
                shutil.rmtree(f"{save_path}/{scenario_id}_{str(save_id).zfill(3)}")
                os.remove(f"{save_path}/gifs/{scenario_id}_{str(save_id - 1).zfill(3)}.gif")
            except:
                pass
        except Exception as e:
            tqdm.write(f"{e}! Failed to save gif at {gif_path}")

    return pil_images


def match_token_map(data):

    # init map token
    argmin_sample_len = 3
    map_token_traj_path = 'tokens/map_traj_token5.pkl'

    map_token_traj = pickle.load(open(map_token_traj_path, 'rb'))
    map_token = {'traj_src': map_token_traj['traj_src'], }
    traj_end_theta = np.arctan2(map_token['traj_src'][:, -1, 1] - map_token['traj_src'][:, -2, 1],
                                map_token['traj_src'][:, -1, 0] - map_token['traj_src'][:, -2, 0])
    indices = torch.linspace(0, map_token['traj_src'].shape[1]-1, steps=argmin_sample_len).long()
    map_token['sample_pt'] = torch.from_numpy(map_token['traj_src'][:, indices]).to(torch.float)
    map_token['traj_end_theta'] = torch.from_numpy(traj_end_theta).to(torch.float)
    map_token['traj_src'] = torch.from_numpy(map_token['traj_src']).to(torch.float)

    traj_pos = data['map_save']['traj_pos'].to(torch.float)
    traj_theta = data['map_save']['traj_theta'].to(torch.float)
    pl_idx_list = data['map_save']['pl_idx_list']
    token_sample_pt = map_token['sample_pt'].to(traj_pos.device)
    token_src = map_token['traj_src'].to(traj_pos.device)
    max_traj_len = map_token['traj_src'].shape[1]
    pl_num = traj_pos.shape[0]

    pt_token_pos = traj_pos[:, 0, :].clone()
    pt_token_orientation = traj_theta.clone()
    cos, sin = traj_theta.cos(), traj_theta.sin()
    rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
    rot_mat[..., 0, 0] = cos
    rot_mat[..., 0, 1] = -sin
    rot_mat[..., 1, 0] = sin
    rot_mat[..., 1, 1] = cos
    traj_pos_local = torch.bmm((traj_pos - traj_pos[:, 0:1]), rot_mat.view(-1, 2, 2))
    distance = torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1)) ** 2, dim=(-2, -1))
    pt_token_id = torch.argmin(distance, dim=1)

    noise = False
    if noise:
        topk_indices = torch.argsort(torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1)) ** 2, dim=(-2, -1)), dim=1)[:, :8]
        sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
        pt_token_id = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)

    # cos, sin = traj_theta.cos(), traj_theta.sin()
    # rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
    # rot_mat[..., 0, 0] = cos
    # rot_mat[..., 0, 1] = sin
    # rot_mat[..., 1, 0] = -sin
    # rot_mat[..., 1, 1] = cos
    # token_src_world = torch.bmm(token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
    #                             rot_mat.view(-1, 2, 2)).reshape(pl_num, token_src.shape[0], max_traj_len, 2) + traj_pos[:, None, [0], :]
    # token_src_world_select = token_src_world.view(-1, 1024, 11, 2)[torch.arange(pt_token_id.view(-1).shape[0]), pt_token_id.view(-1)].view(pl_num, max_traj_len, 2)

    pl_idx_full = pl_idx_list.clone()
    token2pl = torch.stack([torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()])
    count_nums = []
    for pl in pl_idx_full.unique():
        pt = token2pl[0, token2pl[1, :] == pl]
        left_side = (data['pt_token']['side'][pt] == 0).sum()
        right_side = (data['pt_token']['side'][pt] == 1).sum()
        center_side = (data['pt_token']['side'][pt] == 2).sum()
        count_nums.append(torch.Tensor([left_side, right_side, center_side]))
    count_nums = torch.stack(count_nums, dim=0)
    num_polyline = int(count_nums.max().item())
    traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, num_polyline), dtype=bool)
    idx_matrix = torch.arange(traj_mask.size(2)).unsqueeze(0).unsqueeze(0)
    idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)
    counts_num_expanded = count_nums.unsqueeze(-1)
    mask_update = idx_matrix < counts_num_expanded
    traj_mask[mask_update] = True

    data['pt_token']['traj_mask'] = traj_mask
    data['pt_token']['position'] = torch.cat([pt_token_pos, torch.zeros((data['pt_token']['num_nodes'], 1),
                                                                        device=traj_pos.device, dtype=torch.float)], dim=-1)
    data['pt_token']['orientation'] = pt_token_orientation
    data['pt_token']['height'] = data['pt_token']['position'][:, -1]
    data[('pt_token', 'to', 'map_polygon')] = {}
    data[('pt_token', 'to', 'map_polygon')]['edge_index'] = token2pl  # (2, num_points)
    data['pt_token']['token_idx'] = pt_token_id
    return data


@safe_run
def plot_tokenize(data, save_path: str):

    shift = 5
    token_size = 2048
    pl2seed_radius = 75

    # transformation
    transform = WaymoTargetBuilder(num_historical_steps=11,
                                   num_future_steps=80,
                                   max_num=32,
                                   training=False)

    grid_range = 150.
    grid_interval = 3.
    angle_interval = 3.
    attr_tokenizer = Attr_Tokenizer(grid_range=grid_range,
                                    grid_interval=grid_interval,
                                    radius=pl2seed_radius,
                                    angle_interval=angle_interval)

    # tokenization
    token_processor = TokenProcessor(token_size,
                                     training=False,
                                     predict_motion=True,
                                     predict_state=True,
                                     predict_map=True,
                                     state_token={'invalid': 0, 'valid': 1, 'enter': 2, 'exit': 3},
                                     pl2seed_radius=pl2seed_radius)
    CONSOLE.log(f"Loaded token processor with token_size: {token_size}")

    # preprocess
    data: HeteroData = transform(data)
    tokenized_data = token_processor(data)
    CONSOLE.log(f"Keys in tokenized data:\n{tokenized_data.keys()}")

    # plot
    agent_data = tokenized_data['agent']
    map_data = tokenized_data['map_point']
    # CONSOLE.log(f"Keys in agent data:\n{agent_data.keys()}")

    av_index = agent_data['av_index']
    raw_traj = agent_data['position'][..., :2].contiguous()  # [n_agent, n_step, 2]
    raw_heading = agent_data['heading']  # [n_agent, n_step]

    traj = agent_data['traj_pos'][..., :2].contiguous()  # [n_agent, n_step, 6, 2]
    traj = traj[:, :, 1:, :].flatten(1, 2)
    traj = torch.cat([raw_traj[:, :1], traj], dim=1)
    heading = agent_data['traj_heading']  # [n_agent, n_step, 6]
    heading = heading[:, :, 1:].flatten(1, 2)
    heading = torch.cat([raw_heading[:, :1], heading], dim=1)

    agent_state = agent_data['state_idx'].repeat_interleave(repeats=shift, dim=-1)
    agent_state = torch.cat([torch.zeros_like(agent_state[:, :1]), agent_state], dim=1)
    agent_type = agent_data['type']
    ids = np.arange(raw_traj.shape[0])

    return plot_scenario(
                scenario_id=tokenized_data['scenario_id'],
                map_data=tokenized_data['map_point']['position'].numpy(),
                traj=raw_traj.numpy(),
                heading=raw_heading.numpy(),
                state=agent_state.numpy(),
                types=list(map(lambda i: AGENT_TYPE[i], agent_type.tolist())),
                av_index=av_index,
                ids=ids,
                save_path=save_path,
                pl2seed_radius=pl2seed_radius,
                attr_tokenizer=attr_tokenizer,
                color_type='state',
        )


def get_metainfos(folder: str):

    import pandas as pd

    assert os.path.exists(folder), f'Path {folder} does not exist.'
    files = list(fnmatch.filter(os.listdir(folder), 'idx_*_rollouts.pkl'))
    CONSOLE.log(f'Found {len(files)} rollouts files from {folder}.')

    metainfos_path = f'{os.path.normpath(folder)}_metainfos.parquet'
    csv_path = f'{os.path.normpath(folder)}_metainfos.csv'

    if not os.path.exists(metainfos_path):

        data = []
        for file in tqdm(files):
            pkl_data = pickle.load(open(os.path.join(folder, file), 'rb'))
            data.extend((file, scenario_id, index) for index, scenario_id in enumerate(pkl_data['_scenario_id']))

        df = pd.DataFrame(data, columns=('rollout_file', 'scenario_id', 'index'))
        df.to_parquet(metainfos_path)
        df.to_csv(csv_path)
        CONSOLE.log(f'Successfully saved to {metainfos_path}.')

    else:
        CONSOLE.log(f'File {metainfos_path} already exists!')
        return


def plot_comparison(methods: List[str], rollouts_paths: List[str], gt_folders: List[str],
                    save_path: str, scenario_ids: Optional[List[str]] = None):
    import pandas as pd
    from collections import defaultdict

    # ! hyperparameter
    fps = 10

    plot_time = [1, 6, 12, 18, 24, 30]
    # plot_time = [1, 5, 10, 15, 20, 25]
    time_idx = [int(time * fps) for time in plot_time]

    limit_size = [75, 60]  # [width, height]

    # ! load metainfos
    metainfos = defaultdict(dict)
    for method, rollout_path in zip(methods, rollouts_paths):
        meta_info_path = f'{os.path.normpath(rollout_path)}_metainfos.parquet'
        metainfos[method]['df'] = pd.read_parquet(meta_info_path)
        CONSOLE.log(f'Loaded {method=} with {len(metainfos[method]["df"]["scenario_id"])=}.')
    common_scenarios = set(metainfos['ours']['df']['scenario_id'])
    for method, meta_info in metainfos.items():
        if method == 'ours':
            continue
        common_scenarios &= set(meta_info['df']['scenario_id'])
    for method, meta_info in metainfos.items():
        df = metainfos[method]['df']
        metainfos[method]['df'] = df[df['scenario_id'].isin(common_scenarios)]
    CONSOLE.log(f'Filter and get {len(common_scenarios)=}.')

    # ! load data and plot
    if scenario_ids is None:
        scenario_ids = metainfos['ours']['df']['scenario_id'].tolist()
    CONSOLE.log(f'Plotting {len(scenario_ids)=} ...')

    for scenario_id in (pbar := tqdm(scenario_ids)):
        pbar.set_postfix(scenario_id=scenario_id)

        figures = dict()
        for method, rollout_path, gt_folder in zip(methods, rollouts_paths, gt_folders):
            df = metainfos[method]['df']
            _df = df.loc[df['scenario_id'] == scenario_id]
            batch_idx = int(_df['index'].tolist()[0])
            rollout_file = _df['rollout_file'].tolist()[0]
            figures[method] = plot_file(
                                gt_folder=gt_folder,
                                files=os.path.join(rollout_path, rollout_file),
                                save_gif=False,
                                batch_idx=batch_idx,
                                time_idx=time_idx,
                                limit_size=limit_size,
                                color_type='insert',
                            )[0][0][0]

        # ! plot figures
        border = 5
        padding_x = 20
        padding_y = 50

        img_width, img_height = figures['ours'][0].size
        img_width = img_width + 2 * border
        img_height = img_height + 2 * border
        n_col = len(time_idx)
        n_row = len(methods)

        W = n_col * img_width + (n_col - 1) * padding_x
        H = n_row * img_height + (n_row - 1) * padding_y

        canvas = Image.new('RGB', (W, H), 'white')
        for i_row, (method, method_figures) in enumerate(figures.items()):
            for i_col, method_figure in enumerate(method_figures):
                x = i_col * (img_width + padding_x)
                y = i_row * (img_height + padding_y)

                padded_figure = Image.new('RGB', (img_width, img_height), 'black')
                padded_figure.paste(method_figure, (border, border))

                canvas.paste(padded_figure, (x, y))

        canvas.save(
            os.path.join(save_path, f'{scenario_id}.png')
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/waymo_processed')
    parser.add_argument('--tfrecord_dir', type=str, default='validation_tfrecords_splitted')
    # plot tokenized data
    parser.add_argument('--save_folder', type=str, default='plot_gt')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--scenario_id', type=str, default=None)
    parser.add_argument('--plot_tokenize', action='store_true')
    # plot generated rollouts
    parser.add_argument('--plot_file', action='store_true')
    parser.add_argument('--folder_path', type=str, default=None)
    parser.add_argument('--file_path', type=str, default=None)
    # metainfos
    parser.add_argument('--get_metainfos', action='store_true')
    # plot comparison
    parser.add_argument('--plot_comparison', action='store_true')
    parser.add_argument('--comparison_folder', type=str, default='comparisons')
    args = parser.parse_args()

    if args.plot_tokenize:

        scenario_id = "74ad7b76d5906d39"
        data_path = os.path.join(args.data_path, args.split, f"{scenario_id}.pkl")
        data = pickle.load(open(data_path, "rb"))
        data['tfrecord_path'] = os.path.join(args.tfrecord_dir, f'{scenario_id}.tfrecords')
        CONSOLE.log(f"Loaded scenario {scenario_id}")

        save_path = os.path.join(args.data_path, args.save_folder, args.split)
        os.makedirs(save_path, exist_ok=True)

        plot_tokenize(data, save_path)

    if args.plot_file:

        plot_file(args.data_path, folder=args.folder_path, files=args.file_path)

    if args.get_metainfos:

        assert args.folder_path is not None, f'`folder_path` should not be None!'
        get_metainfos(args.folder_path)

    if args.plot_comparison:

        methods = ['ours', 'infgen']
        gt_folders = [
        ]
        rollouts_paths = [
        ]
        save_path = f'outputs/scalable_infgen_long/{args.comparison_folder}/'
        os.makedirs(save_path, exist_ok=True)

        scenario_ids = []
        plot_comparison(methods, rollouts_paths, gt_folders,
                        save_path=save_path,
                        scenario_ids=scenario_ids)
