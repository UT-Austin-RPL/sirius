import os
import numpy as np

from robomimic.scripts.vis.vis_utils import get_argparser, playback_dataset
from robomimic.scripts.vis.image_utils import apply_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_intv_and_preintv_inds(ep_info):
    if args.model == 'Q':
        vals = ep_info['q_vals']
    elif args.model == 'V':
        vals = ep_info['v_vals']
    else:
        raise ValueError

    ac_mods = ep_info["action_modes"]
    intv_inds = np.reshape(np.argwhere(ac_mods == 1), -1)

    preintv_inds = []
    intv_start_inds = [i for i in intv_inds if i > 0 and ac_mods[-1] != 1]
    for i_start in intv_start_inds:
        for j in range(i_start-1, 0, -1):
            if j in intv_inds or vals[j] > args.th:
                break

            preintv_inds.append(j)

    return intv_inds, preintv_inds


def plot_helper(ep_num, ep_info):
    fig, ax1 = plt.subplots()

    if args.model == 'Q':
        y_vals = ep_info['q_vals']
        y_label = 'Q'
    elif args.model == 'V':
        y_vals = ep_info['v_vals']
        y_label = 'V'
    else:
        raise ValueError

    color = 'tab:blue'
    ax1.set_xlabel('Timestep')

    ax1.set_ylabel(y_label)
    ax1.plot(y_vals, color = color)
    ax1.tick_params(axis ='y')

    ax1.axhline(y = 0.0, color = 'black')

    ax1.set_ylim(-1.2, 0.2)

    intv_inds, preintv_inds = get_intv_and_preintv_inds(ep_info)
    for i in intv_inds:
        ax1.axvline(x=i, color='green', linewidth=5, alpha=0.10)

    for i in preintv_inds:
        ax1.axvline(x=i, color='red', linewidth=5, alpha=0.10)

    plt.savefig(os.path.join(
        args.vis_path,
        'plot_{}.png'.format(ep_num)
    ))
    plt.close()


def video_helper(ep_num, ep_info):
    intv_inds, preintv_inds = get_intv_and_preintv_inds(ep_info)

    if len(intv_inds) == 0:
        return []

    video_frames = ep_info['video_frames']
    for (i, img) in video_frames:
        if i in intv_inds:
            img[::] = apply_filter(img, color=(0, 255, 0))

        if i in preintv_inds:
            img[::] = apply_filter(img, color=(255, 0, 0))

    return video_frames


if __name__ == "__main__":
    parser = get_argparser()

    parser.add_argument(
        "--th",
        type=float,
        default=-0.35,
        help="threshold for pre-intervention",
    )

    parser.add_argument(
        "--model",
        type=str,
        default='Q',
        choices=['Q', 'V'],
        help="Model to use for determining pre-intv",
    )

    args = parser.parse_args()
    playback_dataset(args, plot_helper=plot_helper, video_helper=video_helper)
