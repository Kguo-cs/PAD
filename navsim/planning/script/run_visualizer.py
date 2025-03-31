from typing import Dict
from pathlib import Path
import logging
import traceback
import pickle
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from navsim.common.dataclasses import SensorConfig
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory, SceneFilter
from navsim.common.dataloader import SceneLoader
import torch
from torch.utils.data import default_collate
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"

import matplotlib.pyplot as plt
import numpy as np
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
import lzma
import pickle
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from matplotlib.patches import Polygon
import cv2
import matplotlib.cm as cm

from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.plots import plot_bev_with_agent,configure_bev_ax,configure_ax
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax,add_oriented_box_to_bev_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG,ELLIS_5
from matplotlib.gridspec import GridSpec

from navsim.visualization.camera import _transform_points_to_image,add_lidar_to_camera_ax,_transform_pcs_to_images
from nuplan.common.actor_state.car_footprint import CarFootprint
from navsim.visualization.config import BEV_PLOT_CONFIG, MAP_LAYER_CONFIG, AGENT_CONFIG, LIDAR_CONFIG
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.transform import translate_longitudinally

proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)


def plot_front(proposals,initial_proposals,human_poses,poses,camera, ax,pdm_score ):
    all_traj=np.concatenate([proposals[..., :2], initial_proposals[..., :2],human_poses[None],poses[None,:,:2]],axis=0)

    all_traj=np.concatenate([all_traj[..., :2], np.zeros_like(all_traj)[..., :1]], -1)
    lidar_pc=all_traj.reshape(-1,3).T
    image = camera.image.copy()

    image_height, image_width = image.shape[:2]

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )

    pc_in_cam=pc_in_cam.reshape(-1,8,2)
    pc_in_fov_mask=pc_in_fov_mask.reshape(-1,8)

    config=TRAJECTORY_CONFIG["human"]

    ax.plot(
        pc_in_cam[-2,:, 0],
        pc_in_cam[-2,:, 1],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
        label="Human Trajectory"
    )


    config=TRAJECTORY_CONFIG["agent"]#red

    ax.plot(
        pc_in_cam[-1,:, 0],
        pc_in_cam[-1,:, 1],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
        label="Planning"
    )

    for i in range(64):
        points=pc_in_cam[i]#[pc_in_fov_mask[i],None].reshape(-1,2)
        color = cm.Reds(pdm_score[i])  # This returns an RGBA color
        init_points=pc_in_cam[i+64]

        if i==0:
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=1,
                linestyle=config["line_style"],
                marker='.',
                markersize=2,
                zorder=2,
                label="Proposal"
            )
            ax.plot(
                init_points[:, 0],
                init_points[:, 1],
                color="#edc948",
                linewidth=1,
                linestyle=config["line_style"],
                marker='.',
                markersize=2,
                zorder=2,
                label="Initial Proposal"
            )
        else:
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                alpha=1,
                linewidth=1,
                linestyle=config["line_style"],
                marker='.',
                markersize=2,
                # markeredgecolor=config["marker_edge_color"],
                zorder=2
            )
            ax.plot(
                init_points[:, 0],
                init_points[:, 1],
                color='#edc948',
                linewidth=1,
                linestyle=config["line_style"],
                marker='.',
                markersize=2,
                zorder=2
            )
    ax.axis('off')
    ax.imshow(image)

def plot_bev(scene,frame_idx):

    rel_pose=scene.rel_pose
    frame = scene.frames[frame_idx]

    fig, ax = plt.subplots(figsize=CAMERAS_PLOT_CONFIG["figure_size"])
    if frame_idx>3:
        pos=rel_pose[frame_idx-4]

        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(pos[0], pos[1], pos[2]),
            vehicle_parameters=get_pacifica_parameters(),
        )

        config= {
                "fill_color":  "#B0E685",
                "fill_color_alpha": 0.5,
                "line_color": "black",
                "line_color_alpha":1.0,
                "line_width": 1.0,
                "line_style": "--",
                "zorder": 4,
            }
        
        box=car_footprint.oriented_box
        add_heading=True

        box_corners = box.all_corners()
        corners = [[corner.x, corner.y] for corner in box_corners]
        corners = np.asarray(corners + [corners[0]])

        corners=corners[:,::-1]

        polygon = Polygon(
            corners, 
            closed=True, 
            facecolor=config["fill_color"], 
            edgecolor=config["line_color"], 
            linewidth=config["line_width"], 
            linestyle=config["line_style"], 
            alpha=config["fill_color_alpha"], 
            zorder=config["zorder"],
            label="Planning Poses"
        )

        # Add the polygon to the axes
        ax.add_patch(polygon)

        if add_heading:
            future = translate_longitudinally(box.center, distance=box.length / 2 + 1)
            line = np.array([[box.center.x, box.center.y], [future.x, future.y]])
            ax.plot(
                line[:, 1],
                line[:, 0],
                color=config["line_color"],
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                zorder=config["zorder"],
            )
        ax.legend(loc=1)


    add_configured_bev_on_ax(ax, scene.map_api, frame)
    ax.set_aspect("equal")
    ax.set_xlim(-36, 36)
    ax.set_ylim(-8, 64)
    ax.invert_xaxis()
    fig.suptitle(str(frame_idx*0.5-1.5)+' s', fontsize=16)
    configure_ax(ax)
    plt.tight_layout()

    return fig,ax


def run_test_evaluation(
    agent: AbstractAgent, scene_filter: SceneFilter, data_path: Path, sensor_blobs_path: Path
) -> Dict[str, Trajectory]:
    """
    Function to create the output file for evaluation of an agent on the testserver
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param sensor_blobs_path: pathlib path to sensor blobs
    :param save_path: pathlib path to folder where scores are stored as .csv
    """
    if agent.requires_scene:
        raise ValueError(
            """
            In evaluation, no access to the annotated scene is provided, but only to the AgentInput.
            Thus, agent.requires_scene has to be False for the agent that is to be evaluated.
            """
        )
    logger.info("Building Agent Input Loader")
    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs_path,
        sensor_config=SensorConfig.build_all_sensors(include=[3])
    )
    agent.initialize()

    output: Dict[str, Trajectory] = {}


    agent.eval()
    agent.cuda()

    for token in tqdm(input_loader, desc="Running evaluation"):
        # try:
        # if token not in ["51e697d3f5255ac3","83212ddd15375812", "fc6dc98b89a95817", "dabc9043dbb9560b", "ce516bdfc6e45d5b"]:
        #     continue
        # print(token)
        # from navsim.visualization.plots import plot_cameras_frame_with_annotations

        scene=input_loader.get_scene_from_token(token)


        # continue

        agent_input = input_loader.get_agent_input_from_token(token)

        #print(agent_input.ego_statuses[-1])

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v for k, v in features.items()}

        feature_list=[features]
        
        features=default_collate(feature_list)

        features={key:value.cuda() for key,value in features.items()}
        # forward pass
        with torch.no_grad():
            predictions = agent.forward(features)

        trajectory=predictions["trajectory"].cpu().numpy()[0]
        human_trajectory = scene.get_future_trajectory(8).poses
        
        rel_traj=trajectory-human_trajectory

        theta=human_trajectory[:,2]

        c, s = np.cos(theta), np.sin(theta)
        mat = np.array([[c, -s],
                        [s, c]])
        rel_traj[:,:2]=np.einsum('ti,ijt->tj',rel_traj[:,:2],mat )

        from navsim.visualization.plots import frame_plot_to_gif

        #plot_bev(scene,rel_traj)
        frame_indices = [idx for idx in range(12)]  # all frames in scene
        file_name = f"./exp/navsim_test/{token}.gif"
        scene.rel_pose=rel_traj
        images = frame_plot_to_gif(file_name, plot_bev, scene, frame_indices)

        continue

        poses = trajectory[:,:2]

        proposals = predictions["proposals"].cpu().numpy()[0]
        pdm_score = predictions["pdm_score"].cpu().numpy()[0]
        initial_proposals=predictions["proposal_list"][0].cpu().numpy()[0]
        human_trajectory = scene.get_future_trajectory(8)

        human_poses = human_trajectory[:, :2]

        pred_area = torch.sigmoid(predictions["pred_area_logit"]).cpu().numpy()[0].reshape(-1,40,3)

        on_road=1-pred_area[:,::5,-2]
        on_route=1-pred_area[:,::5,-1]

        pred_agents_states = predictions["pred_agents_states"][0]

        pred_agents_label=torch.sigmoid(pred_agents_states[..., -41:]).cpu().numpy()
        pred_agents_corners=pred_agents_states[..., :-41].cpu().numpy()

        cameras=agent_input.cameras[-1]
        frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame

        frame_idx = scene.scene_metadata.num_history_frames - 1
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0, wspace=0,height_ratios=[0.36,0.64])

        # Create axes

        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0])

        ax01 = fig.add_subplot(gs[0, 1])
        ax11 = fig.add_subplot(gs[1, 1])

        ax02 = fig.add_subplot(gs[0, 2])
        ax12 = fig.add_subplot(gs[1, 2])

        points=proposals[...,:2].reshape(-1,2)

        on_road=on_road.reshape(-1)
        on_route=on_route.reshape(-1)

        for i in range(len(on_road)):

            color = cm.Reds(on_road[i])  # This returns an RGBA color

            if i ==0:
                ax10.scatter(
                    points[i, 1],
                    points[i, 0],
                    color=color,
                    marker='.',
                    s=2,
                    zorder=2,
                    label='On-road Prediction'
                )
            else:
                ax10.scatter(
                    points[i, 1],
                    points[i, 0],
                    color=color,
                    marker='.',
                    s=2,
                    zorder=2
                )

            color = cm.Reds(on_route[i])  # This returns an RGBA color

            if i ==0:
                ax12.scatter(
                    points[i, 1],
                    points[i, 0],
                    color=color,
                    marker='.',
                    s=2,
                    zorder=2,
                    label='On-route Prediction'
                )
            else:
                ax12.scatter(
                    points[i, 1],
                    points[i, 0],
                    color=color,
                    marker='.',
                    s=2,
                    zorder=2
                )


        plot_front(proposals,initial_proposals,human_poses,poses,cameras.cam_l0, ax00,pdm_score )
        plot_front(proposals,initial_proposals,human_poses,poses,cameras.cam_f0, ax01,pdm_score )
        plot_front(proposals,initial_proposals,human_poses,poses,cameras.cam_r0, ax02,pdm_score )

        add_configured_bev_on_ax(ax10, scene.map_api, scene.frames[frame_idx])
        add_configured_bev_on_ax(ax11, scene.map_api, scene.frames[frame_idx])
        add_configured_bev_on_ax(ax12, scene.map_api, scene.frames[frame_idx])

        config=TRAJECTORY_CONFIG["human"]

        ax11.plot(
            human_poses[:, 1],
            human_poses[:, 0],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config["marker"],
            markersize=config["marker_size"],
            markeredgecolor=config["marker_edge_color"],
            zorder=config["zorder"]
        )


        config=TRAJECTORY_CONFIG["agent"]#red

        ax11.plot(
            poses[:, 1],
            poses[:, 0],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config["marker"],
            markersize=config["marker_size"],
            markeredgecolor=config["marker_edge_color"],
            zorder=config["zorder"]
        )

        ttc_marking=True
        col_marking=True


        for i in range(64):
            proposal=proposals[i]
            color = cm.Reds(pdm_score[i])  # This returns an RGBA color

            ax11.plot(
                proposal[:, 1],
                proposal[:, 0],
                color=color,
                linewidth=1,
                linestyle=config["line_style"],
                marker='.',
                markersize=2,
                # markeredgecolor=config["marker_edge_color"],
                zorder=2
            )

            initial_proposal=initial_proposals[i]

            ax11.plot(
                initial_proposal[:, 1],
                initial_proposal[:, 0],
                color='#edc948',
                linewidth=1,
                linestyle=config["line_style"],
                marker='.',
                markersize=2,
                zorder=2
            )

            col_label=pred_agents_label[i][0]

            if col_label.max()>0.5:
                #print(i)
                col_corner=pred_agents_corners[i][0].reshape(41,4,2)

                #print(col_label)
                #print(col_corner)
                color = cm.Blues(pdm_score[i])  # This returns an RGBA color

                #print(col_corner)
                for t in range(1,41,5):
                    if col_label[t]>0.3:
                        col_corner_t=col_corner[t][:,::-1]
                        if col_marking:
                            p = Polygon(col_corner_t,
                                alpha=1,
                                edgecolor=color,
                                facecolor = "None",
                                zorder=2,
                                label='At-fault Collision Prediction'
                                )
                        else:
                            p = Polygon(col_corner_t,
                                        alpha=1,
                                        edgecolor=color,
                                        facecolor = "None",
                                        zorder=2
                                        )

                        ax11.add_patch(p)
                        col_marking=False

            ttc_label=pred_agents_label[i][1]
            color = cm.Oranges(pdm_score[i])  # This returns an RGBA color

            if ttc_label.max()>0.8:
                #print(i)
                ttc_corner=pred_agents_corners[i][1].reshape(41,4,2)

                #print(ttc_label)
                #print(col_corner)

                #print(col_corner)
                for t in range(1,41,5):
                    if ttc_label[t]>0.5:
                        ttc_corner_t=ttc_corner[t][:,::-1]
                        if ttc_marking and col_marking==False:
                            p = Polygon(ttc_corner_t,
                                alpha=1,
                                edgecolor=color,
                                facecolor = "None",
                                zorder=2,
                                label='Time-to-collision Prediction'
                                )
                            ttc_marking=False

                        else:
                            p = Polygon(ttc_corner_t,
                                alpha=1,
                                edgecolor=color,
                                facecolor = "None",
                                zorder=2
                                )
                        ax11.add_patch(p)

        ax10.set_aspect("equal")
        ax10.set_xlim(-36, 36)
        ax10.set_ylim(-8, 64)
        ax10.invert_xaxis()
        configure_ax(ax10)

        ax11.set_aspect("equal")
        ax11.set_xlim(-36, 36)
        ax11.set_ylim(-8, 64)
        ax11.invert_xaxis()
        configure_ax(ax11)

        ax12.set_aspect("equal")
        ax12.set_xlim(-36, 36)
        ax12.set_ylim(-8, 64)
        ax12.invert_xaxis()
        configure_ax(ax12)

        ax01.legend(loc=1)
        ax10.legend(loc=1)
        ax11.legend(loc=1)
        ax12.legend(loc=1)

        plt.tight_layout()

        plt.savefig('exp/navsim_test/'+str(token)+'.png')  # Saves as a PNG file
        plt.savefig('exp/navsim_test/'+str(token)+'.pdf')  # Saves as a PNG file
        plt.close()

    return output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """
    agent = instantiate(cfg.agent)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    output = run_test_evaluation(
        agent=agent,
        scene_filter=scene_filter,
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
    )

if __name__ == "__main__":
    main()
