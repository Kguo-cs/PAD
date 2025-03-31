from typing import Tuple
from pathlib import Path
import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
from PIL.features import features
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
import torch

from navsim.agents.pad.score_module.compute_b2d_score import compute_corners_torch

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "b2d_training"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")

    logger.info("Using cached data without building SceneLoader")
    assert (
        not cfg.force_cache_computation
    ), "force_cache_computation must be False when using cached data without building SceneLoader"
    assert (
            cfg.cache_path is not None
    ), "cache_path must be provided when using cached data without building SceneLoader"
    train_data = CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=['train'],
    )
    val_data = CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=['val'],
    )

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True,drop_last=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False,drop_last=True)
    logger.info("Num validation samples: %d", len(val_data))


    #agent._checkpoint_path = "/home/ke/PAD/exp/b2d_result/B2d_p32_prev01_inter01_w01_allw10/03.15_19.56/epoch=13-step=10752.ckpt" #"/home/ke/PAD/exp/b2d_result/B2d_prev01_inter01_l101/03.14_12.34/epoch=20-step=16128.ckpt"

    #agent.initialize()

    agent.eval()

    agent.cuda()

    agent.ray=False

    for key, value in agent.map_infos.items():
        agent.map_infos[key] = torch.tensor(value).cuda()

    l2_list=[]

    for batch in val_dataloader:
        features,targets=batch

        for key,feature in features.items():
            features[key]=feature.cuda()

        for key,target in targets.items():
            if type(target) is not list:
                targets[key]=target.cuda()

        pred=agent.forward(features)

        pdm_score = (torch.sigmoid(pred['pred_logit']) + torch.sigmoid(pred['pred_logit2']))[0] / 2

        #proposals = pred["proposals"]
        proposals = targets["trajectory"][:, None] #torch.cat([targets["trajectory"][:, None], proposals], dim=1)
        #proposals=torch.cat([targets["trajectory"][:, None], proposals], dim=1)

        metric_cache_paths = agent.test_metric_cache_paths

        target_trajectory = targets["trajectory"].detach()
        proposals = proposals.detach()

        target_traj = target_trajectory.cpu().numpy()

        data_points = []

        lidar2worlds = targets["lidar2world"]

        all_proposals = torch.cat([proposals, target_trajectory[:, None]], dim=1)

        all_proposals_xy = all_proposals[:, :, :, :2]
        all_proposals_heading = all_proposals[:, :, :, 2:]

        all_pos = all_proposals_xy.reshape(len(target_traj), -1, 2)

        mid_points = (all_pos.amax(1) + all_pos.amin(1)) / 2

        dists = torch.linalg.norm(all_pos - mid_points[:, None], dim=-1).amax(1) + 5

        xyz = torch.cat(
            [mid_points[..., :2], torch.zeros_like(mid_points[..., :1]), torch.ones_like(mid_points[..., :1])],
            dim=-1)

        xys = torch.einsum("nij,nj->ni", lidar2worlds, xyz)[:, :2]

        vel = all_proposals_xy[:, :, 1:] - all_proposals_xy[:, :, :-1]

        vel = torch.cat([all_proposals_xy[:, :, :1], vel], dim=2) / 0.5

        proposals_05 = torch.cat([all_proposals_xy + vel * 0.5, all_proposals_heading], dim=-1)

        proposals_ttc = torch.stack([all_proposals, proposals_05], dim=3)

        ego_corners_ttc = compute_corners_torch(proposals_ttc.reshape(-1, 3)).reshape(proposals_ttc.shape[0],
                                                                                      proposals_ttc.shape[1],
                                                                                      proposals_ttc.shape[2], 2,
                                                                                      4, 2)

        ego_corners_center = torch.cat([ego_corners_ttc[:, :, :, 0], all_proposals_xy[:, :, :, None]], dim=-2)

        ego_corners_center_xyz = torch.cat(
            [ego_corners_center, torch.zeros_like(ego_corners_center[..., :1]),
             torch.ones_like(ego_corners_center[..., :1])], dim=-1)

        global_ego_corners_centers = torch.einsum("nij,nptkj->nptki", lidar2worlds, ego_corners_center_xyz)[...,
                                     :2]

        vel = vel[:, :-1]

        accs = torch.linalg.norm(vel[:, :, 1:] - vel[:, :, :-1], dim=-1) / 0.5

        comforts = (accs < 10).all(-1)


        for token, town_name, proposal,target_traj, comfort, dist, xy, global_conners, local_corners in zip(targets["token"],
                                                                                               targets[
                                                                                                   "town_name"],
                                                                                               proposals.cpu().numpy(),
                                                                                               target_trajectory.cpu().numpy(),
                                                                                               comforts.cpu().numpy(),
                                                                                               dists.cpu().numpy(),
                                                                                               xys,
                                                                                               global_ego_corners_centers,
                                                                                               ego_corners_ttc.cpu().numpy()):
            all_lane_points = agent.map_infos[town_name[:6]]

            dist_to_cur = torch.linalg.norm(all_lane_points[:, :2] - xy, dim=-1)

            nearby_point = all_lane_points[dist_to_cur < dist]

            lane_xy = nearby_point[:, :2]
            lane_width = nearby_point[:, 2]+0.1
            lane_id = nearby_point[:, -1]

            dist_to_lane = torch.linalg.norm(global_conners[None] - lane_xy[:, None, None, None], dim=-1)

            on_road = dist_to_lane < lane_width[:, None, None, None]

            on_road_all = on_road.any(0).all(-1)

            nearest_lane = torch.argmin(dist_to_lane - lane_width[:, None, None, None], dim=0)

            nearest_lane_id = lane_id[nearest_lane]

            center_nearest_lane_id = nearest_lane_id[:, :, -1]

            nearest_road_id = torch.round(center_nearest_lane_id)

            target_road_id = torch.unique(nearest_road_id[-1])

            proposal_center_road_id = nearest_road_id

            on_route_all = torch.isin(proposal_center_road_id, target_road_id)
            # in_multiple_lanes: if
            # - more than one drivable polygon contains at least one corner
            # - no polygon contains all corners
            corner_nearest_lane_id = nearest_lane_id[:, :, :-1]

            batch_multiple_lanes_mask = (corner_nearest_lane_id != corner_nearest_lane_id[:, :, :1]).any(-1)

            ego_areas = torch.stack([batch_multiple_lanes_mask, on_road_all, on_route_all], dim=-1)

            data_dict = {
                "fut_box_corners": metric_cache_paths[token],
                "_ego_coords": local_corners,
                "target_traj": target_traj,
                "proposal": proposal,
                "comfort": comfort,
                "ego_areas": ego_areas.cpu().numpy(),
            }
            data_points.append(data_dict)

            all_res = agent.get_scores(data_points)

            target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

            final_scores = target_scores[:, :, -1]

            best_scores = torch.amax(final_scores, dim=-1)

            # l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4].mean().cpu().numpy()
            #
            # # return final_scores[:, 0].mean(), best_scores.mean(), final_scores[:, 1:], l2_2s.mean(), target_scores[:, 0]
            #
            # l2_list.append(l2_2s)
            #
            # print(np.mean(l2_list))


            key_agent_corners =np.stack([res[1] for res in all_res])[0]

            key_agent_labels = np.stack([res[2] for res in all_res])[0]

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)
            target_dist_to_road = dist_to_lane[:, -1:] - lane_width[:, None, None, None]

            target_on_road = target_dist_to_road.amin(0).amax(-1)


            # if target_on_road.max()>0:
            #     print(target_on_road)
            collision_score=target_scores[0,0,0]
            on_road=target_scores[0,0,1]

           # print(torch.argmax(pdm_score,dim=0))

            #print(pdm_score[torch.argmax(pdm_score,dim=0)[-1]])
            print(collision_score)


            if not on_road.all() :
                # print(features["ego_status"][0])
                # print(pdm_score[torch.argmax(pdm_score[:,-1],dim=0)])
                # print(target_scores[0][0])

                fut_box_corners = metric_cache_paths[token]
                fut_mask = fut_box_corners.any(-1).any(-1)

                fut_box_corners_xyz = np.concatenate(
                    [fut_box_corners, np.zeros_like(fut_box_corners[..., :1]),
                     np.ones_like(fut_box_corners[..., :1])], axis=-1)

                global_fut_box_corners = np.einsum("ij,ptkj->ptki", lidar2worlds[0].cpu().numpy(), fut_box_corners_xyz)[...,:2]

                key_agent_corners_xyz = np.concatenate(
                    [key_agent_corners, np.zeros_like(key_agent_corners[..., :1]),
                     np.ones_like(key_agent_corners[..., :1])], axis=-1)

                global_key_agent_corners = np.einsum("ij,pqtkj->pqtki", lidar2worlds[0].cpu().numpy(), key_agent_corners_xyz)[...,:2]


                local_waypoins=features["ego_status"][0][0][3:5].cpu().numpy()
                local_waypoins = np.concatenate(
                    [local_waypoins, np.zeros_like(local_waypoins[..., :1]),
                     np.ones_like(local_waypoins[..., :1])], axis=-1)

                global_waypoint = np.einsum("ij,j->i", lidar2worlds[0].cpu().numpy(), local_waypoins)[...,:2]

                fig, (ax, ax2) = plt.subplots(2)  # note we must use plt.subplots, not plt.subplot

                ax.scatter(global_waypoint[0], global_waypoint[1], marker='*', zorder=5,s=100)

                global_conners=global_conners.cpu().numpy()

                for i in range(len(global_fut_box_corners)):
                    for t in range(6):
                        if fut_mask[i][t]:
                            polygon = plt.Polygon(
                                global_fut_box_corners[i][t][:4],
                                edgecolor='purple',
                                fill=False,
                                linewidth=2,
                                alpha=1-t*0.15,
                                zorder=3
                            )
                            ax.add_patch(polygon)

                # for i in range(len(global_key_agent_corners)):
                #     for t in range(6):
                #         if key_agent_labels[i][1][t]:
                #             polygon = plt.Polygon(
                #                 global_key_agent_corners[i][1][t][:4],
                #                 edgecolor='orange',
                #                 fill=False,
                #                 linewidth=2,
                #                 alpha=1-t*0.15,
                #                 zorder=6
                #             )
                #             ax.add_patch(polygon)





                for i in range(len(global_conners)):
                    if i==0:
                        color='r'
                        zorder=10
                    elif i==len(global_conners)-1:
                        color='b'
                        zorder=9

                    else:
                        color='g'
                        zorder=2
                    for t in range(6):
                        polygon = plt.Polygon(
                            global_conners[i][t][:4],
                            edgecolor=color,  # 绿色边框
                            fill=False,
                            linewidth=2,  # 线条宽度
                            alpha=1-t*0.1,
                            zorder=zorder
                        )
                        ax.add_patch(polygon)
                    # plt.plot(global_conners[i][:,-1,0],global_conners[i][:,-1,1],color=color)


                lane_xy=lane_xy.cpu().numpy()
                lane_width=lane_width.cpu().numpy()

                for i in range(len(nearby_point)):
                    circle1 = plt.Circle((lane_xy[i][0],lane_xy[i][1]), lane_width[i],fill=True,zorder=1, color='grey')

                       # print((lane_xy[i][0],lane_xy[i][1]), lane_width[i])

                    ax.add_patch(circle1)
                ax.set_aspect('equal')  # 保持比例


                ax.set_xlim(min(lane_xy[:, 0]) - 5, max(lane_xy[:, 0]) + 5)
                ax.set_ylim(min(lane_xy[:, 1]) - 5, max(lane_xy[:, 1]) + 5)


                front_view = features["camera_feature"][0, 0].cpu().numpy()
                front_view = np.transpose(front_view, (1, 2, 0))

                ax2.imshow(front_view)
                plt.show()









if __name__ == "__main__":
    main()
