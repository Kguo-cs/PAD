import torch.nn as nn
from .bevformer.bev_refiner import Bev_refiner
from .bevformer.transformer_decoder import MyTransformeDecoder,MLP
import numpy as np

class Traj_refiner(nn.Module):
    def __init__(self,config,init_p=False):
        super().__init__()

        self.poses_num=config.num_poses
        self.state_size=3

        self.traj_bev = config.traj_bev
        self.b2d = config.b2d

        self.init_p = init_p

        if self.init_p:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
        elif self.traj_bev:
            self.Bev_refiner=Bev_refiner(config,config.proposal_num,self.poses_num,config.traj_proposal_query)

        self.traj_decoder = MLP(config.tf_d_model, config.tf_d_ffn, self.poses_num * self.state_size)

    def forward(self, keyval,proposal_list=None,image_feature=None):

        if self.init_p:
            proposal_feature = keyval + self.init_feature.weight[None]
        else:
            proposals=proposal_list[-1]
            if self.traj_bev:
                keyval = self.Bev_refiner(proposals,keyval,image_feature)

            proposal_feature = keyval[:, :proposals.shape[1] * proposals.shape[2]].reshape(proposals.shape[0],
                                                                                           proposals.shape[1],
                                                                                           proposals.shape[2], -1).amax(
                -2)

        proposals = self.traj_decoder(proposal_feature).reshape(keyval.shape[0], -1, self.poses_num, self.state_size)

        if self.b2d:
            proposals[...,2]+=np.pi/2

        proposal_list.append(proposals)

        return keyval,proposal_list


