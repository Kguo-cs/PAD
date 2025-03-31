import torch
import torch.nn as nn
from ..bevformer.bev_refiner import Bev_refiner
from ..bevformer.transformer_decoder import MyTransformeDecoder,MLP
from .map_head import MapHead
import numpy as np

class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.score_bev = config.score_bev

        self.b2d=config.b2d

        self.proposal_num=config.proposal_num
        self.score_num = 6
        num_poses = config.num_poses

        if self.score_bev:
            self.Bev_refiner=Bev_refiner(config,self.proposal_num,num_poses,proposal_query=config.score_proposal_query)

        self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.double_score=config.double_score

        if self.double_score:
            self.pred_score2 = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.agent_pred= config.agent_pred

        num_agent_pose=config.num_agent_pose

        if self.agent_pred:
            self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, num_agent_pose * 9)
            self.pred_ttc_agent = MLP(config.tf_d_model, config.tf_d_ffn, num_agent_pose * 9)

        self.area_pred=config.area_pred

        if self.area_pred:
            if self.b2d:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 2)
            else:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 5*2)


    def forward(self, proposals,keyval,image_feature):
        batch_size=len(proposals)
        p_size=proposals.shape[1]
        t_size=proposals.shape[2]

        if self.score_bev:
            keyval = self.Bev_refiner(proposals, keyval, image_feature)

        proposal_feature = keyval[:, :p_size * t_size].reshape(batch_size, p_size, t_size, -1).amax(-2)
        pred_logit = self.pred_score(proposal_feature).reshape(batch_size, -1, 6)

        pred_logit2=pred_agents_states=pred_area_logit=bev_semantic_map=agent_states=agent_labels=None

        if self.double_score:
            pred_logit2 = self.pred_score2(proposal_feature).reshape(batch_size, -1, 6)

        if  self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(keyval[:, :p_size * t_size])

            if self.agent_pred:
                col_agents_state = self.pred_col_agent(proposal_feature)
                ttc_agents_state = self.pred_ttc_agent(proposal_feature)
                pred_agents_states = torch.stack([ttc_agents_state, col_agents_state], dim=2)

        return pred_logit,pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels
