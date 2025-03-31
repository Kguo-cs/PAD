from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.image_encoder import ImgEncoder

class PadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self._backbone = ImgEncoder(config)

        self.command_num=config.command_num

        self.hist_encoding = nn.Linear(11, config.tf_d_model)

        traj_refiner0 = Traj_refiner(config,init_p=True)

        ref_num=config.ref_num

        self._trajectory_head=nn.ModuleList([traj_refiner0]+[Traj_refiner(config) for _ in range(ref_num) ] )

        self.scorer = Scorer(config)

        self.b2d=config.b2d

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego_status: torch.Tensor = features["ego_status"][:,-1]
        camera_feature: torch.Tensor = features["camera_feature"]

        batch_size = ego_status.shape[0]

        if self.b2d:
            
            # if self.training:
            #     ego_status[:,:5]+=torch.randn_like(ego_status[:,:5])

            # print(ego_status[:,1:3])
            # print(ego_status[:,:1])

            ego_status[:,1:3]=0#torch.clamp(ego_status[:,1:3], min=-1, max=1)

            # ego_status=torch.clamp(ego_status, min=-15, max=15)
            ego_status[:,:1]=torch.clamp(ego_status[:,:1], min=0, max=15)

            # ego_status[:,1:3]+=torch.randn_like(ego_status[:,1:3])
            # ego_status[:,:1]+=torch.randn_like(ego_status[:,:1])

        image_feature = self._backbone(camera_feature,img_metas=features)  # b,64,64,64

        output={}

        keyval=self.hist_encoding(ego_status)[:,None]

        proposal_list = []
        for i, refine in enumerate(self._trajectory_head):
            keyval, proposal_list = refine(keyval, proposal_list, image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        proposals = proposals.detach()[..., :3]

        pred_logit,pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels = self.scorer(proposals, keyval,image_feature)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"] = agent_states
        output["agent_labels"] =agent_labels

        if pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output



