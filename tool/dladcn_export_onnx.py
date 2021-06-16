import cv2
import numpy as np
import torch
import torch.onnx.utils as onnx
import models.networks.pose_dla_dcn as net
from collections import OrderedDict

model = net.get_pose_net(num_layers=34, heads={'hm': 1, 'wh': 4, 'reg': 2, 'id':128})

checkpoint = torch.load(r"models/fairmot_dla34.pth", map_location="cpu")
checkpoint = checkpoint["state_dict"]
change = OrderedDict()
for key, op in checkpoint.items():
    change[key.replace("module.", "", 1)] = op

model.load_state_dict(change)
model.eval()
model.cuda()

input = torch.zeros((1, 3, 608, 1088)).cuda()
[hm, wh, reg, hm_pool, id_feature] = model(input)
onnx.export(model, (input), "fairmot_dla34_1088x608.onnx", output_names=["hm", "wh", "reg", "hm_pool", "id"], verbose=True)
