from multiprocessing.spawn import freeze_support

import numpy as np
import torch
from torch.autograd import Variable
import h5py

from networks import solver
from networks.net_api.losses import CombinedLoss
from networks.relay_net import ReLayNet
from networks.solver import Solver
from networks.data_utils import get_imdb_data
import matplotlib.pyplot as plt
import torch.nn.functional as F


SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}];


# {"id": 9, "name": "Fluid region", "rgb_values": [64, 128, 0]}];

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)

train_data, test_data = get_imdb_data()
print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))

img_test = train_data.X[10:11]

img_test = np.squeeze(img_test)
print(img_test.shape)
plt.imshow(img_test)
plt.show()

relaynet_model = torch.load('C:/Users/MASTER/PycharmProjects/relaynet_pytorch2//models/Exp06/relaynet_epoch20.model', map_location=torch.device('cuda'))
relaynet_model.cuda()
with torch.no_grad():
    out = relaynet_model(Variable(torch.Tensor(train_data.X[10:11]).cuda()))
loss_func = CombinedLoss()
X = torch.from_numpy(np.array(train_data.X[10:11]))
y = torch.from_numpy(np.array(train_data.y[10:11]))
w = torch.from_numpy(np.array(train_data.w[10:11]))
loss = loss_func(out, y.cuda(), w.cuda())
print(loss)
out = F.softmax(out, dim=1)
max_val, idx = torch.max(out,1)
idx = idx.data.cpu().numpy()
idx = label_img_to_rgb(idx)
plt.imshow(idx)
plt.show()


# if __name__ == '__main__':
#         freeze_support()
#         train_data, test_data = get_imdb_data()
#
#         train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
#         val_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)
#
#         param ={
#                 'num_channels':1,
#                 'num_filters':64,
#                 'kernel_h':3,
#                 'kernel_w':7,
#                 'kernel_c': 1,
#                 'stride_conv':1,
#                 'pool':2,
#                 'stride_pool':2,
#                 'num_class':9
#             }
#
#         exp_dir_name = 'Exp02'
#
#         relaynet_model = ReLayNet(param)
#         solver = Solver(optim_args={"lr": 1e-2})
#         solver.train(relaynet_model, train_loader, val_loader, log_nth=1, num_epochs=20, exp_dir_name=exp_dir_name)