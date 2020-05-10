import numpy as np
import torch
from torch.autograd import Variable
import h5py
from networks.data_utils import get_imdb_data
import matplotlib.pyplot as plt

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


train_data, test_data = get_imdb_data()
print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))

img_test = test_data.X[60:61]

img_test = np.squeeze(img_test)
print(img_test.shape)
plt.imshow(img_test)
plt.show()

relaynet_model = torch.load('C:/Users/krive/PycharmProjects/relaynet_pytorch/models/Exp01/relaynet_epoch18.model', map_location=torch.device('cpu'))
out = relaynet_model(Variable(torch.Tensor(test_data.X[0:1]).cuda(), volatile=True))
