from utils import parse_workloads, paramTuning, plot_loss, evaluate_models
from models import MLP, BaseCNN, AdvCNN
import torch
import matplotlib.pyplot as plt



#parse_workloads()

queries_train = torch.load('parsed_workloads/queries_train.pt')
cardinalities_train = torch.load('parsed_workloads/cardinalities_train.pt')

paramTuning("mlp", [16, 32], [3e-4, 3e-3, 3e-2],[2e-3, 2e-2], queries_train, cardinalities_train)
paramTuning("base_cnn", [16, 32], [3e-4, 3e-3],[2e-3, 2e-2], queries_train, cardinalities_train)
paramTuning("adv_cnn", [16, 32], [3e-4, 3e-3],[2e-3, 2e-2], queries_train, cardinalities_train)

plot_loss()

evaluate_models()