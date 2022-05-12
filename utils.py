import os
from Query2Vec import Query2Vec
import torch
from models import MLP, AdvCNN, BaseCNN
import matplotlib.pyplot as plt
import numpy as np


def parse_workloads():
    """
    Params: None
    Actions: saves test (3 sets) and train queries and workloads to .pt files in /workloads
    Returns: None
    """
    if not os.path.isdir('parsed_workloads'):
        os.mkdir('parsed_workloads')

    with open(os.path.join(os.getcwd(), 'learnedcardinalities/data/train.csv'), 'r') as f:
        data = f.read()
    q_train = Query2Vec('learnedcardinalities/data/train.csv','learnedcardinalities/data/column_min_max_vals.csv',3)
    queries_train, cardinalities = q_train.parse(data)
    queries_train = queries_train.reshape(queries_train.size(0), queries_train.size(2), queries_train.size(1))
    torch.save(queries_train, 'parsed_workloads/queries_train.pt')
    torch.save(cardinalities, 'parsed_workloads/cardinalities_train.pt')

    with open(os.path.join(os.getcwd(), 'learnedcardinalities/workloads/job-light.csv'), 'r') as f:
        job_light = f.read()
    q_job_light = Query2Vec('learnedcardinalities/workloads/job-light.csv','learnedcardinalities/data/column_min_max_vals.csv',3)
    queries_job_light, cardinalities_job_light = q_job_light.parse(job_light) 
    queries_job_light = queries_job_light.reshape(queries_job_light.size(0), queries_job_light.size(2), queries_job_light.size(1)) 
    torch.save(queries_job_light, 'parsed_workloads/queries_job_light.pt')
    torch.save(cardinalities_job_light, 'parsed_workloads/cardinalities_job_light.pt')

    with open(os.path.join(os.getcwd(), 'learnedcardinalities/workloads/scale.csv'), 'r') as f:
        scale = f.read()
    q_scale = Query2Vec('learnedcardinalities/workloads/scale.csv','learnedcardinalities/data/column_min_max_vals.csv',3)
    queries_scale, cardinalities_scale = q_scale.parse(scale)
    queries_scale = queries_scale.reshape(queries_scale.size(0), queries_scale.size(2), queries_scale.size(1)) 
    torch.save(queries_scale, 'parsed_workloads/queries_scale.pt')
    torch.save(cardinalities_scale, 'parsed_workloads/cardinalities_scale.pt')


    with open(os.path.join(os.getcwd(), 'learnedcardinalities/workloads/synthetic.csv'), 'r') as f:
        synth = f.read()
    q_synth = Query2Vec('learnedcardinalities/workloads/synthetic.csv','learnedcardinalities/data/column_min_max_vals.csv',3)
    queries_synth, cardinalities_synth = q_synth.parse(synth) 
    queries_synth = queries_synth.reshape(queries_synth.size(0), queries_synth.size(2), queries_synth.size(1)) 
    torch.save(queries_synth, 'parsed_workloads/queries_synthetic.pt')
    torch.save(cardinalities_synth, 'parsed_workloads/cardinalities_synthetic.pt')

class Data():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    def __len__(self):
        return len(self.X)

# Credit: Andreas Kipf
def qerror_loss(preds, targets):
    qerror = []
    for i in range(len(targets)):
        if (preds[i] > targets[i]).numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))
# End credit 

def train(model_type, in_feat, train, val, n_epochs, batch_size, learning_rate=3e-3, weight_decay=2e-3):
    if model_type=="mlp":
        model = MLP()
    elif model_type=="adv_cnn":
        model = AdvCNN(in_feat)
    elif model_type =="base_cnn":
        model = BaseCNN(in_feat)

    MODEL_PATH = "MODEL/" + model_type + "_bs_" + str(batch_size) + "_eta_" + str(learning_rate) + "_lambda_" + str(weight_decay) + ".pt"
    LOSS_PATH = "LOSS/" + model_type + "_bs_" + str(batch_size) + "_eta_" + str(learning_rate) + "_lambda_" + str(weight_decay) + ".pt"
    tr_load = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    val_load = torch.utils.data.DataLoader(val, 1, shuffle=False)

    tr_losses = []
    val_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for e in range(n_epochs):
        ctr = 0
        loss_total = 0
        model.train()
        for features, targets in tr_load:

            ctr += len(features)
            preds = model(features)
            
            optimizer.zero_grad()
            
            # From Kipf
            loss = qerror_loss(preds, targets.float())
            loss_total += loss.item()
            # End from Kipf

            loss.backward()
            optimizer.step()
        tr_losses.append(loss_total / len(tr_load))

        model.eval()
        val_loss = 0
        for features, targets in val_load:
            preds = model(features)
            loss = qerror_loss(preds, targets.float())
            val_loss += loss.item()
            
        print(f'Epoch {e+1:0f} \t Loss {val_loss/val_load.__len__()}')
        val_losses.append(val_loss/len(val_load))
    losses_all = torch.Tensor((tr_losses, val_losses))
    TUNING_DIR = os.path.join(os.getcwd(), 'tuning')
    torch.save(model.state_dict(), os.path.join(TUNING_DIR ,MODEL_PATH))
    torch.save(losses_all, os.path.join(TUNING_DIR, LOSS_PATH))

    return tr_losses, val_losses        

def paramTuning(model_type,batch_sizes, etas, lambdas, queries_train, cardinalities_train):
    train_conv = Data(queries_train[:70000, :,:], cardinalities_train[:70000])
    val_conv = Data(queries_train[70000:, :,:], cardinalities_train[70000:])

    queries_train_flat = queries_train.view(queries_train.size(0), queries_train.size(1)*queries_train.size(2))
    train_flat = Data(queries_train_flat[:70000, :], cardinalities_train[:70000])
    val_flat = Data(queries_train_flat[70000:, :], cardinalities_train[70000:])

    for b in batch_sizes:
        for e in etas:
            for l in lambdas:
                if model_type=='mlp':
                    mod_loss = train("mlp", 3, train_flat, val_flat, 25, b, e, l)
                else: 
                    mod_loss = train(model_type,3, train_conv, val_conv, 25, b, e, l)
    return

def plot_loss():
    final_val_losses = []
    for f in os.listdir('tuning/LOSS'):
        if not f == '.DS_Store':
            loss = torch.load(os.path.join('tuning/LOSS',f))
            tr_loss = loss[0]
            val_loss = loss[1]
            final_val_losses.append([f, val_loss[-1:]])
            plt.plot(np.arange(1,26), np.log(tr_loss.numpy()), color='r', label='train')
            plt.plot(np.arange(1,26), np.log(val_loss.numpy()), color='b', label='val')
            plt.title(f)
            plt.xlabel('Epoch')
            plt.ylabel('Q error')
            plt.legend(loc='upper right')
            plt.show()

def evalOnTest(model_type):
    queries_scale = torch.load('parsed_workloads/queries_scale.pt')
    queries_synth = torch.load('parsed_workloads/queries_synthetic.pt')
    queries_job_light = torch.load('parsed_workloads/queries_job_light.pt')

    cardinalities_scale = torch.load('parsed_workloads/cardinalities_scale.pt')
    cardinalities_synth = torch.load('parsed_workloads/cardinalities_synthetic.pt')
    cardinalities_job_light = torch.load('parsed_workloads/cardinalities_job_light.pt')

    if model_type == 'ann':
        ann_best = torch.load(os.path.join('tuning/MODEL', 'mlp_bs_16_eta_0.003_lambda_0.002.pt'))
        scale_flat = queries_scale[:, :,:14].reshape(queries_scale.size(0), 14 * 3) 
        synth_flat = queries_synth[:, :,:14].reshape(queries_synth.size(0), 14 * 3) 
        job_flat = queries_job_light[:, :,:14].reshape(queries_job_light.size(0), 14 * 3) 
        ann = MLP()
        ann.load_state_dict(ann_best)
        ann.eval()
        scale_preds =  ann(scale_flat)
        synth_preds = ann(synth_flat)
        job_preds = ann(job_flat)

        scale_loss = qerror_loss(scale_preds, cardinalities_scale)
        synth_loss = qerror_loss(synth_preds, cardinalities_synth)
        job_loss = qerror_loss(job_preds, cardinalities_job_light)
        
        print('ANN')
        print(f'{model_type} Q-error on scale: {scale_loss}')
        print(f'{model_type} Q-error on synth: {synth_loss}')
        print(f'{model_type} Q-error on JOB-Light: {job_loss}')
        print()

    elif model_type == 'bcnn':
        bcnn_best = torch.load(os.path.join('tuning/MODEL', 'base_cnn_bs_16_eta_0.003_lambda_0.002.pt'))
        bcnn = BaseCNN(3)
        bcnn.load_state_dict(bcnn_best)
        bcnn.eval()
        scale_preds =  bcnn(queries_scale[:,:,:14])
        synth_preds = bcnn(queries_synth[:,:,:14])
        job_preds = bcnn(queries_job_light[:,:,:14])

        scale_loss = qerror_loss(scale_preds, cardinalities_scale)
        synth_loss = qerror_loss(synth_preds, cardinalities_synth)
        job_loss = qerror_loss(job_preds, cardinalities_job_light)
        
        print('BaseCNN')
        print(f'{model_type} Q-error on scale: {scale_loss}')
        print(f'{model_type} Q-error on synth: {synth_loss}')
        print(f'{model_type} Q-error on JOB-Light: {job_loss}')     
        print()
    elif model_type == 'acnn':
        acnn_best =torch.load(os.path.join('tuning/MODEL', 'adv_cnn_bs_16_eta_0.003_lambda_0.002.pt'))
        acnn = AdvCNN(3)
        acnn.load_state_dict(acnn_best)
        acnn.eval()
        scale_preds =  acnn(queries_scale[:,:,:14])
        synth_preds = acnn(queries_synth[:,:,:14])
        job_preds = acnn(queries_job_light[:,:,:14])

        scale_loss = qerror_loss(scale_preds, cardinalities_scale)
        synth_loss = qerror_loss(synth_preds, cardinalities_synth)
        job_loss = qerror_loss(job_preds, cardinalities_job_light)
        
        print('AdvCNN')
        print(f'{model_type} Q-error on scale: {scale_loss}')
        print(f'{model_type} Q-error on synth: {synth_loss}')
        print(f'{model_type} Q-error on JOB-Light: {job_loss}')     
        print()
    else: 
        print('Dont recognize that model type')
        return -1   

def evaluate_models():
    models = ['ann', 'bcnn', 'acnn']
    for m in models:
        evalOnTest(m)