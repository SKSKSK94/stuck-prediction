#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset_road import stuckDataset

from network import SRUPRS
from utils import my_print, my_save_model, my_load_model, print_metric
from utils import cal_slip_ratio_metric, cal_gaussian_NLL_loss, evaluate
from utils import writeTrainInfo, setup_logger

from torch.utils.data import DataLoader

import json
import os
from datetime import datetime

from torch.optim.lr_scheduler import StepLR

#%% Config and Parameters
data_keys_cfg_path = '../cfg/data_keys.json'
overall_cfg_path = '../cfg/config.json'
saved_path = '../saved_model/SRUPRS_R2_' + str(datetime.now())

assert os.path.isfile(overall_cfg_path)
with open(overall_cfg_path, 'r') as f:
    cfg = json.load(f)

log_file = os.path.join(saved_path, 'training_log.txt')

roadinfo_mode = cfg["roadinfo_mode"]
batch_size = cfg["batch_size"]
num_epoch = cfg["num_epoch"]
lr = cfg["lr"] 
print_iteration_period = cfg["print_iteration_period"]
loss_weights = cfg["loss_weights"]

#%% Dataset/Loader 
train_dataset = stuckDataset(
    road_info=cfg["roadinfo_mode"],
    train_mode=cfg["train_mode"],
    num_sequence=cfg['num_sequence'],
    data_keys_cfg_path=data_keys_cfg_path
)
val_train_dataset = stuckDataset(
    road_info=cfg["roadinfo_mode"], 
    train_mode=cfg["train_mode"],
    num_sequence=cfg['num_sequence'],
    data_keys_cfg_path=data_keys_cfg_path
)
test_dataset = stuckDataset(
    road_info=cfg["roadinfo_mode"],
    train_mode=False,
    test_file='test/*/*.mat',
    num_sequence=cfg['num_sequence'],
    data_keys_cfg_path=data_keys_cfg_path
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=4
)
val_train_loader = DataLoader(
    dataset=val_train_dataset,
    batch_size=len(train_dataset),
    shuffle=False,
    num_workers=4
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=len(test_dataset),
    shuffle=False,
    num_workers=4
)

#%%
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

network = SRUPRS(
    SRUPRS_cfg=cfg['SRUPRS_cfg'],
    data_keys_cfg_path=data_keys_cfg_path
).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=lr)
scheduler = StepLR(
    optimizer=optimizer,
    step_size=cfg['lr_step_size'],
    gamma=cfg['lr_gamma']
)

cross_entropy_loss_metric = nn.NLLLoss()
#%%
CE_loss_list = []
GNLL_loss_list = []
regression_loss_list = []
total_loss_list = []
iteration_len = len(train_dataset)//batch_size
if cfg["train_mode"]:
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    
    info_logger = setup_logger("info", log_file)
    info = writeTrainInfo(cfg)
    info_logger.info(info)

    for epoch in range(num_epoch):
        for iteration, items in enumerate(train_loader):
            
            data, pos, road_state_label, slip_ratio_label = items
            data = data.to(device)
            slip_ratio_label, road_state_label = slip_ratio_label.to(device), road_state_label.to(device)
                
            predicted_road_state, slip_mu, slip_std = network.forward(
                data,
                device=device,
                use_detach=False
            )

            CE_loss = cross_entropy_loss_metric(predicted_road_state, road_state_label[:, 0])
            GNLL_loss = cal_gaussian_NLL_loss(
                x=slip_ratio_label,
                mu=slip_mu,
                std=slip_std,
                reduction=True
            )
            regression_loss = torch.mean((slip_mu-slip_ratio_label)**2)
            total_loss = loss_weights[0]*CE_loss + loss_weights[1]*GNLL_loss + loss_weights[2]*regression_loss       
            # total_loss = 0.02*CE_loss + 0.01*GNLL_loss + regression_loss             
            # total_loss = 0.2*CE_loss + 0.02*GNLL_loss + regression_loss
            
            # regression_loss = torch.mean(torch.abs((slip_mu-slip_ratio_label)))
            # total_loss = 0.05*CE_loss + 0.05*GNLL_loss + regression_loss

            CE_loss_list.append(CE_loss.data.item())
            GNLL_loss_list.append(GNLL_loss.data.item())
            regression_loss_list.append(regression_loss.data.item())
            total_loss_list.append(total_loss.data.item())

            if iteration % print_iteration_period == 0:
                
                now_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                my_print('|{}|Epoch:{:>4}/{:>4}|\tIteration:{:>5}/{}|\tLoss:{:.6f}(CE:{:.4f}, GNLL:{:.4f}, Regression:{:.6f})|lr: {}|'.format(
                    datetime.now(), 
                    epoch, 
                    num_epoch,
                    iteration, 
                    iteration_len, 
                    np.mean(total_loss_list), 
                    np.mean(CE_loss_list), 
                    np.mean(GNLL_loss_list), 
                    np.mean(regression_loss_list),
                    now_lr),
                    log_file=log_file
                )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        if epoch % 1 == 0:
            my_print('###################### Val ###########################', log_file=log_file)
            network.eval()
            for iteration, items in enumerate(val_train_loader):
                
                data, pos, road_state_label, slip_ratio_label = items
                data = data.to(device)                
                slip_ratio_label, road_state_label = slip_ratio_label.to(device), road_state_label.to(device)
            
                with torch.no_grad():
                    predicted_road_state, slip_mu, slip_std = network.forward(
                        data,
                        device=device,
                        use_detach=False
                    )

                metric_dict = evaluate(
                    predicted_road_state,
                    road_state_label,
                    slip_mu,
                    slip_ratio_label
                )

                print_metric(metric_dict, dataset_name='', log_file=log_file)
            
            my_print('----------', log_file=log_file)
                
            for iteration, items in enumerate(test_loader):
                            
                data, pos, road_state_label, slip_ratio_label = items
                data = data.to(device)                
                slip_ratio_label, road_state_label = slip_ratio_label.to(device), road_state_label.to(device)
            
                with torch.no_grad():
                    predicted_road_state, slip_mu, slip_std = network.forward(
                        data,
                        device=device,
                        use_detach=False
                    )

                metric_dict = evaluate(
                    predicted_road_state,
                    road_state_label,
                    slip_mu,
                    slip_ratio_label
                )

                print_metric(metric_dict, dataset_name='Val_', log_file=log_file)                
            
            my_print('###################### Val ###########################', log_file=log_file)
            my_save_model(network, epoch, saved_path)
        network.train()
        scheduler.step()
                        
#%%
network = my_load_model(model=network, path='saved_model/SRUPRS_R2_2021-10-08 15:50:49.474811/model_epoch_0070.pt')
network.eval()
for idx, items in enumerate(test_loader):
    data, pos, road_state_label, slip_ratio_label = items
    data = data.to(device)                
    slip_ratio_label, road_state_label = slip_ratio_label.to(device), road_state_label.to(device)

    with torch.no_grad():
        predicted_road_state, slip_mu, slip_std = network.forward(data, device=device, use_detach=False)

    slip_mu = slip_mu.cpu().detach().numpy()
    slip_std = slip_std.cpu().detach().numpy()
    slip_ratio_label = slip_ratio_label.cpu().detach().numpy()

rmse_ = cal_slip_ratio_metric(slip_mu, slip_ratio_label, metric='rmse')
mae_ = cal_slip_ratio_metric(slip_mu, slip_ratio_label, metric='mae')
r2_score_ = cal_slip_ratio_metric(slip_mu, slip_ratio_label, metric='r2_score')

print('rmse : ', rmse_)
print('mae : ', mae_)
print('r2_score : ', r2_score_)

#%%
plt.figure(figsize=(15,5))
# start = 31500
# end = start + 500 
# start = 7800
# end = start + 400
# start = 34000
# end = start + 700
# start = 0
# end = start + len(slip_mu)
start = 17000
end = start + 2000

std_scale = 1.0
tire_idx = 1

plt.plot(slip_ratio_label[start:end,tire_idx]*100, 'b-*')
plt.plot(slip_mu[start:end,tire_idx]*100, 'r-*', alpha=0.6)

lower = np.clip(slip_mu[start:end,tire_idx]*100 - std_scale*slip_std[start:end,tire_idx]*100, a_min=0, a_max=100)
upper = np.clip(slip_mu[start:end,tire_idx]*100 + std_scale*slip_std[start:end,tire_idx]*100, a_min=0, a_max=100)
plt.fill_between(np.arange(len(lower)), lower, upper, color='r', alpha=0.3)

# plt.plot(slip_ratio_label[::1,0]*100, 'bx')
# plt.plot(slip_mu[::1,0]*100, 'rx')
plt.legend(['true', 'pred'])
plt.show()
