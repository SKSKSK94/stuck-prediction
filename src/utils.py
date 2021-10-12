import numpy as np
import torch
import torch.nn as nn
import json
import logging
import math

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o

def cfg_read(path):
    with open(path, 'r') as f:
        cfg = json.loads(f.read(), cls=Decoder)
    return cfg

class jsonParser:
    """
    configuration은 *.json 형태이기 때문에
    이를 dictionary형태로 변환시켜주는 class
    """

    def __init__(self, fileName):
        with open(fileName) as jsonFile:
            self.jsonFile = json.load(jsonFile)
            self.jsonFile: dict

    def loadParser(self):
        return self.jsonFile

    def loadAgentParser(self):
        agentData = self.jsonFile.get("agent")
        agentData["sSize"] = self.jsonFile["sSize"]
        agentData["aSize"] = self.jsonFile["aSize"]
        agentData["device"] = self.jsonFile["device"]
        agentData["gamma"] = self.jsonFile["gamma"]
        return agentData

    def loadOptParser(self):
        return self.jsonFile.get("optim")


class Configuration:
    def __init__(self, path: str):
        parser = jsonParser(path)
        self._data = parser.loadParser()
        self.network_info = None
        self.otpim = None
        self.device = None
        self.FTT_mode = None
        for key, value in self._data.items():
            setattr(self, key, value)


def writeDict(info, data, key, n=0):
    tab = ""
    for _ in range(n):
        tab += "\t"
    if type(data) == dict:
        for k in data.keys():
            dK = data[k]
            if type(dK) == dict:
                info += """
        {}{} :
            """.format(
                    tab, k
                )
                writeDict(info, dK, k, n=n + 1)
            else:
                info += """
        {}{} : {}
        """.format(
                    tab, k, dK
                )
    else:
        info += """
        {} : {}
        """.format(
            key, data
        )


class INFO:
    def __init__(self):
        self.info = """
    Configuration for this experiment
    """

    def __add__(self, string):
        self.info += string
        return self

    def __str__(self):
        return self.info


def writeTrainInfo(datas):
    info = INFO()
    key = datas.keys()
    for k in key:
        data = datas[k]
        if type(data) == dict:
            info += """
        {} :
        """.format(
                k
            )

            writeDict(info, data, k, n=1)
        else:
            writeDict(info, data, k)

    return info


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='a')
    stream = logging.StreamHandler()
    # handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream)

    return logger


def my_print(content, log_file):
    with open(log_file, 'a') as writer:
        print(content)
        writer.write(content+'\n')


def my_save_model(model, epoch, saved_path):
    path = '{}/model_epoch_{:04}.pt'.format(saved_path, epoch)
    torch.save(
        {
            'SRUPRS': model.state_dict(),
        }, 
        path
    )
    print('Successfull saved to {}'.format(path))


def my_load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['SRUPRS'])
    print('Successfull loaded from {}'.format(path))
    return model


def cal_slip_ratio_metric(pred_list, true_list, metric='rmse'):
    '''
    Calulate the metric of slip_ratio. 
    
    Metric can be 'r2_score', 'mae', and 'rmse'.
    
    Default metric is 'rmse'.
    '''
    if metric == 'r2_score':
        output = r2_score(true_list, pred_list)
    elif metric == 'mae':
        output = mean_absolute_error(true_list, pred_list)
    else:
        output = np.sqrt(mean_squared_error(true_list, pred_list))
    return output    


def cal_gaussian_NLL_loss(x, mu, std, reduction=True):
    eps = 1e-6
    var = std**2
    loss = 0.5 * (torch.log(var+eps) + (x - mu)**2 / (var+eps)) + 0.5 * math.log(2 * math.pi)
    if reduction:
        loss = loss.mean()
    return loss    


def evaluate(pred_prob, one_hot_label, pred_slip_ratio, slip_ratio_label, threshold=0.5):    
    metric_dict = {}
    metric_dict['road_state'] = {}
    metric_dict['slip_ratio'] = {}
    
    ### metric for road_state classification ###
    correct = 0
    pred = torch.argmax(pred_prob, dim=1)
    
    for index, x in enumerate(one_hot_label):        
        if pred[index] == x:
            correct+=1    
    metric_dict['road_state']['accuracy'] = correct/len(pred_prob)    
    
    for i in range(2):
        cond_True = torch.zeros_like(one_hot_label).view(-1)
        cond_Pos = torch.zeros_like(one_hot_label).view(-1)
        
        mask_True = [index for index, x in enumerate(one_hot_label) if x.data == i] 
        cond_True[mask_True] = 1
    
        mask_Pos = [index for index, x in enumerate(pred_prob) if x.data[i] > 0.5]
        cond_Pos[mask_Pos] = 1

        cond_True_Pos = cond_Pos * cond_True

        positive_num = cond_Pos.sum()
        true_num = cond_True.sum()
        true_positive_num = cond_True_Pos.sum()
                    
        recall = (true_positive_num)/(true_num+1e-3)
        precision = true_positive_num /positive_num
        f1 = 2 * (precision * recall)/(precision + recall + 1e-3)
        
        if i==0:
            metric_dict['road_state']['deep_recalls'] = recall.cpu().numpy()
            metric_dict['road_state']['deep_precisions'] = precision.cpu().numpy()
            metric_dict['road_state']['deep_f1'] = f1.cpu().numpy()
        else:
            metric_dict['road_state']['shallow_recalls'] = recall.cpu().numpy()
            metric_dict['road_state']['shallow_precisions'] = precision.cpu().numpy()
            metric_dict['road_state']['shallow_f1'] = f1.cpu().numpy()
    ### metric for road_state classification ###

    ### metric for slip_ratio prediction ###
    pred_slip_ratio = pred_slip_ratio.cpu().detach().numpy()
    slip_ratio_label = slip_ratio_label.cpu().detach().numpy()
    metric_dict['slip_ratio']['r2_score'] = cal_slip_ratio_metric(
        pred_slip_ratio, slip_ratio_label, metric='r2_score'
    )
    metric_dict['slip_ratio']['mae'] = cal_slip_ratio_metric(
        pred_slip_ratio, slip_ratio_label, metric='mae')
   
    metric_dict['slip_ratio']['rmse'] = cal_slip_ratio_metric(
        pred_slip_ratio, slip_ratio_label, metric='rmse')  
    ### metric for slip_ratio prediction ###  
                        
    return metric_dict   


def print_metric(metric_dict, dataset_name, log_file):    
    ### metric for road_state classification ###
    my_print(dataset_name + 'Deep / Recall : {:3f}, Precision : {:3f}, F1: {:3f}'.format(
        metric_dict['road_state']['deep_recalls'],
        metric_dict['road_state']['deep_precisions'],
        metric_dict['road_state']['deep_f1']), 
        log_file=log_file
    )
    my_print(dataset_name + 'Shallow / Recall : {:3f}, Precision : {:3f}, F1: {:3f}'.format(
        metric_dict['road_state']['shallow_recalls'],
        metric_dict['road_state']['shallow_precisions'],
        metric_dict['road_state']['shallow_f1']),
        log_file=log_file
    )
    my_print(dataset_name + 'Accuracy : {:3f}'.format(
        metric_dict['road_state']['accuracy']), 
        log_file=log_file
    )
    ### metric for road_state classification ###
    
    ### metric for slip_ratio prediction ###  
    my_print(dataset_name + 'Slip Ratio / R2 Score : {:3f}, MAE : {:3f}, RMSE : {:3f}'.format(
        metric_dict['slip_ratio']['r2_score'],
        metric_dict['slip_ratio']['mae'],
        metric_dict['slip_ratio']['rmse']),
        log_file=log_file
    )
    ### metric for slip_ratio prediction ###  


def cal_slip_ratio(Vx, Rw):
    # Vx = (data_length=556,)
    # Rw = (data_length=556,)
    epsilon = 1e-8
    slip_array = np.zeros_like(Vx)
    idx_1 = np.where(Vx >= Rw)
    idx_2 = np.where(Vx < Rw)
    slip_array[idx_1] = (Vx[idx_1] - Rw[idx_1]) / (Vx[idx_1] + epsilon)
    slip_array[idx_2] = (Rw[idx_2] - Vx[idx_2]) / (Rw[idx_2] + epsilon)
    slip_array = np.expand_dims(slip_array, axis=-1)
    slip_array = np.clip(slip_array, a_min=0, a_max=1)
    return slip_array


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def build_mlp(
    input_dim,
    output_dim,
    hidden_dims,
    add_batchNorm=False,
    activation=nn.ReLU(),
    p_dropout=None
):
    '''
    Not include actiavtion of output layer
    '''
    network = nn.ModuleList()
    dims = [input_dim] + hidden_dims
    for in_dim_, out_dim_ in zip(dims[:-1], dims[1:]):
        network.append(
            nn.Linear(
                in_features=in_dim_,
                out_features=out_dim_
            )
        )
        if add_batchNorm:
            network.append(
                nn.BatchNorm1d(
                    num_features=out_dim_
                )
            )
        network.append(activation)
        if p_dropout is not None:
            network.append(
                nn.Dropout(p=p_dropout)
            )
    network.append(
        nn.Linear(
            in_features=hidden_dims[-1],
            out_features=output_dim
        )
    )

    return nn.Sequential(*network)  


def label_to_onehot(inputs, num_classes, device):
    '''
    Params: 

        inputs = (batch_size, num_classes)
        device : torch device

    Outputs:
    
        outputs = (batch_size, num_classes)
    '''
    
    max_indices = torch.argmax(inputs, dim=-1,keepdim=True)
    outputs = (
        max_indices == torch.arange(num_classes).reshape(1, num_classes).to(device)
    ).float()
    return outputs
    