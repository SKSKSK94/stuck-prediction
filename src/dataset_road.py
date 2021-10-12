#%%
import numpy as np

from scipy.io import loadmat
from glob import glob

import torch
from torch.utils.data import Dataset

from utils import cal_slip_ratio, cfg_read

# Deep/Shallow -> 0/1

# mud road      : [0, 1]
# sand road     : [1, 0]
# deep label    : 0
# shallow label : 1

class stuckDataset(Dataset):
    '''
    info :
        mud road      : [0, 1]
        sand road     : [1, 0]
        deep label    : 0
        shallow label : 1

    output : 
        data = (feature_dim, num_sequence)
        pos = (2,)
        road_state_label = (1,) : deep or shallow
        slip_ratio_label = (4) : 0 ~ 1

        road_info=True, FFT_bool=True   -> feature_dim = 10+2+10
        road_info=True, FFT_bool=False  -> feature_dim = 10+2
        road_info=False, FFT_bool=False -> feature_dim = 10
    '''
    def __init__(self, 
        road_info, 
        train_mode, 
        data_keys_cfg_path,
        test_file=None, 
        FFT_bool=False, 
        root='/home/mmc-server3/Server/dataset/Stuck/data/',
        num_sequence=10
    ):
        self.road_info = road_info
        self.train_mode = train_mode
        self.num_sequence = num_sequence
        self.FFT_bool = FFT_bool

        self.root = root

        self.sensor_data=[]
        self.road_state_label_data=[]
        self.slip_ratio_label_data=[]
        self.pos_data = []

        if self.train_mode:
            train_list = sorted(glob(self.root + 'train/*/*.mat'))
        else:            
            assert test_file is not None
            test_list = sorted(glob(self.root + test_file))        
        
        self.data_keys = cfg_read(data_keys_cfg_path)

        print('''
        Used data keys are {}
        '''.format(self.data_keys))

        # self.data_keys = ['CAN_DBC_HMC__AWD_01_20ms__AWD_Tq',           # Nm
        #                   'CAN_DBC_HMC__EMS_01_10ms__Eng_rpm',          # rpm
        #                   'CAN_DBC_HMC__YRS_01_10ms__Long_Acc',         # g
        #                   'CAN_DBC_HMC__EMS_02_10ms__APS',              # Nm
        #                 #   'CAN_DBC_HMC__YRS_01_10ms__Yaw_Rate',         # rad/s
        #                 #   'CAN_DBC_HMC__YRS_01_10ms__Lat_Acc',           # g
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FL',     # kph
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FR',     # kph  
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_RL',     # kph  
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_RR',     # kph
        #                   'RT2502_Stuck__VelocityLevel__VelForward']    # m/s -> kph    

                          
        # self.data_keys = ['CAN_DBC_HMC__AWD_01_20ms__AWD_Tq',           # Nm
        #                   'CAN_DBC_HMC__EMS_01_10ms__Eng_rpm',          # rpm
        #                   'CAN_DBC_HMC__YRS_01_10ms__Long_Acc',         # g
        #                   'CAN_DBC_HMC__EMS_01_10ms__Crct_EngTq',       # Nm
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FL',     # kph
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FR',     # kph  
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_RL',     # kph  
        #                   'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_RR',     # kph
        #                   'CAN_DBC_HMC__YRS_01_10ms__Yaw_Rate',         # rad/s
        #                   'CAN_DBC_HMC__YRS_01_10ms__Lat_Acc',           # g
        #                   'RT2502_Stuck__VelocityLevel__VelForward']    # m/s -> kph  

        self.label_keys = ['RT2502_Stuck__PosLocal__PosLocalX',
                           'RT2502_Stuck__PosLocal__PosLocalY']        

        if self.train_mode:
            self.make_dataset(train_list)
        else:
            self.make_dataset(test_list)


    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, index):
        
        data = torch.FloatTensor(self.sensor_data[index]).permute(1,0) # (num_sequence, feature_dim) -> (feature_dim, num_sequence)
        pos = torch.FloatTensor(self.pos_data[index]) # (pos_dim=2,)
        road_state_label = torch.LongTensor([self.road_state_label_data[index]]) # (1,)
        slip_ratio_label = torch.FloatTensor(self.slip_ratio_label_data[index]) # (1,)

        return data, pos, road_state_label, slip_ratio_label

    def make_dataset(self, data_list):
        print('Data Stacking!')
        
        # mud road      : [0, 1]
        # sand road     : [1, 0]
        # deep label    : 0
        # shallow label : 1

        for data in data_list:
            # If Use Road info 
            road = None
            if self.road_info:
                if "mud" in data:
                    road = [0,1]
                elif "sand" in data:
                    road = [1,0]

            # For Labeling
            if "deep" in data:
                road_state_label = 0
            elif "shallow" in data:
                road_state_label = 1

            mat = loadmat(data)
            value_list = []
            pos_list = []

            for key in self.data_keys:
                value = mat[key][:,0]
                if key == 'RT2502_Stuck__VelocityLevel__VelForward':
                    value = value * 3.6 # m/s to km/h
                mask = np.arange(0, len(value), 10) # Here 10 is downsampling coeff from 10ms to 100ms
                value_list.append(value[mask])                

            value_list = np.transpose(value_list) 
            # Then, value_list = (data_length=556, data_keys_num=11=feature_dim(10)+slip_keys_num)

            for key in self.label_keys:
                value = mat[key][:,0]
                mask = np.arange(0, len(value), 10) # Here 10 is downsampling coeff from 10ms to 100ms
                pos_list.append(value[mask])

            pos_list = np.transpose(pos_list)
            # Then, pos_list = (data_length=556, label_keys_num=2)

            keys_to_num_dict = {
                key : idx for idx, key in enumerate(self.data_keys)
            }

            ### making slip_ratio_labels for 4 wheels respectively
             
            slip_FL = cal_slip_ratio(
                Vx=value_list[:, keys_to_num_dict['CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FL']],
                Rw=value_list[:, keys_to_num_dict['RT2502_Stuck__VelocityLevel__VelForward']]
            )
            slip_FR = cal_slip_ratio(
                Vx=value_list[:, keys_to_num_dict['CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FR']],
                Rw=value_list[:, keys_to_num_dict['RT2502_Stuck__VelocityLevel__VelForward']]
            )
            slip_RL = cal_slip_ratio(
                Vx=value_list[:, keys_to_num_dict['CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_RL']],
                Rw=value_list[:, keys_to_num_dict['RT2502_Stuck__VelocityLevel__VelForward']]
            )
            slip_RR = cal_slip_ratio(
                Vx=value_list[:, keys_to_num_dict['CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_RR']],
                Rw=value_list[:, keys_to_num_dict['RT2502_Stuck__VelocityLevel__VelForward']]
            )

            # slip_FL = cal_slip_ratio(value_list[:, -5], value_list[:, -1])
            # slip_FR = cal_slip_ratio(value_list[:, -4], value_list[:, -1])
            # slip_RL = cal_slip_ratio(value_list[:, -3], value_list[:, -1])
            # slip_RR = cal_slip_ratio(value_list[:, -2], value_list[:, -1])
            # Then, slip_xx = (data_length=556, 1)
            slip_ratio_label = np.concatenate([slip_FL, slip_FR, slip_RL, slip_RR], axis=1)
            # Then, slip_ratio_label = (data_length=556, #num_wheels=4)
            
            # value_list = (data_length=556, data_keys_num=11=feature_dim(10)+slip_keys_num(1))
            # pos_list = (data_length=556, label_keys_num=2)
            # road_state_label = 0(deep) or 1(shallow)
            # slip_ratio_label = (data_length=556, #num_wheels=4)
            # road = [0, 1](mud road) or [1, 0](sand road)
            # self.make_sequence(value_list, pos_list, road_state_label, road)

            value_list = self.normalization(value_list)
            # value_list = self.normalization(value_list[:, :-1])
            # Then, value_list = (data_length=556, feature_dim(10))
            
            self.make_sequence(value_list, pos_list, road_state_label, slip_ratio_label, road)
            

        print('Data Stacking finished!\n')

    def normalization(self, data):        
        mean = np.tile(np.mean(data, axis=0, keepdims=True), (data.shape[0], 1))         
        std = np.tile(np.std(data, axis=0, keepdims=True), (data.shape[0], 1))
        data = (data - mean)/std

        data = np.nan_to_num(data)
        
        return data
    
    def FFT(self, data):    
        strength = np.fft.fft(data)/len(data)
        strength = abs(strength)
        x = np.argmax(strength, axis=0)
        frequency = np.fft.fftfreq(10, 0.001)
        freq = frequency[x]
        # data = torch.from_numpy(abs(freq)).float()
          
        return np.tile(np.array([abs(freq)]), (10, 1))
        

    def make_sequence(self, data, pose, road_state_label, slip_ratio_label, road):
        # data = (data_length=556, data_keys_num=10=feature_dim)
        # pose = (data_length=556, label_keys_num=2)
        # road_state_label = 0(deep) or 1(shallow)
        # slip_ratio_label = (data_length=556, #num_wheels=4)
        # road = [0, 1](mud road) or [1, 0](sand road)

        road_feature = np.tile(road, (self.num_sequence, 1)) # (num_sequence=10, 2)
        for i in range(len(data) - self.num_sequence):

            self.pos_data.append(pose[i+self.num_sequence,:])
            self.road_state_label_data.append(road_state_label)
            self.slip_ratio_label_data.append(slip_ratio_label[i+self.num_sequence,:])

            if self.road_info:
                
                feature = data[i:i+self.num_sequence,:]
                road_feature = np.tile(road, (self.num_sequence, 1))
                feature = np.concatenate([feature, road_feature], 1)
                                
                if self.FFT_bool:
                    new_feature=self.FFT(data[i:i+self.num_sequence,:])
                    feature= np.concatenate([feature, new_feature], 1)

            else:
                if self.FFT_bool:
                    new_feature=self.FFT(data[i:i+self.num_sequence,:])
                    feature= np.concatenate([data[i:i+self.num_sequence,:], new_feature], 1)
        
                else:
                    feature=data[i:i+self.num_sequence,:]
            
            # final_data_len = (data_length=556) - (num_sequence=10) + 1 = 547
            #       FFT     road_info                                                       data_shape
            #        O         O        sensor_data=(547, num_sequence=10, feautre_dim(=10)+road_feature_dim(=2)+feature_dim(=10)), road_state_label_data=(547,), slip_ratio_label_data=(547, 4), pos_data=(547, 2)
            #        O         X        sensor_data=(547, num_sequence=10, feautre_dim(=10)+feature_dim(=10)),                      road_state_label_data=(547,), slip_ratio_label_data=(547, 4), pos_data=(547, 2)
            #        X         O        sensor_data=(547, num_sequence=10, feautre_dim(=10)+road_feature_dim(=2)),                  road_state_label_data=(547,), slip_ratio_label_data=(547, 4), pos_data=(547, 2)
            #        X         X        sensor_data=(547, num_sequence=10, feautre_dim(=10)),                                       road_state_label_data=(547,), slip_ratio_label_data=(547, 4), pos_data=(547, 2)
            
            self.sensor_data.append(feature) 

# #%%
# from utils import cfg_read
# data_keys = cfg_read('cfg/data_keys.json')
# keys_to_num_dict = {
#                 key : idx for idx, key in enumerate(data_keys)
#             }

# non_wheel_idx_list = [idx for idx, key in enumerate(data_keys) if 'Wheel_Spd' not in key]
# # for idx, key in enumerate(data_keys):
# #     if 'Wheel_Spd' in key:
# #         continue
# #     else:
# #         non_wheel_idx_list.append()
        
        
# # keys_to_num_dict['CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FL']

# 'CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FL'.find('FL')
# non_wheel_idx_list

# # #%%
# # import torch
# # x = torch.zeros((64, 9))
# # x_FL = torch.cat(
# #     [x[:, idx].clone().reshape(-1, 1) for idx in non_wheel_idx_list]
# #     + [x[:, keys_to_num_dict['CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FL']].clone().reshape(-1, 1)]
# # , dim=1)
# # # a = [x[:, idx].clone().reshape(-1, 1) for idx in non_wheel_idx_list] + [x[:, keys_to_num_dict['CAN_DBC_HMC__WHL_01_10ms__Wheel_Spd_FL']].reshape(-1, 1)]
# # x_FL.shape
# # # a[0].shape