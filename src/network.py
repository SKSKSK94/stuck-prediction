import torch
import torch.nn as nn

from utils import build_mlp, weights_init, label_to_onehot, cfg_read

class ConvLSTM(nn.Module):
    def __init__(self, 
        ConvLSTM_cfg,
    ):
        super(ConvLSTM, self).__init__()

        self.lstm_hidden_dim = ConvLSTM_cfg['lstm_hidden_dim']
        self.num_layers = ConvLSTM_cfg['lstm_num_layers']

        # Conv1d : (batch_size, input_dim, seq_length) -> (batch_size, out_channels=32, out_seq_length=9)
        # out_seq_length = (seq_length - kernel_size) / stride + 1 = (10 - 2) / 1 + 1 = 9
        self.conv = nn.Conv1d(
            in_channels=ConvLSTM_cfg['input_dim'],
            out_channels=ConvLSTM_cfg['out_channels'],
            kernel_size=ConvLSTM_cfg['kernel_size'],
            stride=ConvLSTM_cfg['stride']
        ) # (batch_size, input_dim, seq_length=10) -> (batch_size, out_channels=32, out_seq_length=9)
        
        self.lstm = nn.LSTM(
            input_size=ConvLSTM_cfg['out_channels'],
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x, device):
        
        x = self.conv(x) # x = (batch_size, input_dim, seq_length=10) -> (batch_size, out_channels=32, out_seq_length=9)
        x = x.permute(0,2,1).contiguous() # x = (batch_size, out_channels=32, out_seq_length=9) -> (batch_size, out_seq_length=9, out_channels=32)

        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_dim).to(device)

        outputs, (ht, _) = self.lstm(x, (h0, c0))

        outputs = outputs[:, -1, :].view(-1, self.lstm_hidden_dim)        

        return outputs    

######### predict Slip Ratio Using Predicted Road State #########
class SRUPRS(nn.Module):
    def __init__(self, 
        SRUPRS_cfg,
        data_keys_cfg_path
    ):
        super(SRUPRS, self).__init__()

        self.num_classes = SRUPRS_cfg['num_classes']
        self.lstm_hidden_dim = SRUPRS_cfg['ConvLSTM_cfg']['lstm_hidden_dim']
        # self.respective_mode = SRUPRS_cfg['respective_mode']
        self.data_keys = cfg_read(data_keys_cfg_path)
        self.keys_to_num_dict = {
            key : idx for idx, key in enumerate(self.data_keys)
        }
        self.non_wheel_idx_list = [
            idx for idx, key in enumerate(self.data_keys) if 'Wheel_Spd' not in key
        ]

        output_dim = 2*4 # mu, std for slip_ratio(2) for each wheel(4)
        SRUPRS_cfg['ConvLSTM_cfg']['input_dim'] = len(self.data_keys)

        self.conv_lstm = ConvLSTM(
            SRUPRS_cfg['ConvLSTM_cfg']
        )
        
        self.road_state_predictor = build_mlp(
            input_dim=self.lstm_hidden_dim,
            output_dim=self.num_classes,
            hidden_dims=SRUPRS_cfg['mlp_hidden_dims'],
            add_batchNorm=True,
            activation=nn.LeakyReLU(),
            p_dropout=SRUPRS_cfg['p_dropout']
        )
        weights_init(self.road_state_predictor)
        
        self.slip_ratio_predictor = build_mlp(
            input_dim=self.lstm_hidden_dim+self.num_classes,
            output_dim=output_dim, 
            hidden_dims=SRUPRS_cfg['mlp_hidden_dims'],
            add_batchNorm=True,
            activation=nn.LeakyReLU(),
            p_dropout=SRUPRS_cfg['p_dropout']
        )
        weights_init(self.slip_ratio_predictor)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, device, use_detach=False):
        '''
        Params: 

            x = (batch_size, input_dim, seq_length=10)
            device : torch device
            use_detach : bool
            
        Outputs:
        
            predicted_road_state = (batch_size, num_classes=2)
            slip_mu = (batch_size, 1)
            slip_std = (batch_size, 1)
        '''
        # x = (batch_size, input_dim, seq_length=10)
        
        encoding = self.conv_lstm.forward(x, device) # encoding = (batch_size, lstm_hidden_dim)

        predicted_road_state = self.road_state_predictor.forward(encoding) # predicted_road_state = (batch_size, num_classes=2)
        predicted_road_state = self.softmax(predicted_road_state)
        
        hard_road_state = label_to_onehot(
            inputs=predicted_road_state.detach() if use_detach else predicted_road_state, 
            num_classes=self.num_classes,
            device=device
        ) # hard_road_state = (batch_size, num_classes=2)
        
        slip_ratio_predictor_inputs = torch.cat([
            encoding, hard_road_state
        ], dim=1)

        outputs = self.slip_ratio_predictor.forward(slip_ratio_predictor_inputs)

        slip_mu, slip_std = self.sigmoid(outputs[:, 0:4]), outputs[:, 4:8].exp()

        return predicted_road_state, slip_mu, slip_std
    

        



        
        


        


        