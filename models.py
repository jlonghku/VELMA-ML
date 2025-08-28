import torch
import torch.nn as nn

class SurrogateModel(nn.Module):
    def __init__(
        self,
        time_series_input_size=2,
        lstm_hidden_size=256,
        param_size=1,
        shared_lstm_layers=2,
        branch_lstm_hidden_size=256,
        branch_lstm_layers=4,
        fc_hidden_dims=(256, 128),
        branch_fc_dims=(256, 128),
        output_size=1,
        use_dropout=True,
        dropout_prob=0.2,
    ):
        super().__init__()
        # shared LSTM
        self.shared_lstm = nn.LSTM(
            input_size=time_series_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=shared_lstm_layers,
            batch_first=True,
            dropout=dropout_prob if (use_dropout and shared_lstm_layers > 1) else 0.0,
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

        # shared FC: [LSTM_out | params] -> features
        self.param_size = param_size
        in_dim = lstm_hidden_size + param_size
        layers = []
        for d in fc_hidden_dims:
            layers += [nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout_prob) if use_dropout else nn.Identity()]
            in_dim = d
        self.fc_layers = nn.Sequential(*layers)

        # first output head
        self.first_output = nn.Linear(fc_hidden_dims[-1], 1)

        # other outputs: per-branch (LSTM + FC)
        self.output_branches = nn.ModuleList()
        if output_size > 1:
            for _ in range(output_size - 1):
                lstm = nn.LSTM(
                    input_size=fc_hidden_dims[-1] + 1,  # concat first output
                    hidden_size=branch_lstm_hidden_size,
                    num_layers=branch_lstm_layers,
                    batch_first=True,
                    dropout=dropout_prob if (use_dropout and branch_lstm_layers > 1) else 0.0,
                )
                in_b = branch_lstm_hidden_size
                b_layers = []
                for d in branch_fc_dims:
                    b_layers += [nn.Linear(in_b, d), nn.ReLU(), nn.Dropout(dropout_prob) if use_dropout else nn.Identity()]
                    in_b = d
                b_layers.append(nn.Linear(branch_fc_dims[-1], 1))
                self.output_branches.append(nn.ModuleDict({"lstm": lstm, "fc": nn.Sequential(*b_layers)}))

        self.output_size = output_size

    def forward(self, time_series, velma_params):
        B, T = time_series.size(0), time_series.size(1)

        # shared LSTM -> norm
        h, _ = self.shared_lstm(time_series)
        h = self.layer_norm(h)

        # concat params (broadcast to time dimension)
        p = velma_params.unsqueeze(1).expand(B, T, self.param_size)
        x = torch.cat([h, p], dim=2)

        # shared FC (time-wise)
        x = self.fc_layers(x.reshape(B * T, -1)).reshape(B, T, -1)

        # first output
        y1 = self.first_output(x)
        outs = [y1]

        # other outputs (each branch gets [x | y1])
        for br in self.output_branches:
            z = torch.cat([x, y1], dim=2)
            z, _ = br["lstm"](z)
            z = br["fc"](z.reshape(B * T, -1)).reshape(B, T, 1)
            outs.append(z)

        return torch.cat(outs, dim=2)
