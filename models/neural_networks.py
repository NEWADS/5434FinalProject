import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):

    def __init__(self, layers: list, n_class: int = 3):
        super(ANN, self).__init__()
        linears = [nn.Linear(250, layers[0]), nn.Tanh()]
        # pay attention to here
        for i in range(len(layers) - 1):
            linears.append(nn.Linear(layers[i], layers[i + 1]))
            linears.append(nn.Tanh())
        self.features = nn.Sequential(*linears)
        self.out = nn.Linear(layers[-1], n_class)

    def forward(self, x):
        out = self.features(x)
        logits = self.out(out)
        return logits


class IdentityBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, **kwargs):
        super(IdentityBlock1D, self).__init__()
        self.down_sample = down_sample
        stride = 2 if down_sample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        # The first conv layer has to follow the original paper so as to make sure feature map downsampled correctly.
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.non_linear1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.non_linear2 = nn.ReLU()
        if down_sample:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=2),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # obviously, up-conv layer must be applied before identity block in decoder block.
        if self.down_sample:
            identity = self.residual(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.non_linear1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.non_linear2(out)

        return out


class ResNet1D(nn.Module):
    def _make_stage(self, in_c, out_c, down_s=True, num_l=2):
        stage_list = [IdentityBlock1D(in_c, out_c, down_s)]
        for _ in range(1, num_l):
            stage_list.append(IdentityBlock1D(out_c, out_c))
        return nn.Sequential(*stage_list)

    def _configure_layers(self, stage=18):
        if stage == 18:
            return [2, 2, 2, 2]
        else:
            return [3, 4, 6, 3]

    def __init__(self, in_channels: int, layers: list, n_class: int = 3, stages: int = 18):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=layers[0], kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm1d(layers[0])
        self.non_linear1 = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        config = self._configure_layers(stages)
        self.stage1 = self._make_stage(layers[0], layers[0], down_s=False, num_l=config[0])
        self.stage2 = self._make_stage(layers[0], layers[1], down_s=True, num_l=config[1])
        self.stage3 = self._make_stage(layers[1], layers[2], down_s=True, num_l=config[2])
        self.stage4 = self._make_stage(layers[2], layers[3], down_s=True, num_l=config[3])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(layers[3], n_class)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.non_linear1(x)
        x = self.max_pool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class GRU(nn.Module):
    # please note that GRU does not accept one_hot data.
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 2, n_class: int = 3):
        # embedding_dim refers to the dimension of the embedding vector to better encode the input vector.
        """
        Pay attention:
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        """
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(4, embedding_dim)  # ATGC
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=0.2,
                          batch_first=True)
        self.fc = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.Linear(hidden_dim, n_class)
        )

    def forward(self, x):
        # x : [batch, sequence]
        embeds = self.embedding(x)
        # embeds : [batch, sequence, embedding_dim]
        r_out, _ = self.gru(embeds, None)
        # r_out : [batch, sequence, hidden_dim]
        # suggestion from poncey.
        # out = self.fc(r_out[:, -1, :])
        r_out = torch.mean(r_out, dim=1)
        out = self.fc(r_out)
        # out : [batch, sequence, output_dim]
        return out
