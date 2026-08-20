"""Implementação explícita da U-Net++ em PyTorch.

Diferença fundamental entre U-Net e U-Net++
--------------------------------------------
Na U-Net original, cada nó do decoder recebe skip connections
diretamente dos nós do encoder no mesmo nível de profundidade.

Na U-Net++ (Zhou et al., 2018 — "UNet++: A Nested U-Net Architecture
for Medical Image Segmentation"), os nós do decoder são organizados
numa grade x^{i,j} onde:
    - i = profundidade (0 = encoder/entrada, L = bottleneck)
    - j = posição na grade horizontal (0 = encoder original)

Cada nó x^{i,j} recebe:
    1. Upsampling de x^{i+1, j-1} (caminho principal de decoder)
    2. TODOS os nós x^{i, 0..j-1} do mesmo nível de profundidade
       (nested skip connections — a principal inovação)

Isso permite que as skip connections transmitam features
progressivamente re-processadas em vez de features brutas do encoder,
o que reduz o gap semântico entre encoder e decoder.

Deep supervision
----------------
Com deep_supervision=True, a rede produz uma saída por coluna do
decoder (j=1, 2, ..., depth). Durante o treinamento, a loss é
calculada sobre todas as saídas e as médias. Na inferência, apenas
a última saída (j=depth) é usada, que agrega todas as nested connections.

Notação dos atributos
---------------------
self.nodes[i][j] = ConvBlock do nó x^{i,j}
    i ∈ {0, ..., depth}  (0=topo/alta-resolução, depth=bottleneck)
    j ∈ {0, ..., depth-i} (j=0 é o encoder; j>0 são nós do decoder)

Exemplo para depth=4, base_channels=32:
    nodes[0][0]: encoder L0  → 32 ch   (256×256)
    nodes[1][0]: encoder L1  → 64 ch   (128×128)
    nodes[2][0]: encoder L2  → 128 ch  (64×64)
    nodes[3][0]: encoder L3  → 256 ch  (32×32)
    nodes[4][0]: bottleneck  → 512 ch  (16×16)

    nodes[0][1]: decoder x^{0,1} = up(x^{1,0}) + x^{0,0}
    nodes[0][2]: decoder x^{0,2} = up(x^{1,1}) + x^{0,0} + x^{0,1}
    nodes[0][3]: decoder x^{0,3} = up(x^{1,2}) + x^{0,0..2}
    nodes[0][4]: decoder x^{0,4} = up(x^{1,3}) + x^{0,0..3}
    etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import ConvBlock, DownBlock, UpBlock


class UNetPlusPlus(nn.Module):
    """U-Net++ para segmentação binária de imagens.

    Parameters
    ----------
    in_channels:
        Canais de entrada (default 3: occupancy, density, distance).
    out_channels:
        Canais de saída (default 1: máscara binária de fileiras).
    base_channels:
        Número de filtros no primeiro nível do encoder. Cada nível
        subsequente dobra (32 → 64 → 128 → 256 → 512).
    depth:
        Número de níveis de downsampling (default 4). Com depth=4 e
        image_size=256, o bottleneck terá resolução 16×16.
    deep_supervision:
        Se True, produz saídas intermediárias durante o treino.
        Na inferência, retorna apenas a saída final.
    dropout_p:
        Dropout aplicado nos blocos intermediários para regularização.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        deep_supervision: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.deep_supervision = deep_supervision

        # Número de canais em cada nível do encoder
        # nivel 0: base_channels, nivel 1: 2×, ..., nivel depth: 2^depth ×
        self.channels: list[int] = [base_channels * (2**i) for i in range(depth + 1)]

        # -------------------------------------------------------------------
        # ENCODER (coluna j=0): nós x^{i,0}
        # -------------------------------------------------------------------
        # nodes[0][0] = primeiro bloco conv (sem pooling)
        # nodes[i][0] para i>0 = pool + conv (DownBlock)
        self.nodes: nn.ModuleList = nn.ModuleList()

        for i in range(depth + 1):
            row: nn.ModuleList = nn.ModuleList()
            if i == 0:
                # Bloco de entrada: apenas conv dupla, sem pool
                row.append(ConvBlock(in_channels, self.channels[0], dropout_p=dropout_p))
            else:
                # Encoder: pool + conv
                row.append(DownBlock(self.channels[i - 1], self.channels[i], dropout_p=dropout_p))
            self.nodes.append(row)

        # -------------------------------------------------------------------
        # DECODER (colunas j=1..depth): nós x^{i,j}
        # -------------------------------------------------------------------
        # Para cada coluna j (1 a depth), e para cada linha i (0 a depth-j):
        #   x^{i,j} recebe:
        #     - upsampling de x^{i+1, j-1}  → channels[i+1] canais
        #     - concatenação de x^{i,0..j-1} → j × channels[i] canais
        #   e produz → channels[i] canais
        #
        # O UpBlock é inicializado com:
        #   in_channels = channels[i+1]  (do upsampling)
        #   skip_channels = j × channels[i]  (nested skip connections)
        #   out_channels = channels[i]

        self.up_blocks: nn.ModuleList = nn.ModuleList()

        for j in range(1, depth + 1):          # coluna do decoder
            col: nn.ModuleList = nn.ModuleList()
            for i in range(depth - j + 1):     # linha (nível de profundidade)
                skip_channels = j * self.channels[i]   # j nós × channels[i]
                col.append(
                    UpBlock(
                        in_channels=self.channels[i + 1],
                        skip_channels=skip_channels,
                        out_channels=self.channels[i],
                    )
                )
            self.up_blocks.append(col)

        # -------------------------------------------------------------------
        # Cabeças de saída (segmentação)
        # -------------------------------------------------------------------
        # Com deep_supervision, há uma cabeça para cada coluna do decoder.
        # Sem deep_supervision, apenas a última coluna tem cabeça.
        if deep_supervision:
            self.output_heads = nn.ModuleList(
                [nn.Conv2d(self.channels[0], out_channels, kernel_size=1) for _ in range(depth)]
            )
        else:
            self.output_heads = nn.ModuleList(
                [nn.Conv2d(self.channels[0], out_channels, kernel_size=1)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass da U-Net++.

        Parameters
        ----------
        x: Tensor (B, in_channels, H, W)

        Returns
        -------
        - Se deep_supervision=False: Tensor (B, out_channels, H, W) — logits
        - Se deep_supervision=True:  list[Tensor] com uma saída por coluna
          de decoder. Durante o treino, calcule a loss sobre todas e faça média.
          Durante a inferência, use apenas a última.
        """
        # -------------------------------------------------------------------
        # Passo 1: Forward do encoder
        # feature_grid[i][0] = x^{i,0} (saída do encoder no nível i)
        # -------------------------------------------------------------------
        feature_grid: list[list[torch.Tensor]] = [[] for _ in range(self.depth + 1)]

        current = x
        for i in range(self.depth + 1):
            # nodes[i] é uma ModuleList; nodes[i][0] é o primeiro (e único) bloco do encoder
            current = self.nodes[i][0](current)
            feature_grid[i].append(current)  # feature_grid[i][0] = x^{i,0}

        # -------------------------------------------------------------------
        # Passo 2: Forward do decoder — preenche as colunas j=1..depth
        # feature_grid[i][j] = x^{i,j}
        # -------------------------------------------------------------------
        outputs: list[torch.Tensor] = []

        for j in range(1, self.depth + 1):         # coluna j
            for i in range(self.depth - j + 1):    # linha i
                # Nó de upsampling: vem de x^{i+1, j-1}
                x_down = feature_grid[i + 1][j - 1]

                # Nested skip connections: x^{i,0}, x^{i,1}, ..., x^{i,j-1}
                # (todos os nós anteriores na mesma linha)
                skips = [feature_grid[i][k] for k in range(j)]

                # up_blocks[j-1][i] corresponde ao bloco x^{i,j}
                node_output = self.up_blocks[j - 1][i](x_down, *skips)
                feature_grid[i].append(node_output)

            # Deep supervision: pegar a saída do nível 0 após cada coluna j
            if self.deep_supervision or j == self.depth:
                head_idx = j - 1 if self.deep_supervision else 0
                out = self.output_heads[head_idx](feature_grid[0][j])
                outputs.append(out)

        if self.deep_supervision:
            return outputs  # lista de tensores para cálculo de loss
        else:
            return outputs[0]  # tensor único (B, 1, H, W)
