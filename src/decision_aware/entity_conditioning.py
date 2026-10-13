"""Global 模型的节点与市场身份条件。"""
from __future__ import annotations

import torch
import torch.nn as nn


class EntityConditioner(nn.Module):
    """把节点和市场 embedding 同时加入历史记忆与未来 query。"""

    def __init__(self, d_model: int, num_nodes: int, num_markets: int):
        super().__init__()
        if num_nodes < 1 or num_markets < 1:
            raise ValueError("启用Global身份embedding时，节点数和市场数必须为正数")
        self.num_nodes = int(num_nodes)
        self.num_markets = int(num_markets)
        self.node_embedding = nn.Embedding(self.num_nodes, d_model)
        self.market_embedding = nn.Embedding(self.num_markets, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.node_embedding.weight, std=0.02)
        nn.init.normal_(self.market_embedding.weight, std=0.02)

    @staticmethod
    def _ids(batch: dict, name: str, batch_size: int, device) -> torch.Tensor:
        if name not in batch:
            raise ValueError(f"Global模型batch缺少{name}")
        values = batch[name]
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        values = values.to(device=device, dtype=torch.long).reshape(-1)
        if len(values) != batch_size:
            raise ValueError(
                f"{name}必须每个样本一个ID；期望{batch_size}个，实际{len(values)}个"
            )
        return values

    def identity(
        self,
        batch: dict,
        batch_size: int,
        device,
    ) -> torch.Tensor:
        """返回每个样本的节点+市场身份向量。"""
        node_ids = self._ids(batch, "node_id", batch_size, device)
        market_ids = self._ids(batch, "market_id", batch_size, device)
        if bool((node_ids < 0).any()) or bool((node_ids >= self.num_nodes).any()):
            raise ValueError("node_id超出正式节点映射范围")
        if bool((market_ids < 0).any()) or bool(
            (market_ids >= self.num_markets).any()
        ):
            raise ValueError("market_id超出正式市场映射范围")
        return self.norm(
            self.node_embedding(node_ids) + self.market_embedding(market_ids)
        )

    def forward(
        self,
        batch: dict,
        memory: torch.Tensor,
        query_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory.shape[0]
        identity = self.identity(batch, batch_size, memory.device)
        return (
            memory + identity.unsqueeze(1),
            query_context + identity.unsqueeze(1),
        )
