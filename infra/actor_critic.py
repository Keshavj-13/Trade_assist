"""Actor-critic RL helper with wallet-aware masking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import (
    RL_ENTROPY_COEF,
    RL_HIDDEN_SIZE,
    RL_LEARNING_RATE,
    RL_MODEL_FILE,
    RL_VALUE_LOSS_COEF,
)


def _safe_div(num: float, denom: float) -> float:
    try:
        return num / denom
    except ZeroDivisionError:
        return 0.0


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class ActionDecision:
    action_id: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    state: torch.Tensor
    mask: torch.Tensor


class RLAgent:
    ACTIONS = ["HOLD", "BUY", "SELL", "ADJUST_STOP", "PARTIAL_EXIT"]

    def __init__(self, input_dim: int):
        self.path = Path(RL_MODEL_FILE)
        self.actor = PolicyNetwork(input_dim, RL_HIDDEN_SIZE, len(self.ACTIONS))
        self.critic = ValueNetwork(input_dim, RL_HIDDEN_SIZE)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=RL_LEARNING_RATE,
        )
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return
        payload = torch.load(self.path)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])

    def save(self) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.path,
        )

    def build_state(
        self,
        features: Dict,
        position: Optional[Dict],
        wallet: float,
    ) -> torch.Tensor:
        if position is None:
            position = {}
        price = features.get("price", 0.0)
        vwap = features.get("vwap", price)
        avg_volume = features.get("avg_volume", 0.0) or 1.0
        percent_from_low = features.get("pct_from_low", 0.0)
        percent_from_high = features.get("pct_from_high", 0.0)
        atr_pct = features.get("atr_pct", 0.0)
        rsi = features.get("rsi", 0.0)
        vol_spike = features.get("vol_spike", 0.0)

        position_qty = position.get("qty", 0.0)
        entry_price = position.get("price") or 0.0
        position_flag = 1.0 if position_qty > 0 else 0.0
        entry_delta = _safe_div(price - entry_price, entry_price) if entry_price else 0.0
        unrealized_pnl = entry_delta
        wallet_util = _safe_div(price * position_qty, wallet if wallet else 1.0)

        state = [
            price / 1000.0,
            _safe_div(vwap, price) if price else 0.0,
            rsi / 100.0,
            atr_pct / 5.0,
            vol_spike / 5.0,
            percent_from_low / 100.0,
            percent_from_high / 100.0,
            position_flag,
            entry_delta,
            unrealized_pnl,
            wallet_util,
            _safe_div(wallet, wallet + 1000.0),
        ]
        return torch.tensor(state, dtype=torch.float32)

    def build_mask(
        self,
        allow_buy: bool,
        wallet: float,
        price: float,
        position_qty: float,
    ) -> torch.Tensor:
        mask = torch.ones(len(self.ACTIONS), dtype=torch.bool)
        if not allow_buy or wallet <= price:
            mask[1] = False
        if position_qty <= 0:
            mask[2] = False
            mask[3] = False
            mask[4] = False
        return mask

    def select_action(self, state: torch.Tensor, mask: torch.Tensor) -> ActionDecision:
        logits = self.actor(state)
        masked_logits = logits.masked_fill(~mask, -1e8)
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        value = self.critic(state)
        return ActionDecision(
            action_id=int(action.item()),
            log_prob=dist.log_prob(action),
            entropy=dist.entropy(),
            value=value,
            state=state,
            mask=mask,
        )

    def action_label(self, action_id: int) -> str:
        return self.ACTIONS[action_id]

    def update(self, decision: ActionDecision, reward: float) -> None:
        advantage = reward - decision.value
        actor_loss = -decision.log_prob * advantage.detach() - RL_ENTROPY_COEF * decision.entropy
        critic_loss = advantage.pow(2)
        loss = actor_loss + RL_VALUE_LOSS_COEF * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer.step()
        self.save()
