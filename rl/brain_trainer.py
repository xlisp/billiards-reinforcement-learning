"""桌球大脑模型自训练框架 (PyTorch)

把 README 的核心思想"用小模型训练你大脑这个大模型"实现成可运行的神经网络系统：

    [小模型 SkillExpert] × N  →  [大脑模型 BrainModel = MoE]  ⇄  [对抗者 Adversary]
              ↑                            ↑                          ↓
         各自定向训练               门控蒸馏整合各专家           生成专攻弱点的难题

四个组成部分
------------
1. **SkillExpert（小模型）**：每个 expert = 一个专项小网络（直线 / 薄球 / 反袋 / 解斯诺克 / 走位），
   只在自己擅长的局面上训练——参数少、专注、好训。

2. **BrainModel（大模型 = 大脑）**：Mixture-of-Experts 结构——
   - 门控网络 (gating) 看局面决定调用哪个 expert（或加权混合）
   - 蒸馏阶段：各 expert 教大脑"什么局面该怎么打"
   - 这就是"用小模型训练大模型"

3. **Adversary（对抗者）**：自我设计的小生成网络，专挑大脑弱点出难题——
   maximize_φ  E_z [ Loss( BrainModel( G_φ(z) ) ) ]
   "小模型打败大模型训练"——大脑被强迫提升弱项。

4. **Curriculum（阶段化组合）**：
   Stage 1 各 expert 单独预训练 → Stage 2 蒸馏到大脑 → Stage 3 对抗微调 →
   反复诊断弱 expert → 重训弱 expert → 重蒸馏 → ……

跑法
----
    python brain_trainer.py                 # 默认完整自训练循环
    python brain_trainer.py --epochs 50
    python brain_trainer.py --device cpu    # 或 cuda / mps
    python brain_trainer.py --no-adversary  # 关闭对抗训练对照
    python brain_trainer.py --eval-only --ckpt brain.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CKPT_PATH = Path(__file__).with_name("brain.pt")
TABLE_W, TABLE_H = 2.54, 1.27   # 标准九球台尺寸（米）
BALL_R = 0.028
POCKETS = torch.tensor([
    [0.0, 0.0], [TABLE_W / 2, 0.0], [TABLE_W, 0.0],
    [0.0, TABLE_H], [TABLE_W / 2, TABLE_H], [TABLE_W, TABLE_H],
])  # 6 个袋口

SCENARIO_DIM = 10   # cue(2) + target(2) + pocket(2) + obstacle(2) + has_obstacle(1) + scenario_type(1)
ACTION_DIM = 2      # angle, force


# ---------------------------------------------------------------------------
# 1. 场景生成器 —— 合成训练数据 (scenario, optimal_action)
# ---------------------------------------------------------------------------
SCENARIO_TYPES = ["straight", "thin_cut", "bank_shot", "snooker_escape", "position"]


def _ghost_ball(target: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
    """鬼球法：从目标球沿 (袋口→目标球) 方向反延伸一个球径。"""
    direction = target - pocket
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return target + direction / norm * (2 * BALL_R)


def _optimal_action(cue: torch.Tensor, target: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
    """几何最优动作 (angle_norm, force_norm)，用作监督训练的 ground truth。"""
    ghost = _ghost_ball(target, pocket)
    delta = ghost - cue
    angle = torch.atan2(delta[..., 1], delta[..., 0]) / math.pi  # → [-1, 1]
    distance = (target - cue).norm(dim=-1)
    # 距离越远要越大力，且加上目标→袋口距离
    force = torch.tanh(0.4 * (distance + (pocket - target).norm(dim=-1)))
    return torch.stack([angle, force], dim=-1)


def sample_scenarios(n: int, scenario_type: str | None = None,
                     seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 n 个 (scenario, optimal_action) 对。

    scenario_type 限定只生成某类——用于训练专项 expert。None = 混合。
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    cue = torch.rand(n, 2, generator=g) * torch.tensor([TABLE_W, TABLE_H])
    target = torch.rand(n, 2, generator=g) * torch.tensor([TABLE_W, TABLE_H])
    # 防止 cue 和 target 重叠
    target = target + (target - cue).sign() * 0.1

    # 每个场景挑一个最近的袋口
    pocket_idx = torch.argmin(torch.cdist(target, POCKETS), dim=1)
    pocket = POCKETS[pocket_idx]

    obstacle = torch.rand(n, 2, generator=g) * torch.tensor([TABLE_W, TABLE_H])
    has_obstacle = torch.zeros(n)
    type_idx = torch.zeros(n)

    types = [scenario_type] * n if scenario_type else \
            [SCENARIO_TYPES[i % len(SCENARIO_TYPES)] for i in range(n)]

    for i, t in enumerate(types):
        type_idx[i] = SCENARIO_TYPES.index(t)
        if t == "straight":
            # 让 cue, target, pocket 接近共线
            mid = (cue[i] + pocket[i]) / 2
            target[i] = mid + torch.randn(2, generator=g) * 0.05
        elif t == "thin_cut":
            # 让母球和袋口在目标球同侧但成大角度
            offset = torch.randn(2, generator=g)
            offset = offset / offset.norm() * 0.4
            cue[i] = target[i] + offset
        elif t == "bank_shot":
            # 目标球贴库
            if torch.rand(1, generator=g).item() > 0.5:
                target[i, 1] = BALL_R + 0.02
            else:
                target[i, 1] = TABLE_H - BALL_R - 0.02
        elif t == "snooker_escape":
            # 障碍球放在 cue→target 之间
            t_pos = 0.4 + 0.2 * torch.rand(1, generator=g).item()
            obstacle[i] = cue[i] * (1 - t_pos) + target[i] * t_pos
            obstacle[i] += torch.randn(2, generator=g) * 0.03
            has_obstacle[i] = 1.0
        elif t == "position":
            # 位置球：和 straight 类似但走位要求高（这里特征上加权重）
            pass

    # 归一化到 [-1, 1]
    def norm_pos(p: torch.Tensor) -> torch.Tensor:
        return p / torch.tensor([TABLE_W, TABLE_H]) * 2 - 1

    actions = _optimal_action(cue, target, pocket)
    scenario = torch.cat([
        norm_pos(cue), norm_pos(target), norm_pos(pocket), norm_pos(obstacle),
        has_obstacle.unsqueeze(-1), (type_idx / len(SCENARIO_TYPES)).unsqueeze(-1),
    ], dim=-1)
    return scenario, actions


# ---------------------------------------------------------------------------
# 2. SkillExpert —— 小模型，每个专攻一类场景
# ---------------------------------------------------------------------------
class SkillExpert(nn.Module):
    """一个 expert = 一个专项小 MLP。参数量很小、训练容易、容易调优。"""

    def __init__(self, name: str, hidden: int = 32):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(SCENARIO_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. BrainModel —— 大模型 = Mixture of Experts
# ---------------------------------------------------------------------------
class BrainModel(nn.Module):
    """大脑 = 门控网络 + 一组 expert + 自身的 baseline 头。

    forward 时：gating 看场景决定每个 expert 权重 → 加权融合 expert 输出 →
    再过一个小的 refinement head 输出最终动作。这是"小模型们共同训练大模型"的核心。
    """

    def __init__(self, expert_names: list[str], hidden: int = 64):
        super().__init__()
        self.experts = nn.ModuleDict({n: SkillExpert(n) for n in expert_names})
        self.expert_names = expert_names
        self.gating = nn.Sequential(
            nn.Linear(SCENARIO_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, len(expert_names)),
        )
        self.refine = nn.Sequential(
            nn.Linear(ACTION_DIM + SCENARIO_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, ACTION_DIM), nn.Tanh(),
        )

    def expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """形状 (B, n_experts, ACTION_DIM)"""
        return torch.stack([self.experts[n](x) for n in self.expert_names], dim=1)

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gating(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.expert_outputs(x)            # (B, E, A)
        w = self.gate_weights(x).unsqueeze(-1)    # (B, E, 1)
        fused = (outs * w).sum(dim=1)             # (B, A)
        return self.refine(torch.cat([fused, x], dim=-1))


# ---------------------------------------------------------------------------
# 4. Adversary —— 小生成网络，造难题打败大脑
# ---------------------------------------------------------------------------
class Adversary(nn.Module):
    """从噪声 z 生成场景，目标是最大化 brain 的预测误差。

    这就是"训练关键参数小模型去打败大模型训练"——
    它生成的难题反过来训练大脑（取它生成的场景 + 几何最优作为新一轮训练数据）。
    """

    def __init__(self, z_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, SCENARIO_DIM), nn.Tanh(),  # 场景特征已归一到 [-1,1]
        )

    def forward(self, batch: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch, self.z_dim, device=device)
        return self.net(z)


def adversary_label(scenario: torch.Tensor) -> torch.Tensor:
    """对抗者生成的场景，用几何最优作为标签——
    本质是从合成场景反算最优动作，让大脑学会处理这些边角情况。"""
    # 反归一化前 6 维：cue / target / pocket（取前 6 维）
    pos = (scenario[..., :6] + 1) / 2 * torch.tensor(
        [TABLE_W, TABLE_H, TABLE_W, TABLE_H, TABLE_W, TABLE_H], device=scenario.device,
    )
    cue, target, pocket = pos[..., :2], pos[..., 2:4], pos[..., 4:6]
    return _optimal_action(cue, target, pocket)


# ---------------------------------------------------------------------------
# 5. 训练阶段化组合
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    expert_steps_per_type: int = 1500
    distill_steps: int = 2000
    adv_steps: int = 800
    lr: float = 3e-4
    device: str = "cpu"
    use_adversary: bool = True


def train_experts(brain: BrainModel, cfg: TrainConfig) -> dict[str, float]:
    """Stage 1：每个 expert 单独训练在自己的专项数据上。"""
    losses = {}
    device = torch.device(cfg.device)
    for name in brain.expert_names:
        expert = brain.experts[name]
        opt = torch.optim.Adam(expert.parameters(), lr=cfg.lr)
        X, y = sample_scenarios(cfg.expert_steps_per_type, scenario_type=name, seed=hash(name) & 0xFFFF)
        X, y = X.to(device), y.to(device)
        loader = DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True)
        last = 0.0
        for _ in range(cfg.epochs):
            ep_loss = 0.0
            for xb, yb in loader:
                pred = expert(xb)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item() * len(xb)
            last = ep_loss / len(X)
        losses[name] = last
        print(f"  [Expert] {name:18s}  final MSE = {last:.5f}")
    return losses


def distill_to_brain(brain: BrainModel, cfg: TrainConfig) -> float:
    """Stage 2：用混合数据训练大脑（experts 已预训练，让 gating + refine 学会调度）。"""
    device = torch.device(cfg.device)
    X, y = sample_scenarios(cfg.distill_steps, scenario_type=None, seed=2024)
    X, y = X.to(device), y.to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True)

    # 蒸馏阶段：experts 学习率较小（fine-tune），gating/refine 正常
    expert_params = [p for e in brain.experts.values() for p in e.parameters()]
    head_params = list(brain.gating.parameters()) + list(brain.refine.parameters())
    opt = torch.optim.Adam([
        {"params": expert_params, "lr": cfg.lr * 0.1},
        {"params": head_params,   "lr": cfg.lr},
    ])
    last = 0.0
    for ep in range(cfg.epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            pred = brain(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        last = ep_loss / len(X)
        if (ep + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"  [Distill] epoch {ep+1:3d}  MSE = {last:.5f}")
    return last


def adversarial_round(brain: BrainModel, cfg: TrainConfig) -> tuple[float, float]:
    """Stage 3：小模型 Adversary 生成难题 → 反过来训练大脑。

    交替优化：
      ▸ Adversary 步：固定 brain，最大化 brain 的预测误差。
      ▸ Brain 步：固定 adversary（用它生成的场景），最小化误差。
    """
    device = torch.device(cfg.device)
    adv = Adversary().to(device)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=cfg.lr)
    opt_brain = torch.optim.Adam(brain.parameters(), lr=cfg.lr * 0.5)

    last_adv_obj, last_brain_loss = 0.0, 0.0
    for step in range(cfg.adv_steps):
        # ── (a) Adversary 步：造难题
        for p in brain.parameters(): p.requires_grad_(False)
        scenario = adv(cfg.batch_size, device)
        with torch.no_grad():
            target = adversary_label(scenario)
        pred = brain(scenario)
        adv_obj = -F.mse_loss(pred, target)  # 最大化误差 = 最小化负误差
        opt_adv.zero_grad(); adv_obj.backward(); opt_adv.step()
        last_adv_obj = -adv_obj.item()
        for p in brain.parameters(): p.requires_grad_(True)

        # ── (b) Brain 步：从难题里学习
        with torch.no_grad():
            scenario = adv(cfg.batch_size, device).detach()
        target = adversary_label(scenario)
        pred = brain(scenario)
        brain_loss = F.mse_loss(pred, target)
        opt_brain.zero_grad(); brain_loss.backward(); opt_brain.step()
        last_brain_loss = brain_loss.item()

        if (step + 1) % max(1, cfg.adv_steps // 5) == 0:
            print(f"  [Adv] step {step+1:4d}  adv_attack={last_adv_obj:.5f}  brain_recover={last_brain_loss:.5f}")
    return last_adv_obj, last_brain_loss


# ---------------------------------------------------------------------------
# 6. 评估 & 弱点诊断
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_per_skill(brain: BrainModel, cfg: TrainConfig, n: int = 1000) -> dict[str, float]:
    """对每个场景类型评估大脑误差——找出最弱的 expert/技能。"""
    device = torch.device(cfg.device)
    brain.eval()
    out = {}
    for t in SCENARIO_TYPES:
        X, y = sample_scenarios(n, scenario_type=t, seed=9999)
        X, y = X.to(device), y.to(device)
        pred = brain(X)
        out[t] = F.mse_loss(pred, y).item()
    brain.train()
    return out


@torch.no_grad()
def gating_distribution(brain: BrainModel, cfg: TrainConfig, n: int = 500) -> dict[str, dict[str, float]]:
    """看大脑遇到不同场景时，门控网络给每个 expert 多大权重——
    可以验证大脑是否学会了"看局面调专家"。"""
    device = torch.device(cfg.device)
    brain.eval()
    out = {}
    for t in SCENARIO_TYPES:
        X, _ = sample_scenarios(n, scenario_type=t, seed=8888)
        X = X.to(device)
        w = brain.gate_weights(X).mean(dim=0)
        out[t] = {n: w[i].item() for i, n in enumerate(brain.expert_names)}
    brain.train()
    return out


def self_design_reweight(brain: BrainModel, per_skill_loss: dict[str, float],
                         cfg: TrainConfig, factor: float = 2.0) -> None:
    """自我设计：找出最弱场景类型，给那个 expert 单独加训。
    这就是"自我设计小模型 引导训练你自己大脑的模型"。"""
    weakest = max(per_skill_loss, key=per_skill_loss.get)
    print(f"  [Self-design] 诊断最弱项: {weakest} (MSE={per_skill_loss[weakest]:.5f})  → 强化训练")
    expert = brain.experts[weakest]
    opt = torch.optim.Adam(expert.parameters(), lr=cfg.lr)
    device = torch.device(cfg.device)
    X, y = sample_scenarios(int(cfg.expert_steps_per_type * factor),
                            scenario_type=weakest, seed=hash(weakest) & 0xFFFF)
    X, y = X.to(device), y.to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            loss = F.mse_loss(expert(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()


# ---------------------------------------------------------------------------
# 7. 主流程：完整自训练循环
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig, ckpt: Path = CKPT_PATH) -> BrainModel:
    torch.manual_seed(0)
    brain = BrainModel(SCENARIO_TYPES).to(cfg.device)

    print("\n=== Stage 1: 各小模型 (Expert) 专项预训练 ===")
    train_experts(brain, cfg)

    print("\n=== Stage 2: 蒸馏 — 小模型们共同训练大脑 (MoE 门控学习) ===")
    distill_to_brain(brain, cfg)

    print("\n=== Stage 3: 自我诊断 + 自我设计弱项强化 ===")
    losses = eval_per_skill(brain, cfg)
    for t, l in losses.items():
        print(f"  {t:18s}  MSE = {l:.5f}")
    self_design_reweight(brain, losses, cfg)

    if cfg.use_adversary:
        print("\n=== Stage 4: 对抗训练 — 小模型造难题打败大脑，倒逼提升 ===")
        adversarial_round(brain, cfg)

    print("\n=== 最终评估：大脑各场景表现 + 门控分布 ===")
    final_losses = eval_per_skill(brain, cfg)
    for t, l in final_losses.items():
        delta = l - losses[t]
        sym = "↓" if delta < 0 else "↑"
        print(f"  {t:18s}  MSE = {l:.5f}  ({sym} {abs(delta):.5f} vs Stage 2)")

    print("\n  Gating 分布（每类场景下大脑调用各 expert 的权重）:")
    gates = gating_distribution(brain, cfg)
    print(f"  {'场景\\专家':<18s} " + " ".join(f"{n:>10s}" for n in brain.expert_names))
    for t, w in gates.items():
        row = " ".join(f"{w[n]:>10.3f}" for n in brain.expert_names)
        print(f"  {t:<18s} {row}")

    torch.save({"state_dict": brain.state_dict(), "experts": brain.expert_names}, ckpt)
    print(f"\n✓ 已保存到 {ckpt}")
    return brain


def evaluate(ckpt: Path, cfg: TrainConfig) -> None:
    data = torch.load(ckpt, map_location=cfg.device, weights_only=False)
    brain = BrainModel(data["experts"]).to(cfg.device)
    brain.load_state_dict(data["state_dict"])
    print("=== 评估已训练大脑 ===")
    losses = eval_per_skill(brain, cfg)
    for t, l in losses.items():
        print(f"  {t:18s}  MSE = {l:.5f}")


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expert-steps", type=int, default=1500,
                        help="每个 expert 的训练样本数")
    parser.add_argument("--distill-steps", type=int, default=2000)
    parser.add_argument("--adv-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-adversary", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        expert_steps_per_type=args.expert_steps,
        distill_steps=args.distill_steps, adv_steps=args.adv_steps,
        lr=args.lr, device=args.device, use_adversary=not args.no_adversary,
    )
    if args.eval_only:
        evaluate(args.ckpt, cfg)
    else:
        train(cfg, args.ckpt)


if __name__ == "__main__":
    main()
