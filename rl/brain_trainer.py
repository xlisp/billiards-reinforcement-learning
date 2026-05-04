"""自我训练框架：用小模型训练你大脑这个大模型 (Self-Trainer for the Brain).

核心思想（对应 README §9.3 与 §17–§20）
----------------------------------------
- **大模型 (BrainModel)**：你大脑里的整体桌球直觉——所有子技能的加权聚合。无法直接训练。
- **小模型 (SkillModule)**：单一参数 / 单一技能的专项小模型——可以直接通过定向训练提升。
  例：直线瞄准、薄球、走位、低杆、解斯诺克……每个都是一个"参数"。
- **课程设计 (CurriculumDesigner)**：把小模型组合成阶段化课程——基础 → 进阶 → 战术 → 实战。
- **对抗者 (Adversary)**：小模型扮演"挑战者"，专挑你最弱的技能出难题——
  小模型"打败"大模型 → 大模型被迫提升那一参数 → 整体水平上一个台阶。
- **自我训练循环 (SelfTrainer)**：每轮 ① 诊断弱项 ② 设计小模型 ③ 训练小模型
  ④ 集成回大模型 ⑤ 对抗验证 ⑥ 记录 → 重复。

跑法
----
    python brain_trainer.py            # 跑一次完整自我训练循环（按当前 mastery 推荐课程）
    python brain_trainer.py --plan 7   # 生成 7 天训练计划
    python brain_trainer.py --eval     # 只评估当前大脑模型的水平，不训练
    python brain_trainer.py --reset    # 清空进度从头来

进度持久化在 ./brain_state.json。这是给"你"用的训练助手——AI 不是在打球，是在帮你设计训练。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

STATE_PATH = Path(__file__).with_name("brain_state.json")


# ---------------------------------------------------------------------------
# 1. SkillModule —— 小模型：单一技能 / 单一参数
# ---------------------------------------------------------------------------
@dataclass
class SkillModule:
    """一个小模型 = 一个可独立训练、可独立评估的技能参数。

    mastery (0.0–1.0) 就是这个参数当前的"权重"。BrainModel 把所有 mastery 加权聚合
    成整体水平。训练这个小模型 = 提升 mastery。
    """

    name: str
    cn_name: str
    section: str           # 在 README 里对应章节
    difficulty: float       # 难度系数（0 = 基础动作，1 = 顶级技巧）
    prerequisites: list[str] = field(default_factory=list)
    drills: list[str] = field(default_factory=list)
    mastery: float = 0.0    # 当前掌握度（小模型的"训练权重"）

    def is_unlockable(self, brain: "BrainModel") -> bool:
        """小模型解锁条件：所有先修技能 mastery >= 0.4（早期可并行渐进）"""
        return all(brain.skills[p].mastery >= 0.4 for p in self.prerequisites)

    def expected_gain(self) -> float:
        """每轮训练这个小模型预期提升多少 mastery（边际递减）。"""
        # 越难的技能学习曲线越陡；越熟练的技能边际递减
        room = 1.0 - self.mastery
        difficulty_penalty = 1.0 - 0.4 * self.difficulty
        return 0.08 * room * difficulty_penalty

    def train_one_round(self, intensity: float = 1.0) -> float:
        """跑一轮训练：返回 mastery 增量。intensity 是"今天练多少"的强度系数。"""
        gain = self.expected_gain() * intensity
        # 加入随机噪声（人类训练本来就不稳定）
        gain *= random.uniform(0.7, 1.2)
        self.mastery = min(1.0, self.mastery + gain)
        return gain


# ---------------------------------------------------------------------------
# 2. BrainModel —— 大模型：你的大脑 = 所有 SkillModule 的加权聚合
# ---------------------------------------------------------------------------
@dataclass
class BrainModel:
    skills: dict[str, SkillModule] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @classmethod
    def fresh(cls) -> "BrainModel":
        """初始化一个新的大脑模型——所有小模型 mastery = 0。

        小模型清单对应 README 的章节，按难度和依赖关系组织。
        """
        skills_def = [
            # name, cn, section, difficulty, prereqs, drills
            ("stance",        "站姿与手架",      "§1.1–1.2", 0.10, [],
             ["双脚 45° 站位 50 次", "V 槽稳定持续 30 秒 ×10 组", "钟摆出杆空挥 100 次"]),
            ("aim_straight",  "直线瞄准",       "§2",        0.15, ["stance"],
             ["直线球 100 颗（近 → 远）", "闭眼出杆直线球 30 颗", "节拍器 60 BPM 出杆 50 次"]),
            ("ghost_ball",    "鬼球透视法",      "§2.1–2.3",  0.30, ["aim_straight"],
             ["硬币标记接触点 50 颗", "三档基准球（1/4, 1/2, 3/4）各 30 颗", "同袋异位 40 颗"]),
            ("squat_check",   "蹲位投影验证",    "§2.7",      0.25, ["ghost_ball"],
             ["蹲位 + 站位双视角 30 颗", "1/4 / 1/2 / 3/4 投影记忆训练"]),
            ("power_control", "力度控制",       "§4–5",      0.30, ["aim_straight"],
             ["三档力度（轻/中/大）走位 50 颗", "落点精度训练 ±20cm 30 次"]),
            ("english",       "高低杆 / 跟缩",   "§9.2.4",    0.45, ["power_control"],
             ["纯中杆/高杆/低杆三档 60 颗", "走位叫位训练 40 次"]),
            ("bank_shot",     "反袋 / 库边",    "§3, §15",   0.50, ["aim_straight", "power_control"],
             ["固定母球反袋 50 颗", "近距离 / 远距离反袋各 30 颗"]),
            ("position_play", "连续走位 / 清台", "§9",        0.55, ["english", "power_control"],
             ["3 球清台 ×20 局", "倒推法预读两颗球 30 局", "母球走短不走长训练"]),
            ("thin_cut",      "薄球",          "§5",        0.60, ["ghost_ball", "power_control"],
             ["1/4 球以下薄球 50 颗", "贴库薄球 30 颗"]),
            ("jump_shot",     "跳球",          "§6, §15",   0.70, ["english"],
             ["近距离跳球（落袋玩法）30 次", "杆角 30°/45°/60° 各 10 次"]),
            ("pinch_shot",    "搓杆 / 戳杆",    "§7",        0.75, ["english"],
             ["贴球状态戳杆 30 次", "录像检查动作合法性"]),
            ("snooker_make",  "做斯诺克",       "§13",       0.60, ["position_play"],
             ["设定情景做斯诺克 20 次", "对方视角验证障碍 20 次"]),
            ("snooker_escape","解斯诺克 / 脱围", "§15",       0.65, ["bank_shot", "english"],
             ["库边绕一库 20 次", "低杆撞库脱围 20 次"]),
            ("threat_handle", "处理威胁球",      "§19",       0.55, ["thin_cut", "position_play"],
             ["切线轻顶贴袋球 30 次", "borrowing combo 20 次"]),
            ("doubles_team",  "双打配合",       "§14",       0.50, ["position_play", "snooker_make"],
             ["送队友好球训练 30 次", "无声沟通训练 20 局"]),
            ("hail_mary",     "败势解局",       "§16",       0.55, ["snooker_make", "threat_handle"],
             ["撞散对方球群训练 20 次", "锁球策略 20 次"]),
            ("tempo_routine", "预击球流程",     "§12.2",     0.20, ["stance"],
             ["固定 2-3 次引杆 100 杆", "节拍器训练 14 天"]),
            ("integrated_jin","松沉稳劲整劲",   "§17, §18.1", 0.65, ["tempo_routine", "english"],
             ["慢动作出杆体会劲路 50 次", "录侧面视频检查中轴 10 次"]),
            ("be_water",      "局面随变心法",   "§18.2",     0.80, ["integrated_jin", "position_play", "snooker_make"],
             ["每杆 3 问练习 100 杆", "节奏 flow↔crash 切换训练"]),
            ("vs_strong",     "抗高手三档决策",  "§20",       0.85, ["be_water", "snooker_escape", "threat_handle"],
             ["和高手对局复盘 ×10 局", "三档决策模拟训练"]),
        ]
        skills = {
            name: SkillModule(
                name=name, cn_name=cn, section=sec, difficulty=diff,
                prerequisites=prereqs, drills=drills,
            )
            for name, cn, sec, diff, prereqs, drills in skills_def
        }
        return cls(skills=skills)

    def overall_score(self) -> float:
        """大模型整体水平——按难度加权的 mastery 平均。难技能权重高。"""
        total_w, weighted = 0.0, 0.0
        for s in self.skills.values():
            w = 0.5 + s.difficulty
            total_w += w
            weighted += w * s.mastery
        return weighted / total_w if total_w else 0.0

    def weakest_unlockable(self, k: int = 3) -> list[SkillModule]:
        """找出当前可训练但最弱的 k 个小模型——这是下轮训练的目标。"""
        candidates = [s for s in self.skills.values()
                      if s.is_unlockable(self) and s.mastery < 0.95]
        candidates.sort(key=lambda s: (s.mastery, s.difficulty))
        return candidates[:k]

    def stage(self) -> str:
        """根据 overall_score 返回当前所处的训练阶段。"""
        s = self.overall_score()
        if s < 0.20: return "Stage 1：基本功（站姿 + 直线球 + 透视瞄准）"
        if s < 0.40: return "Stage 2：进阶杆法（高低杆 + 力度 + 反袋）"
        if s < 0.60: return "Stage 3：连续清台（走位 + 薄球 + 预击球流程）"
        if s < 0.80: return "Stage 4：战术对局（斯诺克 + 解斯诺克 + 威胁球）"
        return "Stage 5：心法与抗高手（松沉稳劲 + Be water + 三档决策）"


# ---------------------------------------------------------------------------
# 3. Adversary —— 小模型扮演挑战者，专攻你的弱点
# ---------------------------------------------------------------------------
@dataclass
class Adversary:
    """对抗式小模型：每轮挑你最弱的小模型出难题，"打败"大模型。
    被打败 → 倒逼你训练那个小模型 → 大模型整体提升。
    """

    @staticmethod
    def challenge(brain: BrainModel) -> dict:
        """生成一个针对当前最弱小模型的挑战。"""
        weakest = brain.weakest_unlockable(1)
        if not weakest:
            return {"verdict": "PASS", "msg": "当前没有可训练的弱项——继续保持。"}

        target = weakest[0]
        # 大模型在这个技能上的胜率（vs 对抗者难度 = 该技能难度 + 0.1）
        adversary_skill = min(1.0, target.difficulty + 0.10)
        win_prob = max(0.0, target.mastery - adversary_skill * 0.5)
        win_prob = min(1.0, win_prob + 0.3)
        defeated = random.random() > win_prob

        return {
            "verdict": "FAIL" if defeated else "PASS",
            "target": target.name,
            "target_cn": target.cn_name,
            "section": target.section,
            "current_mastery": target.mastery,
            "adversary_difficulty": adversary_skill,
            "win_prob": win_prob,
            "msg": (f"对抗者攻你最弱的【{target.cn_name}】(§{target.section})——"
                    f"当前 mastery={target.mastery:.2f}, 胜率={win_prob:.0%}, "
                    f"{'被打败 → 必须强化这一项' if defeated else '勉强守住——但仍是弱项'}"),
        }


# ---------------------------------------------------------------------------
# 4. CurriculumDesigner —— 自我设计小模型并组合成阶段化课程
# ---------------------------------------------------------------------------
class CurriculumDesigner:
    """根据当前大脑状态，自我设计下一阶段的小模型组合。

    "组合化训练"的本质：不是单独练每个技能，是把几个相关小模型串成一个组合训练，
    每天的训练就是几个小模型一起前进，最后再融合到大模型里。
    """

    @staticmethod
    def design_session(brain: BrainModel, n_modules: int = 3) -> dict:
        """设计今天的训练 session：选 n 个小模型组合训练。"""
        targets = brain.weakest_unlockable(n_modules)
        if not targets:
            # 全部解锁？挑 mastery < 1.0 的随机几个
            targets = [s for s in brain.skills.values() if s.mastery < 1.0][:n_modules]

        total_minutes = 0
        plan = []
        for t in targets:
            # 难度越高，建议时长越长（难技能需要更多反复）
            minutes = int(15 + 25 * t.difficulty)
            total_minutes += minutes
            plan.append({
                "skill": t.name,
                "cn": t.cn_name,
                "section": t.section,
                "current_mastery": round(t.mastery, 3),
                "minutes": minutes,
                "drills": t.drills,
            })

        return {
            "stage": brain.stage(),
            "overall_score": round(brain.overall_score(), 3),
            "total_minutes": total_minutes,
            "session": plan,
            "philosophy": CurriculumDesigner._pick_philosophy(brain),
        }

    @staticmethod
    def _pick_philosophy(brain: BrainModel) -> str:
        """每个阶段配一句心法（呼应 README §17–§20）。"""
        s = brain.overall_score()
        if s < 0.20: return "稳 > 力 > 准——动作不稳，再准也是偶然 (§十.1)"
        if s < 0.40: return "稳定来自流程——固定预击球节拍 (§17.稳)"
        if s < 0.60: return "母球走短不走长——走位的鲁棒性 > 精度 (§9.2.6)"
        if s < 0.80: return "打之前先想白球去哪——而不是这球能不能进 (§十.2)"
        return "Be water——形随境变、本质不变 (§18.2)"


# ---------------------------------------------------------------------------
# 5. SelfTrainer —— 主循环：诊断 → 设计 → 训练 → 对抗 → 集成
# ---------------------------------------------------------------------------
class SelfTrainer:
    def __init__(self, brain: BrainModel | None = None):
        self.brain = brain or BrainModel.fresh()

    @classmethod
    def load(cls) -> "SelfTrainer":
        if STATE_PATH.exists():
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            brain = BrainModel.fresh()
            for name, m in data.get("mastery", {}).items():
                if name in brain.skills:
                    brain.skills[name].mastery = m
            brain.history = data.get("history", [])
            return cls(brain)
        return cls()

    def save(self) -> None:
        data = {
            "mastery": {n: s.mastery for n, s in self.brain.skills.items()},
            "history": self.brain.history,
        }
        STATE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_one_cycle(self, intensity: float = 1.0) -> dict:
        """一轮完整循环：诊断 → 设计课程 → 模拟训练 → 对抗挑战 → 集成进度。"""
        before = self.brain.overall_score()
        session = CurriculumDesigner.design_session(self.brain, n_modules=3)

        gains = {}
        for entry in session["session"]:
            skill = self.brain.skills[entry["skill"]]
            gain = skill.train_one_round(intensity)
            gains[skill.name] = round(gain, 4)

        challenge = Adversary.challenge(self.brain)
        if challenge["verdict"] == "FAIL":
            forced = self.brain.skills[challenge["target"]]
            extra = forced.train_one_round(intensity * 0.6)
            gains[forced.name] = round(gains.get(forced.name, 0) + extra, 4)

        after = self.brain.overall_score()
        record = {
            "before": round(before, 3),
            "after": round(after, 3),
            "delta": round(after - before, 4),
            "session": [e["skill"] for e in session["session"]],
            "gains": gains,
            "challenge": challenge["verdict"],
            "challenge_target": challenge.get("target"),
        }
        self.brain.history.append(record)
        return {"session": session, "challenge": challenge, "result": record}

    def make_plan(self, days: int = 7) -> list[dict]:
        """生成 N 天训练计划——每天一个组合 session。不实际训练，只规划。"""
        # 用快照模拟，不动当前 brain
        snapshot = BrainModel.fresh()
        for n, s in self.brain.skills.items():
            snapshot.skills[n].mastery = s.mastery

        plan = []
        for day in range(1, days + 1):
            session = CurriculumDesigner.design_session(snapshot, n_modules=3)
            plan.append({"day": day, **session})
            # 模拟训练以推进快照（让后续天的课程往前演进）
            for entry in session["session"]:
                snapshot.skills[entry["skill"]].train_one_round(intensity=1.0)
        return plan


# ---------------------------------------------------------------------------
# 6. CLI —— 让这套框架真正能用
# ---------------------------------------------------------------------------
def _print_session(out: dict) -> None:
    s = out["session"]
    print(f"\n=== 当前阶段 ===\n{s['stage']}")
    print(f"整体水平 (overall_score) = {s['overall_score']:.3f}")
    print(f"心法：{s['philosophy']}")
    print(f"\n=== 今日训练 session（共 {s['total_minutes']} 分钟）===")
    for i, e in enumerate(s["session"], 1):
        print(f"\n  [{i}] {e['cn']} (§{e['section']})  —  {e['minutes']} 分钟")
        print(f"      当前 mastery: {e['current_mastery']:.2f}")
        print(f"      训练内容:")
        for d in e["drills"]:
            print(f"        • {d}")
    c = out["challenge"]
    print(f"\n=== 对抗者挑战（小模型 vs 大脑）===\n  {c['msg']}")
    r = out["result"]
    print(f"\n=== 本轮收益 ===\n  整体水平 {r['before']:.3f} → {r['after']:.3f}  (Δ {r['delta']:+.4f})")
    for k, v in r["gains"].items():
        print(f"    + {k:18s} mastery +{v:.4f}")


def _print_plan(plan: list[dict]) -> None:
    print(f"\n=== {len(plan)} 天训练计划 ===")
    for day in plan:
        print(f"\n— Day {day['day']} | {day['stage']} | {day['total_minutes']} 分钟")
        print(f"  心法：{day['philosophy']}")
        for e in day["session"]:
            print(f"  • {e['cn']:20s} (§{e['section']:8s}) {e['minutes']:3d} min  "
                  f"mastery={e['current_mastery']:.2f}")


def _print_eval(brain: BrainModel) -> None:
    print(f"\n=== 大脑模型评估 ===")
    print(f"阶段：{brain.stage()}")
    print(f"整体水平：{brain.overall_score():.3f}")
    print(f"\n各小模型 mastery（按依赖顺序）:")
    for s in brain.skills.values():
        bar = "█" * int(s.mastery * 20) + "·" * (20 - int(s.mastery * 20))
        unlocked = "✓" if s.is_unlockable(brain) else "🔒"
        print(f"  {unlocked} {s.cn_name:18s} (§{s.section:8s}) [{bar}] {s.mastery:.2f}  D={s.difficulty:.2f}")
    print(f"\n历史训练轮数：{len(brain.history)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-train your brain billiards model.")
    parser.add_argument("--plan", type=int, metavar="DAYS",
                        help="只生成 N 天训练计划，不实际训练")
    parser.add_argument("--eval", action="store_true",
                        help="只评估当前大脑模型，不训练")
    parser.add_argument("--cycles", type=int, default=1,
                        help="跑几轮自我训练循环（默认 1）")
    parser.add_argument("--intensity", type=float, default=1.0,
                        help="训练强度（默认 1.0；0.5 = 半强度休息日）")
    parser.add_argument("--reset", action="store_true",
                        help="清空进度从头来")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()
        print("进度已清空。")

    trainer = SelfTrainer.load()

    if args.eval:
        _print_eval(trainer.brain)
        return

    if args.plan:
        plan = trainer.make_plan(days=args.plan)
        _print_plan(plan)
        return

    for i in range(args.cycles):
        if args.cycles > 1:
            print(f"\n========== Cycle {i+1}/{args.cycles} ==========")
        out = trainer.run_one_cycle(intensity=args.intensity)
        _print_session(out)
    trainer.save()
    print(f"\n进度已保存到 {STATE_PATH}")


if __name__ == "__main__":
    main()
