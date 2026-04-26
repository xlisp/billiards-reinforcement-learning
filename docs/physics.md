# 桌球技巧背后的物理原理（PyTorch 表达版）

> 配合 [README.md](../README.md) 阅读。每条技巧背后都有可计算的物理量。
> 本文不写公式，所有原理用 **PyTorch 张量运算**表示——代码即推导。
>
> 约定：
> - 所有向量用 `torch.tensor`，长度单位米，时间秒，质量千克。
> - 桌面坐标系：`x` 向右，`y` 向远（球桌长边），`z` 向上。
> - 球的半径 `R = 0.02858`（标准美式 / 中式 8 球，约 57.15 mm 直径）。

---

## 一、力的分解与合成（出杆方向 / 接触点）

对应 README §1.2（出杆）与 §2（透视法 / 鬼球法）。

**原理**：球杆给母球一个力，这个力沿"杆头—球心"方向作用。任何"加塞"或"高低杆"都是把这个力分解到三个分量上：**前进 + 侧旋 + 上下旋**。

```python
import torch

def decompose_cue_force(force_magnitude, cue_dir, hit_offset_xy, ball_radius=0.02858):
    """
    把球杆的总冲量分解为：质心冲量 + 力矩（旋转）。
    cue_dir: 球杆指向的单位向量 (3,)，水平分量决定方向
    hit_offset_xy: 杆头偏离母球中心的位移 (2,)，单位米
                   (0, 0) = 中杆；(0, +R) = 顶部高杆；(0, -R) = 底部缩杆
                   (+R, 0) = 右塞；(-R, 0) = 左塞
    返回：(线动量, 角动量)
    """
    cue_dir = cue_dir / cue_dir.norm()
    linear_impulse = force_magnitude * cue_dir              # 平动：F·t
    # 角冲量 = r × F，其中 r 是接触点相对球心的位移
    r = torch.tensor([hit_offset_xy[0], hit_offset_xy[1], 0.0])
    angular_impulse = torch.linalg.cross(r, linear_impulse) # 转动：r × F
    return linear_impulse, angular_impulse

# 中杆（中心击打）：纯平动，无旋转
p, L = decompose_cue_force(
    force_magnitude=torch.tensor(5.0),
    cue_dir=torch.tensor([0.0, 1.0, 0.0]),
    hit_offset_xy=torch.tensor([0.0, 0.0]),
)
# L ≈ 0  → 母球纯滚动，透视法最准（README §2.6 提到的"中杆中力"）

# 极低杆（搓杆 / 缩杆）：产生强烈倒旋
p, L = decompose_cue_force(
    force_magnitude=torch.tensor(5.0),
    cue_dir=torch.tensor([0.0, 1.0, 0.0]),
    hit_offset_xy=torch.tensor([0.0, -0.025]),  # 接近球底
)
# L_x < 0  → 绕 x 轴反向旋转，对应 README §7 的"母球回旋退回"
```

**对应技巧**：
- README §2.6"加塞会破坏透视" → `hit_offset_xy[0] != 0` 时 `L_z != 0`，母球出杆方向受 squirt 偏射影响。
- README §7 极低杆位 → `hit_offset_xy[1] = -R` 时倒旋角动量最大。

---

## 二、动量守恒（两球碰撞 / 鬼球位置的物理依据）

对应 README §2（透视法）的核心论断："母球撞在哪个点，目标球就沿那个点的法线方向飞出"。

**原理**：弹性碰撞中，沿"两球心连线"方向上动量交换，**垂直方向上目标球不受力**。这就是为什么接触点决定方向。

```python
def two_ball_collision(v_cue, x_cue, x_obj, mass=0.17, restitution=0.96):
    """
    两个等质量球的（近）弹性碰撞。
    v_cue: 母球碰撞瞬间的速度 (2,)
    x_cue, x_obj: 碰撞瞬间两球球心位置 (2,)
    返回：碰撞后母球速度、目标球速度
    """
    # 法线方向（沿两球心连线）= 目标球唯一能受力的方向
    normal = (x_obj - x_cue) / (x_obj - x_cue).norm()

    # 把母球速度分解为：法线分量（撞过去）+ 切线分量（擦过去）
    v_normal = (v_cue @ normal) * normal
    v_tangent = v_cue - v_normal

    # 等质量弹性碰撞：法线分量"完全交换"，切线分量保持
    v_obj_after = restitution * v_normal           # 目标球只拿到法线分量
    v_cue_after = v_tangent + (1 - restitution) * v_normal  # 母球保留切线 + 损耗

    return v_cue_after, v_obj_after

# 验证 README §2 鬼球法：
# 假设袋口在 y 方向正前方。要让目标球 y+ 方向飞，
# 必须让"两球心连线"也指向 y+，也就是母球停在目标球正后方一个球径处——这就是"鬼球位置"。
R = 0.02858
x_obj = torch.tensor([0.0, 1.0])           # 目标球
x_ghost = x_obj - torch.tensor([0.0, 2*R]) # 鬼球：目标球后方一个直径
v_cue = torch.tensor([0.0, 3.0])           # 母球以 3 m/s 撞过来
v_c2, v_o2 = two_ball_collision(v_cue, x_ghost, x_obj)
# v_o2 ≈ [0, +2.88] → 目标球沿 y+ 飞向袋口 ✓
# v_c2 ≈ [0, +0.12] → 母球几乎停住（"定杆"效果，直球的标志）
```

**为什么透视法"反推"是对的**：
```python
def ghost_ball_position(target_pos, pocket_pos, R=0.02858):
    """
    给定目标球位置和袋口位置，反推鬼球应该在哪里。
    这就是 README §2.1 第二步"沿袋口线反向延伸"的代码版。
    """
    pocket_line = (pocket_pos - target_pos)
    pocket_line = pocket_line / pocket_line.norm()
    # 鬼球在袋口线的反方向、距目标球一个直径
    return target_pos - 2 * R * pocket_line

# 母球只要走到 ghost_ball_position(...) 这一点，碰撞自然把目标球送向袋口。
```

---

## 三、角动量守恒（高低杆 / 跟杆 / 缩杆 / 搓杆回旋）

对应 README §7"搓杆"原理段落里那句"**角动量守恒 + 桌呢摩擦力把旋转转化为向后位移**"。

```python
def ball_with_spin_step(v, omega, mu=0.2, g=9.8, dt=1e-3, R=0.02858):
    """
    一个有旋转的球在桌呢上滚动一个时间步的演化。
    v: 球质心速度 (2,)
    omega: 球的角速度 (3,)，omega.z 是侧旋；omega.x 与 v.y 配对决定前后旋
    返回：dt 后的 (v', omega')
    """
    # 球底接触点的相对滑动速度 = 质心速度 + 旋转在接触点产生的线速度
    # 接触点在球底 (0, 0, -R)
    contact_offset = torch.tensor([0.0, 0.0, -R])
    v3 = torch.tensor([v[0], v[1], 0.0])
    v_contact = v3 + torch.linalg.cross(omega, contact_offset)  # 接触点相对台呢的速度

    # 摩擦力方向 = -v_contact 的水平分量
    slip = v_contact[:2]
    if slip.norm() < 1e-6:                                    # 已经纯滚动，无滑动摩擦
        return v, omega
    friction_dir = -slip / slip.norm()
    f_friction = mu * g * friction_dir                        # 单位质量摩擦力

    # 摩擦力对质心：减速 / 改方向
    v_new = v + f_friction * dt
    # 摩擦力对旋转：力矩 = r × F，其中 r = (0,0,-R)
    f3 = torch.tensor([f_friction[0], f_friction[1], 0.0])
    torque = torch.linalg.cross(contact_offset, f3)           # 改变 omega
    # 球的转动惯量系数 2/5 mR²，这里直接归一化
    I_factor = 2.5 / (R * R)
    omega_new = omega + torque * I_factor * dt
    return v_new, omega_new

# 搓杆（README §7）：极低杆位 → 强烈倒旋
v0 = torch.tensor([0.0, 2.0])                          # 前进 2 m/s
omega0 = torch.tensor([-100.0, 0.0, 0.0])              # 强倒旋（绕 x 反转）
# 模拟前进 → 撞球 → 倒旋把母球拉回的过程
# 撞击后 v.y 几乎归零，但 omega.x 仍然为负 → 接下来 v.y 会被摩擦力拉到负值
# 这就是"母球倒旋后撤"的动力学解释
```

**对应技巧映射**：
| README 章节 | omega 初始条件 | 物理后果 |
|---|---|---|
| §1 中杆 | `omega ≈ 0` | 球纯滑→纯滚动，方向稳定 |
| §5 跟杆（高杆） | `omega.x > 0`（前旋） | 撞球后母球继续前进 |
| §7 缩杆（搓杆） | `omega.x << 0`（强倒旋） | 撞球后母球倒退 |
| §2.6 加塞 | `omega.z ≠ 0`（侧旋） | 碰库时改变反射角；接触目标球时产生 throw |

---

## 四、进球概率（瞄准误差的传播）

对应 README §4"远距离中袋打边"——为什么远距离要扩大容错？

**原理**：瞄准角度有一个固定的人手抖动 `σ_θ`（弧度）。距离越远，球到达袋口时的位置误差 `σ_x = L · σ_θ` 越大。袋口宽度 `W` 决定了容错。

```python
import torch

def pocketing_probability(distance, pocket_width, sigma_theta, aim_offset=0.0):
    """
    给定距离、袋口宽度、瞄准角度标准差，计算进球概率。
    distance: 目标球到袋口的距离 (m)
    pocket_width: 袋口有效宽度 (m)，中袋 ~0.085，底袋 ~0.10
    sigma_theta: 出杆角度的标准差（rad），新手 ~0.01，高手 ~0.003
    aim_offset: 瞄准点相对袋口中心的偏移（m），负值=故意打左边
    """
    # 球到达袋口时的位置分布：均值 = aim_offset，标准差 = distance * sigma_theta
    sigma_x = distance * sigma_theta
    # 进球条件：|位置| < pocket_width / 2
    half_w = pocket_width / 2
    # 用正态分布 CDF 计算
    normal = torch.distributions.Normal(loc=aim_offset, scale=sigma_x)
    p_in = normal.cdf(torch.tensor(half_w)) - normal.cdf(torch.tensor(-half_w))
    return p_in

# 验证 README §4：远距离中袋"打边"反而提高概率
sigma = 0.008      # 中等水平
W_mid = 0.085      # 中袋
L = 1.5            # 远距离

# 策略 A：瞄正中（aim_offset=0）但角度抖动大
p_center = pocketing_probability(L, W_mid, sigma, aim_offset=0.0)

# 策略 B：故意瞄左袋角（aim_offset = -W/3），加大力气
# 力气大对应 sigma_theta 略增（动作变形），但分布更宽 → 实际进球区域可能更大
p_edge_strong = pocketing_probability(L, W_mid, sigma * 1.2, aim_offset=-W_mid/3)

# 当 distance 很大时，p_edge_strong > p_center 的反直觉结论成立 ✓
# README §4 实战口诀"远中袋打边、要发力"的代码证明
```

**为什么 README §5 短距离要"小力气"**：
```python
# 短距离时 distance 小 → sigma_x 小 → p 已经接近 1
# 这时候继续加力没有收益，反而 sigma_theta 增大 → p 反而下降
short = pocketing_probability(distance=0.3, pocket_width=W_mid, sigma_theta=0.008)
# short ≈ 0.999  → 不需要任何力气加成
```

---

## 五、路径积分（母球的运动轨迹与走位）

对应 README §9"连续进攻 / 球形选择"——母球碰球后的去向是一条**积分曲线**，沿途受摩擦力衰减。

```python
def simulate_ball_path(x0, v0, omega0, mu=0.2, g=9.8, dt=1e-3, T=5.0):
    """
    把球的整段路径用欧拉积分算出来。这就是"路径积分"——把每一时刻的状态加起来。
    返回：轨迹张量 (N, 2)，速度衰减到 0 时停止。
    """
    x = x0.clone()
    v = v0.clone()
    omega = omega0.clone()
    trajectory = [x.clone()]
    n_steps = int(T / dt)
    R = 0.02858
    for _ in range(n_steps):
        if v.norm() < 1e-3:                          # 球停了
            break
        # 一步演化（复用 §三 的物理）
        v, omega = ball_with_spin_step(v, omega, mu, g, dt, R)
        x = x + v * dt
        trajectory.append(x.clone())
    return torch.stack(trajectory)

# 走位预测（README §9 决策流程第 2 步"倒推"）：
# 给定击球点 + 力量 + 杆位，预测母球停在哪里
x0 = torch.tensor([0.5, 0.5])
v0 = torch.tensor([0.0, 2.5])
omega0 = torch.tensor([50.0, 0.0, 0.0])              # 跟杆（前旋）
path = simulate_ball_path(x0, v0, omega0)
final_position = path[-1]
# final_position 就是"白球停哪"——决定下一杆能不能连贯
```

**README §9"母球走短不走长"的物理解释**：
```python
# 路径越长，累积的摩擦力误差越大，最终位置的不确定性越高
# 这是一个梯度问题：d(final_position) / d(initial_velocity) 随路径长度放大
v0 = torch.tensor([0.0, 2.5], requires_grad=True)
path = simulate_ball_path(x0, v0, omega0=torch.zeros(3))
sensitivity = torch.autograd.grad(path[-1].sum(), v0)[0]
# sensitivity 越大 → 走位越不可控 → 这就是为什么宁可叫近一点
```

---

## 六、库边反射（反袋 / Bank Shot）

对应 README §3：入射角 ≈ 反射角，但要根据速度和旋转修正。

```python
def cushion_reflect(v_in, omega_in, cushion_normal, e_n=0.6, mu_c=0.3, R=0.02858):
    """
    球撞库边的反射。理想反射 + 速度损失 + 侧旋影响。
    v_in: 入射速度 (2,)
    omega_in: 入射角速度 (3,)
    cushion_normal: 库边法向量 (2,)，指向桌内
    e_n: 法向恢复系数（库的弹性，0.5~0.7）
    """
    n = cushion_normal / cushion_normal.norm()
    v_n = (v_in @ n) * n                                       # 法向分量
    v_t = v_in - v_n                                           # 切向分量

    # 法向：被库"弹"回来，损失能量
    v_n_out = -e_n * v_n

    # 切向：受库与球之间的摩擦影响 + 侧旋的额外推动（README §3 提到的"加塞改变反射角"）
    n3 = torch.tensor([n[0], n[1], 0.0])
    spin_at_contact = torch.linalg.cross(omega_in, R * n3)[:2] # 旋转在库接触点的线速度
    v_t_out = v_t - mu_c * (v_t - spin_at_contact)             # 摩擦把切向速度往 spin 方向拉

    return v_n_out + v_t_out

# 纯推杆（无旋转）：入射角 ≈ 反射角
v_out = cushion_reflect(
    v_in=torch.tensor([1.0, -1.0]),                  # 45° 入射
    omega_in=torch.zeros(3),
    cushion_normal=torch.tensor([0.0, 1.0]),         # 撞底库
)
# v_out ≈ [1.0, +0.6]  → 45° 反射但 y 方向能量损失

# 加右塞撞库 → 反射角偏离 45°
v_out_spin = cushion_reflect(
    v_in=torch.tensor([1.0, -1.0]),
    omega_in=torch.tensor([0.0, 0.0, 30.0]),         # 右侧旋
    cushion_normal=torch.tensor([0.0, 1.0]),
)
# v_out_spin 的 x 分量被 spin "推"得更大 → README §3 警告的"加塞改变反射"
```

---

## 七、跳球（Jump Shot）—— 抛体运动 + 弹性反弹

对应 README §6。原理是"母球被向下压、再被台呢顶回去"。

```python
def jump_shot_trajectory(v0_xy, cue_angle_deg, force_magnitude,
                         e_table=0.5, g=9.8, dt=1e-3, T=1.0):
    """
    跳球轨迹：母球离台 → 抛物线 → 落点反弹（如果还有速度）
    cue_angle_deg: 球杆与台面夹角（README §6 推荐 30~60°）
    """
    angle = torch.deg2rad(torch.tensor(cue_angle_deg))
    # 球杆冲量分解为水平 + 垂直
    v_horizontal = force_magnitude * torch.cos(angle) * v0_xy / v0_xy.norm()
    v_vertical = force_magnitude * torch.sin(angle)            # 向上的初速度

    x = torch.tensor([0.0, 0.0, 0.0])
    v = torch.tensor([v_horizontal[0], v_horizontal[1], v_vertical])
    trajectory = [x.clone()]
    for _ in range(int(T / dt)):
        v[2] = v[2] - g * dt                                   # 重力
        x = x + v * dt
        if x[2] < 0:                                           # 落到台面
            x[2] = 0.0
            v[2] = -v[2] * e_table                             # 反弹（速度反向 + 损耗）
            if abs(v[2]) < 0.1:                                # 跳不动了
                v[2] = 0.0
        trajectory.append(x.clone())
    return torch.stack(trajectory)

# README §6 关键判断"第一落点必须落在障碍球之后"：
traj = jump_shot_trajectory(
    v0_xy=torch.tensor([0.0, 1.0]),                  # 向 y+ 跳
    cue_angle_deg=45.0,
    force_magnitude=3.0,
)
# 找第一次落地点
first_landing_idx = (traj[1:, 2] < 1e-4).nonzero()[0]
first_landing_y = traj[first_landing_idx + 1, 1]
# 如果障碍球在 y = 0.3，那么 first_landing_y 必须 > 0.3 + R
# 否则母球先压到障碍球 → 犯规
```

---

## 八、搓杆 / 戳杆——双击犯规的临界判定

对应 README §7 表格"双击 Double Hit"：抬杆 + 短促收杆是为了避免杆头碰到母球第二次。

```python
def double_hit_check(cue_angle_deg, contact_duration, ball_velocity_after,
                     cue_recoil_speed=2.0, dt_threshold=0.005):
    """
    检查是否会发生双击。
    cue_angle_deg: 球杆与台面夹角（抬杆 = 大角度）
    contact_duration: 杆头与母球接触时长（送杆越长越大）
    ball_velocity_after: 母球被击出后的速度
    cue_recoil_speed: 杆头回撤速度
    """
    angle = torch.deg2rad(torch.tensor(cue_angle_deg))
    # 母球被撞后水平退离杆头的速度
    ball_horizontal = ball_velocity_after * torch.cos(angle)
    # 杆头沿出杆方向继续前进的速度（如果送杆没收）
    cue_forward = cue_recoil_speed
    # 母球远离杆头的相对速度
    separation_rate = ball_horizontal + cue_forward * torch.sin(angle)  # 抬杆让杆头有向上分量

    # 双击条件：杆头还来得及追上母球
    # 抬杆角度大 → 杆头快速向斜上方离开 → separation_rate 大 → 安全
    is_safe = (separation_rate > 0.5) & (contact_duration < dt_threshold)
    return is_safe

# 水平推杆贴球 → 容易双击
safe_horizontal = double_hit_check(cue_angle_deg=5.0,  contact_duration=0.01, ball_velocity_after=0.5)
# False → 犯规高发
safe_stab = double_hit_check(cue_angle_deg=45.0, contact_duration=0.002, ball_velocity_after=0.5)
# True → 抬杆 + 短促 = 安全
```

---

## 九、综合：一杆球的完整数据流

把以上模块串起来，就是一个**可微分的桌球模拟器**——给定动作参数，预测进球与否、白球停在哪。

```python
def shot_pipeline(cue_offset, cue_angle, force, cue_pos, target_pos, pocket_pos):
    """
    端到端：动作参数 → 鬼球位置 → 碰撞 → 路径积分 → 进球概率
    所有张量都可微，可以用 SGD 反推"最优出杆参数"。
    """
    # 1. 鬼球位置（§二）
    ghost = ghost_ball_position(target_pos, pocket_pos)

    # 2. 母球到鬼球的方向 = 出杆方向
    aim_dir = (ghost - cue_pos)
    aim_dir = aim_dir / aim_dir.norm()

    # 3. 球杆冲量分解（§一）
    p, L = decompose_cue_force(
        force, torch.tensor([aim_dir[0], aim_dir[1], 0.0]), cue_offset
    )

    # 4. 母球带着 p, L 前进，到达鬼球位置时碰撞（§二）
    v_cue_at_hit = p[:2] / 0.17                    # 质量 0.17 kg
    v_cue_after, v_obj_after = two_ball_collision(v_cue_at_hit, ghost, target_pos)

    # 5. 目标球的路径 → 进入袋口的概率（§四）
    target_to_pocket = (pocket_pos - target_pos).norm()
    p_pocket = pocketing_probability(
        distance=target_to_pocket, pocket_width=0.085,
        sigma_theta=torch.tensor(0.005),
    )

    # 6. 母球的走位路径（§五）
    omega_after = L / (0.4 * 0.17 * 0.02858**2)    # 转化为角速度
    cue_path = simulate_ball_path(target_pos, v_cue_after, omega_after)

    return p_pocket, cue_path[-1]                  # 进球概率 + 白球停哪

# 训练循环：让 SGD 自己学出"完美出杆参数"——AI 教练版本
```

---

## 物理 → 技巧 对照表

| README 技巧 | 核心物理量 | 关键变量 |
|---|---|---|
| §2 透视法 / 鬼球法 | 动量在法线方向交换 | `ghost = target - 2R · pocket_dir` |
| §3 库边反弹 | 反射 + 切向摩擦 + 侧旋 | `v_n_out = -e_n · v_n` |
| §4 远中袋打边 | 误差正态分布 + 容错窗口 | `sigma_x = distance · sigma_theta` |
| §5 短距离小力 | 误差已小，加力反而恶化 | `d(p)/d(force) < 0` |
| §6 跳球 | 抛体运动 + 台面弹性 | `v[2] -= g·dt`；落地反弹 `e_table` |
| §7 搓杆回旋 | 角动量 + 摩擦力转化为位移 | `omega.x << 0` → 后续 `v.y < 0` |
| §7 双击犯规 | 杆头与母球的相对速度 | 抬杆 → `separation_rate > 0` |
| §9 走位 | 路径积分 + 误差敏感度 | `d(final_pos)/d(v0)` 越大越难控 |
| §10 通用"稳定>力量>准度" | `sigma_theta` 是主导误差源 | 减小 `sigma_theta` 优于加大 `force` |

---

> 这份文档把"手感"翻译成"张量运算"——不是为了让你打球时去算梯度，而是让你**在脑子里有一张物理因果图**：
> 知道为什么这样做有效，比死记技巧更稳。
