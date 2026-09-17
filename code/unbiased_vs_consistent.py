#!/usr/bin/env python3
"""无偏性 vs 一致性：三个估计量的蒙特卡洛对比

配合李子奈《计量经济学》（第五版）§2.3 二，书 p.40–42。

教材把无偏性列为小样本性质、一致性列为大样本性质，但没说清楚
**两者在逻辑上互不蕴含**。这个脚本用三个估计量把四种可能里的三种摆出来：

    A  OLS 全样本            → 无偏 + 一致   （正常情形）
    B  只用头两个观测算斜率   → 无偏但不一致 （数据再多也不用）
    C  滞后因变量模型的 OLS   → 有偏但一致   （偏差随 n 消失）

判读方法：
    无偏性 → 看「β̂ 的均值」这一列是否等于真值，**每个 n 都要等**
    一致性 → 看 n 增大时均值是否走向真值、标准差是否趋于 0

    python3 code/unbiased_vs_consistent.py
"""
import numpy as np

RNG = np.random.default_rng(20260917)
REPS = 20_000
BETA1 = 0.5      # A、B 的真实斜率
RHO = 0.5        # C 的真实自回归系数


def ols_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """按列做一元 OLS，返回每列的斜率。"""
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    return (xc * yc).sum(axis=0) / (xc ** 2).sum(axis=0)


def estimator_A(n: int) -> np.ndarray:
    """OLS 全样本。假设 3 成立 → 无偏；假设 2 成立 → 一致。"""
    x = RNG.normal(size=(n, REPS))
    y = 2 + BETA1 * x + RNG.normal(size=(n, REPS))
    return ols_slope(x, y)


def estimator_B(n: int) -> np.ndarray:
    """只用前两个观测连一条直线，其余数据全扔掉。

    这里把这两点的 X 固定在 0 和 1（设计取值），于是
        β̃ = (Y₂−Y₁)/(1−0) = β₁ + (μ₂−μ₁)
    对任何 n 都有 E(β̃) = β₁ → 无偏；
    而 Var(β̃) = 2σ² 是个常数，完全不随 n 变化 → 不一致。

    为什么要固定 X：若 X 也随机，分母 X₂−X₁ 可能接近零，比值成为柯西型
    重尾分布，均值根本不存在，蒙特卡洛的样本均值不收敛、跑出来是噪声。
    那是另一个问题（比值估计量的矩不存在），会盖住这里要演示的点。
    """
    del n  # 故意不用：这正是它不一致的原因
    mu = RNG.normal(size=(2, REPS))
    y = 2 + BETA1 * np.array([[0.0], [1.0]]) + mu
    return (y[1] - y[0]) / (1.0 - 0.0)


def estimator_C(n: int) -> np.ndarray:
    """滞后因变量模型 Yt = ρ·Y(t−1) + εt 的 OLS。

    解释变量 Y(t−1) 受 ε(t−1) 影响，而 ε(t−1) 又是上一期的干扰项，
    所以严格外生不成立 → 有偏。但同期外生仍成立 → 一致。
    """
    out = np.empty(REPS)
    burn = 50
    for r in range(REPS):
        e = RNG.normal(size=n + burn + 1)
        y = np.empty(n + burn + 1)
        y[0] = e[0] / np.sqrt(1 - RHO ** 2)      # 从平稳分布起步
        for t in range(1, n + burn + 1):
            y[t] = RHO * y[t - 1] + e[t]
        y = y[burn:]
        lag, cur = y[:-1], y[1:]
        out[r] = (lag * cur).sum() / (lag ** 2).sum()
    return out


def report(name: str, fn, truth: float, sizes: tuple[int, ...], note: str) -> None:
    print(f"\n{name}")
    print(f"  {note}")
    print(f"  {'n':>7} {'β̂ 的均值':>12} {'偏差':>11} {'标准差':>11}")
    print("  " + "-" * 44)
    for n in sizes:
        b = fn(n)
        bias = b.mean() - truth
        se = b.std() / np.sqrt(REPS)          # 均值的标准误
        if abs(bias) > 4 * se:
            flag = "  ← 偏差显著" + ("（且量级可观）" if abs(bias) > 0.01 else "（但已很小）")
        else:
            flag = ""
        print(f"  {n:>7} {b.mean():>12.4f} {bias:>+11.4f} {b.std():>11.4f}{flag}")


def main() -> None:
    print("=" * 62)
    print(f"蒙特卡洛：每种情形重复抽样 {REPS:,} 次")
    print("=" * 62)

    report("A ｜ OLS 全样本　　　　　【无偏 + 一致】", estimator_A, BETA1,
           (10, 50, 500, 5000),
           f"真值 β₁ = {BETA1}。每个 n 均值都等于真值，且标准差随 n 缩小。")

    report("B ｜ 只用前两个观测　　　【无偏但不一致】", estimator_B, BETA1,
           (10, 50, 500, 5000),
           f"真值 β₁ = {BETA1}。均值一直是真值，但标准差四行完全一样。")

    report("C ｜ 滞后因变量 OLS　　　【有偏但一致】", estimator_C, RHO,
           (10, 50, 500, 5000),
           f"真值 ρ = {RHO}。小 n 时系统性偏低，n 增大后偏差消失。")

    print("""
==============================================================
怎么读这张表
--------------------------------------------------------------
无偏性  看「β̂ 的均值」列：**每一个 n** 都等于真值才算无偏。
        它是关于抽样分布中心位置的陈述，与 n 大小无关。

一致性  看 n 增大时的走向：均值走向真值 **且** 标准差趋于 0。
        它是关于 n→∞ 时去向的陈述。

B 的均值每行都对，但标准差四行几乎一样 —— 无偏，不一致。
C 的前两行均值明显偏低，后两行贴上真值 —— 有偏，一致。

所以两者互不蕴含。教材 p.42 说一致性所需的外生性比无偏性弱，
指的是「同期外生 vs 严格外生」这个方向；但 B 提醒我们，
外生性之外还得有假设 2（X 有变异且被充分利用），一致性才成立。
==============================================================""")


if __name__ == "__main__":
    main()
