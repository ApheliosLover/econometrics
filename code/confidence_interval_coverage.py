#!/usr/bin/env python3
"""置信区间的「95%」到底在说什么 —— 覆盖率模拟

配合李子奈《计量经济学》（第五版）§2.4 三，书 p.45–46。

**几乎人人都会错的读法**：
    ❌ 「β₁ 有 95% 的概率落在 (a, b) 里」
       β₁ 是固定的未知常数，它要么在要么不在，没有概率可言。

**正确读法**：
    ✅ 「重复抽样很多次，每次算一个这样的区间，大约 95% 的区间会盖住真值」
       随机的是区间，不是参数。换个样本，β̂₁ 变、S_β̂₁ 变，区间整个挪。

本脚本把真值钉死，反复抽样，数有多少个区间盖住了它。

    python3 code/confidence_interval_coverage.py
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(922)
B1_TRUE = 0.67          # 真实斜率，全程不变
B0_TRUE = 142.4
N = 10                  # 与教材例 2.1.1 的样本量一致
SIGMA = 50.0


def one_interval(alpha: float) -> tuple[float, float]:
    """抽一个样本，返回该样本算出的 β₁ 置信区间。"""
    tc = stats.t.ppf(1 - alpha / 2, N - 2)
    X = RNG.normal(1000, 300, N)
    Y = B0_TRUE + B1_TRUE * X + RNG.normal(0, SIGMA, N)

    xc, yc = X - X.mean(), Y - Y.mean()
    b = (xc * yc).sum() / (xc ** 2).sum()
    e = yc - b * xc                              # 离差形式下的残差
    se = np.sqrt((e ** 2).sum() / (N - 2) / (xc ** 2).sum())
    return b - tc * se, b + tc * se


def coverage(alpha: float, reps: int = 20_000) -> float:
    hit = sum(lo <= B1_TRUE <= hi
              for lo, hi in (one_interval(alpha) for _ in range(reps)))
    return hit / reps


def main() -> None:
    print(f"真值 β₁ = {B1_TRUE}（全程不变）　n = {N}\n")

    print("置信度 95% 时，前 12 次抽样各自算出的区间：")
    for i in range(12):
        lo, hi = one_interval(0.05)
        mark = "盖住 ✓" if lo <= B1_TRUE <= hi else "没盖住 ✗"
        print(f"  第 {i + 1:>2} 次   ({lo:.3f}, {hi:.3f})   {mark}")
    print("  ↑ 每次的位置和宽窄都不同。真值一动没动，动的是区间。\n")

    print(f"{'名义置信度':>12}{'α':>8}{'t 临界值':>12}{'实测覆盖率':>14}")
    print("-" * 48)
    for alpha in (0.10, 0.05, 0.01):
        tc = stats.t.ppf(1 - alpha / 2, N - 2)
        print(f"{1 - alpha:>11.0%}{alpha:>8.2f}{tc:>12.3f}{coverage(alpha):>14.1%}")

    print("""
读法
----
实测覆盖率贴着名义置信度 —— 这就是「95%」的全部含义：
**100 个这样的区间里，约 95 个会盖住真值。**

α 越小 → t 临界值越大 → 区间越宽 → 覆盖率越高。
想更有把握，就得接受更模糊的结论。没有免费午餐。

缩窄区间的三条路（教材 p.46 末）：
  加大 n        S_β̂₁ 变小，且 t 临界值也变小（双重收益）
  提高拟合优度   RSS 小 → σ̂² 小
  X 更分散      Σxᵢ² 大 —— 常被忽略：样本里 X 的取值范围越宽，斜率估得越准
""")


if __name__ == "__main__":
    main()
