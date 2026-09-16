#!/usr/bin/env python3
"""例 2.6.1 复现：2018 年中国内地居民人均可支配收入与人均消费支出

李子奈《计量经济学》（第五版）§2.6，书 p.49–52。

这个脚本的用法**不是**"跑一下看结果"。正确用法：
  1. 先在纸上手算 β̂₀、β̂₁、R²、σ̂²、S_β̂₁、t、两个预测区间；
  2. 再跑这个脚本，逐个对答案；
  3. 对不上的那一项，就是公式记错或算错的地方——去翻笔记，别急着改。

左列是只用 §2.2–§2.5 的公式、纯 numpy 手写的结果；
右列是 statsmodels 的结果；最后一列是教材图 2.6.1 上印的数。
三列应当一致。

    python3 code/ex2_6_consumption.py
"""
import unicodedata

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA = "code/data/ex2_6_consumption.csv"


def pad(s: str, width: int) -> str:
    """按终端显示宽度补空格（中日韩字符占两列）。"""
    w = sum(0 if unicodedata.combining(c)
            else 2 if unicodedata.east_asian_width(c) in "WF"
            else 1 for c in s)
    return s + " " * max(0, width - w)

# 教材图 2.6.1（EViews 9.0 输出）与正文 p.51 报告的数字
BOOK = {
    "beta0": 2372.623, "beta1": 0.623242,
    "se0": 546.0320, "se1": 0.017997,
    "t1": 34.63066, "R2": 0.976390, "R2_adj": 0.975576,
    "sigma_hat": 1130.176, "RSS": 37041610.0, "F": 1199.283,
}


def hand_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """只用 §2.2–§2.4 的离差形式公式，一步不省。"""
    n = len(x)
    xbar, ybar = x.mean(), y.mean()
    dx, dy = x - xbar, y - ybar          # 离差 xᵢ, yᵢ

    Sxx = (dx ** 2).sum()                # Σxᵢ²
    Sxy = (dx * dy).sum()                # Σxᵢyᵢ

    b1 = Sxy / Sxx                       # (2.2.7)
    b0 = ybar - b1 * xbar                # 回归线过 (X̄, Ȳ)

    yhat = b0 + b1 * x
    e = y - yhat                         # 残差

    TSS = (dy ** 2).sum()
    ESS = b1 ** 2 * Sxx                  # = Σŷᵢ²，用了 ŷᵢ = β̂₁xᵢ (2.2.8)
    RSS = (e ** 2).sum()

    sigma2 = RSS / (n - 2)               # (2.4.6)，除 n−2 不是 n
    se1 = np.sqrt(sigma2 / Sxx)          # (2.4.7)
    se0 = np.sqrt(sigma2 * (x ** 2).sum() / (n * Sxx))

    R2 = ESS / TSS
    return dict(n=n, xbar=xbar, Sxx=Sxx, b0=b0, b1=b1, se0=se0, se1=se1,
                t0=b0 / se0, t1=b1 / se1, TSS=TSS, ESS=ESS, RSS=RSS,
                sigma2=sigma2, sigma_hat=np.sqrt(sigma2), R2=R2,
                R2_adj=1 - (1 - R2) * (n - 1) / (n - 2),
                F=(ESS / 1) / (RSS / (n - 2)))


def predict_intervals(h: dict, x0: float, alpha: float = 0.05) -> dict:
    """§2.5：均值预测区间 vs 个值预测区间。

    两式只差根号里的那个 1 —— 那个 1 就是 σ²，是个体自身的随机性，
    它不随 n→∞ 消失。这是 §2.5 全节的要点。
    """
    n, tc = h["n"], stats.t.ppf(1 - alpha / 2, h["n"] - 2)
    y0 = h["b0"] + h["b1"] * x0
    lever = 1 / n + (x0 - h["xbar"]) ** 2 / h["Sxx"]

    half_mean = tc * h["sigma_hat"] * np.sqrt(lever)        # E(Y₀) 的区间
    half_indiv = tc * h["sigma_hat"] * np.sqrt(1 + lever)   # Y₀ 个别值的区间
    return dict(y0=y0, t_crit=tc,
                mean=(y0 - half_mean, y0 + half_mean), half_mean=half_mean,
                indiv=(y0 - half_indiv, y0 + half_indiv), half_indiv=half_indiv)


def main() -> None:
    df = pd.read_csv(DATA)
    x, y = df["X"].to_numpy(float), df["Y"].to_numpy(float)

    h = hand_ols(x, y)
    fit = sm.OLS(y, sm.add_constant(x)).fit()

    sm_vals = {
        "beta0": fit.params[0], "beta1": fit.params[1],
        "se0": fit.bse[0], "se1": fit.bse[1], "t1": fit.tvalues[1],
        "R2": fit.rsquared, "R2_adj": fit.rsquared_adj,
        "sigma_hat": np.sqrt(fit.mse_resid), "RSS": fit.ssr, "F": fit.fvalue,
    }
    hand_vals = {
        "beta0": h["b0"], "beta1": h["b1"], "se0": h["se0"], "se1": h["se1"],
        "t1": h["t1"], "R2": h["R2"], "R2_adj": h["R2_adj"],
        "sigma_hat": h["sigma_hat"], "RSS": h["RSS"], "F": h["F"],
    }
    labels = {
        "beta0": "β̂₀ 截距", "beta1": "β̂₁ 斜率（边际消费倾向）",
        "se0": "S_β̂₀", "se1": "S_β̂₁", "t1": "t(β̂₁)",
        "R2": "R²", "R2_adj": "调整 R̄²", "sigma_hat": "σ̂（回归标准误）",
        "RSS": "RSS = Σeᵢ²", "F": "F 统计量",
    }

    print(f"例 2.6.1  n = {h['n']}  X̄ = {h['xbar']:.1f}  Σxᵢ² = {h['Sxx']:.0f}\n")
    print(pad("", 30) + f"{'手算':>14}{'statsmodels':>18}{'教材':>18}   一致?")
    print("-" * 88)
    for k, lab in labels.items():
        ok = "✓" if abs(hand_vals[k] - sm_vals[k]) <= 1e-6 * max(1.0, abs(sm_vals[k])) else "✗"
        # 教材数字是四舍五入印出来的，按其有效位数比
        book_ok = "✓" if abs(hand_vals[k] - BOOK[k]) <= 5e-4 * max(1.0, abs(BOOK[k])) else "≈"
        print(pad(lab, 30) + f"{hand_vals[k]:>16,.4f}{sm_vals[k]:>18,.4f}"
              f"{BOOK[k]:>18,.4f}   {ok}{book_ok}")

    print("\n手算 vs statsmodels：✓ 表示逐位相同。")
    print("手算 vs 教材：✓ 表示与书上印的有效位一致；≈ 表示只差书上的舍入。\n")

    print("§2.2 平方和分解：TSS = ESS + RSS")
    print(f"  TSS = {h['TSS']:>18,.1f}")
    print(f"  ESS = {h['ESS']:>18,.1f}")
    print(f"  RSS = {h['RSS']:>18,.1f}")
    print(f"  ESS + RSS - TSS = {h['ESS'] + h['RSS'] - h['TSS']:.6f}   "
          "← 这个 0 是 Σŷᵢeᵢ = 0 的直接后果，不是巧合\n")

    print("OLS 正规方程组的两条数值特征（不是假设，是最小化的必然结果）")
    e = y - (h["b0"] + h["b1"] * x)
    print(f"  Σeᵢ   = {e.sum():.6f}")
    print(f"  ΣXᵢeᵢ = {(x * e).sum():.6f}")
    print("  正因为有这两条约束，n 个残差里独立的只有 n−2 个，σ̂² 才除 n−2\n")

    p = predict_intervals(h, x0=20000.0)
    print("§2.5 预测：X₀ = 20 000 元，95% 置信度")
    rounded = 2372.62 + 0.6232 * 20000
    print(f"  点预测 Ŷ₀ = {p['y0']:,.1f} 元   （教材 p.51：14 836.6）")
    print(f"    差的这 {p['y0'] - rounded:.1f} 元不是算错：教材是拿四舍五入后的 "
          "2 372.62 + 0.623 2×20 000 代进去的，")
    print(f"    用未舍入的系数应得 {p['y0']:,.1f}。报告结果时保留几位有效数字，"
          "在预测环节是会放大的。")
    print(f"  t_{{0.025}}(29) = {p['t_crit']:.3f}   （教材取 2.045）")
    print(f"  均值 E(Y₀) 区间：±{p['half_mean']:,.1f} → "
          f"({p['mean'][0]:,.1f}, {p['mean'][1]:,.1f})"
          "   （教材：±512.5 → 14 324.1, 15 349.1）")
    print(f"  个别值 Y₀ 区间：±{p['half_indiv']:,.1f} → "
          f"({p['indiv'][0]:,.1f}, {p['indiv'][1]:,.1f})"
          "   （教材：±2 367.3 → 12 469.3, 17 204.0）")
    print(f"\n  个值区间是均值区间的 {p['half_indiv'] / p['half_mean']:.1f} 倍。")
    print("  多出来的宽度全部来自根号里那个 1，即 σ² 本身。")
    print("  样本再大，它也不会缩小——预测一个具体家庭，永远比预测一类家庭的平均值难。")


if __name__ == "__main__":
    main()
