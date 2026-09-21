#!/usr/bin/env python3
"""为什么假设 3 的条件必须是全部 X —— 用 k₂ 的分布看

配合李子奈《计量经济学》（第五版）§2.3，书 p.38、p.40。

**问题**：无偏性推导里有一步把 kᵢ 提到条件期望号外面当常数。
凭什么？只给定 Xᵢ 一个值行不行？

**规则**：只有「在给定条件下算得出确切数值」的东西才能提出期望号。
    E(3·Z)       = 3·E(Z)        ✓  3 是已知的数
    E(W·Z)       ≠ W·E(Z)        ✗  W 是随机的
    E(W·Z | W)   = W·E(Z|W)      ✓  条件里把 W 定死了

**做法**：把 X₂ 钉死在 5，其余 X 随机抽，看 k₂ 还动不动。

    python3 code/why_condition_on_all_X.py
"""
import numpy as np

RNG = np.random.default_rng(921)
REPS, N = 100_000, 5
X2_FIXED = 5.0


def k_second(X: np.ndarray) -> np.ndarray:
    """按列算 k₂ = (X₂ − X̄) / Σⱼ(Xⱼ − X̄)²。

    注意分子里的 X̄ 与分母的求和 **各跑一遍全部 n 个 X**，
    所以 k₂ 并不只取决于 X₂。
    """
    d = X - X.mean(axis=0)
    return d[1] / (d ** 2).sum(axis=0)


def main() -> None:
    X = RNG.normal(3, 2, size=(N, REPS))
    X[1] = X2_FIXED                      # X₂ 永远钉在 5
    k2 = k_second(X)

    print(f"情形 A：只知道 X₂ = {X2_FIXED}，其余 {N - 1} 个 X 随机")
    print(f"  k₂ 取值范围 : {k2.min():+.4f}  ~  {k2.max():+.4f}")
    print(f"  k₂ 标准差   : {k2.std():.4f}      <- 不是 0，k₂ 还在变")
    print(f"  k₂ 为正比例 : {(k2 > 0).mean():.1%}  <- 连正负号都不定")

    Xf = np.array([[3.0], [X2_FIXED], [1.0], [4.0], [2.0]])
    print(f"\n情形 B：全部 X 给定，X = {Xf.ravel().tolist()}")
    print(f"  k₂ = {k_second(Xf)[0]:.4f}      <- 一个确定的数，标准差为 0")

    print("""
结论
----
X₂ 从头到尾锁死，k₂ 却到处跑 —— 因为 k₂ 的公式要用到 X₁,X₃,X₄,X₅。

  只给定 Xᵢ    ->  kᵢ 仍是随机变量  ->  提不出期望号  ❌
  给定全部 X   ->  kᵢ 是一个数      ->  提得出来      ✅

所以假设 3 写成 E(μᵢ|X)=0 而非 E(μᵢ|Xᵢ)=0，
不是为了严格而严格 —— 不写全 X，这步推导根本走不下去。

另一层理由（本脚本未演示）：即便放过这一点，E(μᵢ|Xᵢ)=0 也只说 μᵢ 与
自己那个 Xᵢ 无关，管不住 μ₂ 与 X₅ 之间的勾连（见 notes/ 里的同桌讲题例）。
""")


if __name__ == "__main__":
    main()
