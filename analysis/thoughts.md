这篇 Anthropic 的 blog 对你现在这个“用 benchmark 子集去最大化与 SWE-Bench 的相关性 / 做 upper bound”的项目，其实有几条**非常直接的提醒**：很多你现在在做的“找最相关子集”行为，本质上就是一种 *eval-hacking / 过拟合到指标*，如果不加防护，会得到一个“看起来很强但不可用”的 benchmark。

下面我按“对你项目的启示 → 具体怎么落地/避免坑”来讲。

---

## 1) 不要只追一个数字：把 eval 拆成 Task / Trial / Grader / Outcome

Anthropic 强调 eval 由任务、试次（同一任务多次运行）、grader、以及最终 outcome 构成；agent 的灵活性会让“看起来失败但其实更好”的情况出现（比如发现规则漏洞）。([Anthropic][1])
**对你项目的启示**：你现在的 `A[m,i]` 是“单次/单指标”的 0/1。如果你要把它当成 benchmark 的核心信号，至少要想清楚：

* `A` 代表的是 **pass@1** 还是 **pass@k**（给 agent 多次机会）？Anthropic 专门区分了 *pass@k*（k 次中至少成功一次）与 *pass^k*（k 次全成功，强调稳定性）。([Anthropic][1])
* 你要的相关性上界更像“能力上界”（pass@k 会更乐观）还是“可靠性上界”（pass^k 更贴近真实用户体验）？([Anthropic][1])

**落地建议**：你现在的 bootstrap（重采样 instance）可以扩展成两层：

* 外层：重采样 instances（你已经在做）
* 内层：对同一 instance 做多 trial，分别算 pass@k / pass^k，再拿它们和 SWE 做相关性对比（你会得到两条完全不同的“上界曲线”）。

---

## 2) “最大相关子集”很容易变成“钻空子子集”

Anthropic 提到：前沿模型会找到 eval 设计者没预料到的“有效路径”，导致 **eval 说失败但用户角度更好**；以及过于 rigid 的 graders 会错误惩罚合理变体（比如数值精度/格式差一点）。([Anthropic][1])

**对你项目的启示**：你用 greedy/swap 找出来的“最相关子集”，可能包含大量：

* **评分器 bug / 模糊规格** 导致的“噪声题”（这些题对模型排序非常敏感，从而“更容易被相关性算法选中”）([Anthropic][1])
* **类不平衡 / 单向测试** 导致的题（只测“应该触发”，不测“不该触发”），这种题集会诱导系统学会投机策略。([Anthropic][1])

**落地建议（很关键）**：把你选出来的子集当成“候选题库”，然后做一次**task-quality audit**：

* Anthropic 的标准是：两个领域专家应能独立给出一致的 pass/fail；并且每题要有参考解验证 grader 正确。([Anthropic][1])
* 你可以把 audit 变成自动化：对“入选频率最高的题”逐题做 sanity checks（是否存在 ambiguous spec、是否存在 grading brittleness、是否存在环境噪声）。

---

## 3) 你现在的“上界”需要同时报告：能力 vs 回归（capability vs regression）

Anthropic 把 eval 分成两类：

* **Capability/quality eval**：一开始可以低通过率，用来 hill-climb、发现弱点。
* **Regression eval**：应该接近 100% 通过率，用来防止退步。([Anthropic][1])

**对你项目的启示**：
你的“最大相关子集”更像一个 capability-oriented 的 *hill-climb target*（专门挑最能区分模型的题）。但如果你最后想把它当 benchmark，就还需要一个 regression-style 的子集/套件，确保系统不会为了相关性“牺牲基本能力/稳定性”。

**落地建议**：做两套子集并同时报告：

1. **Correlation-max subset**（你现在的目标）
2. **Coverage/regression subset**：从“常见失败/高频用户需求”抽样 + 保证高可复现/低噪声（更像护栏）。([Anthropic][1])

---

## 4) 环境与数据泄漏：你的 bootstrap/选子集也要防“共享状态”污染

Anthropic 很强调每个 trial 要隔离环境，避免共享状态导致虚假提升或相关性失真（例如残留文件、缓存、资源耗尽、甚至从前一次 trial 的 git history 获益）。([Anthropic][1])

**对你项目的启示**：如果你的 `load_model_results_0`/评测流水线在不同模型之间存在任何共享缓存、或某些题的结果是“可记忆/可复用”的，那么：

* 相关性会被人为抬高（尤其是你用算法挑子集时，会更偏好这些“泄漏题”）
* bootstrap 的分布也会变得过于乐观

**落地建议**：在你保存的 `model_outputs` 上加两类元信息（哪怕先做日志）：

* 是否使用了缓存/检索到历史产物
* 失败类型（infra flake vs 真实失败）
  这样你后面可以把“易受环境影响的题”直接剔除或降权。([Anthropic][1])

---

## 5) 你在做的“选子集最大相关”= 一种 grader 设计；别只用单一 grader

Anthropic 总结 agent eval 往往混合三类 graders：代码/规则、模型判分、人工，并强调 LLM-as-judge 需要人类校准，且要允许 “Unknown” 以避免胡判。([Anthropic][1])

**对你项目的启示**：你现在把每题压缩成 `pass@1`（单一、硬二值）非常像只用 code-based grader。对于更开放的 agent 行为，这会让你挑出的子集偏向“容易被硬规则误判”的题。

**落地建议**：即便你最终仍然要二值化，也建议在“选子集”阶段加入额外维度当作约束/正则，例如：

* 题目是否存在多解/自由度（更适合 model-based rubric）
* 是否能稳定复现（多 trial 方差）([Anthropic][1])
  这样选出来的子集更不容易是“脆弱题集”。

---

### 一句话总结给你们团队的提醒

* 你现在做的相关性上界/子集优化很有价值，但它天然会**放大评测设计中的模糊、脆弱与泄漏**；所以要像 Anthropic 说的那样，把 eval 当成一个系统：多 trial、任务可判定性、稳定环境、合适 graders、以及 capability/regression 分离一起做，才能把“上界”变成可靠的 benchmark，而不是漂亮但不可用的数字。([Anthropic][1])

[1]: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents "Demystifying evals for AI agents \ Anthropic"
