# Swarm Stress Report

## Subagent Alpha – Stress Test

The Log-Normal asset cost variance was increased to evaluate the robustness of the revenue model.

Observed behavior:

| Sigma | Result |
|---------|---------|
| 0.35 | Stable positive VaR |
| 0.50 | Increased downside risk |
| 0.80 | Significant profit volatility |
| 1.20 | VaR approaches critical region |
| 1.50 | Negative tail events become frequent |

Conclusion:

As the variance of production asset costs increases, the distribution develops a much heavier downside tail. Extreme cost realizations can dominate revenue generation and eventually push Value-at-Risk toward negative territory.

---

## Subagent Beta – JAX Performance Profiling

Execution results:

| Run | Time (s) |
|-------|-----------|
| First Run | 1.653376 |
| Second Run | 0.226458 |

Speedup:

\[
\text{Speedup} = \frac{1.653376}{0.226458}
\approx 7.30
\]

Conclusion:

The first execution includes JAX tracing and XLA compilation overhead. During the second execution the compiled computation graph is reused, resulting in substantially faster execution.

---

## Overall Assessment

The Monte Carlo pipeline successfully evaluates one million stochastic business scenarios in parallel using JAX vectorization.

The stress test demonstrates that increasing uncertainty in asset costs significantly increases downside financial risk.

The profiling experiment confirms the benefit of JAX compilation, where initial compilation costs are amortized over repeated executions.