import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


N_PATHS = 1_000_000


def simulate_path(key):
    key_demand, key_cost, key_penalty = jax.random.split(key, 3)

    demand = jax.random.normal(key_demand) * 20_000.0 + 100_000.0
    asset_cost = jax.random.lognormal(key_cost, sigma=0.35) * 45_000.0
    penalty_rate = jax.random.uniform(key_penalty, minval=0.02, maxval=0.12)

    price_per_unit = 8.0
    variable_cost_rate = 0.35

    gross_revenue = demand * price_per_unit
    variable_cost = gross_revenue * variable_cost_rate
    regulatory_penalty = gross_revenue * penalty_rate

    net_revenue = gross_revenue - variable_cost - asset_cost - regulatory_penalty
    return net_revenue


@jax.jit
def run_simulation(keys):
    return jax.vmap(simulate_path)(keys)


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    master_key = jax.random.PRNGKey(42)
    keys = jax.random.split(master_key, N_PATHS)

    start_1 = time.perf_counter()
    revenues = run_simulation(keys)
    revenues.block_until_ready()
    end_1 = time.perf_counter()

    start_2 = time.perf_counter()
    revenues = run_simulation(keys)
    revenues.block_until_ready()
    end_2 = time.perf_counter()

    mean_revenue = jnp.mean(revenues)
    var_95 = jnp.percentile(revenues, 5)

    revenues_np = jax.device_get(revenues)
    mean_np = float(mean_revenue)
    var_np = float(var_95)

    plt.figure(figsize=(10, 6))
    plt.hist(revenues_np, bins=100, alpha=0.75)
    plt.axvline(mean_np, color="black", linewidth=2, label=f"Expected revenue: {mean_np:,.2f}")
    plt.axvline(var_np, color="red", linestyle="--", linewidth=2, label=f"VaR 95%: {var_np:,.2f}")

    plt.title("JAX Monte Carlo Revenue Distribution")
    plt.xlabel("Net revenue")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "revenue_dist.png", dpi=200)
    plt.close()

    print(f"Number of paths: {N_PATHS}")
    print(f"Expected revenue: {mean_np:.2f}")
    print(f"VaR 95% threshold: {var_np:.2f}")
    print(f"First run time: {end_1 - start_1:.6f} seconds")
    print(f"Second run time: {end_2 - start_2:.6f} seconds")
    print("Saved plot to data/revenue_dist.png")


if __name__ == "__main__":
    main()