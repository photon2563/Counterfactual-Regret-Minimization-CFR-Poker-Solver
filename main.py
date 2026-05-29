"""
GTO Poker Solver — Main Training & Demonstration Script.

Trains CFR variants on Kuhn Poker and Leduc Hold'em, verifies convergence
to known Nash equilibria, and generates exploitability plots.

Usage:
    python main.py                    # Full demo (all variants, both games)
    python main.py --game kuhn        # Kuhn Poker only
    python main.py --game leduc       # Leduc Hold'em only
    python main.py --variant vanilla  # Only Vanilla CFR
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from game_engine.kuhn_poker import KuhnPoker, KuhnState
from game_engine.leduc_holdem import LeducHoldem
from game_engine.game_state import Action, Player
from cfr.vanilla_cfr import VanillaCFR
from cfr.mccfr import ExternalSamplingMCCFR, OutcomeSamplingMCCFR
from evaluation.best_response import BestResponse


def train_kuhn_poker():
    """
    Train and verify CFR on Kuhn Poker.
    
    Kuhn Poker has a known analytical Nash equilibrium, making it
    the perfect verification target for CFR correctness.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: Kuhn Poker — Vanilla CFR Training")
    print("=" * 70)
    
    game = KuhnPoker()
    results = {}
    
    # ─── Vanilla CFR ───
    print("\n▸ Training Vanilla CFR (10,000 iterations)...")
    solver = VanillaCFR()
    start = time.time()
    metrics = solver.train_and_track(
        game, num_iterations=10000, eval_every=1000, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    
    results['vanilla'] = {
        'metrics': metrics,
        'time': elapsed,
        'final_exploitability': metrics['exploitability'][-1],
    }
    
    # Print converged strategy
    solver.print_strategy("Kuhn Poker (Vanilla CFR)")
    
    # Verify against known Nash equilibrium
    known_nash = KuhnPoker.known_nash_equilibrium()
    avg_strategy = solver.get_average_strategy()
    
    print("▸ Verification against known Nash equilibrium:")
    print(f"  Known game value (P0): {KuhnPoker.GAME_VALUE:.6f}")
    
    br = BestResponse()
    game_value = br.compute_game_value(game, avg_strategy, Player.PLAYER_0)
    print(f"  Computed game value:   {game_value:.6f}")
    print(f"  Error:                 {abs(game_value - KuhnPoker.GAME_VALUE):.6f}")
    
    exploit = br.compute_exploitability(game, avg_strategy)
    print(f"  Final exploitability:  {exploit:.6f}")
    
    # ─── CFR+ ───
    print("\n▸ Training CFR+ (10,000 iterations)...")
    solver_plus = VanillaCFR(cfr_plus=True)
    start = time.time()
    metrics_plus = solver_plus.train_and_track(
        game, num_iterations=10000, eval_every=1000, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    
    results['cfr_plus'] = {
        'metrics': metrics_plus,
        'time': elapsed,
        'final_exploitability': metrics_plus['exploitability'][-1],
    }
    
    # ─── DCFR ───
    print("\n▸ Training DCFR (α=1.5, β=0.0, γ=2.0, 10,000 iterations)...")
    solver_dcfr = VanillaCFR(
        dcfr_params={'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0}
    )
    start = time.time()
    metrics_dcfr = solver_dcfr.train_and_track(
        game, num_iterations=10000, eval_every=1000, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    
    results['dcfr'] = {
        'metrics': metrics_dcfr,
        'time': elapsed,
        'final_exploitability': metrics_dcfr['exploitability'][-1],
    }
    
    # ─── External Sampling MCCFR ───
    print("\n▸ Training External Sampling MCCFR (20,000 iterations)...")
    solver_es = ExternalSamplingMCCFR()
    solver_es.set_seed(42)
    start = time.time()
    metrics_es = solver_es.train_and_track(
        game, num_iterations=20000, eval_every=2000, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    
    results['es_mccfr'] = {
        'metrics': metrics_es,
        'time': elapsed,
        'final_exploitability': metrics_es['exploitability'][-1],
    }
    
    return results


def train_leduc_holdem():
    """
    Train CFR on Leduc Hold'em — a richer testbed.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1-2: Leduc Hold'em — Multi-Variant CFR Training")
    print("=" * 70)
    
    game = LeducHoldem()
    results = {}
    
    # ─── Vanilla CFR ───
    print("\n▸ Training Vanilla CFR on Leduc Hold'em (5,000 iterations)...")
    solver = VanillaCFR()
    start = time.time()
    metrics = solver.train_and_track(
        game, num_iterations=5000, eval_every=500, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s, Info sets: {len(solver.info_sets)}")
    
    results['vanilla'] = {
        'metrics': metrics,
        'time': elapsed,
        'final_exploitability': metrics['exploitability'][-1],
        'info_sets': len(solver.info_sets),
    }
    
    # ─── CFR+ ───
    print("\n▸ Training CFR+ on Leduc Hold'em (5,000 iterations)...")
    solver_plus = VanillaCFR(cfr_plus=True)
    start = time.time()
    metrics_plus = solver_plus.train_and_track(
        game, num_iterations=5000, eval_every=500, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    
    results['cfr_plus'] = {
        'metrics': metrics_plus,
        'time': elapsed,
        'final_exploitability': metrics_plus['exploitability'][-1],
    }
    
    # ─── DCFR ───
    print("\n▸ Training DCFR on Leduc Hold'em (5,000 iterations)...")
    solver_dcfr = VanillaCFR(
        dcfr_params={'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0}
    )
    start = time.time()
    metrics_dcfr = solver_dcfr.train_and_track(
        game, num_iterations=5000, eval_every=500, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    
    results['dcfr'] = {
        'metrics': metrics_dcfr,
        'time': elapsed,
        'final_exploitability': metrics_dcfr['exploitability'][-1],
    }
    
    # ─── External Sampling MCCFR ───
    print("\n▸ Training External Sampling MCCFR on Leduc Hold'em (10,000 iterations)...")
    solver_es = ExternalSamplingMCCFR()
    solver_es.set_seed(42)
    start = time.time()
    metrics_es = solver_es.train_and_track(
        game, num_iterations=10000, eval_every=1000, verbose=True
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    
    results['es_mccfr'] = {
        'metrics': metrics_es,
        'time': elapsed,
        'final_exploitability': metrics_es['exploitability'][-1],
    }
    
    return results


def print_summary(kuhn_results: dict, leduc_results: dict):
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print("  CONVERGENCE SUMMARY")
    print("=" * 70)
    
    print("\n  Kuhn Poker (target: exploitability → 0):")
    print(f"  {'Variant':<25s} {'Final Exploit':>15s} {'Time':>10s}")
    print(f"  {'─'*25} {'─'*15} {'─'*10}")
    for name, data in kuhn_results.items():
        print(f"  {name:<25s} {data['final_exploitability']:>15.6f} {data['time']:>9.2f}s")
    
    if leduc_results:
        print(f"\n  Leduc Hold'em:")
        print(f"  {'Variant':<25s} {'Final Exploit':>15s} {'Time':>10s}")
        print(f"  {'─'*25} {'─'*15} {'─'*10}")
        for name, data in leduc_results.items():
            print(f"  {name:<25s} {data['final_exploitability']:>15.6f} {data['time']:>9.2f}s")
    
    print()


def save_results(kuhn_results: dict, leduc_results: dict, output_dir: str):
    """Save training metrics to JSON for the dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    
    def serialize(results):
        """Convert results to JSON-serializable format."""
        serialized = {}
        for name, data in results.items():
            serialized[name] = {
                'iterations': data['metrics']['iterations'],
                'exploitability': data['metrics']['exploitability'],
                'time': data['time'],
                'final_exploitability': data['final_exploitability'],
            }
        return serialized
    
    with open(os.path.join(output_dir, 'kuhn_results.json'), 'w') as f:
        json.dump(serialize(kuhn_results), f, indent=2)
    
    if leduc_results:
        with open(os.path.join(output_dir, 'leduc_results.json'), 'w') as f:
            json.dump(serialize(leduc_results), f, indent=2)
    
    print(f"  Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='GTO Poker Solver — CFR Training')
    parser.add_argument('--game', choices=['kuhn', 'leduc', 'both'], default='both',
                       help='Which game to train on')
    parser.add_argument('--output', default='benchmarks/data',
                       help='Output directory for results')
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     GTO POKER SOLVER — Counterfactual Regret Minimization  ║")
    print("║     Building the Mathematical Framework for SIG Trading    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    kuhn_results = {}
    leduc_results = {}
    
    if args.game in ('kuhn', 'both'):
        kuhn_results = train_kuhn_poker()
    
    if args.game in ('leduc', 'both'):
        leduc_results = train_leduc_holdem()
    
    print_summary(kuhn_results, leduc_results)
    save_results(kuhn_results, leduc_results, args.output)
    
    print("✅ Training complete! Results ready for dashboard visualization.")


if __name__ == '__main__':
    main()
