"""
Generate JSON data for the dashboard from training runs.

Runs all variants on both games and saves results in
dashboard-ready JSON format.

Usage:
    python benchmarks/generate_data.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_engine.kuhn_poker import KuhnPoker
from game_engine.leduc_holdem import LeducHoldem
from game_engine.game_state import Player
from cfr.vanilla_cfr import VanillaCFR
from cfr.mccfr import ExternalSamplingMCCFR
from evaluation.best_response import BestResponse


def run_kuhn():
    """Train all variants on Kuhn Poker."""
    print("═" * 50)
    print("  KUHN POKER BENCHMARKS")
    print("═" * 50)

    game = KuhnPoker()
    results = {}
    
    configs = [
        ('vanilla',  VanillaCFR(),                                    10000, 1000),
        ('cfr_plus', VanillaCFR(cfr_plus=True),                       10000, 1000),
        ('dcfr',     VanillaCFR(dcfr_params={'alpha':1.5,'beta':0,'gamma':2}), 10000, 1000),
    ]
    
    for name, solver, iters, eval_every in configs:
        print(f"\n▸ {name} ({iters:,} iterations)...")
        t0 = time.time()
        m = solver.train_and_track(game, num_iterations=iters, eval_every=eval_every, verbose=True)
        elapsed = time.time() - t0
        
        results[name] = {
            'iterations': m['iterations'],
            'exploitability': m['exploitability'],
            'time': round(elapsed, 2),
            'final_exploitability': m['exploitability'][-1],
        }
        print(f"  → {elapsed:.2f}s, final exploit: {m['exploitability'][-1]:.6f}")
    
    # MCCFR separately (different API)
    print(f"\n▸ es_mccfr (20,000 iterations)...")
    solver_es = ExternalSamplingMCCFR()
    solver_es.set_seed(42)
    t0 = time.time()
    m = solver_es.train_and_track(game, num_iterations=20000, eval_every=2000, verbose=True)
    elapsed = time.time() - t0
    results['es_mccfr'] = {
        'iterations': m['iterations'],
        'exploitability': m['exploitability'],
        'time': round(elapsed, 2),
        'final_exploitability': m['exploitability'][-1],
    }
    print(f"  → {elapsed:.2f}s, final exploit: {m['exploitability'][-1]:.6f}")
    
    return results


def run_leduc():
    """Train all variants on Leduc Hold'em."""
    print("\n" + "═" * 50)
    print("  LEDUC HOLD'EM BENCHMARKS")
    print("═" * 50)

    game = LeducHoldem()
    results = {}
    
    configs = [
        ('vanilla',  VanillaCFR(),                                    1000, 200),
        ('cfr_plus', VanillaCFR(cfr_plus=True),                       1000, 200),
        ('dcfr',     VanillaCFR(dcfr_params={'alpha':1.5,'beta':0,'gamma':2}), 1000, 200),
    ]
    
    for name, solver, iters, eval_every in configs:
        print(f"\n▸ {name} ({iters:,} iterations)...")
        t0 = time.time()
        m = solver.train_and_track(game, num_iterations=iters, eval_every=eval_every, verbose=True)
        elapsed = time.time() - t0
        
        results[name] = {
            'iterations': m['iterations'],
            'exploitability': m['exploitability'],
            'time': round(elapsed, 2),
            'final_exploitability': m['exploitability'][-1],
        }
        print(f"  → {elapsed:.2f}s, final exploit: {m['exploitability'][-1]:.6f}")
    
    return results


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    kuhn = run_kuhn()
    with open(os.path.join(output_dir, 'kuhn_results.json'), 'w') as f:
        json.dump(kuhn, f, indent=2)
    print(f"\n  → Saved kuhn_results.json")
    
    leduc = run_leduc()
    with open(os.path.join(output_dir, 'leduc_results.json'), 'w') as f:
        json.dump(leduc, f, indent=2)
    print(f"\n  → Saved leduc_results.json")
    
    print("\n" + "═" * 50)
    print("  ALL BENCHMARKS COMPLETE ✅")
    print("═" * 50)


if __name__ == '__main__':
    main()
