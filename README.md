# GTO Poker Solver

A from-scratch implementation of **Counterfactual Regret Minimization (CFR)** for computing Game-Theoretically Optimal strategies in imperfect-information poker games.

Built as a deep technical project demonstrating mastery of game theory, algorithm design, and quantitative reasoning.

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install numpy

# Full training (Kuhn Poker + Leduc Hold'em)
python main.py

# Run test suite (22 tests)
python tests/test_phase1.py && python tests/test_phase3.py

# Open visualization dashboard
open dashboard/index.html
```

---

## 🏗️ Architecture

```
gto-poker-solver/
├── main.py                          # Training entrypoint
├── src/
│   ├── game_engine/                 # Game environments
│   │   ├── game_state.py            # Abstract GameState / PokerGame
│   │   ├── card.py                  # Card / Deck primitives
│   │   ├── kuhn_poker.py            # Kuhn Poker (3 cards, verified ✅)
│   │   └── leduc_holdem.py          # Leduc Hold'em (6 cards, 2 rounds)
│   ├── cfr/                         # CFR algorithm variants
│   │   ├── vanilla_cfr.py           # Vanilla CFR, CFR+, DCFR
│   │   └── mccfr.py                 # External / Outcome Sampling MCCFR
│   ├── evaluation/                  # Strategy evaluation
│   │   └── best_response.py         # Info-set-level best response + exploitability
│   ├── hand_eval/                   # Hand evaluation
│   │   └── evaluator.py             # Bitmask 5-7 card evaluator + EHS calculator
│   ├── abstraction/                 # Card & action abstraction
│   │   ├── emd.py                   # Earth Mover's Distance
│   │   ├── clustering.py            # K-Medoids (PAM) with BUILD init
│   │   └── equity_histogram.py      # Equity distribution histograms
│   └── solving/                     # Real-time solving
│       ├── action_translation.py    # Pseudo-harmonic bet mapping
│       ├── blueprint.py             # Blueprint strategy serialization
│       └── subgame_solver.py        # Depth-limited subgame solving
├── dashboard/                       # Interactive web visualization
│   ├── index.html
│   ├── index.css
│   └── app.js
├── tests/                           # Test suites
│   ├── test_phase1.py               # 9 tests — game mechanics & CFR
│   └── test_phase3.py               # 13 tests — hand eval, abstraction, translation
└── benchmarks/data/                 # Training results (JSON)
```

---

## 🔬 Core Algorithms

### Counterfactual Regret Minimization

CFR iteratively minimizes *regret* at each information set. The key insight: if each player minimizes their average counterfactual regret, the average strategy profile converges to a **Nash Equilibrium**.

**Regret Matching**: At each information set, the strategy is proportional to accumulated positive regrets:

```
σ^{T+1}(I, a) = R_+^T(I, a) / Σ_b R_+^T(I, b)
```

**Counterfactual Value**: The expected utility weighted by opponent reach probability:

```
v_i(σ, I) = Σ_{h∈I} π_{-i}^σ(h) · Σ_{z∈Z_h} π^σ(h,z) · u_i(z)
```

### Algorithm Variants

| Variant | Key Modification | Convergence |
|---------|-----------------|-------------|
| **Vanilla CFR** | Full tree traversal, standard regret matching | O(1/√T) |
| **CFR+** | Non-negative regrets, linear averaging | Faster constant |
| **DCFR** | Temporal discounting (α, β, γ hyperparams) | Tunable |
| **External Sampling MCCFR** | Sample chance + opponent, traverse all player actions | O(1/√T), lower per-iteration cost |

### Best Response & Exploitability

Exploitability measures how far a strategy is from Nash:

```
exploit(σ) = max_{σ'_0} u_0(σ'_0, σ_1) + max_{σ'_1} u_1(σ_0, σ'_1)
```

> **Critical implementation detail**: The best response must operate at the *information set level*, not the state level. A state-level BR allows the responder to distinguish between states in the same info set (omniscient play), giving inflated exploitability values.

This solver implements two BR methods:
1. **Exhaustive enumeration** (≤20 info sets): evaluates all 2^k pure strategies
2. **Recursive accumulation** (>20 info sets): aggregates counterfactual values per info set, picks argmax

---

## 📊 Verification Results

### Kuhn Poker (Analytical Nash Equilibrium)

| Metric | Expected | Achieved | Status |
|--------|----------|----------|--------|
| Nash exploitability | 0.000000 | **0.000000** | ✅ |
| Game value (P0) | -0.055556 | **-0.055556** | ✅ |
| Vanilla CFR (10K iters) | < 0.01 | **0.003** | ✅ |
| CFR+ (10K iters) | < 0.01 | **0.003** | ✅ |
| DCFR (10K iters) | < 0.01 | **0.005** | ✅ |
| Subgame solver (5K iters) | < 0.01 | **0.002** | ✅ |

### Test Suite

```
ALL PHASE 1 TESTS PASSED ✅  (9/9)
ALL PHASE 3 TESTS PASSED ✅  (13/13)
Total: 22/22 tests passing
```

---

## 🧠 Key Technical Decisions

### 1. Information-Set-Level Best Response

The standard recursive BR walks the game tree and returns `max(action_values)` at BR player nodes. However, this is a **state-level** best response — the BR player can act differently at different states within the same information set, which is impossible in actual imperfect-information play.

The fix uses two approaches:
- **Small games** (Kuhn): enumerate all pure strategies, each mapping info sets to actions
- **Large games** (Leduc+): accumulate counterfactual-weighted action values per info set, then pick the single best action per info set

### 2. Hand Evaluation with Inverted Ranking

Hand ranks use `lower = better`. Within each category, sub-rankings are inverted: `(12 - rank)` so that higher poker values (Ace-high) produce lower numerical ranks. This was a subtle bug that initially caused incorrect EHS calculations.

### 3. Pseudo-Harmonic Action Translation

For off-tree bet sizes, the mapping satisfies five axioms:
- **Exact mapping** at boundary actions
- **Monotonicity** — larger bets map more to the higher abstract action  
- **Scale invariance** — pot-normalized, so absolute values don't matter
- **Smoothness** — bounded derivative ensures numerical stability
- **Partition of unity** — probabilities sum to 1

### 4. Depth-Limited Subgame Solving

Real-time re-solving uses CFR within a bounded subtree:
- Blueprint strategy provides opponent range estimates at the root
- At depth-limited leaves, blueprint values serve as EV estimates
- CFR+ with linear averaging provides fast convergence within the subgame

---

## 📚 References

1. Zinkevich et al. (2007). "Regret Minimization in Games with Incomplete Information"
2. Tammelin (2014). "Solving Large Imperfect Information Games Using CFR+"
3. Brown & Sandholm (2019). "Solving Imperfect-Information Games via Discounted Regret Minimization"
4. Johanson (2007). "Robust Strategies and Counter-Strategies"
5. Ganzfried & Sandholm (2013). "Action Translation in Extensive-Form Games"
6. Brown & Sandholm (2017). "Safe and Nested Subgame Solving for Imperfect-Information Games"
