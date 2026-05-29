"""
Phase 3 Test Suite — Hand Evaluation, Abstraction, and Action Translation.

Run: python tests/test_phase3.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hand_eval.evaluator import HandEvaluator, EHSCalculator, str_to_card, card_to_str
from abstraction.emd import earth_movers_distance, build_distance_matrix
from abstraction.clustering import KMedoids
from solving.action_translation import ActionTranslator, pseudo_harmonic_map


def test_hand_ranking_order():
    """Verify correct hand ranking hierarchy."""
    ev = HandEvaluator()
    
    # Create representative hands from each category
    hands = {
        'Straight Flush': ['As', 'Ks', 'Qs', 'Js', 'Ts'],
        'Four of a Kind': ['Ah', 'Ad', 'Ac', 'As', '2c'],
        'Full House': ['Kh', 'Kd', 'Ks', 'Qh', 'Qd'],
        'Flush': ['Ah', '9h', '7h', '5h', '3h'],
        'Straight': ['Ts', '9h', '8d', '7c', '6s'],
        'Three of Kind': ['Jh', 'Jd', 'Jc', 'Ks', '2h'],
        'Two Pair': ['Qh', 'Qd', 'Jc', 'Js', '9h'],
        'One Pair': ['Th', 'Td', 'Ac', 'Kh', '5s'],
        'High Card': ['As', 'Kh', 'Qd', 'Jc', '9s'],
    }
    
    ranks = {}
    for name, cards in hands.items():
        card_ints = [str_to_card(c) for c in cards]
        ranks[name] = ev.evaluate_hand(card_ints)
    
    # Verify ordering (lower rank = better hand)
    categories = list(hands.keys())
    for i in range(len(categories) - 1):
        assert ranks[categories[i]] < ranks[categories[i+1]], \
            f"{categories[i]} ({ranks[categories[i]]}) should beat {categories[i+1]} ({ranks[categories[i+1]]})"
    
    print("✅ test_hand_ranking_order passed")


def test_hand_category_names():
    """Verify category detection."""
    ev = HandEvaluator()
    
    tests = [
        (['As', 'Ks', 'Qs', 'Js', 'Ts'], 'Straight Flush'),
        (['Ah', 'Ad', 'Ac', 'As', '2c'], 'Four of a Kind'),
        (['Kh', 'Kd', 'Ks', 'Qh', 'Qd'], 'Full House'),
        (['Ah', '9h', '7h', '5h', '3h'], 'Flush'),
        (['Ts', '9h', '8d', '7c', '6s'], 'Straight'),
        (['As', 'Kh', 'Qd', 'Jc', '9s'], 'High Card'),
    ]
    
    for cards, expected_cat in tests:
        card_ints = [str_to_card(c) for c in cards]
        rank = ev.evaluate_hand(card_ints)
        category = ev.hand_category(rank)
        assert category == expected_cat, \
            f"Expected {expected_cat}, got {category} for {cards}"
    
    print("✅ test_hand_category_names passed")


def test_7card_evaluation():
    """Verify 7-card hand finds the best 5-card combination."""
    ev = HandEvaluator()
    
    # 7 cards containing a straight flush
    cards = [str_to_card(c) for c in ['As', 'Ks', 'Qs', 'Js', 'Ts', '2c', '3d']]
    rank = ev.evaluate_7card(cards)
    assert ev.hand_category(rank) == 'Straight Flush', \
        f"Expected Straight Flush, got {ev.hand_category(rank)}"
    
    # 7 cards where best is full house
    cards = [str_to_card(c) for c in ['Ah', 'Ad', 'Ac', 'Ks', 'Kh', '5c', '3d']]
    rank = ev.evaluate_7card(cards)
    assert ev.hand_category(rank) == 'Full House', \
        f"Expected Full House, got {ev.hand_category(rank)}"
    
    print("✅ test_7card_evaluation passed")


def test_wheel_straight():
    """Verify A-2-3-4-5 (wheel) is detected as a straight."""
    ev = HandEvaluator()
    cards = [str_to_card(c) for c in ['Ah', '2d', '3c', '4s', '5h']]
    rank = ev.evaluate_hand(cards)
    assert ev.hand_category(rank) == 'Straight', \
        f"Expected Straight (wheel), got {ev.hand_category(rank)}"
    
    # A-5 should lose to 6-high straight
    cards2 = [str_to_card(c) for c in ['6h', '5d', '4c', '3s', '2h']]
    rank2 = ev.evaluate_hand(cards2)
    assert rank > rank2, "Wheel should lose to 6-high straight"
    
    print("✅ test_wheel_straight passed")


def test_ehs_basic():
    """Basic EHS sanity checks."""
    ehs_calc = EHSCalculator()
    
    # AA on a board where 72o can't improve much
    # Board with no 2s or 7s, so 72o has nothing
    hole = [str_to_card(c) for c in ['Ah', 'Ad']]
    board = [str_to_card(c) for c in ['5c', '8s', 'Th', 'Kd', 'Jc']]
    ehs = ehs_calc.compute_ehs_exact(hole, board)
    assert ehs > 0.6, f"AA should have high EHS on this board, got {ehs:.3f}"
    
    # 72o on same board should have very low EHS (no pairs, no draws)
    hole2 = [str_to_card(c) for c in ['7d', '2d']]
    ehs2 = ehs_calc.compute_ehs_exact(hole2, board)
    assert ehs > ehs2, f"AA ({ehs:.3f}) should beat 72o ({ehs2:.3f})"
    
    print(f"✅ test_ehs_basic passed (AA={ehs:.3f}, 72={ehs2:.3f})")


def test_emd_identical():
    """EMD of identical distributions should be 0."""
    hist = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    assert abs(earth_movers_distance(hist, hist)) < 1e-10
    print("✅ test_emd_identical passed")


def test_emd_opposite():
    """EMD of maximally different distributions should be large."""
    hist1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    hist2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    emd = earth_movers_distance(hist1, hist2)
    assert emd > 3.5, f"EMD should be large for opposite distributions, got {emd}"
    print(f"✅ test_emd_opposite passed (EMD={emd:.3f})")


def test_emd_triangle_inequality():
    """EMD should satisfy triangle inequality."""
    h1 = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    h2 = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
    h3 = np.array([0.1, 0.1, 0.1, 0.3, 0.4])
    
    d12 = earth_movers_distance(h1, h2)
    d23 = earth_movers_distance(h2, h3)
    d13 = earth_movers_distance(h1, h3)
    
    assert d13 <= d12 + d23 + 1e-10, \
        f"Triangle inequality violated: d(1,3)={d13} > d(1,2)+d(2,3)={d12+d23}"
    print("✅ test_emd_triangle_inequality passed")


def test_kmedoids_basic():
    """K-Medoids should cluster similar items together."""
    # Create 2 clear clusters
    distances = np.array([
        [0, 0.1, 0.1, 5.0, 5.0],
        [0.1, 0, 0.1, 5.0, 5.0],
        [0.1, 0.1, 0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 0, 0.1],
        [5.0, 5.0, 5.0, 0.1, 0],
    ])
    
    km = KMedoids(n_clusters=2)
    labels = km.fit(distances)
    
    # Items 0-2 should be in one cluster, 3-4 in another
    assert labels[0] == labels[1] == labels[2], "Items 0-2 should be in same cluster"
    assert labels[3] == labels[4], "Items 3-4 should be in same cluster"
    assert labels[0] != labels[3], "The two clusters should be different"
    
    score = km.silhouette_score(distances)
    assert score > 0.5, f"Silhouette score should be high for clear clusters, got {score:.3f}"
    
    print(f"✅ test_kmedoids_basic passed (silhouette={score:.3f})")


def test_action_translation_exact_mapping():
    """Axiom 1: exact mapping at boundary sizes."""
    p_low, p_high = pseudo_harmonic_map(0.5, 0.5, 1.0)
    assert abs(p_low - 1.0) < 1e-10, f"P(low|low) should be 1.0, got {p_low}"
    
    p_low, p_high = pseudo_harmonic_map(1.0, 0.5, 1.0)
    assert abs(p_high - 1.0) < 1e-10, f"P(high|high) should be 1.0, got {p_high}"
    
    print("✅ test_action_translation_exact_mapping passed")


def test_action_translation_monotonicity():
    """Axiom 2: monotonicity — larger bets map more to high action."""
    probs = []
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        bet = 0.5 + t * 0.5  # Between 0.5 and 1.0
        p_low, _ = pseudo_harmonic_map(bet, 0.5, 1.0)
        probs.append(p_low)
    
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i+1], \
            f"Monotonicity violated: P(low) at {i}: {probs[i]} < P(low) at {i+1}: {probs[i+1]}"
    
    print("✅ test_action_translation_monotonicity passed")


def test_action_translation_sum_to_one():
    """Axiom 5: probabilities sum to 1."""
    for bet in [0.55, 0.7, 0.8, 0.95]:
        p_low, p_high = pseudo_harmonic_map(bet, 0.5, 1.0)
        assert abs(p_low + p_high - 1.0) < 1e-10, \
            f"Probabilities don't sum to 1 at bet={bet}: {p_low + p_high}"
    
    print("✅ test_action_translation_sum_to_one passed")


def test_action_translator_class():
    """Test the ActionTranslator class with multiple abstract sizes."""
    translator = ActionTranslator([0.5, 1.0, 2.0])
    
    # Exact matches
    result = translator.translate(0.5)
    assert abs(result[0.5] - 1.0) < 1e-10
    
    result = translator.translate(2.0)
    assert abs(result[2.0] - 1.0) < 1e-10
    
    # Between sizes
    result = translator.translate(0.75)
    assert 0.5 in result and 1.0 in result
    assert abs(sum(result.values()) - 1.0) < 1e-10
    
    # Below minimum
    result = translator.translate(0.3)
    assert abs(result[0.5] - 1.0) < 1e-10
    
    print("✅ test_action_translator_class passed")


if __name__ == '__main__':
    print("=== Hand Evaluator Tests ===")
    test_hand_ranking_order()
    test_hand_category_names()
    test_7card_evaluation()
    test_wheel_straight()
    test_ehs_basic()
    
    print("\n=== Abstraction Tests ===")
    test_emd_identical()
    test_emd_opposite()
    test_emd_triangle_inequality()
    test_kmedoids_basic()
    
    print("\n=== Action Translation Tests ===")
    test_action_translation_exact_mapping()
    test_action_translation_monotonicity()
    test_action_translation_sum_to_one()
    test_action_translator_class()
    
    print("\n" + "=" * 60)
    print("  ALL PHASE 3 TESTS PASSED ✅")
    print("=" * 60)
