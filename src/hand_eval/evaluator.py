"""
High-Speed Poker Hand Evaluation.

Uses a lookup table based approach for fast 5-7 card hand evaluation.
Supports standard Texas Hold'em hand rankings:
    Royal Flush > Straight Flush > Four of a Kind > Full House > Flush >
    Straight > Three of a Kind > Two Pair > One Pair > High Card

Hand categories are encoded as integers (lower = better):
    0: Straight Flush  (includes Royal Flush)
    1: Four of a Kind
    2: Full House
    3: Flush
    4: Straight
    5: Three of a Kind
    6: Two Pair
    7: One Pair
    8: High Card

Within each category, hands are ranked by relevant kickers.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations


# Card encoding: 0-51
# card = rank * 4 + suit
# rank: 0=2, 1=3, 2=4, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
# suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades

RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUIT_NAMES = ['c', 'd', 'h', 's']
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']

NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52


def card_to_str(card: int) -> str:
    """Convert card integer to string like 'Ah', 'Ts', '2c'."""
    rank = card // 4
    suit = card % 4
    return RANK_NAMES[rank] + SUIT_NAMES[suit]


def str_to_card(s: str) -> int:
    """Convert string like 'Ah' to card integer."""
    rank_char = s[0].upper()
    suit_char = s[1].lower()
    rank = RANK_NAMES.index(rank_char)
    suit = SUIT_NAMES.index(suit_char)
    return rank * 4 + suit


def card_rank(card: int) -> int:
    """Get the rank (0-12) of a card."""
    return card // 4


def card_suit(card: int) -> int:
    """Get the suit (0-3) of a card."""
    return card % 4


class HandEvaluator:
    """
    Fast poker hand evaluator using bitmask techniques.
    
    Evaluates 5-card hands and finds the best 5-card hand from 7 cards.
    
    Returns a hand rank where LOWER is BETTER.
    Category is encoded in upper bits, rank within category in lower bits.
    
    Usage:
        evaluator = HandEvaluator()
        rank1 = evaluator.evaluate_hand([0, 4, 8, 12, 16])  # 5 cards
        rank2 = evaluator.evaluate_7card([0, 4, 8, 12, 16, 20, 24])  # 7 cards
        
        # Lower rank = better hand
        if rank1 < rank2:
            print("Hand 1 wins!")
    """
    
    # Hand categories (lower = better)
    STRAIGHT_FLUSH = 0
    FOUR_OF_A_KIND = 1
    FULL_HOUSE = 2
    FLUSH = 3
    STRAIGHT = 4
    THREE_OF_A_KIND = 5
    TWO_PAIR = 6
    ONE_PAIR = 7
    HIGH_CARD = 8
    
    CATEGORY_NAMES = [
        'Straight Flush', 'Four of a Kind', 'Full House',
        'Flush', 'Straight', 'Three of a Kind',
        'Two Pair', 'One Pair', 'High Card'
    ]
    
    # Multiplier for category encoding
    CATEGORY_SHIFT = 1 << 20  # Each category separated by 2^20
    
    def __init__(self):
        """Initialize with precomputed lookup tables."""
        self._flush_table = self._build_flush_table()
        self._unique5_table = self._build_unique5_table()
    
    def _build_flush_table(self) -> dict:
        """Build lookup for 5-card combinations with all same suit."""
        # Maps a bitmask of 5 ranks to the hand rank
        table = {}
        for combo in combinations(range(13), 5):
            bitmask = sum(1 << r for r in combo)
            rank = self._evaluate_flush_ranks(list(combo))
            table[bitmask] = rank
        return table
    
    def _build_unique5_table(self) -> dict:
        """Build lookup for 5 unique ranks (potential straight)."""
        table = {}
        for combo in combinations(range(13), 5):
            bitmask = sum(1 << r for r in combo)
            rank = self._evaluate_unique_ranks(list(combo))
            table[bitmask] = rank
        return table
    
    def _evaluate_flush_ranks(self, ranks: List[int]) -> int:
        """Evaluate a flush hand given 5 sorted ranks."""
        ranks = sorted(ranks, reverse=True)
        # Check for straight flush
        if self._is_straight(ranks):
            # Straight flush rank = category + inverted high card
            # Lower rank = better, higher high card = better, so invert
            high = ranks[0] if not (ranks == [12, 3, 2, 1, 0]) else 3  # A-5 wheel
            return self.STRAIGHT_FLUSH * self.CATEGORY_SHIFT + (12 - high)
        
        # Regular flush — rank by individual cards (inverted so higher cards = lower rank)
        value = 0
        for i, r in enumerate(ranks):
            value += (12 - r) * (13 ** (4 - i))
        return self.FLUSH * self.CATEGORY_SHIFT + value
    
    def _evaluate_unique_ranks(self, ranks: List[int]) -> int:
        """Evaluate 5 unique ranks (no pairs) — could be a straight or high card."""
        ranks = sorted(ranks, reverse=True)
        if self._is_straight(ranks):
            high = ranks[0] if not (ranks == [12, 3, 2, 1, 0]) else 3
            return self.STRAIGHT * self.CATEGORY_SHIFT + (12 - high)
        
        # High card (inverted: higher cards = lower/better rank)
        value = 0
        for i, r in enumerate(ranks):
            value += (12 - r) * (13 ** (4 - i))
        return self.HIGH_CARD * self.CATEGORY_SHIFT + value
    
    @staticmethod
    def _is_straight(sorted_ranks: List[int]) -> bool:
        """Check if sorted (descending) ranks form a straight."""
        if sorted_ranks[0] - sorted_ranks[4] == 4:
            return True
        # A-2-3-4-5 wheel
        if sorted_ranks == [12, 3, 2, 1, 0]:
            return True
        return False
    
    def evaluate_hand(self, cards: List[int]) -> int:
        """
        Evaluate a 5-card poker hand.
        
        Args:
            cards: List of 5 card integers (0-51)
        
        Returns:
            Hand rank (lower = better hand)
        """
        assert len(cards) == 5, f"Expected 5 cards, got {len(cards)}"
        
        ranks = [card_rank(c) for c in cards]
        suits = [card_suit(c) for c in cards]
        
        is_flush = len(set(suits)) == 1
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        unique_ranks = len(rank_counts)
        
        if is_flush:
            bitmask = sum(1 << r for r in set(ranks))
            return self._flush_table[bitmask]
        
        if unique_ranks == 5:
            # No pairs — straight or high card
            bitmask = sum(1 << r for r in ranks)
            return self._unique5_table[bitmask]
        
        # Has pairs/trips/quads
        return self._evaluate_paired(ranks, rank_counts)
    
    def _evaluate_paired(self, ranks: List[int], rank_counts: dict) -> int:
        """Evaluate a hand with pairs, trips, or quads."""
        count_groups = {}
        for rank, count in rank_counts.items():
            if count not in count_groups:
                count_groups[count] = []
            count_groups[count].append(rank)
        
        # Sort each group by rank (descending)
        for count in count_groups:
            count_groups[count].sort(reverse=True)
        
        if 4 in count_groups:
            # Four of a kind (inverted: higher quad rank = better = lower value)
            quad = count_groups[4][0]
            kicker = count_groups[1][0]
            return self.FOUR_OF_A_KIND * self.CATEGORY_SHIFT + (12 - quad) * 13 + (12 - kicker)
        
        if 3 in count_groups:
            if 2 in count_groups:
                # Full house
                trips = count_groups[3][0]
                pair = count_groups[2][0]
                return self.FULL_HOUSE * self.CATEGORY_SHIFT + (12 - trips) * 13 + (12 - pair)
            else:
                # Three of a kind
                trips = count_groups[3][0]
                kickers = count_groups[1]
                return (self.THREE_OF_A_KIND * self.CATEGORY_SHIFT + 
                       (12 - trips) * (13**2) + (12 - kickers[0]) * 13 + (12 - kickers[1]))
        
        if 2 in count_groups:
            pairs = count_groups[2]
            if len(pairs) == 2:
                # Two pair
                high_pair = max(pairs)
                low_pair = min(pairs)
                kicker = count_groups[1][0]
                return (self.TWO_PAIR * self.CATEGORY_SHIFT + 
                       (12 - high_pair) * (13**2) + (12 - low_pair) * 13 + (12 - kicker))
            else:
                # One pair
                pair = pairs[0]
                kickers = count_groups[1]
                return (self.ONE_PAIR * self.CATEGORY_SHIFT + 
                       (12 - pair) * (13**3) + (12 - kickers[0]) * (13**2) + 
                       (12 - kickers[1]) * 13 + (12 - kickers[2]))
        
        # Should not reach here if logic is correct
        raise ValueError("Unexpected hand structure")
    
    def evaluate_7card(self, cards: List[int]) -> int:
        """
        Find the best 5-card hand from 7 cards.
        
        Evaluates all C(7,5) = 21 combinations and returns the best.
        
        Args:
            cards: List of 7 card integers (0-51)
        
        Returns:
            Best hand rank (lower = better)
        """
        assert len(cards) == 7, f"Expected 7 cards, got {len(cards)}"
        
        best_rank = float('inf')
        for combo in combinations(cards, 5):
            rank = self.evaluate_hand(list(combo))
            best_rank = min(best_rank, rank)
        return best_rank
    
    def evaluate_best(self, cards: List[int]) -> int:
        """
        Evaluate the best possible hand from any number of cards (5-7).
        """
        if len(cards) == 5:
            return self.evaluate_hand(cards)
        elif len(cards) == 6:
            best = float('inf')
            for combo in combinations(cards, 5):
                best = min(best, self.evaluate_hand(list(combo)))
            return best
        elif len(cards) == 7:
            return self.evaluate_7card(cards)
        else:
            raise ValueError(f"Expected 5-7 cards, got {len(cards)}")
    
    def hand_category(self, rank: int) -> str:
        """Get the category name from a hand rank."""
        category = rank // self.CATEGORY_SHIFT
        return self.CATEGORY_NAMES[category]
    
    def hand_description(self, cards: List[int]) -> str:
        """Get a human-readable description of a hand."""
        rank = self.evaluate_best(cards)
        cards_str = ' '.join(card_to_str(c) for c in cards)
        category = self.hand_category(rank)
        return f"{cards_str} → {category}"


class EHSCalculator:
    """
    Expected Hand Strength (EHS) Calculator.
    
    EHS(hand, board) = P(win) + 0.5 * P(tie)
    
    Computed via Monte Carlo enumeration of opponent hands
    and remaining board cards.
    
    This is the foundation of card abstraction — hands with similar
    EHS values are strategically similar and can be grouped together.
    """
    
    def __init__(self, evaluator: Optional[HandEvaluator] = None):
        self.evaluator = evaluator or HandEvaluator()
        self.rng = np.random.RandomState(42)
    
    def compute_ehs(
        self,
        hole_cards: List[int],
        board: List[int],
        num_rollouts: int = 1000,
    ) -> float:
        """
        Compute Expected Hand Strength via Monte Carlo.
        
        Args:
            hole_cards: Player's 2 hole cards
            board: Community cards (0 for preflop, 3 for flop, 4 for turn, 5 for river)
            num_rollouts: Number of random opponent hands + boards to sample
        
        Returns:
            EHS value in [0, 1]
        """
        used_cards = set(hole_cards + board)
        remaining = [c for c in range(52) if c not in used_cards]
        
        wins = 0.0
        total = 0.0
        
        cards_to_deal = 5 - len(board)  # Board cards still needed
        
        for _ in range(num_rollouts):
            # Sample remaining cards: 2 for opponent + board cards
            sampled = self.rng.choice(remaining, size=2 + cards_to_deal, replace=False)
            opp_cards = list(sampled[:2])
            extra_board = list(sampled[2:])
            
            full_board = board + extra_board
            
            my_hand = hole_cards + full_board
            opp_hand = opp_cards + full_board
            
            my_rank = self.evaluator.evaluate_best(my_hand)
            opp_rank = self.evaluator.evaluate_best(opp_hand)
            
            if my_rank < opp_rank:
                wins += 1.0
            elif my_rank == opp_rank:
                wins += 0.5
            
            total += 1.0
        
        return wins / total
    
    def compute_ehs_exact(
        self,
        hole_cards: List[int],
        board: List[int],
    ) -> float:
        """
        Compute exact EHS by enumerating all possible opponent hands.
        
        Only feasible on the river (no more board cards to deal).
        """
        assert len(board) == 5, "Exact EHS only available on the river"
        
        used_cards = set(hole_cards + board)
        remaining = [c for c in range(52) if c not in used_cards]
        
        my_hand = hole_cards + board
        my_rank = self.evaluator.evaluate_best(my_hand)
        
        wins = 0.0
        total = 0.0
        
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                opp_cards = [remaining[i], remaining[j]]
                opp_hand = opp_cards + board
                opp_rank = self.evaluator.evaluate_best(opp_hand)
                
                if my_rank < opp_rank:
                    wins += 1.0
                elif my_rank == opp_rank:
                    wins += 0.5
                total += 1.0
        
        return wins / total
    
    def compute_ehs_vector(
        self,
        board: List[int],
        num_rollouts: int = 500,
    ) -> dict:
        """
        Compute EHS for ALL possible hole card pairs given a board.
        
        Returns:
            Dict mapping (card1, card2) → EHS value
        """
        used = set(board)
        available = [c for c in range(52) if c not in used]
        
        result = {}
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                hole = [available[i], available[j]]
                ehs = self.compute_ehs(hole, board, num_rollouts)
                result[(available[i], available[j])] = ehs
        
        return result
