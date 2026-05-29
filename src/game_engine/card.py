"""
Card and Deck representations for poker games.

Supports both simple integer encoding (for Kuhn/Leduc) and standard 52-card
encoding for Texas Hold'em variants.
"""

from __future__ import annotations
import random
from typing import List, Optional


class Card:
    """
    A playing card with rank and suit.
    
    For Kuhn Poker: ranks are integers (0=Jack, 1=Queen, 2=King), suit is None.
    For Leduc Hold'em: ranks 0-2 (Jack, Queen, King), suits 0-1.
    For Texas Hold'em: ranks 0-12 (2 through Ace), suits 0-3.
    """
    
    # Standard rank names for display
    RANK_NAMES_KUHN = {0: 'J', 1: 'Q', 2: 'K'}
    RANK_NAMES_STANDARD = {
        0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8',
        7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'
    }
    SUIT_NAMES = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
    
    __slots__ = ('rank', 'suit', '_hash')
    
    def __init__(self, rank: int, suit: Optional[int] = None):
        self.rank = rank
        self.suit = suit
        self._hash = hash((rank, suit))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        return self._hash
    
    def __lt__(self, other: Card) -> bool:
        if self.rank != other.rank:
            return self.rank < other.rank
        if self.suit is None or other.suit is None:
            return False
        return self.suit < other.suit
    
    def __repr__(self) -> str:
        rank_str = self.RANK_NAMES_KUHN.get(self.rank, 
                   self.RANK_NAMES_STANDARD.get(self.rank, str(self.rank)))
        if self.suit is not None:
            suit_str = self.SUIT_NAMES.get(self.suit, str(self.suit))
            return f'{rank_str}{suit_str}'
        return rank_str
    
    def to_index(self, num_suits: int = 4) -> int:
        """Convert card to a unique integer index."""
        if self.suit is None:
            return self.rank
        return self.rank * num_suits + self.suit


class Deck:
    """
    A deck of cards that can be shuffled and dealt from.
    
    Supports creation of standard decks for different game variants.
    """
    
    def __init__(self, cards: List[Card]):
        self._original_cards = list(cards)
        self._cards = list(cards)
        self._dealt_index = 0
    
    @classmethod
    def kuhn_deck(cls) -> 'Deck':
        """Create a 3-card Kuhn Poker deck: Jack, Queen, King."""
        return cls([Card(rank=r) for r in range(3)])
    
    @classmethod
    def leduc_deck(cls) -> 'Deck':
        """Create a 6-card Leduc Hold'em deck: J, Q, K in 2 suits."""
        cards = [Card(rank=r, suit=s) for r in range(3) for s in range(2)]
        return cls(cards)
    
    @classmethod
    def standard_deck(cls) -> 'Deck':
        """Create a standard 52-card deck."""
        cards = [Card(rank=r, suit=s) for r in range(13) for s in range(4)]
        return cls(cards)
    
    def shuffle(self, rng: Optional[random.Random] = None) -> None:
        """Shuffle the deck and reset the deal position."""
        self._cards = list(self._original_cards)
        if rng:
            rng.shuffle(self._cards)
        else:
            random.shuffle(self._cards)
        self._dealt_index = 0
    
    def deal(self, n: int = 1) -> List[Card]:
        """Deal n cards from the top of the deck."""
        if self._dealt_index + n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self._cards) - self._dealt_index} remain")
        cards = self._cards[self._dealt_index:self._dealt_index + n]
        self._dealt_index += n
        return cards
    
    def remaining(self) -> List[Card]:
        """Return all undealt cards."""
        return self._cards[self._dealt_index:]
    
    def remove(self, cards_to_remove: List[Card]) -> None:
        """Remove specific cards from the deck (for enumeration)."""
        remove_set = set((c.rank, c.suit) for c in cards_to_remove)
        self._cards = [c for c in self._cards 
                       if (c.rank, c.suit) not in remove_set]
    
    def __len__(self) -> int:
        return len(self._cards) - self._dealt_index
    
    def __repr__(self) -> str:
        remaining = self._cards[self._dealt_index:]
        return f"Deck({', '.join(str(c) for c in remaining)})"
