"""
Leduc Hold'em — A richer imperfect-information poker game.

Rules:
- 6-card deck: Jack(0), Queen(1), King(2) × 2 suits
- 2 players each ante 1 chip
- Each player is dealt 1 private card
- Round 1 (pre-flop):
  - P0 acts: CHECK or BET (2 chips)
  - P1 responds accordingly (FOLD/CALL or CHECK/BET)
  - If CHECK-BET, P0 can FOLD or CALL
  - Max 2 bets per round (bet + raise)
- 1 community card is dealt (the "flop")
- Round 2 (post-flop):
  - Same betting structure but bet size is 4 chips
  - Max 2 bets per round
- Showdown:
  - Pair (private card matches community) beats no pair
  - If both pair or neither pairs, higher card wins
  - If equal, it's a draw (split pot)

This game has ~936 information sets, making it a good intermediate
testbed between Kuhn (12 info sets) and full Hold'em (~10^14 info sets).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from .game_state import GameState, PokerGame, Player, Action
from .card import Card


class LeducState(GameState):
    """
    A state in Leduc Hold'em.
    
    Attributes:
        cards: Tuple of (P0_card, P1_card) — Card objects with rank and suit
        community_card: Optional community card
        history: List of action strings per round [round1_actions, round2_actions]
        pot: List of [P0_contribution, P1_contribution]
        round_num: Current round (0 = pre-flop, 1 = post-flop)
        round_bets: Number of bets/raises in current round
        is_dealt: Whether hole cards are dealt
        community_dealt: Whether community card is dealt
    """
    
    # Bet sizes per round
    BET_SIZES = [2, 4]  # Round 1: 2 chips, Round 2: 4 chips
    MAX_BETS_PER_ROUND = 2
    
    CARD_NAMES = {0: 'J', 1: 'Q', 2: 'K'}
    
    ACTION_CHARS = {
        Action.CHECK: 'p',
        Action.BET: 'b',
        Action.FOLD: 'f',
        Action.CALL: 'c',
        Action.RAISE: 'r',
    }
    
    def __init__(
        self,
        cards: Optional[Tuple[int, int]] = None,       # (P0_rank, P1_rank) 
        suits: Optional[Tuple[int, int]] = None,        # (P0_suit, P1_suit)
        community_card: Optional[int] = None,            # community rank
        community_suit: Optional[int] = None,            # community suit
        history: Optional[List[str]] = None,
        pot: Optional[List[int]] = None,
        round_num: int = 0,
        round_bets: int = 0,
        is_dealt: bool = False,
        community_dealt: bool = False,
        _all_cards: Optional[List[Tuple[int, int]]] = None,  # Full deck for chance enumeration
    ):
        self.cards = cards
        self.suits = suits
        self.community_card = community_card
        self.community_suit = community_suit
        self.history = history if history is not None else ['', '']
        self.pot = pot if pot is not None else [1, 1]
        self.round_num = round_num
        self.round_bets = round_bets
        self.is_dealt = is_dealt
        self.community_dealt = community_dealt
        self._all_cards = _all_cards
    
    @property
    def num_players(self) -> int:
        return 2
    
    def _clone(self, **kwargs) -> 'LeducState':
        """Create a copy with specified fields overridden."""
        defaults = {
            'cards': self.cards,
            'suits': self.suits,
            'community_card': self.community_card,
            'community_suit': self.community_suit,
            'history': [h for h in self.history],  # Deep copy
            'pot': list(self.pot),
            'round_num': self.round_num,
            'round_bets': self.round_bets,
            'is_dealt': self.is_dealt,
            'community_dealt': self.community_dealt,
            '_all_cards': self._all_cards,
        }
        defaults.update(kwargs)
        return LeducState(**defaults)
    
    def is_terminal(self) -> bool:
        if not self.is_dealt:
            return False
        
        # Check for fold in current round history
        for round_h in self.history:
            if 'f' in round_h:
                return True
        
        # Check for completed second round
        if self.round_num == 1 and self._round_complete():
            return True
        
        return False
    
    def _round_complete(self) -> bool:
        """Check if the current betting round is complete."""
        h = self.history[self.round_num]
        if len(h) < 2:
            return False
        
        # Round ends after: pp, bc, brc, brf, bf
        if h == 'pp':
            return True
        if h.endswith('c') and len(h) >= 2:
            return True
        if h.endswith('f'):
            return True
        
        return False
    
    def is_chance_node(self) -> bool:
        # Need to deal hole cards
        if not self.is_dealt:
            return True
        # Need to deal community card between rounds
        if self.round_num == 0 and self._round_complete() and not self.community_dealt:
            return True
        return False
    
    def current_player(self) -> Player:
        if self.is_chance_node():
            return Player.CHANCE
        
        h = self.history[self.round_num]
        # Player 0 acts at even positions, Player 1 at odd positions
        return Player.PLAYER_0 if len(h) % 2 == 0 else Player.PLAYER_1
    
    def legal_actions(self) -> List[Action]:
        if self.is_terminal() or self.is_chance_node():
            return []
        
        h = self.history[self.round_num]
        
        if len(h) == 0:
            # First to act: check or bet
            return [Action.CHECK, Action.BET]
        
        last_action = h[-1]
        
        if last_action == 'p':
            # After a check: check or bet
            return [Action.CHECK, Action.BET]
        elif last_action == 'b' or last_action == 'r':
            # After a bet/raise: fold, call, or raise (if under max bets)
            actions = [Action.FOLD, Action.CALL]
            if self.round_bets < self.MAX_BETS_PER_ROUND:
                actions.append(Action.RAISE)
            return actions
        
        return []
    
    def apply_action(self, action: Action) -> 'LeducState':
        char = self.ACTION_CHARS[action]
        new_history = [h for h in self.history]
        new_history[self.round_num] += char
        new_pot = list(self.pot)
        new_round_bets = self.round_bets
        new_round_num = self.round_num
        
        bet_size = self.BET_SIZES[self.round_num]
        acting_player = int(self.current_player())
        
        if action == Action.BET:
            new_pot[acting_player] += bet_size
            new_round_bets += 1
        elif action == Action.RAISE:
            # Call the previous bet + add a raise
            opponent = 1 - acting_player
            call_amount = new_pot[opponent] - new_pot[acting_player]
            new_pot[acting_player] += call_amount + bet_size
            new_round_bets += 1
        elif action == Action.CALL:
            opponent = 1 - acting_player
            call_amount = new_pot[opponent] - new_pot[acting_player]
            new_pot[acting_player] += call_amount
        
        new_state = self._clone(
            history=new_history,
            pot=new_pot,
            round_bets=new_round_bets,
            round_num=new_round_num,
        )
        
        # If round is now complete and we're in round 0, prepare for community card
        if new_round_num == 0 and new_state._round_complete():
            # Community card dealing will happen at the next chance node
            pass
        
        return new_state
    
    def terminal_utility(self, player: Player) -> float:
        assert self.is_terminal()
        
        # Check for fold
        for round_idx, round_h in enumerate(self.history):
            if 'f' in round_h:
                # Who folded?
                fold_pos = round_h.index('f')
                folder = Player.PLAYER_0 if fold_pos % 2 == 0 else Player.PLAYER_1
                winner = Player(1 - folder)
                
                if player == winner:
                    return float(self.pot[1 - player])
                else:
                    return float(-self.pot[player])
        
        # Showdown
        p0_card = self.cards[0]
        p1_card = self.cards[1]
        comm = self.community_card
        
        # Check for pairs
        p0_pair = (p0_card == comm)
        p1_pair = (p1_card == comm)
        
        if p0_pair and not p1_pair:
            winner = Player.PLAYER_0
        elif p1_pair and not p0_pair:
            winner = Player.PLAYER_1
        elif p0_card > p1_card:
            winner = Player.PLAYER_0
        elif p1_card > p0_card:
            winner = Player.PLAYER_1
        else:
            # Draw — split pot (utility = 0)
            return 0.0
        
        if player == winner:
            return float(self.pot[1 - player])
        else:
            return float(-self.pot[player])
    
    def information_set_key(self) -> str:
        """
        Information set key for Leduc Hold'em.
        
        Player can see:
        - Their own card rank (not opponent's)
        - The community card (if dealt)
        - Full action history
        
        Format: "{my_card}|{community_or_none}|{round1_history}/{round2_history}"
        """
        player = int(self.current_player())
        my_card = self.cards[player]
        
        if self.community_dealt and self.community_card is not None:
            comm_str = str(self.community_card)
        else:
            comm_str = '_'
        
        history_str = '/'.join(self.history)
        return f"{my_card}|{comm_str}|{history_str}"
    
    def chance_outcomes(self) -> List[Tuple['LeducState', float]]:
        assert self.is_chance_node()
        
        if not self.is_dealt:
            return self._deal_hole_cards()
        else:
            return self._deal_community_card()
    
    def _deal_hole_cards(self) -> List[Tuple['LeducState', float]]:
        """Enumerate all possible hole card deals."""
        # Deck: 3 ranks × 2 suits = 6 cards
        all_cards = [(r, s) for r in range(3) for s in range(2)]
        outcomes = []
        
        for i, (r0, s0) in enumerate(all_cards):
            for j, (r1, s1) in enumerate(all_cards):
                if i == j:
                    continue
                state = self._clone(
                    cards=(r0, r1),
                    suits=(s0, s1),
                    is_dealt=True,
                    _all_cards=all_cards,
                )
                # 6 cards, choose 2 in order: 6*5 = 30 deals
                outcomes.append((state, 1.0 / 30.0))
        
        return outcomes
    
    def _deal_community_card(self) -> List[Tuple['LeducState', float]]:
        """Enumerate all possible community cards from remaining deck."""
        # Remove dealt hole cards
        dealt = set()
        dealt.add((self.cards[0], self.suits[0]))
        dealt.add((self.cards[1], self.suits[1]))
        
        remaining = [(r, s) for r in range(3) for s in range(2) 
                      if (r, s) not in dealt]
        
        prob = 1.0 / len(remaining)
        outcomes = []
        
        for r, s in remaining:
            state = self._clone(
                community_card=r,
                community_suit=s,
                community_dealt=True,
                round_num=1,
                round_bets=0,
            )
            outcomes.append((state, prob))
        
        return outcomes
    
    def __repr__(self) -> str:
        names = {0: 'J', 1: 'Q', 2: 'K'}
        if not self.is_dealt:
            return "LeducState(dealing...)"
        c0 = names[self.cards[0]]
        c1 = names[self.cards[1]]
        comm = names.get(self.community_card, '?') if self.community_dealt else '_'
        return (f"LeducState(P0={c0}, P1={c1}, comm={comm}, "
                f"h={self.history}, pot={self.pot}, round={self.round_num})")


class LeducHoldem(PokerGame):
    """
    Leduc Hold'em game factory.
    
    A 6-card poker variant with ~936 information sets. More complex than
    Kuhn Poker but still tractable for full CFR traversal, making it
    ideal for benchmarking CFR variants.
    """
    
    def initial_state(self) -> LeducState:
        return LeducState()
    
    def name(self) -> str:
        return "Leduc Hold'em"
    
    @property
    def num_players(self) -> int:
        return 2
