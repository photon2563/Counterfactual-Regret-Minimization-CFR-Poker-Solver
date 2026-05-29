"""
Kuhn Poker — The simplest nontrivial imperfect-information game.

Rules:
- 3-card deck: Jack (0), Queen (1), King (2)
- 2 players each ante 1 chip
- Each player is dealt 1 private card
- Player 0 acts first: can CHECK or BET (1 chip)
- Player 1 responds:
  - If P0 checked: CHECK (showdown) or BET (1 chip)
  - If P0 bet: FOLD (P0 wins antes) or CALL (1 chip, showdown)
- If P0 checked and P1 bet: P0 can FOLD or CALL
- Showdown: higher card wins the pot

Known Nash Equilibrium (for verification):
- Player 0 with Jack: Check, then fold to a bet (never bet)
- Player 0 with Queen: Check, then call a bet with probability 1/3
- Player 0 with King: Bet with probability 3*alpha, check-call rest
  (where alpha ∈ [0, 1/3] — usually alpha = 1/3 → bet with probability 1)
- Player 1 with Jack: If checked to, bet with probability 1/3
- Player 1 with Queen: Always check/fold
- Player 1 with King: Always bet/call

Game value: Player 0's EV = -1/18 ≈ -0.0556

References:
    Kuhn, H.W. (1950). "Simplified Poker"
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from .game_state import GameState, PokerGame, Player, Action


class KuhnState(GameState):
    """
    A state in Kuhn Poker.
    
    Attributes:
        cards: Tuple of (P0_card, P1_card) where each is 0(J), 1(Q), 2(K)
        history: String of actions taken so far (e.g., "cb" = check, bet)
        pot: List of [P0_contribution, P1_contribution]
    """
    
    # Map actions to history characters for information set keys
    ACTION_CHARS = {
        Action.CHECK: 'p',   # pass
        Action.BET: 'b',
        Action.FOLD: 'f',
        Action.CALL: 'c',
    }
    
    def __init__(
        self,
        cards: Optional[Tuple[int, int]] = None,
        history: str = '',
        pot: Optional[List[int]] = None
    ):
        self.cards = cards
        self.history = history
        self.pot = pot if pot is not None else [1, 1]  # Both ante 1
    
    @property
    def num_players(self) -> int:
        return 2
    
    def is_terminal(self) -> bool:
        """
        Terminal states in Kuhn Poker:
        - "pp"  : both check → showdown
        - "bp"  : bet, fold → bettor wins (but we use 'f' for fold)
        - "bf"  : bet, fold → bettor wins  
        - "bc"  : bet, call → showdown
        - "pbf" : check, bet, fold → bettor wins
        - "pbc" : check, bet, call → showdown
        """
        if self.cards is None:
            return False
        h = self.history
        return h in ('pp', 'bf', 'bc', 'pbf', 'pbc')
    
    def is_chance_node(self) -> bool:
        """Chance node = cards haven't been dealt yet."""
        return self.cards is None
    
    def current_player(self) -> Player:
        if self.cards is None:
            return Player.CHANCE
        # Player 0 acts at history lengths 0 and 2, Player 1 at length 1
        if len(self.history) == 0:
            return Player.PLAYER_0
        elif len(self.history) == 1:
            return Player.PLAYER_1
        else:
            # Only reached when history is "pb" — Player 0 responds
            return Player.PLAYER_0
    
    def legal_actions(self) -> List[Action]:
        if self.is_terminal() or self.is_chance_node():
            return []
        
        h = self.history
        if len(h) == 0:
            # Player 0's first action: check or bet
            return [Action.CHECK, Action.BET]
        elif len(h) == 1:
            if h == 'p':
                # After P0 checks: P1 can check or bet
                return [Action.CHECK, Action.BET]
            else:  # h == 'b'
                # After P0 bets: P1 can fold or call
                return [Action.FOLD, Action.CALL]
        else:  # len(h) == 2, must be "pb"
            # After P0 checks, P1 bets: P0 can fold or call
            return [Action.FOLD, Action.CALL]
    
    def apply_action(self, action: Action) -> 'KuhnState':
        char = self.ACTION_CHARS[action]
        new_history = self.history + char
        new_pot = list(self.pot)
        
        # Update pot contributions
        if action == Action.BET:
            acting_player = self.current_player()
            new_pot[acting_player] += 1
        elif action == Action.CALL:
            acting_player = self.current_player()
            new_pot[acting_player] += 1
        
        return KuhnState(
            cards=self.cards,
            history=new_history,
            pot=new_pot
        )
    
    def terminal_utility(self, player: Player) -> float:
        """
        Calculate payoff at a terminal node.
        
        Returns utility relative to antes (so -1 means losing your ante,
        +1 means winning opponent's ante, +2 means winning ante + bet).
        """
        assert self.is_terminal(), "Not a terminal state"
        assert self.cards is not None
        
        h = self.history
        p0_card, p1_card = self.cards
        
        if h == 'bf' or h == 'pbf':
            # Someone folded after a bet
            if h == 'bf':
                # P0 bet, P1 folded → P0 wins P1's ante
                winner = Player.PLAYER_0
            else:
                # P0 checked, P1 bet, P0 folded → P1 wins P0's ante
                winner = Player.PLAYER_1
            
            if player == winner:
                return 1.0
            else:
                return -1.0
        
        # Showdown: pp, bc, pbc
        # Higher card wins
        if p0_card > p1_card:
            showdown_winner = Player.PLAYER_0
        else:
            showdown_winner = Player.PLAYER_1
        
        # Calculate the pot each player contributed
        if player == showdown_winner:
            return float(self.pot[1 - player])  # Win opponent's contribution
        else:
            return float(-self.pot[player])  # Lose own contribution
    
    def information_set_key(self) -> str:
        """
        Information set = player's card + public action history.
        
        The player can see their own card and all actions taken,
        but NOT the opponent's card.
        
        Format: "{card_rank}:{action_history}"
        Example: "2:pb" means player holds King, action history is pass-bet
        """
        player = self.current_player()
        card = self.cards[player]
        return f"{card}:{self.history}"
    
    def chance_outcomes(self) -> List[Tuple['KuhnState', float]]:
        """
        Enumerate all possible card deals.
        
        3 cards, 2 players: 3 * 2 = 6 possible deals, each with probability 1/6.
        """
        assert self.is_chance_node()
        outcomes = []
        cards = [0, 1, 2]  # Jack, Queen, King
        
        for i, c0 in enumerate(cards):
            for j, c1 in enumerate(cards):
                if i == j:
                    continue  # Can't deal same card twice
                state = KuhnState(
                    cards=(c0, c1),
                    history='',
                    pot=[1, 1]
                )
                outcomes.append((state, 1.0 / 6.0))
        
        return outcomes
    
    def __repr__(self) -> str:
        card_names = {0: 'J', 1: 'Q', 2: 'K'}
        if self.cards is None:
            return "KuhnState(dealing...)"
        c0 = card_names[self.cards[0]]
        c1 = card_names[self.cards[1]]
        return f"KuhnState(P0={c0}, P1={c1}, h='{self.history}', pot={self.pot})"


class KuhnPoker(PokerGame):
    """
    Kuhn Poker game factory.
    
    The simplest nontrivial imperfect-information game, ideal for
    verifying CFR correctness against known analytical Nash equilibria.
    """
    
    def initial_state(self) -> KuhnState:
        """Return the root chance node (cards not yet dealt)."""
        return KuhnState()
    
    def name(self) -> str:
        return "Kuhn Poker"
    
    @property
    def num_players(self) -> int:
        return 2
    
    # Known Nash Equilibrium values for verification
    GAME_VALUE = -1.0 / 18.0  # ≈ -0.0556 for Player 0
    
    @staticmethod
    def known_nash_equilibrium() -> dict:
        """
        Return the known Nash equilibrium strategy for Kuhn Poker.
        
        Uses alpha = 1/3 (one of the family of equilibria).
        
        Player 0:
        - Jack: bet with α, check with 1-α. At pb: always fold.
        - Queen: always check. At pb: call with α+1/3.
        - King: bet with 3α. At pb: always call.
        
        Player 1:
        - Jack: after check: bet 1/3. Facing bet: fold.
        - Queen: after check: check. Facing bet: call 1/3.
        - King: after check: bet. Facing bet: call.
        
        Returns dict mapping information_set_key → {action: probability}
        """
        alpha = 1.0 / 3.0
        return {
            # Player 0 strategies
            '0:':   {Action.CHECK: 1.0 - alpha, Action.BET: alpha},           # J: bet with α=1/3
            '0:pb': {Action.FOLD: 1.0, Action.CALL: 0.0},                    # J after check-bet: always fold
            '1:':   {Action.CHECK: 1.0, Action.BET: 0.0},                    # Q: always check
            '1:pb': {Action.FOLD: 1.0 - (alpha + 1.0/3.0),                   # Q after check-bet: call α+1/3
                     Action.CALL: alpha + 1.0/3.0},
            '2:':   {Action.CHECK: 1.0 - 3*alpha, Action.BET: 3*alpha},      # K: bet 3α=1
            '2:pb': {Action.FOLD: 0.0, Action.CALL: 1.0},                    # K after check-bet: always call
            
            # Player 1 strategies  
            '0:p':  {Action.CHECK: 2.0/3.0, Action.BET: 1.0/3.0},            # J after check: bet 1/3
            '0:b':  {Action.FOLD: 1.0, Action.CALL: 0.0},                    # J facing bet: always fold
            '1:p':  {Action.CHECK: 1.0, Action.BET: 0.0},                    # Q after check: always check
            '1:b':  {Action.FOLD: 2.0/3.0, Action.CALL: 1.0/3.0},            # Q facing bet: call 1/3
            '2:p':  {Action.CHECK: 0.0, Action.BET: 1.0},                    # K after check: always bet
            '2:b':  {Action.FOLD: 0.0, Action.CALL: 1.0},                    # K facing bet: always call
        }

