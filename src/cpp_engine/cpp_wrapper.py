"""
Python wrapper for the C++ CFR engine.

Flattens the GameState tree into C-compatible arrays,
invokes the high-performance C++ CFR+ algorithm, and
unflattens the resulting strategy back into Python dictionaries.
"""

import ctypes
import os
import time
from typing import Dict, List, Any, Tuple

from game_engine.game_state import GameState, Player, Action

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), 'libcfr_engine.dylib')
if not os.path.exists(lib_path):
    lib_path = os.path.join(os.path.dirname(__file__), 'libcfr_engine.so')

if not os.path.exists(lib_path):
    raise FileNotFoundError("C++ CFR engine library not found. Please run 'make' in src/cpp_engine/")

engine = ctypes.CDLL(lib_path)

# Define C-API signatures
engine.init_tree.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), 
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.POINTER(ctypes.c_int)
]
engine.init_tree.restype = None

engine.train.argtypes = [ctypes.c_int]
engine.train.restype = None

engine.get_strategy.argtypes = [ctypes.POINTER(ctypes.c_double)]
engine.get_strategy.restype = None

engine.cleanup_tree.argtypes = []
engine.cleanup_tree.restype = None


class CPPCFRWrapper:
    def __init__(self):
        self.nodes = []        # type, info_set_id, num_actions, edge_start, utility
        self.edges = []        # target_node, probability
        self.infosets = {}     # info_set_key -> (id, num_actions, actions_list)
        self.infoset_keys = [] # list of info_set_key ordered by id
        
    def _get_infoset_id(self, key: str, actions: List[Action]) -> int:
        if key not in self.infosets:
            iid = len(self.infosets)
            self.infosets[key] = (iid, len(actions), actions)
            self.infoset_keys.append(key)
        return self.infosets[key][0]

    def _flatten_tree(self, state: GameState, memo: Dict[str, int]) -> int:
        # To avoid duplicating subtrees, we memoize exactly the same chance/history states.
        # For simplicity in imperfect information, tree nodes are unique histories.
        # But for memory efficiency, we just walk the tree completely.
        # In a real generic implementation, we'd hash the history.
        
        node_idx = len(self.nodes)
        self.nodes.append(None) # Placeholder
        
        if state.is_terminal():
            self.nodes[node_idx] = (3, -1, 0, 0, state.terminal_utility(Player.PLAYER_0))
            return node_idx
            
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            edge_start = len(self.edges)
            for _ in outcomes:
                self.edges.append((0, 0.0)) # Placeholder
            
            for i, (next_state, prob) in enumerate(outcomes):
                child_idx = self._flatten_tree(next_state, memo)
                self.edges[edge_start + i] = (child_idx, prob)
                
            self.nodes[node_idx] = (2, -1, len(outcomes), edge_start, 0.0)
            return node_idx
            
        # Player node
        actions = state.legal_actions()
        info_key = state.information_set_key()
        info_id = self._get_infoset_id(info_key, actions)
        
        edge_start = len(self.edges)
        for _ in actions:
            self.edges.append((0, 0.0))
            
        for i, a in enumerate(actions):
            next_state = state.apply_action(a)
            child_idx = self._flatten_tree(next_state, memo)
            self.edges[edge_start + i] = (child_idx, 0.0)
            
        node_type = 0 if state.current_player() == Player.PLAYER_0 else 1
        self.nodes[node_idx] = (node_type, info_id, len(actions), edge_start, 0.0)
        return node_idx

    def train_and_get_strategy(self, root: GameState, num_iterations: int) -> Dict[str, Dict[Action, float]]:
        print("▸ Flattening game tree for C++ engine...")
        t0 = time.time()
        self._flatten_tree(root, {})
        print(f"  Flattened {len(self.nodes)} nodes, {len(self.infosets)} infosets in {time.time()-t0:.2f}s")
        
        # Prepare C arrays
        num_nodes = len(self.nodes)
        node_types = (ctypes.c_int * num_nodes)()
        node_infosets = (ctypes.c_int * num_nodes)()
        node_num_actions = (ctypes.c_int * num_nodes)()
        node_edge_starts = (ctypes.c_int * num_nodes)()
        node_utilities = (ctypes.c_double * num_nodes)()
        
        for i, (ntype, iset, nact, edge, util) in enumerate(self.nodes):
            node_types[i] = ntype
            node_infosets[i] = iset
            node_num_actions[i] = nact
            node_edge_starts[i] = edge
            node_utilities[i] = util
            
        num_edges = len(self.edges)
        edge_targets = (ctypes.c_int * num_edges)()
        edge_probs = (ctypes.c_double * num_edges)()
        
        for i, (target, prob) in enumerate(self.edges):
            edge_targets[i] = target
            edge_probs[i] = prob
            
        num_infosets = len(self.infosets)
        infoset_num_actions = (ctypes.c_int * num_infosets)()
        for key in self.infoset_keys:
            iid, nact, _ = self.infosets[key]
            infoset_num_actions[iid] = nact
            
        print("▸ Initializing C++ CFR+ engine...")
        engine.init_tree(
            num_nodes, node_types, node_infosets, node_num_actions, node_edge_starts, node_utilities,
            num_edges, edge_targets, edge_probs,
            num_infosets, infoset_num_actions
        )
        
        print(f"▸ Running {num_iterations} iterations in C++...")
        t0 = time.time()
        engine.train(num_iterations)
        print(f"  Completed in {time.time()-t0:.4f}s")
        
        # Retrieve strategy
        total_actions = sum(nact for _, nact, _ in self.infosets.values())
        out_strategy = (ctypes.c_double * total_actions)()
        engine.get_strategy(out_strategy)
        
        # Unflatten strategy
        strategy_dict = {}
        idx = 0
        for iid, key in enumerate(self.infoset_keys):
            _, nact, actions = self.infosets[key]
            
            strat_sum = [out_strategy[idx + i] for i in range(nact)]
            idx += nact
            
            normalizer = sum(strat_sum)
            if normalizer > 0:
                probs = [s / normalizer for s in strat_sum]
            else:
                probs = [1.0 / nact for _ in range(nact)]
                
            strategy_dict[key] = {a: p for a, p in zip(actions, probs)}
            
        engine.cleanup_tree()
        return strategy_dict
