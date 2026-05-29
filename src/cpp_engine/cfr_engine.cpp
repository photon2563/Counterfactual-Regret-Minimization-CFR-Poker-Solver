#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

extern "C" {

struct Node {
    int type; // 0=P0, 1=P1, 2=CHANCE, 3=TERMINAL
    int info_set_id;
    int num_actions;
    int edge_start;
    double utility;
};

struct Edge {
    int target_node;
    double probability;
};

// Flattened Game Tree
Node* g_nodes = nullptr;
Edge* g_edges = nullptr;

// Information Sets
int* g_infoset_num_actions = nullptr;
double* g_regrets = nullptr;      // Flattened: sum of num_actions
double* g_strategy_sum = nullptr; // Flattened
int* g_infoset_start = nullptr;   // Offset into g_regrets

int g_num_nodes = 0;
int g_num_infosets = 0;

void init_tree(
    int num_nodes, 
    int* node_types, 
    int* node_infosets, 
    int* node_num_actions, 
    int* node_edge_starts, 
    double* node_utilities,
    int num_edges, 
    int* edge_targets, 
    double* edge_probs,
    int num_infosets, 
    int* infoset_num_actions) 
{
    g_num_nodes = num_nodes;
    g_num_infosets = num_infosets;

    g_nodes = new Node[num_nodes];
    for (int i = 0; i < num_nodes; ++i) {
        g_nodes[i].type = node_types[i];
        g_nodes[i].info_set_id = node_infosets[i];
        g_nodes[i].num_actions = node_num_actions[i];
        g_nodes[i].edge_start = node_edge_starts[i];
        g_nodes[i].utility = node_utilities[i];
    }

    g_edges = new Edge[num_edges];
    for (int i = 0; i < num_edges; ++i) {
        g_edges[i].target_node = edge_targets[i];
        g_edges[i].probability = edge_probs[i];
    }

    g_infoset_num_actions = new int[num_infosets];
    g_infoset_start = new int[num_infosets];
    
    int total_actions = 0;
    for (int i = 0; i < num_infosets; ++i) {
        g_infoset_num_actions[i] = infoset_num_actions[i];
        g_infoset_start[i] = total_actions;
        total_actions += infoset_num_actions[i];
    }

    g_regrets = new double[total_actions]();
    g_strategy_sum = new double[total_actions]();
}

void cleanup_tree() {
    delete[] g_nodes;
    delete[] g_edges;
    delete[] g_infoset_num_actions;
    delete[] g_infoset_start;
    delete[] g_regrets;
    delete[] g_strategy_sum;
}

void get_strategy(double* out_strategy) {
    int total_actions = g_infoset_start[g_num_infosets - 1] + g_infoset_num_actions[g_num_infosets - 1];
    for (int i = 0; i < total_actions; ++i) {
        out_strategy[i] = g_strategy_sum[i];
    }
}

// Recursive CFR Traversal
double cfr_traverse(int node_idx, int traverser, double reach_0, double reach_1, double chance_reach, int iteration) {
    Node& node = g_nodes[node_idx];

    if (node.type == 3) { // TERMINAL
        return (traverser == 0) ? node.utility : -node.utility;
    }

    if (node.type == 2) { // CHANCE
        double value = 0.0;
        for (int i = 0; i < node.num_actions; ++i) {
            Edge& edge = g_edges[node.edge_start + i];
            value += edge.probability * cfr_traverse(
                edge.target_node, traverser, reach_0, reach_1, chance_reach * edge.probability, iteration
            );
        }
        return value;
    }

    int info_set = node.info_set_id;
    int num_actions = node.num_actions;
    int start = g_infoset_start[info_set];

    // Regret Matching
    double sum_positive_regrets = 0.0;
    std::vector<double> strategy(num_actions, 0.0);
    for (int i = 0; i < num_actions; ++i) {
        if (g_regrets[start + i] > 0) {
            sum_positive_regrets += g_regrets[start + i];
        }
    }

    for (int i = 0; i < num_actions; ++i) {
        if (sum_positive_regrets > 0) {
            strategy[i] = (g_regrets[start + i] > 0) ? g_regrets[start + i] / sum_positive_regrets : 0.0;
        } else {
            strategy[i] = 1.0 / num_actions;
        }
    }

    std::vector<double> action_values(num_actions, 0.0);
    double node_value = 0.0;

    for (int i = 0; i < num_actions; ++i) {
        Edge& edge = g_edges[node.edge_start + i];
        
        double new_reach_0 = (node.type == 0) ? reach_0 * strategy[i] : reach_0;
        double new_reach_1 = (node.type == 1) ? reach_1 * strategy[i] : reach_1;
        
        action_values[i] = cfr_traverse(
            edge.target_node, traverser, new_reach_0, new_reach_1, chance_reach, iteration
        );
        node_value += strategy[i] * action_values[i];
    }

    if (node.type == traverser) {
        double opp_reach = (traverser == 0) ? reach_1 : reach_0;
        for (int i = 0; i < num_actions; ++i) {
            double regret = action_values[i] - node_value;
            g_regrets[start + i] += opp_reach * regret;
            // CFR+ modification
            if (g_regrets[start + i] < 0) {
                g_regrets[start + i] = 0;
            }
        }
    }

    double my_reach = (node.type == 0) ? reach_0 : reach_1;
    for (int i = 0; i < num_actions; ++i) {
        g_strategy_sum[start + i] += iteration * my_reach * strategy[i]; // Linear averaging for CFR+
    }

    return node_value;
}

void train(int iterations) {
    for (int i = 1; i <= iterations; ++i) {
        cfr_traverse(0, 0, 1.0, 1.0, 1.0, i);
        cfr_traverse(0, 1, 1.0, 1.0, 1.0, i);
    }
}

} // extern "C"
