import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from visualization import *
import requests
import os
import gzip
import requests
import gzip
import community as community_louvain  # python-louvain package
from collections import defaultdict


#TODO : Improve get effective beta for multiple debunkers
# , get temporal data, tune parameters, fix memory feedback(?).
#TODO : infecion of rumors increases infection probability (again improbe effective beta) 
"""
Here's how you do it. You set a few debunking nodes.
we know for each timestep what nodes are going to be infected. for a deb_prob of p there is a p
chance that the to be infected node doesnt infact get infected and the infection gets "derailed"
"""
"""
for the nodes that aren't derailed and are infected as
scheduled the infection goes on as it should have based on pre obtained data
"""
debunker_selection_strategies = [
    'random', 
    'high_degree', 
    'betweenness_centrality', 
    'pagerank', 
    'community_aware'
]

class Simulation:
    def __init__(self, seed=55, simul_params=None):
        self.current_time = 0
        self.params = {
            'beta': 0.7,
            'gamma': 0.1,
            't_intervention': 5,
            'alpha': 0.9,
            'eta': 0.01,
            'num_debunkers': 1,
            'credibility_range': (0.7, 1.0),
            'enable_debunking': True,
            'memory_initial': 0.9,         
            'memory_decay_rate': 0.1,
            'hostile_prob' : 0.1, #Probability that an infected node becomes a spreader
            'R_time' : 3, #Duration which a temp debunker / hostile node is a spreader
            'cooldown_time': 5,       # cooldown time between each debunker post
            'deb_prob': 0.03, #Probability that a debunked node becomes a debunker himself
            'n_intervention' : 0.1, #Percentage of infected individuals before debunking starts
            'top_n_percent' : 0.1, #top 0.1% of nodes
            'debunker_selection_strategy' : 'random', #Strategy for selecting debunkers
        }
        self.spread_stats = {
            'max_infected': 1,
            'peak_time': 0,
            'r0_values': [],
            'time_steps': [],
            'beta_values': [],
            'active_infections_list': [],
            'active_hostile_list' : [],
            'temp_debunker_list' : [],
            'cumulative_infections_list': [],
        }
        if simul_params != None:
            self.params.update(simul_params)
        self.G = self.initialize_network(seed=seed)
        self.pos = nx.spring_layout(self.G, seed=seed)
        self.start_debunking = False
        self.all_nodes_ever_infected = set()
        self.print_simulation_description()
    def initialize_network(self, seed=69):
        np.random.seed(seed)
        print("Generating a synthetic 100-node dense directed network...")
    
        # Generate a denser random directed graph
        G = nx.erdos_renyi_graph(n=40, p=0.15, directed=True, seed=seed)  # p=0.1 gives ~1000 edges
    
        # Add some reciprocity manually if needed
        for u, v in list(G.edges()):
            if np.random.rand() < 0.3:  # 30% chance to add reverse edge
                G.add_edge(v, u)

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        print("Processing dataset...")
        G = G.subgraph(max(nx.weakly_connected_components(G), key=len))
        G = nx.convert_node_labels_to_integers(G)

        # Set layout for visualization
        self.pos = nx.spring_layout(G, seed=seed)

        # Assign dummy weights if needed later
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

        self.initialize_nodes(G, seed)
        return G
    
    def download_dataset(self):
        print("Downloading dataset...")
        txt_url = "https://snap.stanford.edu/data/higgs-social_network.edgelist.gz"
        response = requests.get(txt_url)
        with open("higgs-social_network.edgelist.gz", 'wb') as f:
            f.write(response.content)
            print("Download complete.")
            
    def load_network(self, filename):
        filepath = "higgs-social_network.edgelist.gz"
        print("Loading dataset...")
        with gzip.open(filepath, 'rb') as f:
            return nx.read_edgelist(f, comments='#', create_using=nx.DiGraph(), nodetype=int)

        
    def process_network(self, G):
        print("Processing dataset...")
        G = G.subgraph(max(nx.weakly_connected_components(G), key=len))
        return nx.convert_node_labels_to_integers(G)
        
    def set_edge_weights(self, G):
        edge_weights = {}
        for u, v in G.edges():
            weight = len(list(nx.common_neighbors(G, u, v))) + 1
            edge_weights.update({(u, v): weight, (v, u): weight})
        nx.set_edge_attributes(G, edge_weights, 'weight')
    

    def initialize_nodes(self, G, seed):
        print("Setting node states...")
        np.random.seed(seed)
        
        for node in G.nodes:
            G.nodes[node]['state'] = 'S'
            G.nodes[node]['credibility'] = 0
            G.nodes[node]['is_hostile'] = False
            G.nodes[node]['is_debunker'] = False
            G.nodes[node]['was_debunked'] = False

        debunker_candidates = self.get_debunker_candidates(G=G, strategy=self.params['debunker_selection_strategy'])
        debunkers = []
        if self.params['enable_debunking']:
            debunkers = debunker_candidates[0:self.params['num_debunkers']]
            for _, node in enumerate(debunkers):
                G.nodes[node]['state'] = 'D'
                G.nodes[node]['credibility'] = np.random.uniform(
                    *self.params['credibility_range']
                )
                G.nodes[node]['is_debunker'] = True
                G.nodes[node]['debunker_phase_end'] = self.current_time + self.params['R_time']
                cycle_length = self.params['R_time'] + self.params['cooldown_time']
                offset = np.random.randint(0, cycle_length)
                if offset < self.params['R_time']:
                    G.nodes[node]['debunker_active'] = True
                    G.nodes[node]['debunker_phase_end'] = self.current_time + self.params['R_time'] - offset
                else:
                    G.nodes[node]['debunker_active'] = False
                    G.nodes[node]['debunker_phase_end'] = self.current_time + cycle_length - offset

        
        available_nodes = [n for n in G.nodes if n not in debunkers]
        infected_nodes = np.random.choice(
            available_nodes, int(1), replace=False
        )

        for node in infected_nodes:
            G.nodes[node]['state'] = 'I'
            G.nodes[node]['is_hostile'] = True
            G.nodes[node]['hostile_until_time'] = self.current_time + self.params['R_time']
            G.nodes[node]['is_debunker'] = False

        return G
    
         
    def update_debunker_phases(self):
        for node in [n for n in self.G if self.G.nodes[n]['state'] == 'D']:
            data = self.G.nodes[node]
            if self.current_time >= data['debunker_phase_end']:
                data['debunker_active'] = not data['debunker_active']
                if data['debunker_active']:
                    duration = self.params['R_time']
                else:
                    duration = self.params['cooldown_time']
                data['debunker_phase_end'] = self.current_time + duration       
            
    def get_debunker_candidates(self, G, strategy='high_degree'):

        if strategy == 'random':
            nodes = list(G.nodes)
            np.random.shuffle(nodes)
            return nodes

        elif strategy == 'high_degree':
            degrees = dict(G.in_degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            return [n for n, _ in sorted_nodes]
        
        
        elif strategy == 'closeness_centrality':
            print("Computing closeness centrality...")
            centrality = nx.closeness_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return [n for n, _ in sorted_nodes]
        
        elif strategy == 'katz_centrality':
            print("Computing Katz centrality...")
            centrality = nx.katz_centrality(G, alpha=0.005, beta=1.0)
            sorted_nodes = sorted(centrality.items(), key=lambda x : x[1], reverse=True)
            return [n for n, _ in sorted_nodes]
        
        elif strategy == 'betweenness_centrality':
            print("Computing directed betweenness centrality...")
            # Use directed betweenness centrality
            centrality = nx.betweenness_centrality(G, normalized=True, weight=None, endpoints=False)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return [n for n, _ in sorted_nodes]

        elif strategy == 'pagerank':
            print("Computing PageRank...")
            centrality = nx.pagerank(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return [n for n, _ in sorted_nodes]

        elif strategy == 'eigenvector_centrality':
            print("Computing directed (left) eigenvector centrality...")
            adj_matrix = nx.adjacency_matrix(G).T.toarray()
            eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)
            dominant_idx = np.argmax(np.abs(eigenvalues))
            centrality = np.abs(eigenvectors[:, dominant_idx])
            centrality = centrality / centrality.sum()
            sorted_nodes = sorted(zip(G.nodes(), centrality), key=lambda x: x[1], reverse=True)
            return [n for n, _ in sorted_nodes]
        
        elif strategy == 'leaderrank':
            print("Computing LeaderRank centrality...")
            G_lr = G.copy()
            ground_node = -1
            G_lr.add_node(ground_node)
            for node in G.nodes():
                G_lr.add_edge(ground_node, node)
                G_lr.add_edge(node, ground_node)
        
            centrality = nx.pagerank(G_lr, personalization={ground_node:0})
    
            del centrality[ground_node]
            total = sum(centrality.values())
            centrality = {k: v/total for k, v in centrality.items()}
        
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return [n for n, _ in sorted_nodes]


            
            



               
    def print_simulation_description(self):
        """Print detailed simulation parameters and graph statistics for directed network"""
        print("\n" + "="*50)
        print("DIRECTED NETWORK SIMULATION".center(50))
        print("="*50)
        
        print("\n[GRAPH STATISTICS]")
        print(f"{'Number of nodes:':<25} {len(self.G.nodes())}")
        print(f"{'Number of edges:':<25} {len(self.G.edges())}")
        
        # Degree statistics
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
        print(f"{'Average in-degree:':<25} {sum(in_degrees)/len(self.G.nodes()):.2f}")
        print(f"{'Average out-degree:':<25} {sum(out_degrees)/len(self.G.nodes()):.2f}")
        print(f"{'Max in-degree:':<25} {max(in_degrees)}")
        print(f"{'Max out-degree:':<25} {max(out_degrees)}")
        
        # Weakly/strongly connected components
        print(f"{'Weakly connected:':<25} {nx.is_weakly_connected(self.G)}")
        print(f"{'# Weakly CC:':<25} {nx.number_weakly_connected_components(self.G)}")
        
        print("\n[SIMULATION PARAMETERS]")
        for param, value in sorted(self.params.items()):
            if isinstance(value, (int, float, str, bool)):
                print(f"{param+':':<25} {value}")
            elif isinstance(value, tuple):
                print(f"{param+':':<25} {', '.join(map(str, value))}")
        
        print("\n[INITIAL STATE COUNTS]")
        states = nx.get_node_attributes(self.G, 'state')
        state_counts = {state: list(states.values()).count(state) for state in set(states.values())}
        for state, count in sorted(state_counts.items()):
            print(f"{state+':':<25} {count}")
        
        # Add initial infected/debunker locations
        infected = [n for n in self.G.nodes() if self.G.nodes[n]['state'] == 'I']
        debunkers = [n for n in self.G.nodes() if self.G.nodes[n]['state'] == 'D']
        print(f"\n{'Initial infected nodes:':<25} {infected}")
        print(f"{'Initial debunker nodes:':<25} {debunkers}")
        
        # Add degree info for initial infected/debunkers
        if infected:
            print(f"\n{'Infected in-degree:':<25} {[self.G.in_degree(n) for n in infected]}")
            print(f"{'Infected out-degree:':<25} {[self.G.out_degree(n) for n in infected]}")
        if debunkers:
            print(f"{'Debunker in-degree:':<25} {[self.G.in_degree(n) for n in debunkers]}")
            print(f"{'Debunker out-degree:':<25} {[self.G.out_degree(n) for n in debunkers]}")
        
        print("="*50 + "\n")
        
        
    def run(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        self.ani = FuncAnimation(
            fig, 
            self.update, 
            frames=None,
            fargs=(ax,), 
            interval=3000, 
            save_count=200,
            repeat=False
        )
        plt.show()
        return self.spread_stats.copy()

    def update(self,_,ax):
        ax.clear()
        self.spread_stats['time_steps'].append(self.current_time)
        new_infections = set()
        new_recoveries = set()
        new_debunked = set()
        current_infected = [n for n in self.G if self.G.nodes[n]['state'] == 'I']
        current_infected_count = len(current_infected)
    
        self.update_spread_stats()
        new_infections, new_recoveries = self.infections_and_recoveries()
        
        self.params['beta'] *= (1 - self.params['eta'])
        self.params['hostile_prob'] *= (1 - self.params['eta'])
        self.params['deb_prob'] *= (1 - self.params['eta'])
        
        
        if (len(self.all_nodes_ever_infected) / len(self.G.nodes())) > self.params['n_intervention'] and self.params['enable_debunking'] and not self.start_debunking:
            print("Started debunking when more than 10% of the population where infected")
            self.start_debunking = True
        
        if self.start_debunking:
            new_debunked = self.debunk_nodes()
        
        r0 = len(new_infections) / current_infected_count if current_infected_count > 0 else 0
        self.spread_stats['r0_values'].append(r0)
    
        self.update_node_states(new_infections, new_debunked, new_recoveries)
        self.diminish_memory()
        self.update_debunker_phases()
        self.handle_temp_debunkers()
        
        print("Time:", self.current_time,"Number of Infected nodes: ", current_infected_count)
        draw_network(self.G, self.pos, ax, self.params, self.current_time, self.start_debunking)
    
        self.current_time += 1
        if current_infected_count == 0:
            self.ani.event_source.stop()
           
             
             
    def end_hostile_status(self,node_data):
        node_data['is_hostile'] = False
        node_data.pop('hostile_until_time', None)
        
    def set_hostile_status(self,node_data):
        node_data['is_hostile'] = True
        node_data['hostile_until_time'] = self.current_time + self.params['R_time']
                
    def infections_and_recoveries(self):
        new_infections, new_recoveries = set(), set()
        for node in [n for n in self.G if self.G.nodes[n]['state'] == 'I']:
            self.all_nodes_ever_infected.add(node)
            data = self.G.nodes[node]
            is_hostile = data.get('is_hostile', False)
            hostile_until = data.get('hostile_until_time', -1)

            # Check if hostile status should end now
            if is_hostile and self.current_time >= hostile_until:
                is_hostile = False
                self.end_hostile_status(data)

            # Try to become hostile if not already
            if not is_hostile:
                if np.random.rand() < self.params['hostile_prob']:
                    self.set_hostile_status(data)
                    is_hostile = True

            # Only allow hostile nodes to infect
            if is_hostile:             
                neighbors = list(self.G.predecessors(node))
                np.random.shuffle(neighbors)
                for neighbor in neighbors:
                    infection_prob = self.calculate_infection_prob(node, neighbor)
                    if infection_prob > 0 and np.random.rand() < infection_prob:
                        new_infections.add(neighbor)
                        
            if np.random.rand() < self.params['gamma']:
                new_recoveries.add(node)
                        
        return new_infections, new_recoveries
        
    def calculate_infection_prob(self, infector, neighbor):
        neighbor_state = self.G.nodes[neighbor]['state']
        if neighbor_state == 'S':
            return self.get_effective_beta(infector, neighbor)
        elif neighbor_state == 'R':
            current_memory = self.G.nodes[neighbor].get('memory', 0)
            return self.get_effective_beta(infector, neighbor) * (1 - current_memory)
        return 0
    
    def get_effective_beta(self, u, v):
        base_beta = self.params['beta']
        debunker_effect = sum(
            self.params['alpha'] * self.G.nodes[neighbor]['credibility']
            for neighbor in self.G.successors(v)
            if self.G.nodes[neighbor]['state'] == 'D' or 
                (self.G.nodes[neighbor]['state'] == 'R' and 
                self.G.nodes[neighbor].get('is_debunker', False))
        )
        if self.params['enable_debunking'] and self.start_debunking:
            return base_beta * (1 - min(debunker_effect, 0.5))
        else:
            return base_beta
        
    def diminish_memory(self):
        for node in self.G.nodes:
            if self.G.nodes[node]['state'] == 'R':
                current_memory = self.G.nodes[node].get('memory', 0)
                decayed_memory = current_memory * (1 - self.params['memory_decay_rate'])
                self.G.nodes[node]['memory'] = max(decayed_memory, 0)
                
    def handle_temp_debunkers(self):
        for node in [n for n in self.G if self.G.nodes[n].get('is_debunker', False) and\
            self.G.nodes[n]['state'] == 'R']:
            if 'debunker_until_time' in self.G.nodes[node] and self.current_time >= self.G.nodes[node]['debunker_until_time']:
                self.G.nodes[node]['is_debunker'] = False
                self.G.nodes[node].pop('debunker_until_time', None)
                self.G.nodes[node]['credibility'] = 0

        for node in [
                n for n in self.G
                if  self.G.nodes[n]['state'] == 'R'
                and self.G.nodes[n].get('was_debunked', False)
                and not self.G.nodes[n].get('is_debunker', False)
        ]:
            self.maybe_turn_into_debunker(node)

       
    def debunk_nodes(self):
        new_debunked = set()
        active_debunkers = [n for n in self.G.nodes 
                    if ((self.G.nodes[n]['state'] == 'D' and self.G.nodes[n]['debunker_active'] == True) or 
                    (self.G.nodes[n]['state'] == 'R' and 
                        self.G.nodes[n].get('is_debunker', False)))]
        
        for debunker in active_debunkers:
            neighbors = list(self.G.predecessors(debunker))
            np.random.shuffle(neighbors)    
            for neighbor in neighbors:
                if self.G.nodes[neighbor]['state'] in ['S', 'I']:
                    if np.random.rand() < self.G.nodes[debunker]['credibility'] * self.params['alpha']:
                        new_debunked.add(neighbor)
        return new_debunked
        
        
        
    def set_infection_states(self,new_infections):
        for node in new_infections:
            self.G.nodes[node]['state'] = 'I'
            self.G.nodes[node]['is_hostile'] = False
            self.G.nodes[node].pop('hostile_until_time', None)
            self.G.nodes[node]['is_debunker'] = False
            self.G.nodes[node]['credibility'] = False
            self.G.nodes[node]['was_debunked'] = False
            
    def set_recovered_states(self, new_recoveries):
        for node in new_recoveries:
            self.G.nodes[node]['state'] = 'R'
            self.G.nodes[node]['memory'] = self.params['memory_initial']
            self.G.nodes[node]['is_hostile'] = False
            self.G.nodes[node].pop('hostile_until_time', None)
            self.G.nodes[node]['is_debunker'] = False
            self.G.nodes[node]['was_debunked'] = False
            self.G.nodes[node]['credibility'] = False
            
            
    def set_debunked_states(self, new_debunked):
        for node in new_debunked:
            self.G.nodes[node]['state'] = 'R'
            self.G.nodes[node]['memory'] = self.params['memory_initial']
            self.G.nodes[node]['is_hostile'] = False
            self.G.nodes[node].pop('hostile_until_time', None)
            self.G.nodes[node]['was_debunked'] = True
            self.G.nodes[node]['is_debunker'] = False
            self.G.nodes[node]['credibility'] = False
            
    def maybe_turn_into_debunker(self, node):
        if np.random.rand() < self.params['deb_prob']:
            self.G.nodes[node]['is_debunker'] = True
            self.G.nodes[node]['credibility'] = np.random.uniform(*self.params['credibility_range'])
            self.G.nodes[node]['debunker_until_time'] = self.current_time + self.params['R_time']
        
    def update_node_states(self, new_infections, new_debunked, new_recoveries):
        self.set_infection_states(new_infections)
        self.set_recovered_states(new_recoveries)
        self.set_debunked_states(new_debunked)




    def update_spread_stats(self):
        current_infected_count = 0
        cumulative_infection_count = len(self.all_nodes_ever_infected)
        current_hostile_count = 0
        current_temp_debunker_count = 0

        for n, data in self.G.nodes(data=True):
            state = data.get('state')
            if state == 'I':
                current_infected_count += 1
                if data.get('is_hostile'):
                    current_hostile_count += 1
            elif state == 'R' and data.get('is_debunker'):
                current_temp_debunker_count += 1

        self.spread_stats['active_infections_list'].append(current_infected_count)
        self.spread_stats['cumulative_infections_list'].append(cumulative_infection_count)
        self.spread_stats['active_hostile_list'].append(current_hostile_count)
        self.spread_stats['temp_debunker_list'].append(current_temp_debunker_count)
        self.spread_stats['beta_values'].append(self.params['beta'])

        if current_infected_count > self.spread_stats['max_infected']:
            self.spread_stats['max_infected'] = current_infected_count
            self.spread_stats['peak_time'] = self.current_time

    def bii_centrality(G, alpha=1.0, max_iter=26, tol=1.0e-6, weight='weight'):
        N = nx.number_of_nodes(G)
        bo = N * tol
        W = G.copy()
            
        # Initialize weights uniformly if not already present
        for u, v, d in W.edges(data=True):
            d[weight] = 1
            
        biiv = dict.fromkeys(W, 1.0 / N)  # initial value
            
        # Calculate in-degree centrality (normalized in-degree)
        in_degrees = dict(G.in_degree())
        total = sum(in_degrees.values())
        ind = {k: v/total for k, v in in_degrees.items()}
            
            # Power iteration
        for _ in range(max_iter):
            xlast = biiv
            biiv = dict.fromkeys(xlast.keys(), 0)
                
            for n in biiv:
                for nbr in W[n]:
                    biiv[nbr] += alpha * xlast[n] * W[n][nbr][weight]
                biiv[n] += ind[n]
                
            # Normalize
            bii_sum = np.linalg.norm(np.array(list(biiv.values())))
            biiv = {k: v/bii_sum for k, v in biiv.items()}
                
                # Check convergence
            err = sum([abs(biiv[n] - xlast[n]) for n in biiv])
            if err < bo:
                break
            
        return biiv


"""Strategies : 
Debunker placement

High-degree nodes : Place debunkers among users with many followers (e.g., influencers)
Betweenness centrality : Target bridges between communities - great for slowing cascades
PageRank / Eigenvector Centrality : Place them in positions of high influence
Random placement : Baseline strategy
Community-aware seeding : One debunker per community to ensure wide coverage

canceled
Edge censorship : Finding key bridges and removing them (ban)

canceled
Node censorship : 
Finding key nodes and removing them (ban)
Lowering the edge weights from that node (flagging)


"""

    
if __name__ == "__main__":
    
    sim1 = Simulation(seed=67, simul_params={'enable_debunking' : True,'debunker_selection_strategy' : 'betweenness_centrality'})
    spread_stats_1 = sim1.run()
    
    sim2 = Simulation(seed=67, simul_params={'enable_debunking' : True, 'debunker_selection_strategy' : 'high_degree'})
    spread_stats_2 = sim2.run()
    
    compare_simulations(sim1, sim2, label1='en', label2='dis')
    
"""
Update : the debunking starts when the slope of the I-t plot becomes less than a parameter
e.g : 0.1 --- the reason is that we assume that if a debunker does great at that time which is
considered latestage debunking it will also be a good debunker at the times before.

Update : (No change needed) : debunker choice is made at the beginning based on the rumorless
network. Later it will come into play


Things to check :
Edge directions correctly working
R_time not being violated
permanent debunker debunking in phases
natural recoveries not debunking
unhostile nodes not infecting
memory correctly working (seems so)


strategies to maybe add :
voterank
leaderrank (correct it)
BII
RRWG

"""

