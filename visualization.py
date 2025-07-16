from matplotlib import gridspec
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib import gridspec
import matplotlib.pyplot as plt

from matplotlib import gridspec
import matplotlib.pyplot as plt

def compare_simulations(sim1, sim2, label1='Sim 1', label2='Sim 2'):
    stats1 = sim1.spread_stats
    stats2 = sim2.spread_stats
    params1 = sim1.params
    params2 = sim2.params

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3)

    # --- Row 0 ---
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(stats1['time_steps'], stats1['active_infections_list'], color='red', label=label1)
    ax0.plot(stats2['time_steps'], stats2['active_infections_list'], color='blue', label=label2)
    ax0.set_title("Active Infections Over Time")
    ax0.set_xlabel("Time Step")
    ax0.set_ylabel("Infected Count")
    ax0.grid(True)
    ax0.legend()

    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(stats1['time_steps'], stats1['cumulative_infections_list'], color='red', label=label1)
    ax1.plot(stats2['time_steps'], stats2['cumulative_infections_list'], color='blue', label=label2)
    ax1.set_title("Cumulative Infections Over Time")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Total Infected")
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(gs[0, 2])
    ax2.plot(stats1['time_steps'], stats1['r0_values'], color='red', label=label1)
    ax2.plot(stats2['time_steps'], stats2['r0_values'], color='blue', label=label2)
    ax2.axhline(1, color='black', linestyle='--', label='R₀=1')
    ax2.set_title("Effective R₀ Over Time")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("R₀")
    ax2.grid(True)
    ax2.legend()

    # --- Row 1 ---
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(stats1['time_steps'], stats1['beta_values'], color='red', label=label1)
    ax3.plot(stats2['time_steps'], stats2['beta_values'], color='blue', label=label2)
    ax3.set_title("Beta Decay Over Time")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Beta")
    ax3.grid(True)
    ax3.legend()

    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(stats1['time_steps'], stats1['active_hostile_list'], color='orange', label=label1)
    ax4.plot(stats2['time_steps'], stats2['active_hostile_list'], color='brown', label=label2)
    ax4.set_title("Number of Hostile Nodes Over Time")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Hostile Count")
    ax4.grid(True)
    ax4.legend()

    ax5 = plt.subplot(gs[1, 2])
    ax5.plot(stats1['time_steps'], stats1['temp_debunker_list'], color='cyan', label=label1)
    ax5.plot(stats2['time_steps'], stats2['temp_debunker_list'], color='purple', label=label2)
    ax5.set_title("Number of Temporary Debunkers Over Time")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Temp Debunkers")
    ax5.grid(True)
    ax5.legend()

    # --- Row 2: Summary Info ---
    ax6 = plt.subplot(gs[2, :])  # Full width bottom row
    ax6.axis('off')  # Hide axes

    def get_peak_info(stats):
        max_infected = max(stats['active_infections_list'])
        peak_time = stats['time_steps'][stats['active_infections_list'].index(max_infected)]
        return max_infected, peak_time

    max1, time1 = get_peak_info(stats1)
    max2, time2 = get_peak_info(stats2)

    summary_text = (
        "Simulation Comparison Summary\n"
        "=============================\n\n"
        f"Peak Infection:\n"
        f"  {label1}: {max1} at t={time1}\n"
        f"  {label2}: {max2} at t={time2}\n\n"
        f"Final Infections:\n"
        f"  {label1}: {stats1['cumulative_infections_list'][-1]}\n"
        f"  {label2}: {stats2['cumulative_infections_list'][-1]}\n\n"
        # f"Parameters:\n"
        # f"  Beta: {params1['beta']} → {round(stats1['beta_values'][-1], 4)} | "
        # f"{params2['beta']} → {round(stats2['beta_values'][-1], 4)}\n"
        # f"  Gamma: {params1['gamma']} | {params2['gamma']}\n"
        # f"  Hostile Prob: {params1.get('hostile_prob', 'N/A')} | {params2.get('hostile_prob', 'N/A')}\n"
        # f"  Debunk Prob: {params1.get('deb_prob', 'N/A')} | {params2.get('deb_prob', 'N/A')}"
    )

    ax6.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    
state_color = {
    'S': 'green',          # Susceptible
    'I': 'red',            # Infected (Unhostile)
    'H': 'purple',         # Hostile Infected
    'R': 'grey',           # Recovered via Gamma
    'D': 'blue',           # Permanent Debunker
    'TD': 'cyan',          # Temporary Debunker (Active)
    'RD': 'magenta'          # Recovered via Debunking (Inactive)
}
def draw_network(G, pos, ax, simulation_params, current_time, start_debunking):
    node_colors = []

    # Count node types for this time step
    count_S = count_I = count_H = count_R = count_RD = count_TD = count_D = 0

    for n in G:
        data = G.nodes[n]

        if data['state'] == 'R':
            if data.get('was_debunked', False):

                if data.get('is_debunker', False):
                    node_colors.append(state_color['TD'])
                    count_TD += 1
                else:  
                    node_colors.append(state_color['RD'])
                    count_RD += 1
            else:
                # Regular recovered via gamma
                node_colors.append(state_color['R'])
                count_R += 1
        elif data['state'] == 'I':
            if data.get('is_hostile', False):
                # Hostile infected
                node_colors.append(state_color['H'])
                count_H += 1
            else:
                # Non-hostile infected
                node_colors.append(state_color['I'])
                count_I += 1
        elif data['state'] == 'D':
            # Permanent debunker
            if data['debunker_active']:
                node_colors.append(state_color['D'])
            else:
                node_colors.append("black")
            # node_colors.append(state_color['D'])
            count_D += 1
        else:
            # Susceptible
            node_colors.append(state_color['S'])
            count_S += 1

    node_sizes = [50 if G.nodes[n]['state'] != 'D' else 150 for n in G]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.3,
        ax=ax
    )
    # nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    status = "ACTIVE" if (simulation_params['enable_debunking'] and start_debunking) else "INACTIVE"

    stats_text = (
        f"Time: {current_time}\n"
        f"Susceptible: {count_S} | Infected: {count_I} | Hostile: {count_H}\n"
        f"Recovered (Gamma): {count_R} | Recovered (Debunk): {count_RD}\n"
        f"Temp Debunkers: {count_TD} | Original Debunkers: {count_D}\n"
        f"Debunking: {status} | Beta: {simulation_params['beta']:.4f}"
    )

    ax.set_title(stats_text)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=state_color['S'], edgecolor='black', label='Susceptible (S)'),
        Patch(facecolor=state_color['I'], edgecolor='black', label='Infected (I)'),
        Patch(facecolor=state_color['H'], edgecolor='black', label='Hostile Infected (H)'),
        Patch(facecolor=state_color['R'], edgecolor='black', label='Recovered (Gamma)'),
        Patch(facecolor=state_color['RD'], edgecolor='black', label='Recovered (Debunk)'),
        Patch(facecolor=state_color['TD'], edgecolor='black', label='Temporary Debunker (TD)'),
        Patch(facecolor=state_color['D'], edgecolor='black', label='Original Debunker (D)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)