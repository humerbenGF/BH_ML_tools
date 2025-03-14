# import libraries
#################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D




def plot_normal_dist_by_variable(training_data_filename):
    # read in training data
    training_df = pd.read_csv(training_data_filename)
    
    unique_labels = training_df['label'].unique()
    
    print(unique_labels)
    
    for col in training_df.columns:
        plotting_data = []
        for l in range(len(unique_labels)):
            df_filtered = training_df[training_df['label'] == unique_labels[l]]
            plotting_data.append(df_filtered[col])

        # Choose a colormap
        cmap = plt.get_cmap("jet")  # Change to any colormap you prefer

        # Create histogram with KDE
        plt.figure(figsize=(10, 6))
        colors = [cmap(i) for i in np.linspace(0, 1, len(plotting_data))]
        for i in range(len(plotting_data)):
            sns.histplot(
                plotting_data[i], bins=30, kde=False, stat="density", 
                color=colors[i], alpha=0.2, label=f"{unique_labels[i]} - {col}"  # Low alpha for bars
            )
            sns.kdeplot(
                plotting_data[i], color=colors[i], alpha=0.9, lw=2, label=f"{unique_labels[i]} KDE"  # High alpha for KDE lines
            )

        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"KDE of Counted Normal Distribution for {col}")
        plt.legend()
        plt.grid(True)
        plt.show()
        

def plot_normal_dist_by_variable_by_max_li_coat_sustain_only(training_data_filename, max_ss_li_array, linestyles=['--', '-']):
    # Read in training data
    training_df = pd.read_csv(training_data_filename)
    unique_labels = training_df['label'].unique()
    
    # Normalize colors based on max_ss_li_array
    cmap = plt.get_cmap("jet")  # Choose colormap
    norm = mcolors.Normalize(vmin=min(max_ss_li_array), vmax=max(max_ss_li_array))

    for col in training_df.columns:
        plt.figure(figsize=(10, 6))
        plotting_data = []
        legend_arr = []

        for i, ss_li in enumerate(max_ss_li_array):
            df_filtered = training_df[training_df['ss_li_pot'] < ss_li]
            df_filtered = df_filtered[df_filtered['ss_li_gun'] < ss_li]
            plotting_data.append([])

            for j, label in enumerate(unique_labels):
                df_label_filtered = df_filtered[df_filtered['label'] == label]
                plotting_data[-1].append(df_label_filtered[col])

                # Add legend entry (only once per label)
                if i == 0:  
                    legend_arr.append(Line2D([0], [0], color='grey', linestyle=linestyles[j], label=f"{label} KDE"))

        # Generate colors
        colors = [cmap(norm(ss_li)) for ss_li in max_ss_li_array]

        # Plot KDEs
        for i in range(len(plotting_data)):
            for j in range(len(plotting_data[i])):
                sns.kdeplot(
                    plotting_data[i][j], color=colors[i], linestyle=linestyles[j], alpha=0.7
                )

        # Create colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = plt.colorbar(sm)
        cbar.set_label("Max ss_li Values")

        # Labels and legend
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"KDE of Counted Normal Distribution for {col}")
        plt.legend(handles=legend_arr)
        plt.grid(True)
        plt.show()

