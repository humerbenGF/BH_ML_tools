# import libraries
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_feature_importances_different_ss_li_max(function_to_test, csv_filename, model_filename_base, max_ss_li_coat_array, scale=False, num_features_to_compare=1000, polyfit=True):
    plotting_dict = {}
    for ss_li in max_ss_li_coat_array:
        feature_importances = function_to_test(csv_filename, f"{model_filename_base}_{ss_li}ssli", scale=scale, max_shots_since_lithium_coat=ss_li, show_output_plots=False)
        for feature in feature_importances['Feature']:
            importance_value = feature_importances.loc[feature_importances['Feature'] == feature, 'Importance'].values[0]
            if feature not in plotting_dict.keys():
                plotting_dict[feature] = [importance_value]
            else:
                plotting_dict[feature].append(importance_value)
    
    
    # Choose a colormap
    cmap = plt.get_cmap("jet")  # Change to any colormap you prefer

    # Create histogram with KDE
    plt.figure(figsize=(10, 6))
    colors = [cmap(i) for i in np.linspace(0, 1, min(len(plotting_dict), num_features_to_compare))]
    
    keys = list(plotting_dict.keys())
    for i in range(len(plotting_dict)):
        if i < num_features_to_compare:
            if polyfit:
                x = np.linspace(min(max_ss_li_coat_array), max(max_ss_li_coat_array), 1000)
                coefficients = np.polyfit(max_ss_li_coat_array, plotting_dict[keys[i]], deg=4)
                poly = np.poly1d(coefficients)
                plt.plot(x, poly(x), label=keys[i], color=colors[i], alpha=0.9)
                plt.scatter(max_ss_li_coat_array, plotting_dict[keys[i]], color=colors[i], marker='x', alpha=0.3)
            else:
                plt.plot(max_ss_li_coat_array, plotting_dict[keys[i]], label=keys[i], color=colors[i])
    
    plt.title("Feature Importance Parameter Over Maximum Number of Shots Since Lithium Coat")
    plt.ylabel("Feature Importance Parameter")
    plt.xlabel("Maximum Allowable Number of Shots Since Lithium Coat")
    plt.legend()
    plt.grid(True)
    plt.show()

    
    return