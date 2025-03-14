# import personal files
#################################################################
import formatting_scripts.axuv_categorization as axuv_cat_formatting
import formatting_scripts.thomson_categorization as thomson_cat_formatting
# training scripts
import training_scripts.train_axuv_dome_non_dome_categorization as train_axuv_cat_dome_non_dome
import training_scripts.train_machine_settings_model as train_machine_settings
# comparison
import training_scripts.compare_feature_evolution_over_one_specific_feature as compare_feature_evolution
# plotting scripts
import training_data_plots.plot_mu_sig_by_cat as plot_mu_sigma




if __name__ == '__main__':
    # dome non-dome classification
    ##################################################################################################################################
    # data_filename = 'formatted_training_data/axuv_categorization/axuv_dome_non_dome_binary_categorization.csv'
    # axuv_cat_formatting.reformat_axuv_categorization_dome_non_dome_binary_training_data(data_filename)
    # train_axuv_cat_dome_non_dome.train_dome_non_dome_binary_classification(data_filename, "models/axuv_categorization/dome_non_dome_binary.pkl")
    
    
    # crash timing and machine setttings
    ##################################################################################################################################
    data_filename = 'formatted_training_data/axuv_categorization/axuv_first_crash_machine_settings_categorization.csv'
    separation_timing = 0.005
    separation_width = 0.002
    ss_li=5000
    min_sustain_kV = 4
    axuv_cat_formatting.reformat_axuv_categorization_first_crash_timing_machine_settings_binary_training_data(data_filename, separation_timing, separation_width)
    model_filename = f"models/axuv_categorization/first_crash_timing/first_crash_timing_{separation_timing}_{separation_width}_{ss_li}ssli"
    # train_machine_settings.train_machine_setting_with_ss_li(data_filename, model_filename, max_shots_since_lithium_coat=ss_li, min_sustain=min_sustain_kV)
    
    model_filename_base = f"models/axuv_categorization/first_crash_timing/first_crash_timing_{separation_timing}_{separation_width}"
    min_ss_li = 5
    ss_li_array = []
    for i in range(100):
        ss_li_array.append(i+min_ss_li)
    ss_li_array.reverse()
    # compare_feature_evolution.compare_feature_importances_different_ss_li_max(train_machine_settings.train_machine_setting_with_ss_li, data_filename, model_filename_base, ss_li_array, num_features_to_compare=6)
    # plot_mu_sigma.plot_normal_dist_by_variable(data_filename)
    ss_li_array = [10, 20, 30, 40]
    # plot_mu_sigma.plot_normal_dist_by_variable_by_max_li_coat_sustain_only(data_filename, ss_li_array)
    
    
    # hot shots and machine settings
    ##################################################################################################################################
    separation_temp = 250
    seperation_temp_width = 25
    ss_li = 10000
    data_filename = f'formatted_training_data/thomson/hot_cold_thomson_{separation_temp}_{seperation_temp_width}.csv'
    # thomson_cat_formatting.reformat_thomson_temp_machine_settings_binary_training_data(data_filename, separation_temp, seperation_temp_width)
    model_filename = f"models/thomson/hot_cold_thomson/hot_cold_thomson_{separation_temp}_{seperation_temp_width}_{ss_li}ss_li"
    # train_machine_settings.train_machine_setting_with_ss_li(data_filename, model_filename)
    plot_mu_sigma.plot_normal_dist_by_variable(data_filename)