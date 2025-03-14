# import libraries
#################################################################
import numpy as np
import pandas as pd


# import personal files
#################################################################
import load_save_data.load_data as load_data
import load_save_data.save_data as save_data


def reformat_axuv_categorization_first_crash_timing_machine_settings_binary_training_data(filename, crash_timing_threshold, crash_timing_deadzone, example_shot='20681', undesired_columns=['gas']):
    # cat data dict
    cat_data_dict = load_data.load_json_file("unformatted_data/axuv_categorization/categorization_data.json")
    # machine settings
    machine_settings_dict = load_data.load_json_file("unformatted_data/machine_settings_and_state/control_settings.json")
    ss_dict = load_data.load_json_file("unformatted_data/machine_settings_and_state/shots_since.json")
    time_since_last_shot_dict = load_data.load_json_file("unformatted_data/machine_settings_and_state/time_since_last_shot.json")
    
    # jprep dictionary headings
    pre_csv_data_dict = {}
    for k in machine_settings_dict[str(example_shot)].keys():
        if k not in undesired_columns:
            pre_csv_data_dict[k] = np.array([])
            
    for k in ss_dict[str(example_shot)].keys():
        if k not in undesired_columns:
            pre_csv_data_dict[k] = np.array([])
    
    
    pre_csv_data_dict['time_since_last_shot'] = np.array([])
    pre_csv_data_dict['label'] = np.array([])
    
    
    for s in cat_data_dict.keys():
        if 'first_crash_info_time' in cat_data_dict[s].keys():
            if ((cat_data_dict[s]['first_crash_info_time'] > (crash_timing_threshold + (crash_timing_deadzone/2))) or (cat_data_dict[s]['first_crash_info_time'] < (crash_timing_threshold - (crash_timing_deadzone/2)))) and s in ss_dict.keys():
                for k in pre_csv_data_dict.keys():
                    if k in machine_settings_dict[s].keys():
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], machine_settings_dict[s][k])
                    elif k in ss_dict[s].keys():
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], ss_dict[s][k])
                    elif k not in ['label', 'time_since_last_shot']:
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], np.nan)

                # sort out actual labels
                if (cat_data_dict[s]['first_crash_info_time'] > (crash_timing_threshold + (crash_timing_deadzone/2))):
                    pre_csv_data_dict['label'] = np.append(pre_csv_data_dict['label'], 'late_crash')
                else:
                    pre_csv_data_dict['label'] = np.append(pre_csv_data_dict['label'], 'early_crash')
                    
                pre_csv_data_dict['time_since_last_shot'] = np.append(pre_csv_data_dict['time_since_last_shot'], time_since_last_shot_dict[s]['minutes'])


    for k in pre_csv_data_dict.keys():
        print(k, len(pre_csv_data_dict[k]))

    df = pd.DataFrame(pre_csv_data_dict)
    df.to_csv(filename, index=False)
    
    print(df)
    
    return



def reformat_axuv_categorization_dome_non_dome_binary_training_data(filename, example_shot='19734', undesired_columns=['slope_at_t']):
    # cat data dict
    cat_data_dict = load_data.load_json_file("unformatted_data/axuv_categorization/categorization_data.json")
    # dome data dict
    dome_data_dict = load_data.load_json_file("unformatted_data/axuv_categorization/dome_non_dome_manual.json")
    
    # prep dictionary headings
    pre_csv_data_dict = {}
    for k in cat_data_dict[str(example_shot)].keys():
        if k not in undesired_columns:
            pre_csv_data_dict[k] = np.array([])

    pre_csv_data_dict['label'] = np.array([])
    
    
    # iterate by category
    for cat in dome_data_dict.keys():
        for s in dome_data_dict[cat]:
            if str(s) in cat_data_dict.keys():
                # append metadata to arrays
                for k in pre_csv_data_dict.keys():
                    if k in cat_data_dict[str(s)].keys():
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], cat_data_dict[str(s)][k])
                
                    # get empty slots
                    elif k != 'label':
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], np.nan)
                
                # append cat to arrays
                pre_csv_data_dict['label'] = np.append(pre_csv_data_dict['label'], cat)

        
    df = pd.DataFrame(pre_csv_data_dict)
    
    df.to_csv(filename, index=False)
    
    return