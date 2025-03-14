# import libraries
#################################################################
import numpy as np
import pandas as pd


# import personal files
#################################################################
import load_save_data.load_data as load_data


def reformat_thomson_temp_machine_settings_binary_training_data(filename, thomson_temp_threshold, thomson_temp_deadzone, example_shot='20681', undesired_columns=['gas']):
    # cat data dict
    thomson_data = load_data.load_json_file("unformatted_data/thomson/thomson_data_by_shot.json")
    # machine settings
    machine_settings_dict = load_data.load_json_file("unformatted_data/machine_settings_and_state/control_settings.json")
    ss_dict = load_data.load_json_file("unformatted_data/machine_settings_and_state/shots_since.json")
    
    # prep dictionary headings
    pre_csv_data_dict = {}
    for k in machine_settings_dict[str(example_shot)].keys():
        if k not in undesired_columns:
            pre_csv_data_dict[k] = np.array([])
            
    for k in ss_dict[str(example_shot)].keys():
        if k not in undesired_columns:
            pre_csv_data_dict[k] = np.array([])
            
    pre_csv_data_dict['label'] = np.array([])
    
    print(pre_csv_data_dict.keys())
    
    for s in thomson_data.keys():
        if 'thomson_600_temps' in thomson_data[s].keys():
            if ((max(thomson_data[s]['thomson_600_temps']) > (thomson_temp_threshold + (thomson_temp_deadzone/2))) or (max(thomson_data[s]['thomson_600_temps']) < (thomson_temp_threshold - (thomson_temp_deadzone/2)))) and s in ss_dict.keys():
                for k in pre_csv_data_dict.keys():
                    if k in machine_settings_dict[s].keys():
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], machine_settings_dict[s][k])
                    elif k in ss_dict[s].keys():
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], ss_dict[s][k])
                    elif k not in ['label']:
                        pre_csv_data_dict[k] = np.append(pre_csv_data_dict[k], np.nan)

                        
                # sort out actual labels
                if (max(thomson_data[s]['thomson_600_temps']) > (thomson_temp_threshold + (thomson_temp_deadzone/2))):
                    pre_csv_data_dict['label'] = np.append(pre_csv_data_dict['label'], 'hot_shot')
                else:
                    pre_csv_data_dict['label'] = np.append(pre_csv_data_dict['label'], 'cold_shot')
                    
                    
    for k in pre_csv_data_dict.keys():
        print(k, len(pre_csv_data_dict[k]))
        
    df = pd.DataFrame(pre_csv_data_dict)
    
    df.to_csv(filename, index=False)
    
    print(df)
    
    return