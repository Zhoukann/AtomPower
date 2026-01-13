import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd

design_name = sys.argv[1]
testbench = sys.argv[2]

# input
reg_waveform_file = f"./{testbench}/{design_name}_{testbench}_reg_waveform.csv"
reg_clusters_file = f"./Design/Structure/{design_name}_reg_clusters.csv"
reg_features_and_weight_pre_file = f"./{testbench}/{design_name}_{testbench}_reg_features_and_weight_pre.csv"

# temp
reg_features_and_weight_pre_wavesame_file = f"./{testbench}/{design_name}_{testbench}_reg_features_and_weight_pre_wavesame.csv"
reg_features_and_weight_pre_wavenotsame_file = f"./{testbench}/{design_name}_{testbench}_reg_features_and_weight_pre_wavenotsame.csv"

# output
reg_features_and_weight_pre_combine_file = f"./{testbench}/{design_name}_{testbench}_reg_features_and_weight_pre_combine.csv"

def find_highly_correlated_signals(similarity_threshold=0.9):
    
    waveform_df = pd.read_csv(reg_waveform_file)
    waveform_data = waveform_df.iloc[:, 1:] 
    valid_regs = waveform_data.columns.tolist()  

    cluster_df = pd.read_csv(reg_clusters_file)
    filtered_cluster_df = cluster_df[cluster_df['reg_name'].isin(valid_regs)]

    reg_feature_df = pd.read_csv(reg_features_and_weight_pre_file)

    similarity_matrix = cosine_similarity(waveform_data.T) 
    distance_matrix = 1 - similarity_matrix  
    np.fill_diagonal(distance_matrix, 0)  
    distance_matrix[distance_matrix < 0] = 0  

    compressed_distance_matrix = ssd.squareform(distance_matrix)

    linkage_matrix = linkage(compressed_distance_matrix, method='average')
    clusters = fcluster(linkage_matrix, t=1 - similarity_threshold, criterion='distance')

    waveform_clusters = pd.DataFrame({'reg_name': valid_regs, 'Cluster': clusters})

    pattern_to_signals = {}
    for cluster_id, grouped_signals in waveform_clusters.groupby('Cluster'):

        signal_names = grouped_signals['reg_name'].tolist()

        matching_clusters = filtered_cluster_df[filtered_cluster_df['reg_name'].isin(signal_names)]
        unique_clusters = matching_clusters['Cluster'].nunique()

        if unique_clusters > 1:
            pattern_to_signals[cluster_id] = signal_names

    output_data = []
    pattern_id = 1

    for cluster_id, signals in pattern_to_signals.items():
        signal_features = reg_feature_df[reg_feature_df['reg_name'].isin(signals)]

        for signal in signals:
            if signal not in signal_features['reg_name'].values:
                print(f"Warning: Signal {signal} not found in {reg_features_and_weight_pre_file}")
                continue
            
            signal_row = signal_features[signal_features['reg_name'] == signal].iloc[0].tolist()
            signal_row.append(pattern_id)
            output_data.append(signal_row)

        pattern_id += 1

    output_columns = reg_feature_df.columns.tolist() + ['pattern_id']
    output_df = pd.DataFrame(output_data, columns=output_columns)
    output_df.to_csv(reg_features_and_weight_pre_wavesame_file, index=False)
    print(f"same pattern regs saved to {reg_features_and_weight_pre_wavesame_file}")

    processed_signals = output_df['reg_name'].tolist()
    remaining_features = reg_feature_df[~reg_feature_df['reg_name'].isin(processed_signals)]
    remaining_features.to_csv(reg_features_and_weight_pre_wavenotsame_file, index=False)
    print(f"not same pattern regs saved to {reg_features_and_weight_pre_wavenotsame_file}")


def process_and_combine_outputs():
    output1 = pd.read_csv(reg_features_and_weight_pre_wavesame_file)
    output2 = pd.read_csv(reg_features_and_weight_pre_wavenotsame_file)

    if 'PowerWeight' not in output1.columns or 'pattern_id' not in output1.columns:
        raise ValueError("output1 must contain 'PowerWeight' and 'pattern_id' columns")
    if 'tree_feature_node_num' not in output1.columns or 'tree_feature_intree_node_num' not in output1.columns:
        raise ValueError("output1 must contain 'tree_feature_node_num' and 'tree_feature_intree_node_num' columns")

    processed_rows = []
    for pattern_id, group in output1.groupby('pattern_id'):

        total_power_weight = group['PowerWeight'].sum()

        node_num_ratio = group['tree_feature_node_num'] / group['tree_feature_node_num'].sum()
        intree_node_num_ratio = group['tree_feature_intree_node_num'] / group['tree_feature_intree_node_num'].sum()

        allocation_ratio = (node_num_ratio + intree_node_num_ratio) / 2

        group.loc[:,'PowerWeight'] = allocation_ratio * total_power_weight

        group = group.drop(columns=['pattern_id'])

        processed_rows.append(group)

    processed_output1 = pd.concat(processed_rows, ignore_index=True)

    combined_output = pd.concat([output2, processed_output1], ignore_index=True)

    combined_output.to_csv(reg_features_and_weight_pre_combine_file, index=False)
    print(f"Combined output saved to {reg_features_and_weight_pre_combine_file}")

find_highly_correlated_signals()
process_and_combine_outputs()