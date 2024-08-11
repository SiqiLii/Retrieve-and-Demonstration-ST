import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Load the YAML data
with open('/home/sli/DPR_SONAR/results/dev_origin_with_retrieved_example_dpr_sonar_finetune_q_n_p_audio_audio_example_dist.yaml') as f:
    my_dict_dev = yaml.load(f,Loader=yaml.BaseLoader) 


# Create DataFrame
df = pd.DataFrame(my_dict_dev)

# Convert columns to appropriate data types
df['index'] = df['index'].astype(int)
df['similarity'] = df['similarity'].astype(float)

# Create type categories
df['type'] = df.apply(lambda row: 1 if ((not row['rare_word']) and (not row['rare_word_in_example'])) else (2 if (row['rare_word'] and (not row['rare_word_in_example'])) else 3), axis=1)

# Plotting
plt.figure(figsize=(10, 6))
for t, group in df.groupby('type'):
    label = {1: 'not rare word', 2: 'rare word & not retrieved', 3:'rare word & retrieved'}[t]
    size=3 if t==1 else 10
    plt.scatter(group['index'], group['similarity'], s=size, label=label)

plt.xlabel('Index')
plt.ylabel('Similarity')
plt.legend(title='Categories', loc='upper left')
plt.title('Scatter Plot of Similarity by Index with Categories-original-dev-set')
plt.grid(True)
plt.savefig('scatter_plot_similarity_dev.png')
plt.show()