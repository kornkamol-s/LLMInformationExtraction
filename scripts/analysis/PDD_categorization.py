
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("connected_components_groups.csv")

# Define the two project sectors you want to plot separately
sector1 = 'Forestry and Land Use'  # Example sector 1
sector2 = 'Renewable Energy'        # Example sector 2

all_groups = pd.Series(range(1, 12))

# Filter data for each sector
df_sector1 = df[df['project_sector'] == sector1]
df_sector2 = df[df['project_sector'] == sector2]

# Group and pivot for sector 1
grouped_data_sector1 = df_sector1.groupby(['group', 'project_year']).count().reset_index()
pivot_data_sector1 = grouped_data_sector1.pivot(index='group', columns='project_year', values='id').fillna(0)

# Group and pivot for sector 2
grouped_data_sector2 = df_sector2.groupby(['group', 'project_year']).count().reset_index()
pivot_data_sector2 = grouped_data_sector2.pivot(index='group', columns='project_year', values='id').fillna(0)

# Ensure all groups are represented
pivot_data_sector1 = pivot_data_sector1.reindex(all_groups, fill_value=0)
pivot_data_sector2 = pivot_data_sector2.reindex(all_groups, fill_value=0)

# Create a continuous color gradient from blue to red for sector 1
num_years1 = len(pivot_data_sector1.columns)
colors_sector1 = plt.cm.coolwarm(np.linspace(0, 1, num_years1))

# Create a continuous color gradient from blue to red for sector 2
num_years2 = len(pivot_data_sector2.columns)
colors_sector2 = plt.cm.coolwarm(np.linspace(0, 1, num_years2))

# Create subplots for the two sectors
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)

# Plot for sector 1
pivot_data_sector1.plot(kind='bar', stacked=True, ax=axes[0], color=colors_sector1, legend=False)
axes[0].set_title(f'PDDs Structure in {sector1} Sector', fontsize=20) 
axes[0].set_ylabel('Project Count', fontsize=20) 
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot for sector 2
pivot_data_sector2.plot(kind='bar', stacked=True, ax=axes[1], color=colors_sector2, legend=False)
axes[1].set_title(f'PDDs Structure in {sector2} Sector', fontsize=20) 
axes[1].set_ylabel('Project Count', fontsize=20) 
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

for ax in axes:
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

# Finalize layout
plt.xlabel('Group', fontsize=20) 
plt.xticks(rotation=0) 

# Set the x-ticks to show all groups (1 to 13)
axes[0].set_xticks(all_groups)
axes[1].set_xticks(all_groups)

# Adjust layout to prevent clipping
plt.tight_layout()
plt.show()