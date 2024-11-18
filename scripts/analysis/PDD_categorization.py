import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load data
toc = pd.read_csv('data/training/data_analysis/table_of_contents.csv')
scraped_df = pd.read_csv('data/training/data_collection/verra_data.csv')
project_sector = pd.read_csv('data/training/data_collection/AlliedOffsets_project_info.csv', dtype='str')

# Filtering data only VCS and under Forestry and Energy sector
project_sector = project_sector[['UID', 'Project Sector', 'Project Country', 'Year Founded']]
project_sector = project_sector[(project_sector['UID'].str.startswith('VCS')) & 
                                (project_sector['Project Sector'].isin(['Forestry and Land Use', 'Renewable Energy']))]
project_sector['id'] = project_sector['UID'].str.split('VCS', n=1).str[1].astype('int64')

# Filter only date column
scraped_df = scraped_df[['id', 'Project Registration Date']]

# Merge to final dataset
toc = toc.merge(project_sector, how='left', on='id').merge(scraped_df, how='left', on='id')

# Extract registration year, and fill empty value with year founded retreived from AlliedsOffset
toc['Project Registration Date'] = pd.to_datetime(toc['Project Registration Date'], errors='coerce', dayfirst=True)
toc['Project Registration Year'] = toc['Project Registration Date'].dt.to_period('Y').astype(str)
toc['Project Registration Year'] = toc['Project Registration Year'].replace('NaT', pd.NA)
toc['Year Founded'] = toc['Year Founded'].astype(str).str.split('-').str[0].str.replace(',', '')
toc['Project Registration Year'] = toc['Project Registration Year'].fillna(toc['Year Founded'])

# Preprocess data to not take small variations like ':' into account 
toc['section'] = toc['section'].str.lower().str.replace(':', '').replace('.', '')

toc['id'] = toc['id'].astype('str')

# Create dataframe to build network graph
df = pd.DataFrame([{'id': _id, 'section': list(group['section']), 
                    'project_country': group['Project Country'].iloc[0], 
                    'project_sector': group['Project Sector'].iloc[0],
                    'project_year': group['Project Registration Year'].iloc[0]} 
                    for _id, group in toc.groupby('id')])
G = nx.Graph()

# Build the graph by comparing set of headings from any two projects
# If the set of headings overlap at least 10 headings, group those two projects into the same group
for i, row_i in df.iterrows():
    id_i = row_i['id']
    sections_i = set(row_i['section'])
    
    for j in range(i + 1, len(df)):
        row_j = df.iloc[j]
        id_j = row_j['id']
        sections_j = set(row_j['section'])
        
        # Check for overlapping
        common_sections = sections_i.intersection(sections_j)
        
        # if overlapped elements >= 10, then they belong to the same group
        if len(common_sections) >= 10:
            G.add_edge(id_i, id_j)

# List the network groups
groups = list(nx.connected_components(G))

group_data = []

# Store project ID and group number in dataframe
for idx, group in enumerate(groups, start=1):
    for node_id in group:
        group_data.append({'group': idx, 'id': node_id})
group_df = pd.DataFrame(group_data)

# Merge with the original df
df = pd.merge(group_df, df, on='id', how='left')

## =====================================
## Visualisation
## =====================================

plt.figure(figsize=(12, 12))

# Generate node positions
position = nx.spring_layout(G, seed=42)

# Draw the nodes and edges
nx.draw_networkx_nodes(G, position, node_size=50, node_color='#654321', alpha=0.5)
nx.draw_networkx_edges(G, position, width=0.5, edge_color='gray', alpha=0.5)

# Define colormap to differentiate groups
colors = plt.cm.get_cmap('rainbow', len(groups))

# Add patch for each group
for idx, group in enumerate(groups):

    # Collect position for entire group
    group_position = np.array([position[node] for node in group]) 

    # Calculate centre point of each group
    group_centre = group_position.mean(axis=0)

    # Define radius to draw the circle
    radius = np.linalg.norm(group_position - group_centre, axis=1).max() + 0.05

    # Draw the circle around each group
    patch = patches.Circle(group_centre, radius, edgecolor=colors(idx), facecolor=colors(idx), lw=0,alpha=0.5)
    plt.gca().add_patch(patch)

plt.axis('off')
plt.title("Network Graph showing the structural variations of PDDs", fontsize=15)
plt.show()


# To visualise the PDD structure on different project sectors
forestry_sector = 'Forestry and Land Use'
energy_sector = 'Renewable Energy'

# Filter data for each sector
forestry_df = df[df['project_sector'] == forestry_sector]
energy_df = df[df['project_sector'] == energy_sector]

# Group and pivot data
forestry_group = forestry_df.groupby(['group', 'project_year']).count().reset_index()
forestry_pivot = forestry_group.pivot(index='group', columns='project_year', values='id').fillna(0)
energy_group = energy_df.groupby(['group', 'project_year']).count().reset_index()
energy_pivot = energy_group.pivot(index='group', columns='project_year', values='id').fillna(0)

# Align both sectors to the same year and group range
all_years = sorted(set(forestry_pivot.columns).union(set(energy_pivot.columns)))
all_groups = pd.Series(range(1, len(groups)+1))
forestry_pivot = forestry_pivot.reindex(index=all_groups, columns=all_years, fill_value=0)
energy_pivot = energy_pivot.reindex(index=all_groups, columns=all_years, fill_value=0)

# Define colorbar for range of years
colors = plt.cm.coolwarm(np.linspace(0, 1, len(all_years)))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)

# Plot forestry sector visualisation
forestry_pivot.plot(kind='bar', stacked=True, ax=axes[0], color=colors, legend=False)
axes[0].set_title(f'PDDs Structure in {forestry_sector} Sector', fontsize=20) 
axes[0].set_ylabel('Project Count', fontsize=20) 
axes[0].grid(axis='y')

# Plot energy sector visualisation
energy_pivot.plot(kind='bar', stacked=True, ax=axes[1], color=colors, legend=False)
axes[1].set_title(f'PDDs Structure in {energy_sector} Sector', fontsize=20) 
axes[1].set_ylabel('Project Count', fontsize=20) 
axes[1].grid(axis='y')

# Set configurations
for ax in axes:
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

plt.xlabel('Group', fontsize=20) 
plt.xticks(rotation=0) 

plt.tight_layout()
plt.show()