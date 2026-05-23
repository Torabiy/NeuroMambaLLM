def plot_figure(X, y, test_idx, preds):
    atlas = datasets.fetch_atlas_msdl(data_dir="XXX") #Choose your directory
    coords = atlas.region_coords
    
    # Extract test data
    X_test_set = X[test_idx]
    
    # Compute averages
    avg_tc = np.mean(X_test_set[preds == 0], axis=0)
    avg_asd = np.mean(X_test_set[preds == 1], axis=0)
    
    # Using 'PiYG' colormap: Positive = Pink/Red, Negative = Green
    combined_matrix = (avg_asd - avg_tc) 
    combined_matrix = (combined_matrix + combined_matrix.T) / 2 

    fig = plt.figure(figsize=(16, 10), facecolor='white')
    
    # 3. 'PiYG' colormap gives you the Red/Green distinction
    plotting.plot_connectome(
        combined_matrix, 
        coords, 
        edge_threshold="99%", 
        edge_cmap='PiYG',      # Green for TC (-), Red/Pink for ASD (+)
        display_mode='ortho', 
        node_size=25,
        node_color='black',
        title="",
        figure=fig,
        colorbar=False
    )
    plt.savefig("mamba.png", facecolor='white', dpi=300, bbox_inches='tight')
    plt.show()


if name == "__main__":
    X_final, y_final, test_indices, final_preds = run()
    plot_figure(X_final, y_final, test_indices, final_preds)
