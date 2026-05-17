import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image

def plot_clean_brain_regions():
    # Load MSDL Atlas
    atlas = datasets.fetch_atlas_msdl(data_dir="./abide_fast_cache")
    atlas_img = image.load_img(atlas.maps)
    labels = atlas.labels # List of 39 region names
    
    # NEW INDICES for MSDL Atlas:
    # Index 25 in MSDL is 'R Inf Temp' (Right Inferior Temporal)
    # Index 7 in MSDL is 'L Cing post' (Left Posterior Cingulate)
    indices = [25, 7] 
    
    fig = plt.figure(figsize=(15, 10), facecolor='white') # Match reference dark bg

    for i, idx in enumerate(indices):
        # Extract the specific 3D map for this ROI from the 4D atlas
        reg_img = image.index_img(atlas_img, idx)
        
        ax = fig.add_subplot(2, 1, i+1)
        
        # Plot orthographic slices (MNI coordinates)
        display = plotting.plot_stat_map(
            reg_img,
            display_mode='ortho',
            title=f"Region: {labels[idx]}",
            axes=ax,
            colorbar=False,
            cmap=plotting.cm.blue_transparent, # Matches blue highlight in reference
            draw_cross=True,
            annotate=True,
            black_bg=True # Matches the Figure 3 reference aesthetic
        )

    plt.subplots_adjust(hspace=0.2)
    plt.savefig("mamba1.png", facecolor='white', dpi=300, bbox_inches='tight')
    plt.show()

plot_clean_brain_regions()
