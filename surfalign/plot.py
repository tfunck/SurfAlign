
from matplotlib_surface_plotting import plot_surf
import matplotlib.pyplot as plt
from surfalign.utils import utils
import numpy as np
import os
import nibabel as nib

def plot_surface_metrics(metrics_filename, cortex_filename, output_dir, title='', labels=[], cmap='RdBu_r', threshold=[2,98], clobber=False):
    """Plot receptor profiles on the cortical surface"""

    coords, faces = utils.load_mesh_ext(cortex_filename)
    surf = nib.load(metrics_filename)
    n_darrays = len(surf.darrays)

    if labels == []:
        labels = [f'{i}' for i in range(n_darrays)]

    for i, label in enumerate(labels):

        plot_surf(  
            coords, 
            faces, 
            surf.darrays[i].data, 
            rotate=[90, 270], 
            title=title,  
            filename=f'{output_dir}/{label}_{title}.png',
            pvals=np.ones(surf.darrays[i].data.shape[0]),
            vmin=np.percentile(surf.darrays[i].data, threshold[0]),
            vmax=np.percentile(surf.darrays[i].data, threshold[1]),
            cmap=cmap,
            cmap_label=label
        ) 
        plt.close()



def plot_metrics(
        metrics_list,
        cortex_filename, 
        output_dir, 
        medial_wall_mask=None,
        threshold=[2,98],
        label='', 
        cmap='RdBu_r',
        scale=None,
        clobber:bool=False
        ):
    """Plot receptor profiles on the cortical surface"""

    os.makedirs(output_dir, exist_ok=True)

    for i, metric_filename in enumerate(metrics_list):

        filename = f"{output_dir}/{label}_metrics-{i}.png" 
        
        if not os.path.exists(filename) or clobber :
            coords, faces = utils.load_mesh_ext(cortex_filename)
            

            receptor_all = np.array([ nib.load(fn).darrays[0].data.reshape(-1,1) for fn in receptor_surfaces ])
            receptor = np.mean( receptor_all,axis=(0,2))

            if scale is not None:
                receptor = scale(receptor)


            pvals = np.ones(receptor.shape[0])
            if medial_wall_mask is not None :
                pvals[medial_wall_mask] = np.nan

            #vmin, vmax = np.nanmax(receptor)*threshold[0], np.nanmax(receptor)*threshold[1]
            vmin, vmax = np.percentile(receptor[~np.isnan(receptor)], threshold)

            if not os.path.exists(filename) or clobber :
                print(f'\tWriting {filename}')
                plot_surf(  coords, 
                            faces, 
                            receptor, 
                            rotate=[90, 270], 
                            filename=filename,
                            pvals=pvals,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            cmap_label=label
                            )
                plt.close() 
