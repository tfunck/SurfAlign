
from matplotlib_surface_plotting import plot_surf
from surfalign.utils import utils
import numpy as np
import os
import nibabel as nib


def plot_receptor_surf(
        receptor_surfaces, 
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
    print('Receptor surfaces', receptor_surfaces)

    filename = f"{output_dir}/{label}_surf.png" 
    
    if not os.path.exists(filename) or clobber :
        coords, faces = utils.load_mesh_ext(cortex_filename)
        
        try :
            ndepths=nib.load(receptor_surfaces[0]).darrays[0].shape[1]
        except IndexError:
            ndepths=1

        receptor_all = np.array([ load_gifti(fn).reshape(-1,1) for fn in receptor_surfaces ])
        receptor = np.mean( receptor_all,axis=(0,2))

        if scale is not None:
            receptor = scale(receptor)


        pvals = np.ones(receptor.shape[0])
        if medial_wall_mask is not None :
            pvals[medial_wall_mask] = np.nan

        #vmin, vmax = np.nanmax(receptor)*threshold[0], np.nanmax(receptor)*threshold[1]
        vmin, vmax = np.percentile(receptor[~np.isnan(receptor)], threshold)
        print('real threshold', threshold)
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

        if ndepths > 3 :
            bins = np.rint(np.linspace(0, ndepths,4)).astype(int)
            for i, j in zip(bins[0:-1], bins[1:]):
                receptor = np.mean( np.array([ np.load(fn)[:,i:j] for fn in receptor_surfaces ]),axis=(0,2))

                vmin, vmax = np.nanmax(receptor)*threshold[0], np.nanmax(receptor)*threshold[1]

                filename = f"{output_dir}/surf_profiles_{label}_layer-{i/ndepths}.png" 
                
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

