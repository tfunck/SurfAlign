import os
import subprocess
import numpy as np
import nibabel as nib
import shutil
import argparse

from nibabel.freesurfer import read_morph_data, read_geometry
import brainbuilder.utils.mesh_utils as mesh_utils
import surfalign.utils as utils
import surfalign.msm as msm


def surfalign( 
        fixed_sphere:str,
        fixed_mid_cortex:str,
        moving_sphere:str,
        moving_mid_cortex:str,
        output_dir:str,
        radius:float=1,
        fixed_mask:str=None,
        moving_mask:str=None,
        visualize:bool=False,
        clobber:bool=False
    ):
    """
        Align two surfaces using MSM (Multimodal Surface Matching).
        Parameters:
        -----------
        fixed_sphere : str
            Path to the fixed sphere surface file.
        fixed_mid_cortex : str
            Path to the fixed mid-cortex surface file.
        moving_sphere : str
            Path to the moving sphere surface file.
        moving_mid_cortex : str
            Path to the moving mid-cortex surface file.
        output_dir : str
            Directory where the output files will be saved.
        radius : float, optional
            Radius for surface modification, by default 1.
        fixed_mask : str
            Path to the fixed mask file, default is None.
        moving_mask : str
            Path to the moving mask file, default is None.
        visualize : bool, optional
            If True, visualize the final alignment using wb_view, by default False.
        clobber : bool, optional
            If True, overwrite existing files, by default False.
        Returns:
        --------
        tuple
            A tuple containing the paths to the warped sphere, fixed sphere, and moving sphere.
        Notes:
        ------
        This function performs the following steps:
        1. Creates necessary output directories.
        2. Converts the moving sphere to GIFTI format if needed.
        3. Remeshes and resamples the moving surface to match the fixed surface.
        4. Modifies the sphere surfaces based on the given radius.
        5. Computes initial and final surface metrics.
        6. Performs initial and final surface alignment using MSM.
        7. Optionally visualizes the final alignment.
        Example:
        --------
        >>> warped_sphere, fixed_sphere, moving_sphere = surfalign(
        ...     fixed_sphere='fixed_sphere.gii',
        ...     fixed_mid_cortex='fixed_mid_cortex.gii',
        ...     fixed_mask='fixed_mask.nii',
        ...     moving_sphere='moving_sphere.gii',
        ...     moving_mid_cortex='moving_mid_cortex.gii',
        ...     moving_mask='moving_mask.nii',
        ...     output_dir='/path/to/output',
        ...     radius=1.5,
        ...     visualize=True,
        ...     clobber=True
        ... )
    """
        
    output_init_dir = output_dir+'/init/'
    output_mid_dir = output_dir+'/mid/'
    output_final_dir = output_dir+'/final/'
    os.makedirs(output_final_dir, exist_ok=True)
    os.makedirs(output_mid_dir, exist_ok=True)
    os.makedirs(output_init_dir, exist_ok=True)

    # check if moving_sphere is fs or gii
    moving_sphere = utils.convert_fs_to_gii(moving_sphere, output_dir, clobber=clobber)

    n_fixed_vertices = nib.load(fixed_sphere).darrays[0].data.shape[0]
    moving_sphere_orig = utils.convert_fs_to_gii(moving_sphere, output_dir, clobber=clobber)
    moving_sphere = utils.remesh_surface(moving_sphere, output_dir, n_fixed_vertices , clobber=clobber)
    moving_mid_cortex = utils.resample_surface(moving_mid_cortex, moving_sphere_orig, moving_sphere, output_dir, n_fixed_vertices, clobber=clobber)

    #moving_sphere_orig = utils.remesh_surface(moving_sphere, output_dir, radius=radius, clobber=clobber)
    moving_sphere = utils.surface_modify_sphere(moving_sphere, output_dir, radius=radius, clobber=clobber)
    fixed_sphere = utils.surface_modify_sphere(fixed_sphere, output_dir, radius=radius, clobber=clobber)

    init_moving_metrics = utils.get_surface_metrics(moving_mid_cortex, output_mid_dir, metrics=['x', 'y', 'z'],  moving_mask=None, clobber=clobber )
    init_fixed_metrics= utils.get_surface_metrics(fixed_mid_cortex,  output_mid_dir, metrics=['x', 'y', 'z'], fixed_mask=None, clobber=clobber )

    moving_metrics = utils.get_surface_metrics(moving_mid_cortex, moving_mask, output_final_dir, [ 'sulc'], n_sulc=10, n_curv=30, clobber=clobber) 
    print('\nwb_view', moving_sphere, moving_mid_cortex, moving_metrics, '\n' ) 
    fixed_metrics= utils.get_surface_metrics(fixed_mid_cortex, fixed_mask, output_final_dir, [ 'sulc'], n_sulc=10, n_curv=10, clobber=clobber) 
    print('\nwb_view', fixed_sphere, fixed_mid_cortex, fixed_metrics, '\n' )

    # Quality control for surface alignment
    #for label, metric in fixed_metrics_dict.items():
    #    plot_receptor_surf([metric], fixed_mid_cortex, output_dir, label='fx_'+label, cmap='nipy_spectral', clobber=clobber)
    #    plot_receptor_surf([metric], fixed_sphere, output_dir, label='fx_sphere_'+label, cmap='nipy_spectral', clobber=clobber)
    
    #for label, metric in moving_metrics_dict.items():
    #    plot_receptor_surf([metric], moving_mid_cortex, output_dir, label='mv_'+label, cmap='nipy_spectral', clobber=clobber)
    #    plot_receptor_surf([metric], moving_sphere, output_dir, label='mv_sphere_'+label, cmap='nipy_spectral', clobber=clobber)
    warped_sphere_init, data_rsl_init = msm.msm_align(
        fixed_sphere,
        init_fixed_metrics, 
        fixed_mask,
        moving_sphere, 
        init_moving_metrics,
        moving_mask,
        output_init_dir, 
        levels=2,
        clobber=clobber
    )

    warped_sphere, data_rsl_final  = msm.msm_align(
        fixed_sphere,
        fixed_metrics, 
        fixed_mask,
        moving_sphere, 
        moving_metrics,
        moving_mask,
        output_final_dir, 
        levels=2,
        trans=warped_sphere_init,
        clobber=clobber
    )
    if visualize:
        subprocess.run(f'wb_view {fixed_mid_cortex} {fixed_metrics} {data_rsl_final}', shell=True, executable="/bin/bash")

    return warped_sphere, fixed_sphere, moving_sphere


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align two surfaces using MSM.')
    parser.add_argument('fixed_sphere', type=str, help='Path to the fixed sphere file')
    parser.add_argument('fixed_mid_cortex', type=str, help='Path to the fixed mid cortex file')
    parser.add_argument('fixed_mask', type=str, help='Path to the fixed mask file')
    parser.add_argument('moving_sphere', type=str, help='Path to the moving sphere file')
    parser.add_argument('moving_mid_cortex', type=str, help='Path to the moving mid cortex file')
    parser.add_argument('moving_mask', type=str, help='Path to the moving mask file')
    parser.add_argument('output_dir', type=str, help='Directory to save the output')
    parser.add_argument('--radius', type=float, default=1, help='Radius for surface modification')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results using wb_view')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files')

    args = parser.parse_args()

    surfalign(
        fixed_sphere=args.fixed_sphere,
        fixed_mid_cortex=args.fixed_mid_cortex,
        fixed_mask=args.fixed_mask,
        moving_sphere=args.moving_sphere,
        moving_mid_cortex=args.moving_mid_cortex,
        moving_mask=args.moving_mask,
        output_dir=args.output_dir,
        radius=args.radius,
        visualize=args.visualize,
        clobber=args.clobber
    )
