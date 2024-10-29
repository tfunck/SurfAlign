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
import surfalign.metrics as metrics
import surfalign.plot as plot



def resample_to_n_vertices(sphere:str,  mid_cortex:str, output_dir:str, n_fixed_vertices:int, clobber:bool=False)->tuple:
    """Resample the input surfaces to have the same number of vertices."""   
    sphere_orig = sphere

    n_vertices = utils.load_mesh_ext(sphere)[0].shape[0]
    
    if n_vertices != n_fixed_vertices:
        sphere = utils.remesh_surface(sphere, output_dir, n_fixed_vertices , clobber=clobber)

        mid_cortex = utils.resample_surface(mid_cortex, sphere_orig, sphere, output_dir, n_fixed_vertices, clobber=clobber)


    return sphere, mid_cortex

def preprocess(
        fixed_sphere:str,
        fixed_mid_cortex:str,
        moving_sphere:str,
        moving_mid_cortex:str,
        output_dir:str,
        radius:float=1,
        n_vertices:int=None,
        fixed_mask:str=None,
        moving_mask:str=None,
        clobber:bool=False
    )->tuple:
    """
        Preprocesses the input surfaces and metrics for alignment.
        Parameters:
        - fixed_sphere (str): Path to the fixed sphere file.
        - fixed_mid_cortex (str): Path to the fixed mid cortex file.
        - moving_sphere (str): Path to the moving sphere file.
        - moving_mid_cortex (str): Path to the moving mid cortex file.
        - output_dir (str): Directory to save the output files.
        - radius (float, optional): Radius for surface modification. Default is 1.
        - fixed_mask (str, optional): Path to the fixed mask file. Default is None.
        - moving_mask (str, optional): Path to the moving mask file. Default is None.
        - clobber (bool, optional): Whether to overwrite existing files. Default is False.
        Returns:
        - tuple: Contains the following elements:
        - moving_sphere (str): Path to the processed moving sphere file.
        - fixed_sphere (str): Path to the processed fixed sphere file.
        - moving_mid_cortex (str): Path to the processed moving mid cortex file.
        - fixed_mid_cortex (str): Path to the processed fixed mid cortex file.
        - init_moving_metrics (dict): Initial metrics for the moving mid cortex.
        - init_fixed_metrics (dict): Initial metrics for the fixed mid cortex.
        - moving_metrics (dict): Final metrics for the moving mid cortex.
        - fixed_metrics (dict): Final metrics for the fixed mid cortex.
    """

    if n_vertices is None:
        n_fixed_vertices = utils.load_mesh_ext(fixed_sphere)[0].shape[0]

        fixed_sphere_rsl = fixed_sphere
        fixed_mid_cortex_rsl = fixed_mid_cortex
    else :
        n_fixed_vertices = n_vertices

        fixed_sphere_rsl, fixed_mid_cortex_rsl = resample_to_n_vertices(fixed_sphere, fixed_mid_cortex, output_dir, n_fixed_vertices, clobber=clobber)
        
    moving_sphere_rsl, moving_mid_cortex_rsl = resample_to_n_vertices(moving_sphere, moving_mid_cortex, output_dir, n_fixed_vertices, clobber=clobber)

    moving_sphere = utils.surface_modify_sphere(moving_sphere, output_dir, radius=radius, clobber=clobber)
    fixed_sphere = utils.surface_modify_sphere(fixed_sphere, output_dir, radius=radius, clobber=clobber)
    print('fixed_mid_cortex_rsl', fixed_mid_cortex_rsl);

    return moving_sphere_rsl, fixed_sphere_rsl, moving_mid_cortex_rsl, fixed_mid_cortex_rsl


def surfalign( 
        fixed_sphere:str,
        fixed_mid_cortex:str,
        moving_sphere:str,
        moving_mid_cortex:str,
        output_dir:str,
        radius:float=1,
        levels:int=2,
        moving_metrics_list:list = None,
        fixed_metrics_list:list = None,
        mov_param={'n_sulc':10, 'n_curv':10},
        fix_param={'n_sulc':10, 'n_curv':10},
        metric_list_heir:list = [['y'],['z'], ['curv','sulc']],
        n_vertices:int=None,
        fixed_mask:str=None,
        moving_mask:str=None,
        title:str='',
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
        levels : int, optional
            Number of levels for MSM, by default 2.
        moving_metrics_fn : str, optional
            Path to the moving metrics file, by default None.
        fixed_metrics_fn : str, optional    
            Path to the fixed metrics file, by default None.
        mov_param : dict, optional
            Parameters for the moving surface, by default {'n_sulc':10, 'n_curv':10}.
        fix_param : dict, optional  
            Parameters for the fixed surface, by default {'n_sulc':10, 'n_curv':10}.
        fixed_mask : str
            Path to the fixed mask file, default is None.
        moving_mask : str
            Path to the moving mask file, default is None.
        n_vertices : int, optional
            Number of vertices for to which to resample the surfaces, default=None
        wb_visualize : bool, optional
            If True, wb_visualize the final alignment using wb_view, by default False.
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
        7. Optionally wb_visualizes the final alignment.
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
        ...     wb_visualize=True,
        ...     clobber=True
        ... )
    """
        
    output_surf_dir = output_dir+'/surfaces/'
    os.makedirs(output_surf_dir, exist_ok=True)

    print('fixed_sphere', fixed_sphere)
    print('fixed_mid_cortex', fixed_mid_cortex)
    print('moving_sphere', moving_sphere)
    print('moving_mid_cortex', moving_mid_cortex)

    # check if moving_sphere is fs or gii
    moving_sphere = utils.convert_fs_to_gii(moving_sphere, output_dir, clobber=clobber)

    # Preprocess the input surfaces and metrics so that they can be aligned with msm.
    moving_sphere_rsl, fixed_sphere_rsl, moving_mid_cortex_rsl, fixed_mid_cortex_rsl = preprocess(
        fixed_sphere = fixed_sphere,
        fixed_mid_cortex = fixed_mid_cortex,
        moving_sphere = moving_sphere,
        moving_mid_cortex = moving_mid_cortex,
        output_dir = output_dir,
        radius = radius,
        n_vertices = n_vertices,
        fixed_mask = fixed_mask,
        moving_mask = moving_mask,
        clobber = clobber
    )
    print()
    print('moving_sphere', moving_sphere)
    print('moving_sphere_rsl', moving_sphere_rsl)
    print('metrics_list', moving_metrics_list)
    print()
    print('fixed_sphere', fixed_sphere)
    print('fixed sphere_rsl', fixed_sphere_rsl)
    print('metrics_list', fixed_metrics_list)
    print('fixed_mid_cortex_rsl', fixed_mid_cortex_rsl)

    moving_metrics_list = metrics.extract_metrics(
             moving_mid_cortex_rsl,  
             output_dir, 
             params = mov_param, 
             sphere_filename = moving_sphere, 
             sphere_rsl_filename = moving_sphere_rsl, 
             mask_filename = moving_mask, 
             metrics_file_list = moving_metrics_list, 
             metric_list_heir = metric_list_heir, 
            #clobber = clobber
            clobber=True
        )


    fixed_metrics_list = metrics.extract_metrics(
            fixed_mid_cortex_rsl, 
            output_dir, 
            params = fix_param, 
            sphere_filename = fixed_sphere, 
            sphere_rsl_filename=fixed_sphere_rsl, 
            mask_filename=fixed_mask, 
            metrics_file_list = fixed_metrics_list, 
            metric_list_heir = metric_list_heir, 
            clobber=clobber
        )
    print(fixed_metrics_list);
    print(moving_metrics_list)

    warped_sphere_init = None

    for i, (moving_metrics, fixed_metrics) in enumerate(zip(moving_metrics_list, fixed_metrics_list)):

        warped_sphere, warped_metrics= msm.msm_align(
            fixed_sphere_rsl,
            fixed_metrics, 
            fixed_mask,
            moving_sphere_rsl, 
            moving_metrics,
            moving_mask,
            output_dir+f'/{i}/', 
            trans = warped_sphere_init,
            levels = levels,
            clobber = clobber
        )

        warped_sphere_init = warped_sphere
        n0 = nib.load(warped_metrics).darrays[0].data.shape[0]
        n1 = nib.load(fixed_mid_cortex_rsl).darrays[0].data.shape[0]
        assert n0 == n1, f"Number of vertices in warped metrics {n0} does not match fixed mid cortex {n1},\n\t{warped_metrics},\n\t{fixed_mid_cortex_rsl}"
        plot.plot_surface_metrics(warped_metrics, fixed_mid_cortex_rsl, output_dir, title=title, clobber=clobber)

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
    parser.add_argument('--fixed-metrics', type=str, default=None, help='Path to the fixed metrics file')
    parser.add_argument('--moving-metrics', type=str, default=None, help='Path to the moving metrics file')
    parser.add_argument('--fix-n-sulc', type=int, default=10, help='Number of sulcal depth iterations for fixed surface')
    parser.add_argument('--fix-n-curv', type=int, default=10, help='Number of curvature iterations for fixed surface')
    parser.add_argument('--mov-n-sulc', type=int, default=10, help='Number of sulcal depth iterations for moving surface')
    parser.add_argument('--mov-n-curv', type=int, default=10, help='Number of curvature iterations for moving surface')
    parser.add_argument('--wb-visualize', action='store_true', help='wb_visualize the results using wb_view')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files')

    args = parser.parse_args()

    mov_param = {'n_sulc': args.mov_n_sulc, 'n_curv': args.mov_n_curv}
    fix_param = {'n_sulc': args.fix_n_sulc, 'n_curv': args.fix_n_curv}

    surfalign(
        fixed_sphere=args.fixed_sphere,
        fixed_mid_cortex=args.fixed_mid_cortex,
        fixed_mask=args.fixed_mask,
        moving_sphere=args.moving_sphere,
        moving_mid_cortex=args.moving_mid_cortex,
        moving_mask=args.moving_mask,
        output_dir=args.output_dir,
        fixed_metrics = args.fixed_metrics,
        moving_metrics = args.moving_metrics,
        fix_param=fix_param,
        mov_param=mov_param,
        radius=args.radius,
        wb_visualize=args.wb_visualize,
        clobber=args.clobber
    )
