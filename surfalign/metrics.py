import os
import subprocess
import shutil
import numpy as np
import nibabel as nib
from surfalign import utils

def create_xyz_axis_file(surf_filename, mask_filename, output_dir, axis, clobber=False):
    """Create a gifti file with the product of z and y coordinates as the data."""
    base = os.path.basename(surf_filename).replace('.surf.gii','')
    output_filename = f'{output_dir}/{base}_axis-{axis}.func.gii'

    if not os.path.exists(output_filename) or clobber:
        mask = nib.load(mask_filename).darrays[0].data
        coords, _ = utils.load_mesh_ext(surf_filename)
        zyaxis = coords[:,axis] 
        zyaxis = (zyaxis - zyaxis.min()) / (zyaxis.max() - zyaxis.min())
        zyaxis[mask==0] = -3 * np.abs(np.min(zyaxis))
        utils.write_gifti(zyaxis, output_filename)
    return output_filename

def get_surface_sulcal_depth(surf_filename: str, output_dir: str, n: int = 10, dist: float = 0.1, clobber: bool = False) -> tuple:
    """
    Get sulcal depth using mris_inflate.

    Parameters:
    surf_filename (str): The filename of the surface file.
    output_dir (str): The directory where the output files will be saved.
    n (int, optional): The number of iterations for mris_inflate. Default is 10.
    dist (float, optional): The distance parameter for mris_inflate. Default is 0.1.
    clobber (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
    tuple: A tuple containing the sulcal depth filename and the inflated surface filename.

    Raises:
    AssertionError: If the sulcal depth file does not exist after processing.
    """
    base = os.path.basename(surf_filename).replace('.surf.gii','')

    if 'lh.' == utils.get_fs_prefix(surf_filename):
        prefix='lh'
    else :
        prefix='rh'

    sulc_suffix = f'{base}.sulc'
    temp_sulc_filename = f'{output_dir}/{prefix}.{sulc_suffix}'
    sulc_filename = f'{output_dir}/lh.{sulc_suffix}'
    inflated_filename = f'{output_dir}/lh.{base}.inflated'

    if not os.path.exists(sulc_filename) or clobber:
        cmd = f"mris_inflate  -dist {dist} -n {n} -sulc {sulc_suffix} {surf_filename} {inflated_filename}"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        shutil.move(temp_sulc_filename, sulc_filename)

    assert os.path.exists(sulc_filename), f"Could not find sulcal depth file {sulc_filename}"
    return sulc_filename, inflated_filename


def get_surface_curvature(surf_filename:str, output_dir ,n=10, clobber=False)->str:
    """Get surface curvature using mris_curvature."""

    target_prefix = utils.get_fs_prefix(surf_filename)
    prefix=''
    if 'lh.' not in target_prefix and 'rh.' not in target_prefix: 
        prefix='unknown.'

    print()
    print(target_prefix)
    print(prefix)
    print()

    base = prefix+os.path.basename(surf_filename)#.replace('.surf.gii','')
    dirname = os.path.dirname(surf_filename)
    curv_filename = f'{dirname}/{base}.H'
    output_filename = f'{output_dir}/{base}.H'
    if not os.path.exists(output_filename) or clobber :
        cmd = f"mris_curvature -w -a {n}  {surf_filename}"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        shutil.move(curv_filename, output_filename)
    
    assert os.path.exists(output_filename), f"Could not find curvature file {output_filename}"

    return output_filename



def get_surface_curvature(surf_filename:str, output_dir:str ,n:int=10, clobber:bool=False)->str:
    """Get surface curvature using mris_curvature."""

    target_prefix = utils.get_fs_prefix(surf_filename)
    prefix=''
    if 'lh.' not in target_prefix and 'rh.' not in target_prefix: 
        prefix='unknown.'

    print()
    print(target_prefix)
    print(prefix)
    print()

    base = prefix+os.path.basename(surf_filename)#.replace('.surf.gii','')
    dirname = os.path.dirname(surf_filename)
    curv_filename = f'{dirname}/{base}.H'
    output_filename = f'{output_dir}/{base}.H'
    if not os.path.exists(output_filename) or clobber :
        cmd = f"mris_curvature -w -a {n}  {surf_filename}"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        shutil.move(curv_filename, output_filename)
    
    assert os.path.exists(output_filename), f"Could not find curvature file {output_filename}"

    return output_filename



def get_surface_metrics(
        surf_filename: str, 
        output_dir: str, 
        metrics: list = ['sulc'], 
        n_sulc: int = 10, 
        dist: float = 0.1, 
        n_curv: int = 100, 
        mask_filename: str = None, 
        clobber: bool = False
        ) -> str:
    """
    Compute and retrieve various surface metrics for a given surface file.
    Parameters:
    -----------
    surf_filename : str
        Path to the input surface file in .surf.gii format.
    output_dir : str
        Directory where the output files will be saved.
    metrics : list of str, optional
        List of metrics to compute. Default is ['sulc'].
        Possible values include 'sulc', 'curv', 'mask', 'x', 'y', 'z'.
    n_sulc : int, optional
        Number of sulcal depth iterations. Default is 10.
    dist : float, optional
        Distance parameter for sulcal depth computation. Default is 0.1.
    n_curv : int, optional
        Number of curvature iterations. Default is 100.
    mask_filename : str, optional
        Path to the mask file. Default is None.
    clobber : bool, optional
        If True, overwrite existing files. Default is False.
    Returns:
    --------
    str
        Path to the output file containing the merged metrics in .func.gii format.
    Notes:
    ------
    This function computes various surface metrics such as sulcal depth, curvature,
    and axis coordinates (x, y, z) for a given surface file. The computed metrics
    are saved in the specified output directory and merged into a single .func.gii file.
    """
    base = os.path.basename(surf_filename).replace('.surf.gii','')
    output_file = f'{output_dir}/lh.{base}_metrics.func.gii'

    metrics_dict = {}

    if 'sulc' in metrics : 
        fs_sulc_filename, _ = get_surface_sulcal_depth(surf_filename, output_dir, n=n_sulc, dist=dist, clobber=clobber)
        sulc_filename = utils.convert_fs_morph_to_gii(fs_sulc_filename, mask_filename, output_dir, clobber=clobber)
        
        if mask_filename is not None:
            utils.mask_func_gii(sulc_filename, mask_filename)

        metrics_dict['sulc'] = sulc_filename
    
    if 'curv' in metrics:
        fs_curv_filename = get_surface_curvature(surf_filename, output_dir, n=n_curv, clobber=clobber)
        curv_filename = utils.convert_fs_morph_to_gii(fs_curv_filename, mask_filename, output_dir, clobber=clobber)

        if mask_filename is not None:
            utils.mask_func_gii(curv_filename, mask_filename)

        metrics_dict['curv'] = curv_filename

    if 'mask' in metrics :
        metrics_dict['mask'] = mask_filename 

    if 'x' in metrics :
        axis_filename = create_xyz_axis_file(surf_filename, mask_filename, output_dir, 0, clobber=clobber)
        metrics_dict['x'] = axis_filename

    if 'y' in metrics :
        axis_filename = create_xyz_axis_file(surf_filename,  mask_filename, output_dir, 1, clobber=clobber)
        metrics_dict['y'] = axis_filename

    if 'z' in metrics :
        axis_filename = create_xyz_axis_file(surf_filename,  mask_filename, output_dir, 2, clobber=clobber)
        metrics_dict['z'] = axis_filename

    if not os.path.exists(output_file) or clobber:
        # merge input metrics
        metric_string=''
        for metric in metrics_dict.values() :
            metric_string += f' -metric {metric} '

        cmd = f"wb_command -metric-merge {output_file} {metric_string} "

        subprocess.run(cmd, shell=True, executable="/bin/bash")
    
    return output_file 