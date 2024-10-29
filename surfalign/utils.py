import nibabel as nib
import os
import numpy as np
import subprocess
import shutil

from matplotlib_surface_plotting import plot_surf
from nibabel.freesurfer import read_morph_data, read_geometry
from surfalign import utils

def create_mid_surface(smoothwm, pial, output):
    """Create a mid surface between smoothwm and pial surfaces"""
    smoothwm = nib.load(smoothwm)
    pial = nib.load(pial)
    smoothwm_data = smoothwm.darrays[0].data
    pial_data = pial.darrays[0].data
    mid_data = (smoothwm_data + pial_data) / 2

    coordsys = nib.gifti.GiftiCoordSystem(dataspace='NIFTI_XFORM_TALAIRACH', xformspace='NIFTI_XFORM_TALAIRACH')
    #g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar[0].astype(np.float32), intent='NIFTI_INTENT_POINTSET',coordsys=coordsys))
    darrays = [nib.gifti.GiftiDataArray(data=mid_data, intent='NIFTI_INTENT_POINTSET',coordsys=coordsys), pial.darrays[1]]
    mid = nib.gifti.GiftiImage(darrays=darrays)
    nib.save(mid, output)


def load_mesh_ext(in_fn:str, faces_fn:str="", correct_offset:bool=False)->np.ndarray:
    """Load a mesh file with the correct function based on the file extension.
    
    :param in_fn: Filename of the mesh
    :param faces_fn: Filename of the faces file, defaults to 
    :param correct_offset: Whether to correct the offset of the mesh, defaults to False
    :return: Coordinates and faces of the mesh
    """
    ext = os.path.splitext(in_fn)[1]
    faces = None
    volume_info = None
    print(ext); 
    if ext in [".pial", ".white", ".gii", ".sphere", ".inflated"]:
        try : # try to load as gifti
            surf = nib.load(in_fn)
            coords, faces, volume_info = surf.darrays[0].data, surf.darrays[1].data, surf.header
        except Exception as e: 
            # try to load as freesurfer
            coords, faces = read_geometry(in_fn)
    elif ext == ".npz":
        coords = np.load(in_fn)["points"]
    else:
        coords = h5.File(in_fn)["data"][:]
        if os.path.splitext(faces_fn)[1] == ".h5":
            faces_h5 = h5.File(faces_fn, "r")
            faces = faces_h5["data"][:]
    return coords, faces

def convert_fs_morph_to_gii(input_filename, mask_filename, output_dir, clobber=False)  :
    """Convert FreeSurfer surface to GIFTI."""
    base = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f'{output_dir}/{base}_sulc.shape.gii'

    if not os.path.exists(output_filename) or clobber:
        ar = utils.read_morph_data(input_filename).astype(np.float32)

        mask = nib.load(mask_filename).darrays[0].data
        #ar[mask==0] = -1.5*np.abs(np.min(ar))

        g = nib.gifti.GiftiImage()
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar))
        nib.save(g, output_filename)
    return output_filename


def get_surface_curvature(surf_filename, output_dir ,n=10, clobber=False):
    """Get surface curvature using mris_curvature."""

    target_prefix = get_fs_prefix(surf_filename)
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

def convert_fs_morph_to_gii(input_filename, mask_filename, output_dir, clobber=False)  :
    """Convert FreeSurfer surface to GIFTI."""
    base = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f'{output_dir}/{base}_sulc.shape.gii'

    if not os.path.exists(output_filename) or clobber:
        ar = read_morph_data(input_filename).astype(np.float32)

        if mask_filename is not None:
            mask = nib.load(mask_filename).darrays[0].data

        g = nib.gifti.GiftiImage()
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar))
        nib.save(g, output_filename)
    return output_filename


def convert_fs_to_gii(input_filename, output_dir, clobber=False):
    """Convert FreeSurfer surface to GIFTI."""
    base = '_'.join( os.path.basename(input_filename).split('.')[0:-2])
    output_filename = f'{output_dir}/{base}.surf.gii'

    if not os.path.exists(output_filename) or clobber:
        try :
            ar = read_geometry(input_filename)
            print('Freesurfer')
        except ValueError:
            darrays = nib.load(input_filename).darrays
            ar = [ darrays[0].data, darrays[1].data ]
            print('Gifti')

        coordsys = nib.gifti.GiftiCoordSystem(dataspace='NIFTI_XFORM_TALAIRACH', xformspace='NIFTI_XFORM_TALAIRACH')
        g = nib.gifti.GiftiImage()
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar[0].astype(np.float32), intent='NIFTI_INTENT_POINTSET',coordsys=coordsys))
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar[1].astype(np.int32), intent='NIFTI_INTENT_TRIANGLE', coordsys=None))
        nib.save(g, output_filename)
    return output_filename


def fix_surf(surf_fn, output_dir ):
    """Fix surface by using surface-modify-sphere command."""
    base = os.path.basename(surf_fn).replace('.surf.gii','')
    out_fn = f"{output_dir}/{base}.surf.gii" 
    cmd = f"wb_command -surface-modify-sphere {surf_fn} 100 {out_fn}"

    subprocess.run(cmd, shell=True, executable="/bin/bash")    
    return out_fn

def get_fs_prefix(surf_filename):
    prefix=''
    target_prefix=os.path.basename(surf_filename)[0:3]
    return target_prefix
    
def resample_label(label_in, sphere_fn, sphere_rsl_fn, output_dir, clobber=False):
    n = nib.load(sphere_rsl_fn).darrays[0].data.shape[0]
    label_out = f'{output_dir}/n-{n}_{os.path.basename(label_in)}'

    if not os.path.exists(label_out) or clobber:
        cmd = f'wb_command -label-resample {label_in} {sphere_fn} {sphere_rsl_fn} BARYCENTRIC {label_out} -largest'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    assert os.path.exists(label_out), f"Could not find resampled label {label_out}" 

    return label_out

def resample_surface(surface_in, sphere_fn, sphere_rsl_fn, output_dir, n, clobber=False):
    surface_out = f'{output_dir}/n-{n}_{os.path.basename(surface_in)}'

    if not os.path.exists(surface_out) or clobber:
        cmd = f'wb_command -surface-resample {surface_in} {sphere_fn} {sphere_rsl_fn} BARYCENTRIC {surface_out}'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}" 

    return surface_out

def remesh_surface(surface_in,  output_dir, n=10000, radius=1, clobber=False):
    # run command line 
    base = os.path.basename(surface_in)
    surface_out=f'{output_dir}/n-{n}_{base}'
    if not os.path.exists(surface_out) or clobber:
        n_moving_vertices = utils.load_mesh_ext(surface_in)[0].shape[0]
        #cmd = f'mris_remesh --nvert {n} -i {surface_in} -o /tmp/{base} && mris_convert /tmp/{base} {surface_out}'
        #not sure about this->cmd = f'mris_remesh --nvert {n} -i {surface_in} -o {temp_surface_out} && wb_command  -surface-modify-sphere  {temp_surface_out} {radius} {surface_out} -recenter'
        cmd = f'wb_command  -surface-modify-sphere  {surface_in} {radius} {surface_out} -recenter'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")

    assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}"
    
    return surface_out

def get_fs_prefix(surf_filename):
    prefix=''
    target_prefix=os.path.basename(surf_filename)[0:3]
    return target_prefix

def write_gifti(array, filename, intent='NIFTI_INTENT_NORMAL'):
    gifti_img = nib.gifti.gifti.GiftiImage()
    gifti_array = nib.gifti.GiftiDataArray(array.astype(np.float32), intent=intent)
    gifti_img.add_gifti_data_array(gifti_array)
    print('Mean:', array.mean(), 'Std:', array.std())
    print('Writing to\n\t', filename)
    gifti_img.to_filename(filename)

def load_gifti(filename):
    return nib.load(filename).darrays[0].data

def surface_modify_sphere(surface_in, output_dir, radius=1, clobber:bool=False):

    surface_out = output_dir+'/'+os.path.basename(surface_in).replace(".surf.gii","_mod.surf.gii")

    if not os.path.exists(surface_out) or clobber :
        cmd=f'wb_command  -surface-modify-sphere  {surface_in} {radius} {surface_out} -recenter'

        subprocess.run(cmd, shell=True, executable="/bin/bash")

        assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}"

    return surface_out

def normalize_func_gii(fn):
    """read a .func.gii file and z-score it, then save as out_fn """
    ar = nib.load(fn).darrays[0].data
    print('MEAN', np.mean(ar), 'STD', np.std(ar));
    ar = (ar - np.mean(ar)) / np.std(ar)
    
    data = nib.gifti.GiftiDataArray(data=ar, intent='NIFTI_INTENT_SHAPE', datatype='NIFTI_TYPE_FLOAT32')
    img = nib.gifti.GiftiImage(darrays=[data])

    img.to_filename(fn)


def create_sphere(cortex_filename:str, output_dir:str, n:int=100, title:str='', clobber:bool=False):

    base = os.path.basename(cortex_filename).replace('.surf.gii','')

    inflated_filename = f'{output_dir}/{base}.inflated'
    sphere_filename = f'{output_dir}/{base}.sphere'
    sphere_gii_filename = f'{output_dir}/{base}.sphere.gii'
    sphere_png = f'{output_dir}/{base}.png'

    if not os.path.exists(inflated_filename) or clobber:
        cmd = f"mris_inflate -n {n} {cortex_filename} {inflated_filename}"
        print(cmd)

        subprocess.run(cmd, shell=True, executable="/bin/bash")
    
    assert os.path.exists(inflated_filename), f"Could not find inflated file {inflated_filename}"

    if not os.path.exists(sphere_filename) or clobber:
        cmd = f"mris_sphere -q {inflated_filename} {sphere_filename}"
        print(cmd)

        subprocess.run(cmd, shell=True, executable="/bin/bash")
    
    assert os.path.exists(sphere_filename), f"Could not find sphere file {sphere_filename}"

    if not os.path.exists(sphere_gii_filename) or clobber : 
        cmd = f"mris_convert {sphere_filename} {sphere_gii_filename} "
        print(cmd)

        subprocess.run(cmd, shell=True, executable="/bin/bash")    

    assert os.path.exists(sphere_gii_filename), f"Could not find sphere file {sphere_filename}"

    if not os.path.exists(sphere_png) or clobber:
        coords, faces = utils.load_mesh_ext(sphere_filename)
        v = np.prod(utils.load_mesh_ext(cortex_filename)[0], axis=1) 
        v = (v - np.mean(v)) / np.std(v)
        plot_surf(
            coords,
            faces,
            v, 
            rotate=[90, 270], 
            title=title,  
            filename=sphere_png,
            pvals=np.ones(coords.shape[0]),
            vmin=np.percentile(v, 2),
            vmax=np.percentile(v, 98),
            cmap='nipy_spectral',
            cmap_label='Sphere'
        )

    return inflated_filename, sphere_gii_filename

def resample_metrics(metrics_list:str, moving_sphere:str, fixed_sphere:str, output_dir:str, clobber:bool=False):
    """Resample metrics from moving to fixed sphere."""
    resampled_metrics_list = []
    for metric in metrics_list:
        resampled_metric = f'{output_dir}/{os.path.basename(metric)}'
        if not os.path.exists(resampled_metric) or clobber:
            cmd = f'wb_command -metric-resample {metric} {moving_sphere} {fixed_sphere} ADAP_BARY_AREA {resampled_metric}'
            print(cmd)
            subprocess.run(cmd, shell=True, executable="/bin/bash")
        assert os.path.exists(resampled_metric), f"Could not find resampled metric {resampled_metric}"
        resampled_metrics_list.append(resampled_metric)
    
    return resampled_metrics_list

