import nibabel as nib
import os
import numpy as np
import subprocess
import shutil

from nibabel.freesurfer import read_morph_data, read_geometry

from surfalign import utils, plot


def msm_align(
        fixed_sphere, 
        fixed_data, 
        fixed_mask,
        moving_sphere, 
        moving_data, 
        moving_mask,
        output_dir, 
        levels=3,
        trans=None,
        verbose=True,
        clobber=False
        ):
    """Align two surfaces using MSM."""
    os.makedirs(output_dir,exist_ok=True)
    TFM_STR='transformed_and_reprojected.func.gii'

    data_out = f"{output_dir}/{'.'.join(os.path.basename(moving_data).split('.')[0:-2])}_warped_"
    
    data_out_rsl = f'{data_out}{TFM_STR}'

    out_sphere = f'{data_out}sphere.reg.surf.gii'

    if not os.path.exists(out_sphere) or not os.path.exists(data_out_rsl) or clobber :
        cmd = f"msm --inmesh={moving_sphere} --indata={moving_data} --inweight={moving_mask} "
        cmd += f"  --refmesh={fixed_sphere} --refdata={fixed_data} --refweight={fixed_mask} " 
        if trans is not None :
            cmd += f' --trans={trans}'

        cmd += f" --out={data_out} --levels={levels} --verbose=0"       

        if verbose :
            print()
            print('Inputs:')
            print(f'\tMesh {moving_sphere}')
            print(f'\tMask: {moving_mask}')
            print(f'\tData: {moving_data}')
            print('Reference:')
            print(f'\tMesh {fixed_sphere}')
            print(f'\tMask: {fixed_mask}')
            print(f'\tData: {fixed_data}')
            print(f'\tOptions:')
            print(f'\ttrans: {trans}')
            print('Out')
            print(f'\tOutput Data: {data_out_rsl}')
            print(f'\tOutput Sphere: {out_sphere}')
            print()
            print(cmd);
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        assert os.path.exists(data_out_rsl)
        print('\nwb_view', fixed_sphere, fixed_data, data_out_rsl, '\n' )

        plot.plot_receptor_surf(
            [fixed_data], fixed_sphere, output_dir, label='fx_orig', cmap='nipy_spectral', clobber=True
        )
        plot.plot_receptor_surf(
            [data_out_rsl], fixed_sphere, output_dir, label='mv_rsl', cmap='nipy_spectral', clobber=True
        )
    

    return out_sphere, data_out_rsl

def msm_resample_list(rsl_mesh, fixed_mesh, labels, output_dir, clobber=False):
    """Apply MSM to labels."""
    labels_rsl_list = []
    for i, label in enumerate(labels):
        if i % 50 == 0 : 
            print(f'\nCompleted: {(i/len(labels))*100:.2f}%\n')
        label_rsl_filename = msm_resample(rsl_mesh, fixed_mesh, label, output_dir=output_dir, clobber=clobber)
        labels_rsl_list.append(label_rsl_filename) #FIXME

    return labels_rsl_list

def msm_resample(rsl_mesh, fixed_mesh, label=None, output_dir:str='', write_darrays:bool=False, clobber=False):
    output_label_basename = os.path.basename(label).replace('.func','').replace('.gii','') + '_rsl'
    if output_dir == '':
        output_dir = os.path.dirname(label)
    output_label = f'{output_dir}/{output_label_basename}'
    output_label_ext = f'{output_label}.func.gii'
    template_label_rsl_filename = f'{output_label_basename}.func.gii'

    cmd = f"msmresample {rsl_mesh} {output_label} -project {fixed_mesh} -adap_bary"

    if not os.path.exists(output_label_ext) or clobber:
        if label is not None :
            cmd += f" -labels {label}"
        print(cmd);
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    else :
        pass
    
    return output_label_ext