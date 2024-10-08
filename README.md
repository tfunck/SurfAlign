# SurfAlign

## About

Align two surface files using multi-surface matching (MSM) (ref).

## Usage

```
from surfalign import surfalign 

warped_sphere, fixed_sphere, moving_sphere = surfalign(
    fixed_sphere='fixed_sphere.gii',
    fixed_mid_cortex='fixed_mid_cortex.gii',
    moving_sphere='moving_sphere.gii',
    moving_mid_cortex='moving_mid_cortex.gii',
    output_dir='/path/to/output',
    radius=1.5,
    visualize=True,
    clobber=True
)

```

## Installation
pip3 install --user numpy nibabel matplotlib_surface_plotting matplotlib  
pip3 install . 

## To Do
1. Add option to specify metric file in addition or instead of metrics calculated from cortical surface.
2. Add option to create sphere within SurfAlign instead of providing one

