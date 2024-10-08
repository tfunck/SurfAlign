import unittest
import os
import shutil
from surfalign.surfalign import surfalign

class TestSurfAlign(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'data/'
        self.output_dir = 'output_test'
        os.makedirs(self.output_dir, exist_ok=True)

        self.fixed_sphere = os.path.join(self.data_dir, 'yerkes19/MacaqueYerkes19.L.sphere.32k_fs_LR.surf.gii')
        self.fixed_mid_cortex = os.path.join(self.data_dir, 'yerkes19/MacaqueYerkes19.L.mid.32k_fs_LR.surf.gii')
        self.moving_sphere = os.path.join(self.data_dir, 'd99/D99_L_AVG_T1_v2.L.MID.SPHERE.SIX.167625.surf.gii')
        self.moving_mid_cortex = os.path.join(self.data_dir, 'd99/D99_L_AVG_T1_v2.L.MID.167625.surf.gii')
        #self.fixed_mask = os.path.join(self.data_dir, 'fixed_mask.nii')
        #self.moving_mask = os.path.join(self.data_dir, 'moving_mask.nii')

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_surfalign_basic(self):
        warped_sphere, fixed_sphere, moving_sphere = surfalign(
            fixed_sphere=self.fixed_sphere,
            fixed_mid_cortex=self.fixed_mid_cortex,
            fixed_mask=self.fixed_mask,
            moving_sphere=self.moving_sphere,
            moving_mid_cortex=self.moving_mid_cortex,
            moving_mask=self.moving_mask,
            output_dir=self.output_dir,
            radius=1.0,
            visualize=False,
            clobber=True
        )
        self.assertTrue(os.path.exists(warped_sphere))
        self.assertTrue(os.path.exists(fixed_sphere))
        self.assertTrue(os.path.exists(moving_sphere))

    def test_surfalign_with_visualize(self):
        warped_sphere, fixed_sphere, moving_sphere = surfalign(
            fixed_sphere=self.fixed_sphere,
            fixed_mid_cortex=self.fixed_mid_cortex,
            fixed_mask=self.fixed_mask,
            moving_sphere=self.moving_sphere,
            moving_mid_cortex=self.moving_mid_cortex,
            moving_mask=self.moving_mask,
            output_dir=self.output_dir,
            radius=1.0,
            visualize=True,
            clobber=True
        )
        self.assertTrue(os.path.exists(warped_sphere))
        self.assertTrue(os.path.exists(fixed_sphere))
        self.assertTrue(os.path.exists(moving_sphere))

    def test_surfalign_with_different_radius(self):
        warped_sphere, fixed_sphere, moving_sphere = surfalign(
            fixed_sphere = self.fixed_sphere,
            fixed_mid_cortex = self.fixed_mid_cortex,
            moving_sphere = self.moving_sphere,
            moving_mid_cortex = self.moving_mid_cortex,
            output_dir = self.output_dir,
            radius=2.0,
            visualize=False,
            clobber=True
        )
        self.assertTrue(os.path.exists(warped_sphere))
        self.assertTrue(os.path.exists(fixed_sphere))
        self.assertTrue(os.path.exists(moving_sphere))

if __name__ == '__main__':
    unittest.main()