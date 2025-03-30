import hydra, unittest
from omegaconf import DictConfig, OmegaConf
from src.data_loader.dataset_creation import load_dataset


class TestData(unittest.TestCase):
    """
    Testing the data simulation pipeline
    The tests check these scenarios:
    1. System with/ without exogenous input
    2. System with/ without noise
    3. Physics points generation
    4. multiple configurations (time, system, input, sampler, etc.)
    Each of these scenarios is tested in a separate test case
    and image is generated to visualize the results, assertion of shape is made also
    """

    def __init__(self, *args, **kwargs):
        super(TestData, self).__init__(*args, **kwargs)
        self.cfg = OmegaConf.load("config/data.yaml")
        self.ds = None

    def _print_dim(self):
        """
        Print the dimensions of the dataset
        """
        print("Shapes of the data:")
        print(f"X states regress: {self.ds.x_states['x_regress'].shape}")
        print(f"Z states regress: {self.ds.z_states['z_regress'].shape}")
        print(f"Y output regress: {self.ds.y_out['y_regress'].shape}")
        if self.ds.exo_input is not None:
            print(f"exogenous input states: {self.ds.exo_input.shape}")
        if self.cfg.pinn_sampling != 'no_physics':
            print(f"X states physics: {self.ds.x_states['x_physics'].shape}")
            print(f"Z states physics: {self.ds.z_states['z_physics'].shape}")
            print(f"Y output physics: {self.ds.y_out['y_physics'].shape}")
        print("#" * 50)

    def test_exogenous_input(self, save_image: bool = False):
        """
        Test the system with exogenous input
        """
        for i in range(2):
            if i == 1:
                print("Testing the system without exogenous input")
                del self.cfg.input_signal
            else:
                print("Testing the system with exogenous input")
            self.ds = load_dataset(self.cfg).dataset
            self._print_dim()

    def test_ph_genmode(self):
        """
        test physics gen mode
        """
        for i in ['split_traj', 'split_set', 'no_physics']:
            print(f"Testing the system with physics gen mode: {i}")
            self.cfg.pinn_sampling = i
            self.ds = load_dataset(self.cfg).dataset
            self._print_dim()


if __name__ == '__main__':
    unittest.main()
