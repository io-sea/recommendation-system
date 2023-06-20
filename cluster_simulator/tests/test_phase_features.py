import unittest
import time
import numpy as np
import simpy
from loguru import logger
import sys
from cluster_simulator.phase_features import Pattern, Operation, PhaseFeatures



class TestPhaseFeatures(unittest.TestCase):
    """Test phases that happens on tier of type Ephemeral when eviction policy is activated."""

    def test_default_phase_features(self):
        # Test with only default arguments
        # This test checks if the default values are correctly set when no arguments are provided to the PhaseFeatures class.
        pf = PhaseFeatures()
        assert pf.cores == 1
        assert pf.operation is None
        assert pf.read_volume == 0
        assert pf.write_volume == 0
        assert pf.read_io_pattern == Pattern.SEQ
        assert pf.write_io_pattern == Pattern.SEQ
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3

    def test_init_phase_features_read(self):
        # This test checks if the read operation is correctly set when operation="read" and a volume is provided.
        pf = PhaseFeatures(operation="read", volume=5e9)
        assert pf.cores == 1
        assert pf.operation == Operation.READ
        assert pf.read_volume == 5e9
        assert pf.write_volume == 0
        assert pf.read_io_pattern == Pattern.SEQ
        assert pf.write_io_pattern == Pattern.SEQ
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3

    def test_init_phase_features_write(self):
        # This test checks if the write operation is correctly set when operation="write" and a volume is provided.
        pf = PhaseFeatures(operation="write", volume=7e9)
        assert pf.cores == 1
        assert pf.operation == Operation.WRITE
        assert pf.read_volume == 0
        assert pf.write_volume == 7e9
        assert pf.read_io_pattern == Pattern.SEQ
        assert pf.write_io_pattern == Pattern.SEQ
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3

    def test_init_phase_features_read_with_pattern(self):
        # This test checks if the read pattern is correctly set when operation="read", a volume, and a pattern are provided.
        pf = PhaseFeatures(operation="read", volume=6e9, pattern="rand")
        assert pf.cores == 1
        assert pf.operation == Operation.READ
        assert pf.read_volume == 6e9
        assert pf.write_volume == 0
        assert pf.read_io_pattern == Pattern.RAND
        assert pf.write_io_pattern == Pattern.SEQ
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3
        
    def test_init_phase_features_read_with_float_pattern_seq(self):
        # This test checks if the read pattern is correctly set when operation="read", a volume, and a pattern are provided.
        pf = PhaseFeatures(operation="read", volume=6e9, pattern=1)
        assert pf.cores == 1
        assert pf.operation == Operation.READ
        assert pf.read_volume == 6e9
        assert pf.write_volume == 0
        assert pf.read_io_pattern == Pattern.SEQ
        assert pf.write_io_pattern == Pattern.SEQ
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3
        
    def test_init_phase_features_read_with_float_pattern_rnd(self):
        # This test checks if the read pattern is correctly set when operation="read", a volume, and a pattern are provided.
        pf = PhaseFeatures(operation="read", volume=6e9, pattern=0.3)
        assert pf.cores == 1
        assert pf.operation == Operation.READ
        assert pf.read_volume == 6e9
        assert pf.write_volume == 0
        assert pf.read_io_pattern == Pattern.RAND
        assert pf.write_io_pattern == Pattern.SEQ
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3
        
    def test_init_phase_features_write_with_float_pattern_seq(self):
        # This test checks if the read pattern is correctly set when operation="read", a volume, and a pattern are provided.
        pf = PhaseFeatures(operation="write", volume=6e9, pattern=1)
        assert pf.cores == 1
        assert pf.operation == Operation.WRITE
        assert pf.read_volume == 0
        assert pf.write_volume == 6e9
        assert pf.read_io_pattern == Pattern.SEQ
        assert pf.write_io_pattern == Pattern.SEQ
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3
        
    def test_init_phase_features_write_with_float_pattern_rnd(self):
        # This test checks if the read pattern is correctly set when operation="read", a volume, and a pattern are provided.
        pf = PhaseFeatures(operation="write", volume=6e9, pattern=0.3)
        assert pf.cores == 1
        assert pf.operation == Operation.WRITE
        assert pf.read_volume == 0
        assert pf.write_volume == 6e9
        assert pf.read_io_pattern == Pattern.SEQ
        assert pf.write_io_pattern == Pattern.RAND
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3
        
    def test_init_phase_features_write_with_pattern(self):
        # This test checks if the write pattern is correctly set when operation="write", a volume, and a pattern are provided.
        pf = PhaseFeatures(operation="write", volume=8e9, pattern="rand")
        assert pf.cores == 1
        assert pf.operation == Operation.WRITE
        assert pf.read_volume == 0
        assert pf.write_volume == 8e9
        assert pf.read_io_pattern == Pattern.SEQ
        assert pf.write_io_pattern == Pattern.RAND
        assert pf.read_io_size == 4e3
        assert pf.write_io_size == 4e3
        
    def test_init_phase_features_with_bandwidth(self):
        # This test checks if the bandwidth is correctly set when operation="read", a volume, and a bandwidth are provided.
        pf = PhaseFeatures(operation="read", volume=5e9, bw=100)
        assert pf.bw == 100e6        
        
    def test_get_attributes(self):
        phase_features = PhaseFeatures(cores=1, read_io_size=8e6, 
                                       write_io_size=8e6, 
                                       read_volume=169e6, 
                                       write_volume=330e6, 
                                       read_io_pattern='stride', 
                                       write_io_pattern='seq')
        attribute_dict = phase_features.get_attributes()
        expected_dict = {'cores': [1], 'read_io_size': [8e6], 'write_io_size': [8e6],
                         'read_volume': [169e6], 'write_volume': [330e6], 
                         'read_io_pattern': ['stride'], 'write_io_pattern': ['seq']}
        self.assertDictEqual(attribute_dict, expected_dict)       


  

if __name__ == '__main__':
    unittest.main(verbosity=2)