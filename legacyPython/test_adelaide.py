import unittest
import numpy as np
import subprocess
import os
import sys

# Ensure we can import adelaide_bridge
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adelaide_bridge import AdelaideBridge

class TestAdelaideCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = AdelaideBridge.get_instance()
        # Verify the bridge was started successfully
        if cls.bridge.process is None:
            raise unittest.SkipTest("Adelaide_Lite binary not built or not available.")

    def test_cosine_similarity_basic(self):
        # 1. Identical vectors
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        np_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        self.assertIsNotNone(ada_sim)
        self.assertAlmostEqual(ada_sim, np_sim, places=5)
        self.assertAlmostEqual(ada_sim, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        self.assertIsNotNone(ada_sim)
        self.assertAlmostEqual(ada_sim, 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        v1 = [1.0, -1.0, 0.5]
        v2 = [-1.0, 1.0, -0.5]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        self.assertIsNotNone(ada_sim)
        self.assertAlmostEqual(ada_sim, -1.0, places=5)

    def test_cosine_similarity_zero_vector(self):
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        self.assertIsNotNone(ada_sim)
        self.assertEqual(ada_sim, 0.0)

    def test_parity_generate_and_verify(self):
        # Test RAID-5 parity generation and verification via CLI directly
        binary_path = self.bridge.binary_path
        
        # We will write:
        # parity_generate
        # 3 4
        # aaaaaaaabbbbbbbbcccccccc
        #
        # Output should contain "PARITY:" followed by 8 hex characters (4 bytes)
        # representing XOR of aaaaaaaa, bbbbbbbb, cccccccc
        # XOR of aa, bb, cc:
        # aa = 10101010, bb = 10111011, cc = 11001100
        # aa XOR bb = 00010001 = 11
        # 11 XOR cc = 11011101 = dd
        # So parity block should be dddddddd
        
        p = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        
        # Read the initial ready line
        p.stdout.readline()
        
        p.stdin.write("parity_generate\n")
        p.stdin.write("3 4\n")
        p.stdin.write("aaaaaaaabbbbbbbbcccccccc\n")
        p.stdin.flush()
        
        line = p.stdout.readline().strip()
        self.assertTrue(line.startswith("PARITY:"))
        parity_hex = line.split(":")[1].strip()
        self.assertEqual(parity_hex, "dddddddd")
        
        p.stdin.close()
        p.wait()

if __name__ == "__main__":
    unittest.main()
