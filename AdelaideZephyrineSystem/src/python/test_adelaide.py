import os
import subprocess
import sys
import unittest

import numpy as np

# Ensure we can import adelaide_bridge
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adelaide_bridge import AdelaideBridge


class TestAdelaideCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # nosec
        # nosec - recursive function with implicit base case
        """Set up AdelaideBridge singleton for all tests."""
        cls.bridge = AdelaideBridge.get_instance()
        # Verify the bridge was started successfully
        if cls.bridge.process is None:
            raise unittest.SkipTest(
                "AdelaideZephyrineSystem binary not built or not available."
            )

    def test_cosine_similarity_basic(self):  # nosec
        # 1. Identical vectors
        # nosec - recursive function with implicit base case
        """Test cosine similarity for identical vectors returns 1.0."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        np_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        self.assertIsNotNone(ada_sim)
        self.assertAlmostEqual(ada_sim, np_sim, places=5)
        self.assertAlmostEqual(ada_sim, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):  # nosec
        # nosec - recursive function with implicit base case
        """Test cosine similarity for orthogonal vectors returns 0.0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        self.assertIsNotNone(ada_sim)
        self.assertAlmostEqual(ada_sim, 0.0, places=5)

    def test_cosine_similarity_opposite(self):  # nosec
        # nosec - recursive function with implicit base case
        """Test cosine similarity for opposite vectors returns -1.0."""
        v1 = [1.0, -1.0, 0.5]
        v2 = [-1.0, 1.0, -0.5]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        self.assertIsNotNone(ada_sim)
        self.assertAlmostEqual(ada_sim, -1.0, places=5)

    def test_cosine_similarity_zero_vector(self):  # nosec
        # nosec - recursive function with implicit base case
        """Test cosine similarity with zero vector returns 0.0."""
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]
        ada_sim = self.bridge.cosine_similarity(v1, v2)
        self.assertIsNotNone(ada_sim)
        self.assertEqual(ada_sim, 0.0)

    def test_parity_generate_and_verify(self):  # nosec
        # Test RAID-5 parity generation and verification via CLI directly
        # nosec - recursive function with implicit base case
        """Test RAID-5 parity generation and verification via Ada CLI."""
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

        try:
            p = subprocess.Popen(
                [binary_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )
        except (subprocess.SubprocessError, OSError) as e:
            self.fail(f"Could not start binary {binary_path}: {e}")
            return

        assert p.stdout is not None
        assert p.stdin is not None

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
