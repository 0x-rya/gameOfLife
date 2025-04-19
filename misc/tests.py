import unittest
import numpy as np
import networkx as nx
from delaySensitiveGOL import NodeAttr, EdgeAttr, delaySensitiveGOL

class TestNodeAttr(unittest.TestCase):
    def test_initialization(self):
        node = NodeAttr(max_delay=3, value=1)
        self.assertEqual(node.prev_values, [1, 1, 1])
    
    def test_add_new_value(self):
        node = NodeAttr(max_delay=3, value=0)
        node.add_new_value(1)
        self.assertEqual(node.prev_values, [0, 0, 1])
    
    def test_get_delayed_value(self):
        node = NodeAttr(max_delay=3, value=0)
        node.add_new_value(1)
        node.add_new_value(1)
        self.assertEqual(node.get_delayed_value(2), 1)

class TestEdgeAttr(unittest.TestCase):
    def test_initialization(self):
        edge = EdgeAttr(delay=2, node1=(0, 0), lkv1=1, node2=(0, 1), lkv2=0)
        self.assertEqual(edge.getLkv((0, 0)), 1)
        self.assertEqual(edge.getLkv((0, 1)), 0)
    
    def test_invalid_node(self):
        edge = EdgeAttr(delay=2, node1=(0, 0), lkv1=1, node2=(0, 1), lkv2=0)
        with self.assertRaises(Exception):
            edge.getLkv((1, 1))

class TestDelaySensitiveGOL(unittest.TestCase):
    def setUp(self):
        self.game = delaySensitiveGOL(sizeXY=(5, 5), timeStamps=10, initConfig=None, rule="B3/S2,3",
                                      seed=42, density=0.5, alpha=0.5, delay=2, tau=0.1,
                                      averageTimeStamp=5, extremes=(0, 1))
    
    def test_validate_rule(self):
        self.assertTrue(self.game.validateRule("B3/S2,3"))
        self.assertFalse(self.game.validateRule("InvalidRule"))
    
    def test_grid_initialization(self):
        self.assertEqual(len(self.game.automata), 5)
        self.assertEqual(len(self.game.automata[0]), 5)
    
    def test_get_neighbours(self):
        neighbours = self.game.getNeighbours(2, 2)
        self.assertEqual(len(neighbours), 8)
    
    def test_update_automata(self):
        prev_state = np.array(self.game.automata)
        self.game.updateAutomata()
        new_state = np.array(self.game.automata)
        self.assertFalse(np.array_equal(prev_state, new_state))

if __name__ == "__main__":
    unittest.main()

