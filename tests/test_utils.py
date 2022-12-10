import unittest

from detextify.text_detector import TextBox
import detextify.utils as utils


class UtilsTestCase(unittest.TestCase):
    def test_intersection_over_union(self):
        box1 = TextBox(x=100, y=200, w=50, h=10)
        self.assertEqual(utils.intersection_over_union(box1, box1), 1.0)

        # box2 is half of box1.
        box2 = TextBox(x=100, y=200, w=25, h=10)
        self.assertAlmostEqual(utils.intersection_over_union(box1, box2), 0.5, places=1)

    def test_merge_nerby_identical_boxes(self):
        """Tests that `merge_nearby_boxes` collapses identical boxes into one."""
        box1 = TextBox(x=100, y=200, w=50, h=10)
        result = utils.merge_nearby_boxes([box1, box1], max_distance=10)
        self.assertEqual(result, [box1])

    def test_merge_nerby_close_boxes(self):
        """Tests that `merge_nearby_boxes` merges boxes that are nearby."""
        box1 = TextBox(x=100, y=200, h=10, w=50)
        near_x = TextBox(x=95, y=200, h=10, w=50)
        near_y = TextBox(x=100, y=205, h=10, w=50)
        result = utils.merge_nearby_boxes([near_x, near_y, box1], max_distance=10)  # arbitrary order
        expected_merge = TextBox(x=95, y=200, h=15, w=55)
        self.assertEqual(result, [expected_merge])

    def test_merge_nearby_distant_boxes(self):
        """Tests that `merge_nearby_boxes` keeps distant boxes separate."""
        box1 = TextBox(x=100, y=200, h=10, w=50)
        box2 = TextBox(x=0, y=0, h=20, w=30)
        result = utils.merge_nearby_boxes([box1, box2], max_distance=10)
        self.assertEqual(result, [box1, box2])


if __name__ == '__main__':
    unittest.main()
