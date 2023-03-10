import unittest
from tests.tests_methods import test_test_data, all_test_data


class Test1DSSA(unittest.TestCase):

    def test_reconstruct(self):
        what = "reconstruct"
        names = ["co2", "fr1k.nz", "fr1k", "fr50", "fr50.nz"]
        for name in names:
            try:
                test_test_data(what, all_test_data[name])
            except:
                self.fail("Wrong result")

    def test_rforecast(self):
        what = "rforecast"
        names = ["co2", "fr1k.nz", "fr1k", "fr50", "fr50.nz"]
        for name in names:
            try:
                test_test_data(what, all_test_data[name])
            except:
                self.fail("Wrong result")

    def test_vforecast(self):
        what = "vforecast"
        names = ["co2", "fr1k.nz", "fr1k", "fr50", "fr50.nz"]
        for name in names:
            try:
                test_test_data(what, all_test_data[name])
            except:
                self.fail("Wrong result")


if __name__ == '__main__':
    unittest.main()
