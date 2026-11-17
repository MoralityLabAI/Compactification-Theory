import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from research.shaft_compactification.comp24_audit import validate,PATH

class COMP24ReceiptTests(unittest.TestCase):
    def test_exact_receipt(self):
        data=validate()
        self.assertGreater(data["learned_bit_accuracy"],.98)
        self.assertLess(data["zero_bit_accuracy"],.83)
        self.assertGreater(data["primary_difference"],.15)
    def test_corrupt_receipt_rejected(self):
        original=json.loads(PATH.read_text())
        original["metrics_by_depth"]["40"]["learned"]["bits_correct"]+=1
        with TemporaryDirectory() as folder:
            path=Path(folder)/"corrupted.json"
            path.write_text(json.dumps(original))
            with self.assertRaises(AssertionError):validate(path)
if __name__=="__main__":unittest.main()
