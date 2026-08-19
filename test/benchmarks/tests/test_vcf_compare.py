import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcf_compare import (  # noqa: E402
    canonical_gt_digest,
    compare_paths,
    load_dataset,
    scientific_gt_digest,
    validate_allele_count_metadata,
)


HEADER = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb
"""


def records(*genotypes: tuple[str, str]) -> str:
    return "".join(
        f"1\t{index}\t.\tA\tC\t.\tPASS\t.\tGT\t{a}\t{b}\n"
        for index, (a, b) in enumerate(genotypes, 1)
    )


class VcfComparisonTest(unittest.TestCase):
    def compare(self, left_body: str, right_body: str, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.vcf"
            right = Path(directory) / "right.vcf"
            left.write_text(HEADER + left_body)
            right.write_text(HEADER + right_body)
            return compare_paths(left, right, **kwargs)

    def test_exact_genotypes(self):
        body = records(("0|1", "1|0"), ("1|0", "0|1"))
        result = self.compare(body, body)
        self.assertTrue(result.exact_gt)
        self.assertTrue(result.scientifically_equivalent)

    def test_global_haplotype_flip_is_equivalent(self):
        left = records(("0|1", "1|0"), ("1|0", "0|1"), ("0|1", "1|0"))
        right = records(("1|0", "0|1"), ("0|1", "1|0"), ("1|0", "0|1"))
        result = self.compare(left, right)
        self.assertFalse(result.exact_gt)
        self.assertTrue(result.scientifically_equivalent)
        self.assertEqual(result.switch_errors, 0)

        with tempfile.TemporaryDirectory() as directory:
            left_path = Path(directory) / "left.vcf"
            right_path = Path(directory) / "right.vcf"
            left_path.write_text(HEADER + left)
            right_path.write_text(HEADER + right)
            left_dataset = load_dataset(left_path)
            right_dataset = load_dataset(right_path)
            self.assertNotEqual(
                canonical_gt_digest(left_dataset), canonical_gt_digest(right_dataset)
            )
            self.assertEqual(
                scientific_gt_digest(left_dataset), scientific_gt_digest(right_dataset)
            )

    def test_local_phase_change_is_not_equivalent(self):
        left = records(("0|1", "0|0"), ("0|1", "0|0"), ("0|1", "0|0"))
        right = records(("0|1", "0|0"), ("1|0", "0|0"), ("1|0", "0|0"))
        result = self.compare(left, right)
        self.assertFalse(result.scientifically_equivalent)
        self.assertEqual(result.switch_errors, 1)

        with tempfile.TemporaryDirectory() as directory:
            left_path = Path(directory) / "left.vcf"
            right_path = Path(directory) / "right.vcf"
            left_path.write_text(HEADER + left)
            right_path.write_text(HEADER + right)
            self.assertNotEqual(
                scientific_gt_digest(load_dataset(left_path)),
                scientific_gt_digest(load_dataset(right_path)),
            )

    def test_genotype_change_is_not_equivalent(self):
        result = self.compare(records(("0|1", "0|0")), records(("1|1", "0|0")))
        self.assertEqual(result.genotype_errors, 1)
        self.assertFalse(result.scientifically_equivalent)

    def test_right_superset_for_scaffold_check(self):
        left = records(("0|1", "1|0"))
        right = records(("0|1", "1|0"), ("1|1", "0|0"))
        result = self.compare(left, right, allow_right_superset=True)
        self.assertTrue(result.scientifically_equivalent)

    def test_allele_count_metadata_matches_called_genotypes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.vcf"
            path.write_text(
                HEADER
                + "1\t1\t.\tA\tC\t.\tPASS\tAC=2;AN=3\tGT\t0|1\t1|.\n"
            )
            self.assertEqual(validate_allele_count_metadata(path), 1)

    def test_allele_count_metadata_rejects_reference_inflated_an(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.vcf"
            path.write_text(
                HEADER
                + "1\t1\t.\tA\tC\t.\tPASS\tAC=2;AN=7\tGT\t0|1\t1|.\n"
            )
            with self.assertRaisesRegex(ValueError, "INFO allele counts disagree"):
                validate_allele_count_metadata(path)

    def test_allele_count_metadata_rejects_wrong_ac(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.vcf"
            path.write_text(
                HEADER
                + "1\t1\t.\tA\tC\t.\tPASS\tAC=1;AN=3\tGT\t0|1\t1|.\n"
            )
            with self.assertRaisesRegex(ValueError, "INFO allele counts disagree"):
                validate_allele_count_metadata(path)


if __name__ == "__main__":
    unittest.main()
