#!/usr/bin/env python3
"""Compare the scientifically relevant contents of phased VCF/BCF files."""

from __future__ import annotations

import gzip
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


VariantKey = tuple[str, int, str, str]


@dataclass(frozen=True)
class Variant:
    key: VariantKey
    genotypes: tuple[str, ...]


@dataclass(frozen=True)
class PhasedDataset:
    samples: tuple[str, ...]
    variants: tuple[Variant, ...]

    @property
    def by_key(self) -> dict[VariantKey, Variant]:
        return {variant.key: variant for variant in self.variants}


@dataclass(frozen=True)
class Comparison:
    exact_gt: bool
    samples_equal: bool
    variants_equal: bool
    shared_variants: int
    genotype_comparisons: int
    genotype_errors: int
    unphased_heterozygotes: int
    phase_transitions: int
    switch_errors: int
    missing_left_variants: int
    missing_right_variants: int

    @property
    def switch_rate(self) -> float:
        return self.switch_errors / self.phase_transitions if self.phase_transitions else 0.0

    @property
    def scientifically_equivalent(self) -> bool:
        return (
            self.samples_equal
            and self.variants_equal
            and self.genotype_errors == 0
            and self.unphased_heterozygotes == 0
            and self.switch_errors == 0
        )

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "exact_gt": self.exact_gt,
            "scientifically_equivalent": self.scientifically_equivalent,
            "samples_equal": self.samples_equal,
            "variants_equal": self.variants_equal,
            "shared_variants": self.shared_variants,
            "genotype_comparisons": self.genotype_comparisons,
            "genotype_errors": self.genotype_errors,
            "unphased_heterozygotes": self.unphased_heterozygotes,
            "phase_transitions": self.phase_transitions,
            "switch_errors": self.switch_errors,
            "switch_rate": self.switch_rate,
            "missing_left_variants": self.missing_left_variants,
            "missing_right_variants": self.missing_right_variants,
        }


class _ProcessText:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        if process.stdout is None:
            raise RuntimeError("htsfile did not provide stdout")
        self.stream = process.stdout

    def __enter__(self) -> TextIO:
        return self.stream

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stream.close()
        stderr = self.process.stderr.read() if self.process.stderr else ""
        status = self.process.wait()
        if status and exc_type is None:
            raise RuntimeError(f"htsfile failed with status {status}: {stderr.strip()}")


def _open_variant_text(path: Path) -> TextIO | _ProcessText:
    if path.name.endswith(".vcf.gz"):
        return gzip.open(path, "rt")
    if path.suffix == ".vcf":
        return path.open("rt")
    htsfile = shutil.which("htsfile")
    bcftools = shutil.which("bcftools")
    if htsfile:
        command = [htsfile, "--view", str(path)]
    elif bcftools:
        command = [bcftools, "view", "--no-version", str(path)]
    else:
        raise RuntimeError("reading BCF requires either 'htsfile' or 'bcftools'")
    return _ProcessText(
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    )


def load_dataset(path: str | Path) -> PhasedDataset:
    source = Path(path)
    samples: tuple[str, ...] | None = None
    variants: list[Variant] = []
    seen: set[VariantKey] = set()
    with _open_variant_text(source) as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("##"):
                continue
            fields = line.split("\t")
            if line.startswith("#CHROM"):
                samples = tuple(fields[9:])
                continue
            if line.startswith("#"):
                continue
            if samples is None:
                raise ValueError(f"{source}: records precede the #CHROM header")
            if len(fields) != 9 + len(samples):
                raise ValueError(f"{source}: malformed record at {fields[0]}:{fields[1]}")
            formats = fields[8].split(":")
            try:
                gt_index = formats.index("GT")
            except ValueError as error:
                raise ValueError(f"{source}: record lacks GT at {fields[0]}:{fields[1]}") from error
            genotypes = []
            for sample_field in fields[9:]:
                values = sample_field.split(":")
                genotypes.append(values[gt_index] if gt_index < len(values) else ".")
            key = (fields[0], int(fields[1]), fields[3], fields[4])
            if key in seen:
                raise ValueError(f"{source}: duplicate variant {key}")
            seen.add(key)
            variants.append(Variant(key, tuple(genotypes)))
    if samples is None:
        raise ValueError(f"{source}: no #CHROM header")
    return PhasedDataset(samples, tuple(variants))


def canonical_gt_digest(dataset: PhasedDataset, keys: Iterable[VariantKey] | None = None) -> str:
    selected = set(keys) if keys is not None else None
    digest = hashlib.sha256()
    digest.update("\t".join(dataset.samples).encode())
    digest.update(b"\n")
    for variant in dataset.variants:
        if selected is not None and variant.key not in selected:
            continue
        digest.update("\t".join(map(str, variant.key)).encode())
        digest.update(b"\t")
        digest.update("\t".join(variant.genotypes).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _alleles(genotype: str) -> tuple[tuple[str, ...], bool]:
    if "|" in genotype:
        return tuple(genotype.split("|")), True
    if "/" in genotype:
        return tuple(genotype.split("/")), False
    return (genotype,), True


def scientific_gt_digest(dataset: PhasedDataset) -> str:
    """Hash phased GTs after normalizing one global haplotype flip per sample."""
    variants = sorted(dataset.variants, key=lambda variant: variant.key)
    flip = [False] * len(dataset.samples)
    oriented = [False] * len(dataset.samples)
    for variant in variants:
        for sample_index, genotype in enumerate(variant.genotypes):
            if oriented[sample_index]:
                continue
            alleles, phased = _alleles(genotype)
            if phased and len(alleles) == 2 and "." not in alleles and alleles[0] != alleles[1]:
                flip[sample_index] = alleles > tuple(reversed(alleles))
                oriented[sample_index] = True

    digest = hashlib.sha256()
    digest.update("\t".join(dataset.samples).encode())
    digest.update(b"\n")
    for variant in variants:
        digest.update("\t".join(map(str, variant.key)).encode())
        for sample_index, genotype in enumerate(variant.genotypes):
            alleles, phased = _alleles(genotype)
            if flip[sample_index] and phased and len(alleles) == 2:
                genotype = "|".join(reversed(alleles))
            digest.update(b"\t")
            digest.update(genotype.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def compare_datasets(
    left: PhasedDataset,
    right: PhasedDataset,
    *,
    allow_right_superset: bool = False,
) -> Comparison:
    samples_equal = left.samples == right.samples
    left_by_key = left.by_key
    right_by_key = right.by_key
    left_keys = set(left_by_key)
    right_keys = set(right_by_key)
    shared_keys = left_keys & right_keys
    missing_left = right_keys - left_keys
    missing_right = left_keys - right_keys
    variants_equal = not missing_right and (allow_right_superset or not missing_left)

    exact_gt = (
        samples_equal
        and not missing_left
        and not missing_right
        and canonical_gt_digest(left) == canonical_gt_digest(right)
    )
    if not samples_equal:
        return Comparison(
            exact_gt,
            False,
            variants_equal,
            len(shared_keys),
            0,
            0,
            0,
            0,
            0,
            len(missing_left),
            len(missing_right),
        )

    genotype_comparisons = 0
    genotype_errors = 0
    unphased_heterozygotes = 0
    phase_transitions = 0
    switch_errors = 0
    previous_orientation: list[int | None] = [None] * len(left.samples)

    for variant in left.variants:
        other = right_by_key.get(variant.key)
        if other is None:
            continue
        for sample_index, (left_gt, right_gt) in enumerate(zip(variant.genotypes, other.genotypes)):
            genotype_comparisons += 1
            left_alleles, left_phased = _alleles(left_gt)
            right_alleles, right_phased = _alleles(right_gt)
            if sorted(left_alleles) != sorted(right_alleles):
                genotype_errors += 1
                previous_orientation[sample_index] = None
                continue

            informative = (
                len(left_alleles) == 2
                and len(right_alleles) == 2
                and "." not in left_alleles
                and "." not in right_alleles
                and left_alleles[0] != left_alleles[1]
            )
            if not informative:
                continue
            if not left_phased or not right_phased:
                unphased_heterozygotes += 1
                previous_orientation[sample_index] = None
                continue

            orientation = 0 if left_alleles == right_alleles else 1
            previous = previous_orientation[sample_index]
            if previous is not None:
                phase_transitions += 1
                switch_errors += orientation != previous
            previous_orientation[sample_index] = orientation

    return Comparison(
        exact_gt,
        samples_equal,
        variants_equal,
        len(shared_keys),
        genotype_comparisons,
        genotype_errors,
        unphased_heterozygotes,
        phase_transitions,
        switch_errors,
        len(missing_left),
        len(missing_right),
    )


def compare_paths(
    left: str | Path,
    right: str | Path,
    *,
    allow_right_superset: bool = False,
) -> Comparison:
    return compare_datasets(
        load_dataset(left),
        load_dataset(right),
        allow_right_superset=allow_right_superset,
    )
