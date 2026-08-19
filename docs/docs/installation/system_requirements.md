---
layout: default
title: System requirements
nav_order: 1
parent: Build from source
grand_parent: Installation
permalink: /docs/installation/build_from_source/system_requirements
---
# System requirements
{: .no_toc .text-center }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## System requirements
SHAPEIT5 is primarily a set of C++ tools covering the process of haplotype
phasing in large datasets. The source tree supports the following platforms:

| Platform | Compiler | CPU requirements |
|---|---|---|
| Linux x86_64 | A C++20 compiler such as GCC | AVX2 and FMA |
| macOS on Apple Silicon | Apple Clang with C++20 support | Native arm64; Rosetta is not required |

Both platforms also require the Rust toolchain (`cargo` and `rustc` 1.85 or
newer) for the versioned phasing kernels. We recommend current stable compiler
releases. The native macOS build is tested on the arm64 GitHub runner for macOS
15; these instructions do not describe an Intel macOS build.

For example running the following instruction on Ubuntu 20.04 focal:

<div class="code-example" markdown="1">
```bash
sudo apt install build-essential
```
</div>

On macOS, install Apple's command-line developer tools if they are not already
present:

<div class="code-example" markdown="1">
```bash
xcode-select --install
```
</div>


Install Rust 1.85 or newer using
[rustup](https://rustup.rs/) or a sufficiently recent system package. To check
the compiler versions, run:

<div class="code-example" markdown="1">
```bash
g++ --version
cargo --version
```
</div>

Use `clang++ --version` instead of `g++ --version` on macOS.
