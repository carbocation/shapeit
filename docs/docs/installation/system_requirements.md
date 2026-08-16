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
phasing in large datasets. Building from source requires a modern Linux
operating system, a C++20 compiler, and the Rust toolchain (`cargo` and
`rustc` 1.85 or newer) for the versioned phasing kernels. We recommend
using current stable compiler releases.

For example running the following instruction on Ubuntu 20.04 focal:

<div class="code-example" markdown="1">
```bash
sudo apt install build-essential
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
