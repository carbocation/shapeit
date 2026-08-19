---
layout: default
title: Static binaries
nav_order: 2
parent: Installation
has_children: false
permalink: /docs/installation/static_binaries
---

# Static binaries

A complete installation from source is often unnecessary because we provide
static binaries of the latest release. We recommend downloading binaries from
the official release page:

[https://github.com/odelaneau/shapeit/releases](https://github.com/odelaneau/shapeit/releases)

The published static executables support 64-bit Linux on x86_64 CPUs and
require AVX2 and FMA. They are not native macOS executables.

Apple Silicon users should follow the
[source-build instructions](https://odelaneau.github.io/shapeit/docs/installation/build_from_source/compile_shapeit5)
to produce native arm64 executables. macOS does not support the Linux-style
fully static target: the native build links to Homebrew HTSlib, Boost, and their
dependencies. Alternatively, use the
[Docker package provided with SHAPEIT5](https://odelaneau.github.io/shapeit/docs/installation/docker).

The files are released under the MIT license.
