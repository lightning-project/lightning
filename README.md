# Lightning: Fast data processing using GPUs on distributed platforms
[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/lightning-project/lightning/)
[![License](https://img.shields.io/github/license/lightning-project/lightning)](https://github.com/lightning-project/lightning/blob/main/LICENSE)
[![Github](https://img.shields.io/github/workflow/status/lightning-project/lightning/Rust)](https://github.com/lightning-project/lightning)

Lightning is a framework for data processing using GPUs on distributed platforms.
The framework allows distributed multi-GPU execution of compute kernels functions written in CUDA in a way that is similar to programming a single GPU, without worrying about low-level details such as network communication, memory management, and data transfers.
This enables scaling of existing GPU kernels to much larger problem sizes, for beyond the memory capacity of a single GPU.
Lightning efficiently distributes the work/data across GPUS and maximizes efficiency by overlapping scheduling, data movement, and work when possible.

## Installation Guide
The project is written in [Rust](https://www.rust-lang.org/tools/install) and has been tested with Rust 1.56.
To build the project use `cargo`, which is included with the Rust installion.

```bash
cargo build --release
```

## License
Apache 2.0. See [LICENSE](https://github.com/lightning-project/lightning/blob/main/LICENSE).


## Bibliography
S. Heldens,
P. Hijma,
B. van Werkhoven,
J. Maassen,
R.V. van Nieuwpoort,
"Lightning: Scaling the GPU Programming Model Beyond a Single GPU",
in IEEE IPDPS,
2022
