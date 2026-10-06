![https://www.mesoscopy.org](assets/mesoscopy-logo-banner.png)

---

# Getting Started

Mesoscopy is an open-source package for the analysis of mesoscale calcium recordings. It preprocesses mesoscale
recording files to extract ∆F responses, registers recordings to the Allen Brain atlas, and analyses recordings
captured at rest or with behaviour.

## Prerequisites

Mesoscopy works best with data acquired in the [NWB](https://www.nwb.org) format. It also takes recordings
acquired as HDF5 files that follow mesoscopy's [raw data schema](design-principles/data-structure.md).

Dual-channel recordings with a haemodynamic response channel, such as 470nm and 405nm excitation for GCaMP, are
separated during preprocessing and yield a corrected ∆F signal.

An example acquisition script for FLIR Grasshopper cameras is
[here](https://gist.github.com/celefthe/d069e4e90397039b3aaf53292446fbd1).

## Installing mesoscopy

Install `mesoscopy` with [`pipx`](https://pypa.github.io/pipx/). `pipx` creates an isolated python environment for
`mesoscopy` to run from and leaves the system python alone, so it does not interfere with any other python
packages on your system.

### On Linux

1. Install `pipx`.
    - On Ubuntu / Debian:

        ```bash
        sudo apt update
        sudo apt install pipx
        pipx ensurepath
        sudo pipx ensurepath
        ```

    - On Fedora:

        ```bash
        sudo dnf install pipx
        pipx ensurepath
        sudo pipx ensurepath
        ```

    - On other distributions, install `pipx` with `pip`:

        ```bash
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
        sudo pipx ensurepath
        ```

2. Install `mesoscopy`.

    ```bash
    pipx install mesoscopy
    ```

### On MacOS

1. Install `pipx` with [homebrew](https://brew.sh).

    ```bash
    brew install pipx
    pipx ensurepath
    sudo pipx ensurepath
    ```

2. Install `mesoscopy`.

    ```bash
    pipx install mesoscopy
    ```

### On Windows

On Windows, use `mesoscopy` under the
[Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and follow the Linux
instructions above. `mesoscopy` targets Unix-like systems first, so this is the path we recommend.

To run `mesoscopy` directly on Windows instead, install `pipx` with [Scoop](https://scoop.sh) in PowerShell.

```powershell
scoop install pipx
pipx ensurepath
```

Then install `mesoscopy` with `pipx`.

```powershell
pipx install mesoscopy
```

## Next steps

The [typical workflow](typical-workflow.md) guide takes you through the processing steps for analysing mesoscale
recordings with `mesoscopy`.

The [how-to](how-to/index.md) guides go deeper into individual aspects of analysis, including
[quality control](how-to/qa.md). If you are about to start acquiring data, or want to convert existing data to a
format `mesoscopy` reads, start with the [data schema](design-principles/data-structure.md) and
[converting videos to HDF5](how-to/convert-video-to-h5.md) guides.
