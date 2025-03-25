# Getting Started

## Prerequisites


## Installing mesoscopy

The recommended way to install `mesoscopy` is by using `pipx` (https://pypa.github.io/pipx/). `pipx` will create an isolated python environment from which `mesoscopy` will run, leaving the system python alone. This is the recommended way to install `mesoscopy`, as it will not interfere with any other python packages you may have installed on your system.

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

    - For other distributions, install `pipx` with `pip`:

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

With Windows, you have two choices. You can either use `mesoscopy` with the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and follow the instructions above for installing `mesoscopy` on Linux (recommended), or you can run `mesoscopy` directly on Windows via PowerShell. The choice is of course yours, but you should bear in mind that `mesoscopy` primarily targets Unix-like OSs (Linux & MacOS) at the moment, so while everything _should_ work fine on Windows, your milage may vary.

To install `mesoscopy` directly on Windows, first install `pipx` using [Scoop](https://scoop.sh):

```powershell
scoop install pipx
pipx ensurepath
```

Then install `mesoscopy` via `pipx`:

```powershell
pipx install mesoscopy
```

## Next steps
