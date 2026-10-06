# The mesoscopy core module

## Command loading

The root `cli` group lazy-loads commands through `LazyGroup`. Subcommands are listed in `LAZY_SUBCOMMANDS`, a table
of `name -> ("module:attribute", summary)`, and a module is imported only when its subcommand runs.

The summary is what `mesoscopy --help` prints next to each command. It lives in the table rather than in the
command's docstring so that rendering help does not import every subcommand. The subcommand's own `--help` still
comes from its docstring.

To add a subcommand, define it with `@click.command()` or `@click.group()` in its module and add an entry to
`LAZY_SUBCOMMANDS`. Do not register it with `cli.add_command()`. mkdocs-click only walks lazy subcommands when
none are registered eagerly, so an eager command would drop the others from the CLI reference.

## Import layers

The same idea applies to imports. Each stage's command module (`mesoscopy/<stage>/__init__.py`) and `mesoscopy.io`
import only cheap modules at the top, such as `click`, `numpy`, `h5py`, `mesoscopy.io` and `mesoscopy.timer`.
Expensive libraries (pynwb, dask, zarr, pandas, scipy, scikit-learn, pims, napari, plotly, matplotlib) and the
numerics modules that use them are imported inside the function that needs them.

The numerics modules themselves (`preprocess.compute`, `preprocess.qa`, `process.region`, `process.regression`,
`register.transform`, `register.landmarks_gui`, `convert.video`, and so on) keep conventional top-level imports.
`banned-module-level-imports` in `pyproject.toml` lists the expensive libraries, and `per-file-ignores` excludes
the numerics modules from that rule. Names needed only in annotations go in an `if typing.TYPE_CHECKING:` block at
the top of the module.

::: mesoscopy
