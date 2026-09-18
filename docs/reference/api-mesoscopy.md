# The mesoscopy core module

## Command loading

The root `cli` group lazy-loads commands using `LazyGroup`. Subcommands are listed in `LAZY_SUBCOMMANDS`, a table of
`name -> ("module:attribute", summary)`, and the module is imported only when that subcommand is invoked.

The summary is what `mesoscopy --help` prints next to each command; it is stored in the table rather than
read from the command's docstring so that rendering help does not import every subcommand. The subcommand's
own `--help` still comes from its docstring.

To add a subcommand, define it with `@click.command()` or `@click.group()` in its module and add an entry to
`LAZY_SUBCOMMANDS`. Do not register it with `cli.add_command()`: mkdocs-click only walks lazy subcommands
when none are registered eagerly, so an eager command would drop the others from the CLI reference.

::: mesoscopy
