# Releasing new Mesoscopy versions

## Checklist

* Make sure all relevant issues for this release are closed and any PRs merged to main.
* Ensure the Changelog is up to date.
* Make sure the CI isn't failing - build, test and docs should all be working properly.
* Test coverage should be >70% for the main project. Go write some tests if not.
* Double check there aren't any rude comments littering the code.
* Pull any changes to `main`.

## Releasing

* Create a new branch called `release/<version>` (e.g. `git branch release/v0.1.0`). Make sure you're doing this from the `main` branch.
* Checkout the new release branch.
* Bump the version using `hatch version <major|minor|micro>`.
* Commit the version change.
* Tag the version with git using `git tag <version number>` (e.g. `git tag v0.1.0`).
* Push everything with `git push && git push --tags`.
* Create a pull request for the release branch and merge into main.
* Github Actions should handle making a release and pushing to PyPI. If so happy days! If not, go fix it.
