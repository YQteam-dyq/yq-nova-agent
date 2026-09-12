# Contributing

Thank you for taking the time to contribute to this project.

## Language Policy (Mandatory)

Everything added to this repository must be written in English. This is enforced by CI, it is not just a convention.

### Rules

1. Pull request titles and descriptions must be written in English.
2. Every added line of source code must be written in English. This includes code comments (line, block and doc comments), log messages, error messages and assertion messages.
3. Commit messages and branch names must be written in English.
4. Localized resources are exempt, because holding non-English text is their purpose. The exemption covers files under `i18n/`, `locale/`, `locales/`, `lang/` and `translations/`, and files with the extensions `.po`, `.pot`, `.ftl`, `.arb`, `.resx` and `.properties`.

### Enforcement

- The `English Only` workflow runs on every pull request and fails when it finds non-English characters in the pull request title, in the pull request description, or in any added line of the diff.
- `English Only` is registered as a required status check, so a pull request cannot be merged until it passes.
- Only added lines are inspected, so pre-existing text elsewhere in the repository never blocks a pull request.

### Escape hatch

- If a change genuinely needs non-English text outside the allowed paths, add the `english-check-bypass` label to the pull request. The workflow then reports success instead of failing, so the exception stays visible in the pull request timeline.
- Treat this as a last resort and explain in the pull request description why it is needed.

## Opening a Pull Request

Pull requests targeting the `main` branch are welcome. Branch protection is enabled on this repository: a pull request from a regular branch must pass CI before it can be merged, while repository administrators may bypass the protection and push directly.

## Development Workflow

- After forking or cloning, install the dependencies and make sure the existing tests pass.
- Make your change and add or update tests for it.
- Before committing, run lint / typecheck / test (see the README or the CI configuration under `.github/workflows` for the exact commands).
- Push your branch and open a pull request. CI and the Sourcery review bot run automatically.

## Commit Messages

Conventional Commits are recommended, for example feat / fix / refactor / docs / chore / ci. This makes it easier to generate change logs automatically.

## Release and Maintenance

Maintainers merge pull requests with a squash merge, and the source branch is deleted automatically afterwards. Releases are tagged as needed.
