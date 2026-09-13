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

## Pull Request Review Bot

The `PR Review` workflow reviews every pull request automatically, and it reviews the pull request again every time new commits are pushed to the branch.

- The review is posted as a single sticky comment that is updated in place, so the pull request timeline stays clean.
- The bot checks the pull request title and description, the required description sections, the checklist items and the added lines of the diff.
- When the review finds no problem, the bot approves the pull request with the `APPROVE` review state. This relies on the `Allow GitHub Actions to create and approve pull requests` repository setting, which is enabled on this repository.
- When a later push stops passing the checks, the bot dismisses its earlier approval, so the pull request has to be reviewed again.
- Add the `review-bypass` label to skip the review, and explain in the pull request description why the exception is needed.

## Development Workflow

- After forking or cloning, install the dependencies and make sure the existing tests pass.
- Make your change and add or update tests for it.
- Before committing, run lint / typecheck / test (see the README or the CI configuration under `.github/workflows` for the exact commands).
- Push your branch and open a pull request. CI and the `PR Review` bot run automatically.

## Commit Messages

Conventional Commits are recommended, for example feat / fix / refactor / docs / chore / ci. This makes it easier to generate change logs automatically.

## Release and Maintenance

Maintainers merge pull requests with a squash merge, and the source branch is deleted automatically afterwards. Releases are tagged as needed.
