# review_page

A local web page for answering seed cleaning's review cases. Seed cleaning
(`libs/seed_cleaning`, run by `scripts/clean_seed.py`) writes every decision its
rules cannot settle to `data/seed_cleaning/review.jsonl`, one long JSON line per
case. The page:

- lists those cases and shows each case's evidence;
- offers only the choices the placement rules allow;
- saves the answers safely;
- starts a run that applies the answers;
- shows what the run corrected.

The page runs on the reviewer's machine and is reached from that machine
alone, because the page reads and writes local files and starts local runs.

## Contents

- [Terms](#terms)
- [Running it](#running-it)
  - [Settings](#settings)
- [A review, step by step](#a-review-step-by-step)
  - [A new seed release](#a-new-seed-release)
- [The page](#the-page)
  - [Cases tab](#cases-tab)
  - [Corrections tab](#corrections-tab)
  - [Run](#run)
  - [Layout and settings](#layout-and-settings)
- [Saving safely](#saving-safely)
- [Files it reads and writes](#files-it-reads-and-writes)
- [How it is built](#how-it-is-built)
  - [The case source](#the-case-source)
  - [Server and browser](#server-and-browser)
  - [Discovery and production](#discovery-and-production)
- [Tests](#tests)

## Terms

The matching terms (work, provider page, page key, reference page, doubled
provider, split work, linked page, partial source) are defined in the
[`work_matching` README](../../libs/work_matching/README.md#terms), and the
seed cleaning terms (seed correction, review file, provider page store) in the
[`seed_cleaning` README](../../libs/seed_cleaning/README.md#terms). The page
adds:

| Term                | Meaning                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Original answer** | A case's baseline answer recorded by the last pipeline run. "Changed" and "Reset" compare against it.                                |
| **Unsaved edit**    | An unpersisted answer choice staged in the browser until saved or discarded.                                                         |
| **Saved answer**    | A placement answer persisted to the review file on disk. The next Run applies it.                                                    |
| **Suggestion**      | The placement suggested by matching rules. An unanswered case opens on it; Accept saves it.                                          |
| **Case source**     | Interface for reading cases and corrections, saving answers, and tracking runs (`ReviewCaseSource`). The initial source reads files. |

## Running it

With Docker, as part of the dev stack:

```bash
docker compose -f docker/docker-compose.dev.yml up -d review-page
# open http://127.0.0.1:8003
```

The container:

- mounts `data/seed_cleaning` read-write, where saves and runs write;
- mounts `assets/seed_data` read-only, where a run reads the seed file;
- is published on `127.0.0.1` only;
- runs as a non-root user with the host user's uid;
- restarts unless stopped;
- has a health check that calls `GET /health`.

Both folders must exist before Compose starts. `data/seed_cleaning` appears
after the first `scripts/clean_seed.py` run; on a fresh clone, create it with
`mkdir -p data/seed_cleaning`. Compose refuses to start rather than create a
missing folder, because Docker would create it owned by root, where the
container's user cannot write.

Without Docker, from the repository root, so the default data folder and
seed-cleaning script are found:

```bash
./pants run apps/review_page:review_page
# or, without Pants:
PYTHONPATH="apps/review_page/src:$(printf '%s:' libs/*/src)" uv run python -m review_page.main
```

The image (`apps/review_page/Dockerfile.dev`, `Dockerfile.prd`) holds the page,
`scripts/clean_seed.py` and the libraries a run imports. A run from the page
fetches nothing, so the image holds no browser. A run loads no embedding
model either, so the image leaves out the model stack (torch and the packages
built on it).

### Settings

`ReviewPageConfig` in `libs/common/src/common/config/review_page_config.py`,
read like every other sub-config of `Settings`. The settings are trusted: Run
starts the configured script with the server's own Python, so only the person
running the page sets them.

| Environment variable            | Default                 | Meaning                                                                                                |
| ------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `REVIEW_PAGE_HOST`              | `127.0.0.1`             | Address the page listens on. The container sets `0.0.0.0` and publishes the port on `127.0.0.1` alone. |
| `REVIEW_PAGE_PORT`              | `8003`                  | Port the page listens on.                                                                              |
| `REVIEW_PAGE_DATA_DIR`          | `data/seed_cleaning`    | Seed cleaning's data folder: the review file, the page store and the run's results.                    |
| `REVIEW_PAGE_SEED_FILE`         | unset                   | Seed file a Run uses. Unset, Run uses the seed file the last run recorded in `review_run.json`.        |
| `REVIEW_PAGE_CLEAN_SEED_SCRIPT` | `scripts/clean_seed.py` | The script Run starts.                                                                                 |

A run records its seed file as the path it was given. The container finds the
seed folder at `/app/assets/seed_data`, so start the terminal run with the path
relative to the repository (`assets/seed_data/…`), not a full path.

## A review, step by step

1. Run seed cleaning from the terminal. Seed cleaning fetches provider pages and
   writes the review cases, corrections, and run result to
   `data/seed_cleaning/`.
2. Open the review page. The Cases tab lists every review case. Filter by case
   kind, case state, or edit state, and search by title or page key.
3. Open a case and select a work for each provider page, and the partial
   sources of each work, or select a decision for a linked page. Choices are validated against placement rules as
   they are selected, and disallowed options are disabled with an explanation.
4. Click Save (or Save & next, or Save all for all staged edits). Saves that
   violate a placement rule are rejected with the rule message, keeping the
   staged edit in the browser for correction.
5. Click Run. Run invokes seed cleaning without network fetching (`--no-fetch`)
   and applies all saved answers using `REVIEW_PAGE_SEED_FILE` when configured,
   or the seed file recorded by the previous run.
6. Inspect the run summary banner and the Corrections tab to verify the
   resulting entry changes.

### A new seed release

1. Place the new seed file in `assets/seed_data/`.
2. Run seed cleaning from the terminal with the new seed file path:

   ```bash
   uv run python scripts/clean_seed.py assets/seed_data/anime-offline-database-minified.json
   ```

3. Reload the review page. Because the server reads cases and run results on
   every request, no import or rebuild step is needed, and subsequent Runs
   automatically use the new seed file.

The review page's Run button never fetches over the network; the initial run
for a new seed release is always executed from the terminal.

Saved answers carry over to the new seed release by case name:

- A case whose candidate pages are unchanged retains its saved answer, which the
  run applies automatically.
- A case whose candidate pages changed in the new release receives a new case
  name, appears as an open case, and is badged `New` and `Previous answer`. It
  carries the answer of the case it replaces. While the case still holds the
  entries that answer saw, the run keeps applying it to the pages it placed,
  and the case opens on it with the added pages marked `Added`. When the
  release merged those entries or changed the case's kind, the previous answer
  is shown but no longer applies.
- An answer whose case no longer comes up and that no case carries remains in
  the review file without being applied, is marked `Unused answer`, and is
  listed in the run summary under "Unused answers".

Saved answers are kept in the review file because every pipeline run starts
from the original seed file and reapplies all saved answers. A previous answer
is dropped once the case carrying it has a saved answer of its own.

## The page

### Cases tab

- **List.** Displays one row per review case with its title, kind, edit state,
  and status pills (`New`, `Suggested`, `Previous answer`, `No longer comes
up`). Selecting
  multiple chips within the same filter category widens the match (logical OR),
  while combining different filter categories narrows the match (logical AND).
  Paginated at 25 rows per page.
- **Case panel.** Displays the review reason sentences, work entry headers, and
  candidate provider page table. Each page row displays extracted metadata
  (titles, type, episode count, air dates, and external provider links) along
  with a Work dropdown. Below the pages, the Works table lists each work in use,
  and the seed entries of the same series outside the case, with their partial
  sources. Enabling "Show only differences" hides metadata columns and
  comparison rows whose values are identical across all pages.
- **Choices.**
  - A work already holding a page of the same provider is greyed out. While
    the page has no place, a note beneath the dropdown names the page holding
    each such work.
  - `dropped` drops a duplicate page, and is offered only while another page
    of the provider has a work.
  - Selecting `other work` displays a searchable picker for the target seed
    entry. It offers candidate entries from the case that hold no page from
    that provider, and also accepts a pasted page key or provider address
    (normalized to its page key). A typed key shows the seed entry holding it,
    by its title and reference page, unless the picker already shows that
    entry as a suggestion. Choosing "New entry" assigns the page to a new
    entry.
  - In the Works table, each work's partial sources show as chips, removed
    with their ×, and "Add partial source" opens a searchable picker of the
    pages that are a source of another work. Each option shows the page key,
    then the page's title, the work it is in, and its type, episodes and
    start date, or that the page was not fetched. Saving validates the named entry against the seed file, which
    is loaded on first use and cached while its modification time and size
    remain unchanged.
  - "What the choices mean" beneath the table explains each placement value.
    Hovering any dropdown, filter, badge, or button displays its tooltip.
- **States.** Each case displays one edit state: `rejected` (the last save
  violated a placement rule), `edited, not saved`, `saved, applies on Run`, or
  `unchanged`. Rows whose placement differs from the case's baseline answer are
  highlighted. An unanswered case opens on its suggested answer, so an
  unmodified suggestion is not treated as an unsaved edit.
- **Buttons.**
  - Save and Save & next: enabled when the chosen answer differs from the saved answer.
  - Accept: enabled when a complete suggested answer differs from the saved answer.
  - Reset: enabled when the chosen or saved answer differs from the original answer.
  - Every button is disabled while a pipeline run is in progress.
- **Previous answer.** The case replaces one answered before a seed release
  changed its pages. The panel says whether that answer still applies, lists
  the added pages, and marks them `Added` in the pages table; an answer that no
  longer applies is shown above the table. The `Previous answer` filter lists
  these cases.
- **Unused answer.** The review file retains the case's saved answer, but
  the last pipeline run produced no case that uses it, because a seed release
  or another answer already settled those pages. The answer had no effect. No
  action is needed; if a new case holds the same pages, answer that case.

### Corrections tab

Displays entry modifications from the last pipeline run, one corrected entry at a time:

- **List.** Lists each corrected entry with the number of modified pages and a
  "Your answer" badge when a reviewer's answer placed at least one page. Filter
  by change category (pages moved, old addresses dropped, partial sources added,
  duplicate pages dropped, unchanged) and by decision source (reviewer answer or matching rules).
- **Entry panel.** Shows a table of the entry's provider pages: the change
  category, destination (target seed entry, linked page, new entry, current
  address, or unchanged), the page's place with its partial sources, rule
  explanation, and source.
- **Cross-links.** A corrected entry links back to the review cases containing
  it, and a review case links to each of its entries the last run changed,
  counting only the corrections that change the seed.

Corrections reflect the last completed pipeline run; answers saved afterward
take effect on the next Run.

### Run

Run is enabled while saved answers differ from their original answers and no
edit is left unsaved, ensuring a run never misses an in-memory edit. While a run is
active, the toolbar indicates "Run going" and action buttons are disabled. When the run ends,
the summary, the lists, the original answers, and the Corrections tab refresh automatically.

While its tab is visible, the page asks `/run-state` every few seconds, and once
more as soon as a hidden tab is shown again, so a pipeline run started from the
terminal or another browser tab is tracked identically. A hidden tab does not
ask. A failed run
reports its exit code. Subprocess output is written to
`data/seed_cleaning/logs/review_page_runs.log`, which rotates to
`review_page_runs.log.1` once it exceeds 1 MB before starting a new run.

An active run continues if the web server restarts. The restarted server
detects the running process via the operating-system lock on `review.jsonl.lock`,
though its exit code will only be recorded in the run log.

### Layout and settings

- The case list sidebar can be toggled with the button beside the search box
  and resized by dragging the divider or pressing the arrow keys;
  double-clicking restores the default width. The minimum sidebar width keeps
  row badges visible.
- Theme: System (follows the operating system's light or dark setting), Light,
  or Dark.
- Browser `localStorage` preserves session state across reloads, server
  restarts, and pipeline runs: staged unsaved edits, rejected saves, active
  filters, search text, pagination indices, open case and entry selections,
  active tab, color theme, and sidebar width and visibility.

## Saving safely

- **Rule validation first.** The case source validates every candidate answer
  against the `work_matching` placement rules before persisting it, preventing
  invalid answers from being written to disk.
- **Targeted line updates.** Saving rewrites only the `answer` field on
  modified case lines in `review.jsonl`; all other lines remain unchanged byte
  for byte, and the file is replaced atomically.
- **Exclusive file locking.** Saves and pipeline runs acquire a non-blocking
  operating-system lock on `review.jsonl.lock`. Save attempts while a run holds
  the lock are refused with a notice and leave disk files untouched. The lock
  coordinates access across container and host processes.
- **Removed cases.** If a pipeline run removed a case after the browser loaded
  it, saving that case is rejected with an explanatory message while `Save all`
  still persists remaining valid cases. If a concurrent modification removes a
  target case between validation and file rewrite, the write is aborted and a
  notice prompts a reload.
- **Cross-site request protection.** External websites open in the same browser
  cannot submit answers or trigger runs (`request_guards.py`):

  - requests with a `Host` header other than `127.0.0.1`, `localhost`, or `[::1]`
    are rejected with HTTP 400 to prevent DNS rebinding;
  - `POST` requests without the `HX-Request: true` header sent by htmx are
    rejected with HTTP 403, as cross-origin sites cannot attach custom headers
    without a CORS preflight (which the server denies);
  - `POST` requests carrying an `Origin` header from a non-local host are
    rejected with HTTP 403 as defense in depth.

  Health probes or local requests using another hostname must send
  `Host: localhost` or register the hostname in `LOCAL_HOST_NAMES`.

- **Optimistic concurrency conflict detection.** Each unsaved edit records the
  base saved answer (`seen_saved_answer`) present when the edit began. If another
  session or tab modified the saved answer concurrently, the save is rejected under
  the review file lock to prevent accidental overwrites. Submitting again without
  the base revision explicitly forces the update.
- **Multi-tab synchronization.** Open tabs in the same browser synchronize
  staged edits via the `storage` event. Edits made in one tab appear in other
  tabs, and saving in one tab clears the staged edit across all tabs. Active
  filters and open case selections remain independent per tab. Separate browser
  profiles maintain separate state protected by optimistic concurrency checks.
- **Linked-page conflicts.** When two saved answers assign whole pages from the
  same provider to the same work entry, the save succeeds with a warning notice
  because the pipeline skips conflicting additions.

## Files it reads and writes

All in `data/seed_cleaning/` (`REVIEW_PAGE_DATA_DIR`):

| File                                     | Page reads                                                 | Page writes                                |
| ---------------------------------------- | ---------------------------------------------------------- | ------------------------------------------ |
| `review.jsonl`                           | Review cases and their persisted answers.                  | Saved answers on Save, Accept, and Reset.  |
| `review_run.json`                        | Last run result, baseline original answers, and seed file. | Never; written by the seed-cleaning run.   |
| `corrections.jsonl`                      | Seed entry corrections for the Corrections tab.            | Never; written by the seed-cleaning run.   |
| `entries_to_check.json`                  | Titles and metadata of checked seed entries.               | Never; written by the seed-cleaning run.   |
| `review.jsonl.lock`                      | Whether a pipeline run currently holds the lock.           | Acquired during answer saves.              |
| `review_run.pid`, `review_run.exit_code` | Active run process ID and last subprocess exit code.       | Written when starting and completing runs. |
| `logs/review_page_runs.log`              | Never.                                                     | Standard output and error of spawned runs. |

## How it is built

### The case source

The review page accesses data exclusively through an abstract interface
(`ReviewCaseSource` in `review_case_source.py`), whose methods exchange
protobuf messages defined in `protos/shared_proto/v1/review_cases.proto`:

- list and look up review cases, validate and save answers, start a pipeline
  run, and query run execution state;
- read the last run result (`SeedCleaningRunResult`), seed corrections, and
  checked seed entries.

`FileReviewCaseSource` (`file_review_case_source.py`) is the local file-backed
implementation and the only module in `review_page` that imports
`seed_cleaning`. `FileReviewCaseSource`:

- reads and writes the local data files listed above;
- starts runs and reports their state through `SeedCleaningRunProcess`
  (`seed_cleaning_run_process.py`), which is given the paths it needs and
  imports nothing from `seed_cleaning`;
- converts `seed_cleaning` domain records into protobuf messages;
- precomputes correction display labels, rule explanation sentences, and answer
  provenance flags so downstream views remain decoupled from `seed_cleaning`.

### Server and browser

- **Server.** FastAPI with Jinja2 templates. The server holds no session state
  between requests, so server restarts never lose state. `review.jsonl` is read
  again whenever it changes, so every page shows what the file holds, but only
  its new or changed lines are parsed and converted (`ReviewCaseLoader`): a
  save rewrites one line, and a run writes an unchanged case as the same text.
  `load_cases` likewise converts only the case messages it has not converted
  before.
- **Fragments.** htmx updates individual page regions (toolbar, run panel, case
  list, case panel, corrections list, and correction panel). Interactive controls
  carry deterministic element IDs (`build_element_id`) to preserve focus across
  swaps, and scrollable regions disable scroll anchoring (`overflow-anchor: none`).
- **Browser state.** Alpine.js manages client-side state persisted to
  `localStorage` via `@alpinejs/persist` and submits it in the `browser_state`
  form field (`BrowserState`) on every fragment request.
- **Tooltips.** User interface copy for buttons, columns, filters, states, and
  tabs is centralized in `tooltips.py` and injected into templates. Domain rule
  explanations (placement choice meanings, disabling rules, and case kind
  descriptions) live in `work_matching` alongside the rule logic. Automated
  tests verify that every `Tooltip` constant is referenced and that rendered
  fragments contain no empty `title=""` attributes.
- **Styling and components.** A single stylesheet (`static/review_page.css`) defines
  light and dark theme tokens, and `templates/control_macros.html` provides reusable
  Jinja macros (dropdown, chip, pill, status, button, icon button, pager, tabs,
  and notice lines). Static asset URLs include a content hash (`build_static_url`)
  for cache busting.
- **Vendor libraries.** Pinned versions of htmx, Alpine.js, and `@alpinejs/persist`
  are vendored in `static/vendor/` with SHA-256 checksums documented in
  `static/vendor/README.md`; no external network requests are made at runtime.

### Discovery and production

When Discovery replaces the offline seed file, an enrichment-service-backed
`ReviewCaseSource` implementation will exchange the same protobuf messages
without requiring changes to the web application. In production, the review
page will run behind the backend API gateway; during seed cleaning, it runs as
a local development service.

## Tests

```bash
./pants test tests/apps/review_page::
```

`tests/apps/review_page/unit/` tests all modules against fixture review cases
(`tests/libs/work_matching/unit/fixtures/review_cases.jsonl`), corrections, and
checked entries (`tests/apps/review_page/unit/fixtures/`). Route handlers are
tested with FastAPI's `TestClient` against an in-memory `ReviewCaseSource`
test double, and `FileReviewCaseSource` is tested against temporary data
directories including subprocess execution.
