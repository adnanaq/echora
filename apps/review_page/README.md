# review_page

A local web page for answering seed cleaning's review cases. Seed cleaning
(`libs/seed_cleaning`, run by `scripts/clean_seed.py`) writes every decision its
rules cannot settle to `data/seed_cleaning/review.jsonl`, one long JSON line per
case. The page lists those cases, shows each one's evidence, offers only the
choices the placement rules allow, saves the answers safely, starts a run that
applies them, and shows what the run corrected.

It runs on the reviewer's machine and is reached from it alone: it reads and
writes local files and starts local runs.

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
- [Files](#files)

## Terms

The matching terms (work, provider page, page key, reference page, doubled
provider, split work, linked page, part page) are defined in the
[`work_matching` README](../../libs/work_matching/README.md#terms), and the
seed cleaning terms (seed correction, review file, provider page store) in the
[`seed_cleaning` README](../../libs/seed_cleaning/README.md#terms). The page
adds:

| Term               | Meaning                                                                                                                             |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Original value** | A case's answer as the last run wrote it. "Changed" and "Reset" compare against it.                                                 |
| **Unsaved edit**   | A choice made on the page and not saved yet. It lives in the browser until it is saved or discarded.                                |
| **Saved answer**   | An answer written to the review file. The next Run applies it.                                                                      |
| **Suggestion**     | The answer the rules suggest for a case. An unanswered case opens on it; Accept saves it.                                           |
| **Case source**    | Where the page reads cases and corrections, saves answers and starts runs (`ReviewCaseSource`). The first source reads local files. |

## Running it

With Docker, as part of the dev stack:

```bash
docker compose -f docker/docker-compose.dev.yml up -d review-page
# open http://127.0.0.1:8003
```

The container mounts `data/seed_cleaning` read-write (saves and runs write
there) and `assets/seed_data` read-only (a run reads the seed file). It is
published on `127.0.0.1` only, runs as a non-root user with the host user's
uid, and restarts unless stopped. Its health check calls `GET /health`.

Without Docker, from the repository root, so the default data folder and
seed-cleaning script are found:

```bash
uv run python -m review_page.main   # with libs/*/src and apps/review_page/src on PYTHONPATH
```

The image (`apps/review_page/Dockerfile.dev`, `Dockerfile.prd`) holds the page,
`scripts/clean_seed.py` and the libraries a run imports. A run from the page
fetches nothing, so the image holds no browser, and it loads no embedding
model, so it leaves the model stack (torch and the packages built on it) out.

### Settings

`ReviewPageConfig` in `libs/common/src/common/config/review_page_config.py`,
read like every other sub-config of `Settings`:

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

1. Run seed cleaning from the terminal. It fetches the pages and writes the
   review cases, the corrections and the run's summary to `data/seed_cleaning/`.
2. Open the page. The Cases tab lists every case; filter by kind, by whether
   it is answered, or by edit state, and search by title or page key.
3. Open a case and choose a work and mark for each page (or an answer for a
   linked page). Every choice is checked against the rules as it is made; a
   choice the rules refuse is greyed out with the reason.
4. Save (or Save & next, or Save all for every unsaved case). A save that
   breaks a rule is refused, the rule is shown, and the edit stays in the
   browser for fixing.
5. Press Run. It runs seed cleaning without fetching (`--no-fetch`), with the
   seed file of the last run, and applies every saved answer.
6. Read the run's summary at the top, and the Corrections tab, to check that
   the run did what the answers meant.

### A new seed release

1. Put the new seed file in `assets/seed_data/`.
2. Run seed cleaning from the terminal, naming the file:

   ```bash
   uv run python scripts/clean_seed.py assets/seed_data/anime-offline-database-minified.json
   ```

3. Reload the page. It reads the new cases and summary on every request, so
   there is nothing to import and nothing to rebuild; Run follows the new
   release on its own.

The page's Run only re-runs without fetching. The first run of a release,
which fetches its pages, is always started from the terminal.

## The page

### Cases tab

- **List.** One row per case: its title, kind, edit state and tags (New,
  Suggested, No longer comes up). Filters within one group widen the list;
  groups narrow each other. 25 rows a page.
- **Case panel.** Why the case needs review, its evidence, its entries, and a
  table of its pages with their details (titles, type, episodes, air dates,
  links, each linking to the provider page) and a Work and Mark dropdown each.
  "Show only differences" hides the details every page shares.
- **Choices.** Choosing `part` for one page marks the provider's other pages
  in that work as parts too; choosing `main` leaves them unmarked. A choice
  another page's place rules out (a work already holding a MAL page) is greyed
  out, with one note per rule under the dropdown. "What the choices mean" under
  the table explains every word; hovering any dropdown, filter, tag or button
  says what it does.
- **States.** Each case shows one edit state: rejected (its last save broke a
  rule), edited and not saved, saved and applying on Run, or unchanged. Rows
  whose place differs from the case's starting answer are highlighted. An
  unanswered case starts on its suggestion, so its untouched suggestion is not
  an edit.
- **Buttons.** Save and Save & next while the shown answer differs from the
  saved one; Accept while a complete suggestion differs from it; Reset while
  the shown or saved answer differs from the original value. Every button is
  disabled while a run is going.
- **No longer comes up.** A case whose answer the file keeps but the last run
  no longer brought up, because another answer changed its pages. Its answer
  changed nothing; answer the case that replaced it.

### Corrections tab

What the last run changed, one corrected entry at a time:

- **List.** Each corrected entry with how many of its pages changed, and a
  "Your answer" tag when an answer placed any of them. Filter by change (pages
  moved, old addresses dropped, kept as main or part page, unchanged) and by
  source (your answer or the rules).
- **Entry panel.** A table of the entry's pages: the change, where the page
  goes (the seed entry it joins, linked to that entry's page, a new entry, its
  current address, or staying), its work and mark, why, and the source.
- **Links both ways.** A corrected entry links the review cases holding it,
  and a case shows how many corrections the last run made for its entries.

Corrections are the last run's; answers saved since then show after the next
Run.

### Run

Run is enabled while saved answers differ from their original values and no
edit is left unsaved, so a run never misses an edit. While it runs, the page
shows "Run going" and every button waits; when it ends, the summary, the
lists, the original values and the Corrections tab refresh. A failed run shows
its exit code; its output is in `data/seed_cleaning/logs/review_page_runs.log`.

### Layout and settings

- The case list can be hidden and shown (the button beside the search box) and
  resized by dragging the divider or with the arrow keys; double-click resets
  its width. Its narrowest width still shows a row's tags in full.
- Theme: System (follows the computer's light or dark setting), Light or Dark.
- The browser keeps the reviewer's own state across reloads, page restarts and
  runs: unsaved edits, rejected saves, filters, searches, pages, the open case
  and entry, the tab, the theme, and the list's width and visibility.

## Saving safely

- **Rules first.** The source checks every answer with the same rules a run
  uses (`work_matching`) before saving it, so the page never saves an answer
  the run would refuse.
- **One line changed.** A save rewrites only the answered cases' `answer`
  fields; every other line of the review file stays byte for byte, and the file
  is replaced atomically.
- **One writer at a time.** Saves and runs take an operating-system file lock
  on `review.jsonl.lock`. A save while a run holds it is refused with a notice,
  and nothing is written. The lock works between the container and the host.
- **Cases a run removed.** A case a run removed after the page loaded it is
  rejected with that reason, like an answer breaking a rule, and Save All
  still saves the other cases. When the run removes it during the save itself,
  between the check and the write, nothing is saved and a notice asks for a
  reload.
- **Only the page itself.** Other websites open in the same browser cannot
  save answers or start runs (`request_guards.py`):
  - a request naming any host but `127.0.0.1` or `localhost` gets 400, which
    stops DNS rebinding;
  - a POST without the `HX-Request: true` header htmx sends gets 403, and a
    site on another origin cannot add that header without a CORS preflight,
    which the server refuses.

  A request under another name, such as a Kubernetes probe of `/health`,
  must send `Host: localhost`, or that name must be added to
  `LOCAL_HOST_NAMES`.

- **Linked-page conflicts.** Two answers adding whole pages of one provider to
  one entry are saved with a warning, since the run adds neither.

## Files it reads and writes

All in `data/seed_cleaning/` (`REVIEW_PAGE_DATA_DIR`):

| File                                     | Page reads                                          | Page writes                               |
| ---------------------------------------- | --------------------------------------------------- | ----------------------------------------- |
| `review.jsonl`                           | The cases and their saved answers.                  | Answers, on Save, Accept, Reset.          |
| `review_run.json`                        | The last run's summary, original values, seed file. | Never; a run writes it.                   |
| `corrections.jsonl`                      | The Corrections tab.                                | Never; a run writes it.                   |
| `entries_to_check.json`                  | Titles of corrected entries.                        | Never; a run writes it.                   |
| `review.jsonl.lock`                      | Whether a run is going.                             | Taken during a save.                      |
| `review_run.pid`, `review_run.exit_code` | The run the page started, and how it ended.         | When Run starts and ends.                 |
| `logs/review_page_runs.log`              | Never.                                              | The output of runs started from the page. |

## How it is built

### The case source

The page reaches cases only through one swappable source
(`review_case_source.py`), whose methods take and return the protobuf messages
in `protos/shared_proto/v1/review_cases.proto`:

- list and get cases, save answers (checked), start a run and read its state;
- read the last run's summary, its corrections and the entries it checked.

`FileReviewCaseSource` is the first source and the only module that imports
`seed_cleaning`: it reads the files above, converts seed cleaning's models to
the messages, and fills in what the page shows without knowing seed cleaning
(each correction's change, its reason as a sentence, and whether an answer
placed it).

### Server and browser

- **Server.** FastAPI with Jinja templates. It keeps no state between
  requests: every request rereads the files, so reading the cases costs a few
  milliseconds and a page restart loses nothing.
- **Fragments.** htmx redraws one region at a time (toolbar, run panel, list,
  case panel, corrections list and panel). Every control in a redrawn region
  has a stable id, so focus stays where it was, and the regions are never
  scroll anchors, so a redraw never moves the view.
- **Browser state.** Alpine.js holds the reviewer's state, kept in
  localStorage by its persist plugin, and sends it with every request.
- **Look.** One stylesheet (`static/parts.css`) of named colours and sizes,
  with a light and a dark set, and one Jinja macro per shared part
  (`templates/parts.html`: dropdown, chip, pill, status, button, icon button,
  pager, tabs, notes), so each part looks and behaves the same everywhere.
  Static files carry a content hash in their address, so a browser never keeps
  an old copy.
- **Libraries.** htmx, Alpine.js and its persist plugin are copied into
  `static/vendor/` at fixed versions, with their checksums in that folder's
  README; nothing is loaded from the internet.

### Discovery and production

When Discovery replaces the seed, a source backed by the enrichment service
implements the same methods with the same messages, and the page does not
change. In production the page sits behind the backend, which is the only
entry point for the frontend; until then it stays a local dev service.

## Tests

```bash
./pants test tests/apps/review_page::
```

`tests/apps/review_page/unit/` tests every module with real review cases
(`tests/libs/work_matching/unit/fixtures/review_cases.jsonl`) and real
corrections and checked entries (`fixtures/`). Routes are tested with
FastAPI's test client over a fake source that subclasses `ReviewCaseSource`;
the file source is tested on a temporary data folder, runs included.

## Files

| File                         | Holds                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| `main.py`                    | Starts the page: reads the settings, builds the file source, serves the app with uvicorn.             |
| `app.py`                     | `create_app`, `ReviewPageRoutes`: the page, the Cases tab's fragments, choices, saves and Run.        |
| `correction_routes.py`       | `CorrectionRoutes`: the Corrections tab's fragments.                                                  |
| `review_case_source.py`      | `ReviewCaseSource`: the source the page reads and writes through.                                     |
| `file_review_case_source.py` | `FileReviewCaseSource`: the source over seed cleaning's local files, and its conversions to messages. |
| `review_case_messages.py`    | Conversion between `work_matching.ReviewCase` and its message, and the helpers for ordered mappings.  |
| `run_summary.py`             | `RunSummary`, `build_run_summary`: the last run's summary read from its message.                      |
| `page_state.py`              | Edit states, case states, buttons, filters and paging, as plain functions of the answers.             |
| `case_views.py`              | What the case list and case panel show: rows, page rows, choices, notes, columns.                     |
| `correction_views.py`        | What the Corrections tab shows: corrected entries, filters, an entry's pages and destinations.        |
| `browser_state.py`           | `BrowserState`: the reviewer's state as the browser sends it.                                         |
| `request_guards.py`          | The host check and the htmx header check that refuse requests from other websites.                    |
| `template_functions.py`      | `static_url` and `element_id`, given to every template.                                               |
| `templates/`                 | The page, its fragments and `parts.html`, the shared parts.                                           |
| `static/`                    | `parts.css`, `review_page.js` and the copied libraries.                                               |
