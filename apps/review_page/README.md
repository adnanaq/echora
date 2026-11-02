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

1. Run seed cleaning from the terminal. Seed cleaning fetches the pages, then
   writes the review cases, the corrections and the run's summary to
   `data/seed_cleaning/`.
2. Open the page. The Cases tab lists every case. Filter by kind, by whether
   a case is answered, or by edit state, and search by title or page key.
3. Open a case and choose a work and mark for each page, or an answer for a
   linked page. Every choice is checked against the rules as it is made. A
   choice the rules refuse is greyed out, with the reason.
4. Save (or Save & next, or Save all for every unsaved case). A save that
   breaks a rule is refused, the rule is shown, and the edit stays in the
   browser for fixing.
5. Press Run. Run starts seed cleaning without fetching (`--no-fetch`) and
   applies every saved answer. Run uses `REVIEW_PAGE_SEED_FILE` when it is
   set, and otherwise the seed file the last run read.
6. Read the run's summary at the top, and the Corrections tab, to check that
   the run did what the answers meant.

### A new seed release

1. Put the new seed file in `assets/seed_data/`.
2. Run seed cleaning from the terminal, naming the file:

   ```bash
   uv run python scripts/clean_seed.py assets/seed_data/anime-offline-database-minified.json
   ```

3. Reload the page. The page reads the cases and summary on every request,
   so there is nothing to import and nothing to rebuild. Run follows the new
   release on its own.

The page's Run never fetches. The first run of a release, which fetches its
pages, is always started from the terminal.

Saved answers carry over to the new release by case name:

- A case whose pages are the same gets its saved answer back, and the run
  applies that answer again.
- A case whose pages changed in the new release gets a new name. The case is
  asked again and marked New.
- The old answer of a changed case stays in the review file but is not
  applied. The run summary lists the old answer under "Answers whose case no
  longer comes up".

An answered case is never removed from the review file. Every run starts
again from the downloaded seed and applies the saved answers anew, so a
removed answer would be asked again.

## The page

### Cases tab

- **List.** One row per case: its title, kind, edit state and tags (New,
  Suggested, No longer comes up). Filters within one group widen the list;
  groups narrow each other. 25 rows a page.
- **Case panel.** Why the case needs review, its evidence, its entries, and a
  table of its pages. Each page row shows the page's details (titles, type,
  episodes, air dates, links to the provider pages) and a Work and a Mark
  dropdown. "Show only differences" hides the details every page shares.
- **Choices.**
  - Choosing `part` for one page marks the provider's other pages in that
    work as parts too. Choosing `main` leaves the other pages unmarked.
  - A choice another page's place rules out, such as a second MAL page in one
    work, is greyed out, with one note per rule under the dropdown.
  - "What the choices mean" under the table explains every word. Hovering any
    dropdown, filter, tag or button says what it does.
- **States.** Each case shows one edit state: rejected (its last save broke a
  rule), edited and not saved, saved and applying on Run, or unchanged. Rows
  whose place differs from the case's starting answer are highlighted. An
  unanswered case starts on its suggestion, so an untouched suggestion is not
  an edit.
- **Buttons.**
  - Save and Save & next: enabled while the shown answer differs from the
    saved one.
  - Accept: enabled while a complete suggestion differs from the saved answer.
  - Reset: enabled while the shown or saved answer differs from the original
    value.
  - Every button is disabled while a run is going.
- **No longer comes up.** The file keeps the case's answer, but the last run
  no longer brought the case up, because another answer changed the case's
  pages. The kept answer changed nothing. Answer the case that replaced it.

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
edit is left unsaved, so a run never misses an edit. While a run goes, the
page shows "Run going" and every button waits. When the run ends, the
summary, the lists, the original values and the Corrections tab refresh.

The page asks for the run state every few seconds, so a run started from the
terminal or another tab shows the same way. A failed run shows its exit code.
The run's output is in `data/seed_cleaning/logs/review_page_runs.log`. A log
past about 1 MB moves to `review_page_runs.log.1` before the next run.

A run keeps going when the page's server restarts. The restarted server
follows that run through the review file's lock, but cannot read the run's
exit code, so a failure shows only in the run log.

### Layout and settings

- The case list can be hidden and shown (the button beside the search box) and
  resized by dragging the divider or with the arrow keys; double-click resets
  its width. The list's narrowest width still shows a row's tags in full.
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
  rejected with that reason, like an answer breaking a rule. Save all still
  saves the other cases. When the run removes the case during the save itself,
  between the check and the write, nothing is saved and a notice asks for a
  reload.
- **Only the page itself.** Other websites open in the same browser cannot
  save answers or start runs (`request_guards.py`):
  - a request naming any host but `127.0.0.1`, `localhost` or `[::1]` gets
    400, which stops DNS rebinding;
  - a POST without the `HX-Request: true` header htmx sends gets 403. A site
    on another origin cannot add that header without a CORS preflight, which
    the server refuses;
  - a POST whose `Origin` header names another host gets 403, a second check
    in case the header check ever lets a request through.

  A request under another name, such as a Kubernetes probe of `/health`,
  must send `Host: localhost`, or that name must be added to
  `LOCAL_HOST_NAMES`.

- **Edits started from an older answer.** Each unsaved edit keeps the saved
  answer it started from. When another tab or browser has saved a different
  answer since, the save is refused with that reason, checked again under the
  file lock, and nothing is written. Saving again replaces that answer on
  purpose.
- **Several tabs.** Open tabs of the page share unsaved edits through the
  browser's `storage` event. An edit made in one tab shows in the others, and
  a save in one tab clears the edit everywhere. Filters and the open case stay
  per tab. A second browser keeps its own edits, which the check above
  protects.
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
`seed_cleaning`. The file source:

- reads the files above;
- converts seed cleaning's models to the messages;
- fills in what the page shows without knowing seed cleaning: each
  correction's change, the correction's reason as a sentence, and whether an
  answer placed the page.

### Server and browser

- **Server.** FastAPI with Jinja templates. The server keeps no state between
  requests. Every request rereads the files, which costs a few milliseconds,
  so a page restart loses nothing.
- **Fragments.** htmx redraws one region at a time (toolbar, run panel, list,
  case panel, corrections list and panel). Every control in a redrawn region
  has a stable id, so focus stays where it was, and the regions are never
  scroll anchors, so a redraw never moves the view.
- **Browser state.** Alpine.js holds the reviewer's state, kept in
  localStorage by its persist plugin, and sends it with every request.
- **Tooltips.** The page's own wording (buttons, columns, filters, states,
  tabs) lives in `tooltips.py`, which every template reads. Rule wording (what
  an answer word does, why an option is not offered, what a case kind is)
  lives in `work_matching`, next to the rules it explains. A test fails on a
  tooltip text no page uses and on an empty tooltip.
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

| File                         | Holds                                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `main.py`                    | Starts the page: reads the settings, builds the file source, serves the app with uvicorn.                       |
| `app.py`                     | `create_app`, `PageRoutes`: builds the app, serves the page shell and health check, and the broken-line notice. |
| `case_routes.py`             | `CaseRoutes`: the Cases tab's toolbar, list, open case, choices and saves.                                      |
| `correction_routes.py`       | `CorrectionRoutes`: the Corrections tab's fragments.                                                            |
| `run_routes.py`              | `RunRoutes`: starting Run, the run panel, and the run state every page polls.                                   |
| `htmx_responses.py`          | The empty and refusal responses that hand the browser htmx events.                                              |
| `review_case_source.py`      | `ReviewCaseSource`: the source the page reads and writes through.                                               |
| `file_review_case_source.py` | `FileReviewCaseSource`: the source over seed cleaning's local files, and its conversions to messages.           |
| `answer_checks.py`           | The checks an answer passes before the file source saves it, and linked-page conflict warnings.                 |
| `review_case_messages.py`    | Conversion between `work_matching.ReviewCase` and its message, and the helpers for ordered mappings.            |
| `run_summary.py`             | `build_run_summary`: reads the last run's summary from its message.                                             |
| `models.py`                  | Every data shape the page passes between modules and templates: states, rows, views, filters, `BrowserState`.   |
| `page_state.py`              | Decides each case's edit state and case state, the buttons, the filters and paging, from the answers alone.     |
| `case_views.py`              | Builds the case list's rows and assembles the case panel's view.                                                |
| `case_panel_tables.py`       | Builds the case panel's tables: entries, page rows with choices and notes, comparison rows, evidence.           |
| `correction_views.py`        | Builds what the Corrections tab shows: corrected entries, filters, an entry's pages and destinations.           |
| `request_guards.py`          | The host check and the htmx header check that refuse requests from other websites.                              |
| `template_functions.py`      | `static_url`, `element_id` and the tooltips, given to every template.                                           |
| `tooltips.py`                | Every tooltip the page writes itself (`Tooltip`, and the texts for states, changes and themes).                 |
| `templates/`                 | The page, its fragments and `parts.html`, the shared parts.                                                     |
| `static/`                    | `parts.css`, `review_page.js` and the copied libraries.                                                         |
