// Manage the reviewer's browser state and handle server events on the review page.
//
// - Unsaved edits, rejected saves, filters, the open case, the "Show only
//   differences" switch, the theme and the case list's width and visibility
//   are kept in localStorage by Alpine's persist plugin, so a reload, a Run
//   or a server restart loses none of them. The server keeps no session
//   state, and every htmx request sends this state as `browser_state`.
// - Each unsaved edit keeps the saved answer the edit started from
//   (`seen_saved_answers`), so the server refuses a save that would replace
//   an answer saved meanwhile from another tab.
// - Open tabs share unsaved edits: a change in one tab reaches the other tabs
//   through the browser's `storage` event. Filters and the open case stay per
//   tab.
// - A visible page asks for the run state every few seconds, and once more
//   when the page is shown again, so a run started from the terminal or
//   another tab disables the page's buttons too. A hidden page does not ask.
// - The server decides every choice and rule. This script only stores what
//   the server's events report and asks for the regions to redraw.

// Trigger an htmx refresh event on the document body for each named region.
function refreshRegions(...regions) {
  for (const region of regions) {
    htmx.trigger(document.body, "refresh-" + region);
  }
}

// Return a copy of a mapping without the named keys.
function omitKeys(mapping, keys) {
  const keptEntries = { ...mapping };
  for (const key of keys) {
    delete keptEntries[key];
  }
  return keptEntries;
}

// The store fields and the region of each tab's list, so one filter, reset and
// page function serves both tabs.
const TAB_LISTS = {
  cases: {
    filterFields: ["kinds", "case_states", "edit_states"],
    searchField: "search_text",
    pageField: "page_number",
    region: "list",
  },
  corrections: {
    filterFields: ["correction_changes", "correction_sources"],
    searchField: "correction_search_text",
    pageField: "correction_page_number",
    region: "correction-list",
  },
};

// The stored values open tabs share, by their localStorage key.
const SHARED_STORE_FIELDS = {
  review_page_edits: "edits",
  review_page_seen_saved_answers: "seen_saved_answers",
  review_page_rejected: "rejected",
};

// One save in another tab changes several shared values at once, each with
// its own storage event. Redraw the regions once after the last event arrives.
const SHARED_CHANGE_SETTLE_MILLISECONDS = 50;
let sharedChangeRedrawTimer = null;

// Schedule a single redraw of the case regions after shared storage updates settle.
function redrawAfterSharedChanges() {
  clearTimeout(sharedChangeRedrawTimer);
  sharedChangeRedrawTimer = setTimeout(
    () => refreshRegions("toolbar", "list", "case"),
    SHARED_CHANGE_SETTLE_MILLISECONDS,
  );
}

document.addEventListener("alpine:init", () => {
  Alpine.store("review", {
    edits: Alpine.$persist({}).as("review_page_edits"),
    seen_saved_answers: Alpine.$persist({}).as(
      "review_page_seen_saved_answers",
    ),
    rejected: Alpine.$persist({}).as("review_page_rejected"),
    kinds: Alpine.$persist([]).as("review_page_kinds"),
    case_states: Alpine.$persist([]).as("review_page_case_states"),
    edit_states: Alpine.$persist([]).as("review_page_edit_states"),
    search_text: Alpine.$persist("").as("review_page_search_text"),
    page_number: Alpine.$persist(1).as("review_page_page_number"),
    open_case_name: Alpine.$persist(null).as("review_page_open_case_name"),
    show_only_differences: Alpine.$persist(false).as(
      "review_page_show_only_differences",
    ),
    // The shown tab, "cases" or "corrections", and the Corrections tab's own
    // filters, search, page and open entry.
    tab: Alpine.$persist("cases").as("review_page_tab"),
    correction_changes: Alpine.$persist([]).as(
      "review_page_correction_changes",
    ),
    correction_sources: Alpine.$persist([]).as(
      "review_page_correction_sources",
    ),
    correction_search_text: Alpine.$persist("").as(
      "review_page_correction_search_text",
    ),
    correction_page_number: Alpine.$persist(1).as(
      "review_page_correction_page_number",
    ),
    open_correction_entry: Alpine.$persist(null).as(
      "review_page_open_correction_entry",
    ),
    // Allowed values are "system", "light" or "dark". The inline script in
    // base.html applies the theme before the first paint.
    theme: Alpine.$persist("system").as("review_page_theme"),
    // The case list's width in pixels, and whether the list is shown.
    SIDEBAR_DEFAULT_WIDTH: 340,
    // The narrowest list still shows a row's widest common tag line, "doubled
    // provider · unchanged · Suggested" (268 px), with the row's padding
    // (16), the open row's bar (3), the panel's padding (32) and border (2).
    SIDEBAR_MIN_WIDTH: 324,
    SIDEBAR_MAX_WIDTH: 640,
    sidebar_width: Alpine.$persist(340).as("review_page_sidebar_width"),
    is_sidebar_open: Alpine.$persist(true).as("review_page_is_sidebar_open"),
    is_running: false,
    notice: "",

    serializeState() {
      return JSON.stringify({
        edits: this.edits,
        seen_saved_answers: this.seen_saved_answers,
        rejected: this.rejected,
        kinds: this.kinds,
        case_states: this.case_states,
        edit_states: this.edit_states,
        search_text: this.search_text,
        page_number: this.page_number,
        open_case_name: this.open_case_name,
        show_only_differences: this.show_only_differences,
        correction_changes: this.correction_changes,
        correction_sources: this.correction_sources,
        correction_search_text: this.correction_search_text,
        correction_page_number: this.correction_page_number,
        open_correction_entry: this.open_correction_entry,
      });
    },

    toggleFilter(tab, listName, value) {
      const values = this[listName];
      this[listName] = values.includes(value)
        ? values.filter((chosenValue) => chosenValue !== value)
        : [...values, value];
      this.showPage(tab, 1);
    },

    resetFilters(tab) {
      const tabList = TAB_LISTS[tab];
      for (const filterField of tabList.filterFields) {
        this[filterField] = [];
      }
      this[tabList.searchField] = "";
      this.showPage(tab, 1);
    },

    showPage(tab, pageNumber) {
      const tabList = TAB_LISTS[tab];
      this[tabList.pageField] = pageNumber;
      refreshRegions(tabList.region);
    },

    setTab(tab) {
      this.tab = tab;
    },

    openCorrection(referencePage) {
      this.open_correction_entry = referencePage;
      refreshRegions("correction", "correction-list");
    },

    // Move between the tabs: from a corrected entry to a case holding that
    // entry, and from a case to its entry's corrections.
    showCase(caseName) {
      this.tab = "cases";
      this.openCase(caseName);
    },

    showCorrection(referencePage) {
      this.tab = "corrections";
      this.openCorrection(referencePage);
    },

    openCase(caseName) {
      this.open_case_name = caseName;
      refreshRegions("case", "list");
    },

    // Clamp any sidebar width stored before the bounds changed.
    init() {
      this.setSidebarWidth(this.sidebar_width);
    },

    // Move keyboard focus to whichever sidebar toggle button remains visible.
    toggleSidebar() {
      this.is_sidebar_open = !this.is_sidebar_open;
      Alpine.nextTick(() => {
        const shownButtonPlace = this.is_sidebar_open
          ? ".list-panel-top"
          : ".layout-rail";
        document.querySelector(`${shownButtonPlace} .is-icon-only`)?.focus();
      });
    },

    // Keep the width within bounds and leave the open case at least 480 px
    // so neither panel becomes unusable.
    setSidebarWidth(width) {
      const mainLayout = document.querySelector(".layout-main");
      const roomLeft = mainLayout
        ? mainLayout.clientWidth - 480
        : this.SIDEBAR_MAX_WIDTH;
      const maxSidebarWidth = Math.max(
        this.SIDEBAR_MIN_WIDTH,
        Math.min(this.SIDEBAR_MAX_WIDTH, roomLeft),
      );
      this.sidebar_width = Math.round(
        Math.min(maxSidebarWidth, Math.max(this.SIDEBAR_MIN_WIDTH, width)),
      );
    },

    // Follow the pointer until release. Pointer capture keeps the drag active
    // even when the pointer leaves the thin divider.
    startSidebarResize(event) {
      // The width moves by the distance dragged, so grabbing the divider
      // anywhere across its width never makes the list jump.
      const divider = event.currentTarget;
      const startX = event.clientX;
      const startWidth = this.sidebar_width;
      divider.setPointerCapture(event.pointerId);
      divider.classList.add("is-dragging");
      const followPointer = (moveEvent) =>
        this.setSidebarWidth(startWidth + moveEvent.clientX - startX);
      const stopResize = () => {
        divider.classList.remove("is-dragging");
        divider.removeEventListener("pointermove", followPointer);
        divider.removeEventListener("pointerup", stopResize);
        divider.removeEventListener("pointercancel", stopResize);
      };
      divider.addEventListener("pointermove", followPointer);
      divider.addEventListener("pointerup", stopResize);
      divider.addEventListener("pointercancel", stopResize);
    },

    setTheme(choice) {
      this.theme = choice;
      applyReviewPageTheme(choice);
    },

    discardEdits() {
      this.edits = {};
      this.seen_saved_answers = {};
      this.rejected = {};
      refreshRegions("toolbar", "list", "case");
    },
  });

  // Provide a searchable picker for the seed entry an "other work" page moves
  // to and for a work's partial sources. The anchor plugin positions the panel
  // and the focus plugin moves between options (see vendor/README.md). The
  // search box filters the options, while a typed page key is checked by the
  // server, which answers with an option or the reason the key cannot be used.
  Alpine.data("searchPicker", () => ({
    isOpen: false,
    query: "",
    // The options ticked in a picker that adds several at once.
    checkedValues: [],

    togglePicker() {
      this.isOpen = !this.isOpen;
      if (this.isOpen) this.$nextTick(() => this.$refs.search.focus());
    },

    closePicker() {
      this.isOpen = false;
      this.query = "";
    },

    isOptionShown(searchText) {
      const query = this.query.trim().toLowerCase();
      return query === "" || searchText.toLowerCase().includes(query);
    },

    // A typed key's option names a seed entry, and stays hidden while the
    // search shows the same entry as a suggestion, so the entry is offered once.
    isSuggestionShown(entryReferencePage) {
      return [
        ...this.$refs.options.querySelectorAll("[data-entry-reference-page]"),
      ].some(
        (option) =>
          option.dataset.entryReferencePage === entryReferencePage &&
          this.isOptionShown(option.dataset.searchText),
      );
    },

    // The focus plugin counts hidden buttons as focusable, so the arrow keys
    // receive only the options the search leaves visible: option buttons, or
    // the checkboxes of a picker that adds several at once.
    listShownOptions() {
      return [
        ...this.$refs.options.querySelectorAll(
          "button[role=option], [role=option] input[type=checkbox]",
        ),
      ].filter((option) => option.checkVisibility());
    },
  }));
});

document.addEventListener("DOMContentLoaded", () => {
  const body = document.body;

  // "System" follows the operating system when its setting changes.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    applyReviewPageTheme(Alpine.store("review").theme);
  });

  body.addEventListener("answerchanged", (event) => {
    const store = Alpine.store("review");
    const {
      case_name: caseName,
      answer,
      is_unsaved: isUnsaved,
      saved_answer: savedAnswer,
    } = event.detail;
    if (isUnsaved) {
      store.edits = { ...store.edits, [caseName]: answer };
      // The first change of an edit records the saved answer the edit started from.
      if (!(caseName in store.seen_saved_answers)) {
        store.seen_saved_answers = {
          ...store.seen_saved_answers,
          [caseName]: savedAnswer ?? null,
        };
      }
    } else {
      store.edits = omitKeys(store.edits, [caseName]);
      store.seen_saved_answers = omitKeys(store.seen_saved_answers, [
        caseName,
      ]);
    }
    store.rejected = omitKeys(store.rejected, [caseName]);
    refreshRegions("toolbar", "list");
  });

  body.addEventListener("answerssaved", (event) => {
    const store = Alpine.store("review");
    const {
      saved_cases: savedCases,
      cases_saved_elsewhere: casesSavedElsewhere,
    } = event.detail;
    store.edits = omitKeys(store.edits, savedCases);
    // An edit refused because another tab saved meanwhile forgets the answer
    // the edit started from, so saving again replaces that answer on purpose.
    store.seen_saved_answers = omitKeys(store.seen_saved_answers, [
      ...savedCases,
      ...casesSavedElsewhere,
    ]);
    store.rejected = {
      ...omitKeys(store.rejected, savedCases),
      ...event.detail.rejected,
    };
    store.notice = event.detail.warnings.join(" ");
    if (event.detail.next_case_name) {
      store.open_case_name = event.detail.next_case_name;
    }
    refreshRegions("toolbar", "list", "case");
  });

  body.addEventListener("runstarted", () => {
    Alpine.store("review").is_running = true;
    refreshRegions("run", "toolbar", "case");
  });

  body.addEventListener("runstate", (event) => {
    const store = Alpine.store("review");
    const wasRunning = store.is_running;
    store.is_running = event.detail.is_running;
    if (!wasRunning && store.is_running) {
      refreshRegions("run", "toolbar", "case");
    } else if (wasRunning && !store.is_running) {
      refreshRegions(
        "run",
        "toolbar",
        "list",
        "case",
        "correction-list",
        "correction",
      );
    }
  });

  // Apply shared stored values changed by another tab and redraw. Setting the
  // same value back stores nothing new, so the tabs do not echo.
  window.addEventListener("storage", (event) => {
    const field = SHARED_STORE_FIELDS[event.key];
    if (!field) {
      return;
    }
    try {
      Alpine.store("review")[field] = JSON.parse(event.newValue ?? "{}");
    } catch {
      return;
    }
    redrawAfterSharedChanges();
  });

  body.addEventListener("notice", (event) => {
    Alpine.store("review").notice = event.detail.text;
  });

  // htmx does not process response headers on error statuses the same way
  // everywhere, so a refusal's notice is also read from its JSON body.
  // A body without a notice, such as FastAPI's {"detail": ...} for 404 and
  // 422, or a plain-text refusal, shows the general notice.
  body.addEventListener("htmx:responseError", (event) => {
    const generalNotice = "The server could not do that; see its log.";
    try {
      Alpine.store("review").notice =
        JSON.parse(event.detail.xhr.responseText).notice || generalNotice;
    } catch {
      Alpine.store("review").notice = generalNotice;
    }
  });

  // Swap a server error page into its region so the error shows where the
  // failure happened. A notice response is JSON, shown by the notice banner
  // above, so its body is never swapped into a region.
  body.addEventListener("htmx:beforeOnLoad", (event) => {
    const contentType =
      event.detail.xhr.getResponseHeader("Content-Type") || "";
    if (
      event.detail.xhr.status === 500 &&
      !contentType.includes("application/json")
    ) {
      event.detail.shouldSwap = true;
      event.detail.isError = false;
    }
  });
});
