// The browser's part of the review page: the reviewer's own state, and the
// server's events that change it.
//
// - Unsaved edits, rejected saves, filters, the open case, the "Show only
//   differences" switch, the theme and the case list's width and visibility
//   are kept in localStorage by Alpine's persist plugin,
//   so a reload, a Run or a server restart loses none of them. The server
//   keeps no state; every htmx request sends this state as `browser_state`.
// - The server decides every choice and rule. This script only stores what
//   the server's events report and asks for the regions to redraw.

function refreshRegions(...regions) {
  for (const region of regions) {
    htmx.trigger(document.body, "refresh-" + region);
  }
}

document.addEventListener("alpine:init", () => {
  Alpine.store("review", {
    edits: Alpine.$persist({}).as("review_page_edits"),
    rejected: Alpine.$persist({}).as("review_page_rejected"),
    kinds: Alpine.$persist([]).as("review_page_kinds"),
    case_states: Alpine.$persist([]).as("review_page_case_states"),
    edit_states: Alpine.$persist([]).as("review_page_edit_states"),
    search_text: Alpine.$persist("").as("review_page_search_text"),
    page_number: Alpine.$persist(1).as("review_page_page_number"),
    open_case_name: Alpine.$persist(null).as("review_page_open_case_name"),
    show_only_differences: Alpine.$persist(false).as("review_page_show_only_differences"),
    // The shown tab, "cases" or "corrections", and the Corrections tab's own
    // filters, search, page and open entry.
    tab: Alpine.$persist("cases").as("review_page_tab"),
    correction_changes: Alpine.$persist([]).as("review_page_correction_changes"),
    correction_sources: Alpine.$persist([]).as("review_page_correction_sources"),
    correction_search_text: Alpine.$persist("").as("review_page_correction_search_text"),
    correction_page_number: Alpine.$persist(1).as("review_page_correction_page_number"),
    open_correction_entry: Alpine.$persist(null).as("review_page_open_correction_entry"),
    // "system", "light" or "dark"; base.html applies it before the first paint.
    theme: Alpine.$persist("system").as("review_page_theme"),
    // The case list's width in pixels, and whether it is shown.
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

    toJson() {
      return JSON.stringify({
        edits: this.edits,
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

    toggleFilter(listName, value) {
      const values = this[listName];
      this[listName] = values.includes(value)
        ? values.filter((chosen) => chosen !== value)
        : [...values, value];
      this.page_number = 1;
      refreshRegions("list");
    },

    resetFilters() {
      this.kinds = [];
      this.case_states = [];
      this.edit_states = [];
      this.search_text = "";
      this.page_number = 1;
      refreshRegions("list");
    },

    showPage(pageNumber) {
      this.page_number = pageNumber;
      refreshRegions("list");
    },

    setTab(tab) {
      this.tab = tab;
    },

    toggleCorrectionFilter(listName, value) {
      const values = this[listName];
      this[listName] = values.includes(value)
        ? values.filter((chosen) => chosen !== value)
        : [...values, value];
      this.correction_page_number = 1;
      refreshRegions("correction-list");
    },

    resetCorrectionFilters() {
      this.correction_changes = [];
      this.correction_sources = [];
      this.correction_search_text = "";
      this.correction_page_number = 1;
      refreshRegions("correction-list");
    },

    showCorrectionPage(pageNumber) {
      this.correction_page_number = pageNumber;
      refreshRegions("correction-list");
    },

    openCorrection(referencePage) {
      this.open_correction_entry = referencePage;
      refreshRegions("correction", "correction-list");
    },

    // Move between the tabs: from a corrected entry to a case holding it,
    // and from a case to its entry's corrections.
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

    // The button that was pressed is hidden with the list, or with the strip,
    // so keyboard focus moves to the button now shown.
    // A width saved before the bounds changed is brought within them.
    init() {
      this.setSidebarWidth(this.sidebar_width);
    },

    toggleSidebar() {
      this.is_sidebar_open = !this.is_sidebar_open;
      Alpine.nextTick(() => {
        const shownButtonPlace = this.is_sidebar_open ? ".list-panel-top" : ".layout-rail";
        document.querySelector(`${shownButtonPlace} .is-icon-only`)?.focus();
      });
    },

    // Keep the width within its bounds, and leave the open case at least
    // 480 px, so neither panel becomes unusable.
    setSidebarWidth(width) {
      const main = document.querySelector(".layout-main");
      const roomLeft = main ? main.clientWidth - 480 : this.SIDEBAR_MAX_WIDTH;
      const widest = Math.max(this.SIDEBAR_MIN_WIDTH, Math.min(this.SIDEBAR_MAX_WIDTH, roomLeft));
      this.sidebar_width = Math.round(Math.min(widest, Math.max(this.SIDEBAR_MIN_WIDTH, width)));
    },

    // Follow the pointer until it is released; pointer capture keeps the drag
    // going when the pointer leaves the thin divider.
    startSidebarResize(event) {
      // The width moves by the distance dragged, so grabbing the divider
      // anywhere across its width never makes the list jump.
      const divider = event.currentTarget;
      const startX = event.clientX;
      const startWidth = this.sidebar_width;
      divider.setPointerCapture(event.pointerId);
      divider.classList.add("is-dragging");
      const follow = (moveEvent) => this.setSidebarWidth(startWidth + moveEvent.clientX - startX);
      const stop = () => {
        divider.classList.remove("is-dragging");
        divider.removeEventListener("pointermove", follow);
        divider.removeEventListener("pointerup", stop);
        divider.removeEventListener("pointercancel", stop);
      };
      divider.addEventListener("pointermove", follow);
      divider.addEventListener("pointerup", stop);
      divider.addEventListener("pointercancel", stop);
    },

    setTheme(choice) {
      this.theme = choice;
      applyReviewPageTheme(choice);
    },

    discardEdits() {
      this.edits = {};
      this.rejected = {};
      refreshRegions("toolbar", "list", "case");
    },
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const body = document.body;

  // "System" follows the operating system when its setting changes.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    applyReviewPageTheme(Alpine.store("review").theme);
  });

  body.addEventListener("answerchanged", (event) => {
    const store = Alpine.store("review");
    const { case_name: caseName, answer, is_unsaved: isUnsaved } = event.detail;
    const edits = { ...store.edits };
    if (isUnsaved) {
      edits[caseName] = answer;
    } else {
      delete edits[caseName];
    }
    store.edits = edits;
    const rejected = { ...store.rejected };
    delete rejected[caseName];
    store.rejected = rejected;
    refreshRegions("toolbar", "list");
  });

  body.addEventListener("answerssaved", (event) => {
    const store = Alpine.store("review");
    const edits = { ...store.edits };
    const rejected = { ...store.rejected };
    for (const caseName of event.detail.saved) {
      delete edits[caseName];
      delete rejected[caseName];
    }
    Object.assign(rejected, event.detail.rejected);
    store.edits = edits;
    store.rejected = rejected;
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
    if (wasRunning && !store.is_running) {
      refreshRegions("toolbar", "list", "case", "correction-list", "correction");
    }
  });

  body.addEventListener("notice", (event) => {
    Alpine.store("review").notice = event.detail.text;
  });

  // htmx does not process response headers on error statuses the same way
  // everywhere, so a refusal's notice is also read from its JSON body.
  body.addEventListener("htmx:responseError", (event) => {
    try {
      Alpine.store("review").notice = JSON.parse(event.detail.xhr.responseText).notice;
    } catch {
      Alpine.store("review").notice = "The server could not do that; see its log.";
    }
  });
});
