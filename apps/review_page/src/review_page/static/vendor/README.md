# Vendored browser scripts

The review page serves these files itself, so it works without reaching a CDN.
Each file is the package's own minified build, downloaded unchanged from
jsDelivr on 2026-11-01, at the version the npm registry listed as latest.

| File                             | Package             | Version | Source                                                                  | SHA-256                                                            |
| -------------------------------- | ------------------- | ------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `htmx-2.0.11.min.js`             | `htmx.org`          | 2.0.11  | `https://cdn.jsdelivr.net/npm/htmx.org@2.0.11/dist/htmx.min.js`         | `d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717` |
| `alpinejs-3.17.5.min.js`         | `alpinejs`          | 3.17.5  | `https://cdn.jsdelivr.net/npm/alpinejs@3.17.5/dist/cdn.min.js`          | `b4036e9e9ac22fa6f6ae1ba292e015e1f3e9c85d0d79e73e4c8eb9bb139b522e` |
| `alpinejs-persist-3.17.5.min.js` | `@alpinejs/persist` | 3.17.5  | `https://cdn.jsdelivr.net/npm/@alpinejs/persist@3.17.5/dist/cdn.min.js` | `a5dcd888712ef90663b9aedb131515344cf7a7842cd520e75240824a2d105da4` |

Licences:

- `htmx.org` is under the Zero-Clause BSD licence (`0BSD`), which asks for no
  notice (https://github.com/bigskysoftware/htmx/blob/master/LICENSE).
- `alpinejs` and `@alpinejs/persist` come from one repository under the MIT
  licence, which asks that its copyright and permission notice go with every
  copy. The notice is in `alpinejs-LICENSE.md`, copied from
  https://github.com/alpinejs/alpine/blob/main/LICENSE.md.

To upgrade one, download the new version's file from the same path, give it
the version in its name, change the `<script>` tag in `templates/base.html`,
and update this table, checksum included (`sha256sum <file>`).

The persist plugin is loaded before Alpine's core, as its documentation asks
(https://alpinejs.dev/plugins/persist).
