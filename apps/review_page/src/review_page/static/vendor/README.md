# Vendored browser scripts

The review page serves these files itself, so it works without reaching a CDN.
Each file is the package's own minified build, downloaded unchanged from
jsDelivr. htmx, Alpine and the persist plugin were downloaded on 2026-11-01,
at the version the npm registry listed as latest. The anchor and focus plugins
were downloaded on 2026-11-07, at Alpine's version.

| File                             | Package             | Version | Source                                                                  | SHA-256                                                            |
| -------------------------------- | ------------------- | ------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `htmx-2.0.11.min.js`             | `htmx.org`          | 2.0.11  | `https://cdn.jsdelivr.net/npm/htmx.org@2.0.11/dist/htmx.min.js`         | `d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717` |
| `alpinejs-3.17.5.min.js`         | `alpinejs`          | 3.17.5  | `https://cdn.jsdelivr.net/npm/alpinejs@3.17.5/dist/cdn.min.js`          | `b4036e9e9ac22fa6f6ae1ba292e015e1f3e9c85d0d79e73e4c8eb9bb139b522e` |
| `alpinejs-persist-3.17.5.min.js` | `@alpinejs/persist` | 3.17.5  | `https://cdn.jsdelivr.net/npm/@alpinejs/persist@3.17.5/dist/cdn.min.js` | `a5dcd888712ef90663b9aedb131515344cf7a7842cd520e75240824a2d105da4` |
| `alpinejs-anchor-3.17.5.min.js`  | `@alpinejs/anchor`  | 3.17.5  | `https://cdn.jsdelivr.net/npm/@alpinejs/anchor@3.17.5/dist/cdn.min.js`  | `ab491df26e53b7ff6e502e0fa403b1960ce742a1e4498f209d8247ecf3e9e486` |
| `alpinejs-focus-3.17.5.min.js`   | `@alpinejs/focus`   | 3.17.5  | `https://cdn.jsdelivr.net/npm/@alpinejs/focus@3.17.5/dist/cdn.min.js`   | `ea7e215444f5110619549621cd0760cedfe273f708b144d4e658a87b702555f9` |

Licences:

- `htmx.org` is under the Zero-Clause BSD licence (`0BSD`), which asks for no
  notice (https://github.com/bigskysoftware/htmx/blob/master/LICENSE).
- `alpinejs` and its plugins (`@alpinejs/persist`, `@alpinejs/anchor`,
  `@alpinejs/focus`) come from one repository under the MIT licence, which asks
  that its copyright and permission notice go with every copy. The notice is in `alpinejs-LICENSE.md`, copied from
  https://github.com/alpinejs/alpine/blob/main/LICENSE.md.

To upgrade one, download the new version's file from the same path, give it
the version in its name, change the `<script>` tag in `templates/base.html`,
and update this table, checksum included (`sha256sum <file>`).

The plugins are loaded before Alpine's core, as their documentation asks
(https://alpinejs.dev/plugins/persist, https://alpinejs.dev/plugins/anchor,
https://alpinejs.dev/plugins/focus).

- The anchor plugin positions the named entry picker's panel against its
  trigger. It uses fixed positioning, because the page table scrolls.
- The focus plugin moves focus between the picker's options with the arrow
  keys.
