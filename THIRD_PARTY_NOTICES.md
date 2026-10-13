# Third-party notices

Virtual Companion's original code is licensed under the [MIT License](LICENSE).
Third-party adaptations and dependencies retain their own copyright and license
terms. This source repository does not distribute dependency binaries, model
weights, tokenizers, voices, character artwork, Live2D assets, or service images.
References to other projects do not grant rights to their code or assets.

## Adapted source

Local paths below are relative to `apps/backend/src/virtual_companion/`.

| Upstream and fixed source | Local files | Adaptation |
|---|---|---|
| [Cyrene-Agent](https://github.com/Playa-Cyrene/Cyrene-Agent/tree/1ce59f20989f106f0f4bb3920db26858bdef7f30), MIT, copyright 2026 Playa | `cognition/evidence_memory.py`, `cognition/exchange_extraction.py`, `cognition/harness.py`, `cognition/joint_source_proposal.py`, `boundary/sqlite_exchanges.py`, `boundary/sqlite_maintenance.py`, `providers/openai_compatible.py` | Python adaptations of social-context extraction, SocialAtom extraction shapes, open-loop/resolve semantics, post-turn extraction, evidence validation, tool-round/checkpoint recovery and provider mapping. Adds project-specific scope, version, cancellation, source-receipt and atomic-commit constraints; excludes donor role prompts, global settings, credentials and media. |
| [DeepSeek recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea), MIT, copyright 2026 DeepSeek | `text/provider_counter.py` | Bounded Python text rendering adapted from `encoding/src/v4/mod.rs` and `protocol/openai/chat_completion/request/convert.rs`. Does not include the upstream runtime or tokenizer artifact. |
| [pi-web-access 0.39.0](https://github.com/nicobailon/pi-web-access/tree/8e812826269624406458531f9231b679547f5a3a), MIT, copyright 2025 Nico Bailon | `providers/public_search_ddg.py`, `providers/public_search_tavily.py`, `providers/pi_exa.py`, `providers/pi_web_fetch.py` | Python ports/adaptations of `duckduckgo.ts`, `tavily.ts`, `exa.ts`, and the HTTP/readability and URL-validation logic in `extract.ts`/`ssrf-protection.ts`. Host-provided transports enforce authorization, deadlines, cancellation, size limits and evidence admission. No donor Node runtime, key discovery, key pool or automatic provider fallback is included. |

The complete MIT notices for these adaptations are reproduced below. The project
uses its own implementations for state storage and delivery contracts; mechanism
references are not claims that upstream applications are bundled.

## Dependencies

Exact Python artifacts and hashes are recorded in [uv.lock](uv.lock); npm versions
and integrity values are recorded in
[apps/desktop/package-lock.json](apps/desktop/package-lock.json). These tables
include optional and development dependencies, not only production imports.
Original dependency distributions retain their own license and notice files.

`readability-lxml` supplies the Python article extractor; it is not a copy of
Pi's Node `@mozilla/readability` package. [tavily-python 0.8.5](https://pypi.org/project/tavily-python/0.8.5/)
([official source](https://github.com/tavily-ai/tavily-python)) is an optional
unmodified SDK used by `providers/tavily_tools.py`; no SDK implementation is
copied into that adapter. Its MIT notice is also reproduced below.
Service access, pricing and usage terms are separate from SDK/source licenses.

### Python lock inventory

| License expression | Packages and locked versions |
|---|---|
| 0BSD | `chardet 7.6.0` |
| Apache-2.0 | `fastembed 0.9.0`, `flatbuffers 25.12.19`, `hf-xet 1.6.0`, `huggingface-hub 1.33.0`, `readability-lxml 0.8.4.1`, `requests 2.34.2`, `tokenizers 0.23.2` |
| Apache-2.0 AND CNRI-Python | `regex 2026.9.29` |
| Apache-2.0 OR BSD-2-Clause | `packaging 26.3` |
| BSD-2-Clause | `pygments 2.21.0` |
| BSD-3-Clause | `click 8.5.0`, `colorama 0.4.6`, `cssselect 1.6.0`, `fsspec 2026.9.0`, `httpcore 1.0.9`, `httpx 0.28.1`, `idna 3.19`, `lxml 6.1.3`, `lxml-html-clean 0.4.5`, `protobuf 7.36.2`, `zstandard 0.25.0` |
| BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | `numpy 2.5.4` |
| MIT | `annotated-types 0.8.0`, `anyio 4.15.1`, `attrs 26.1.0`, `charset-normalizer 3.5.2`, `filelock 4.0.4`, `h11 0.16.0`, `iniconfig 2.3.0`, `jsonschema 4.26.0`, `jsonschema-specifications 2025.9.1`, `loguru 0.7.3`, `mmh3 5.3.1`, `onnxruntime 1.31.0`, `pluggy 1.6.0`, `py-rust-stemmers 0.1.8`, `pydantic 2.13.4`, `pydantic-core 2.46.4`, `pytest 9.1.1`, `pyyaml 6.0.3`, `referencing 0.37.0`, `rpds-py 2026.6.3`, `ruff 0.16.5`, `tavily-python 0.8.5`, `tiktoken 0.14.0`, `typing-inspection 0.4.4`, `urllib3 2.8.0`, `win32-setctime 1.2.0` |
| MIT-CMU | `pillow 12.3.0` |
| MPL-2.0 | `certifi 2026.7.22` |
| MPL-2.0 AND MIT | `tqdm 4.70.1` |
| PSF-2.0 | `typing-extensions 4.16.0` |

License labels were checked against the corresponding installed distributions;
where metadata was incomplete, the original license file was inspected. In
particular, `py-rust-stemmers 0.1.8` includes an MIT license, and the pinned
`chardet 7.6.0` declares 0BSD. `lxml` also carries a bundled `LICENSES.txt`.

### npm lock inventory

| License expression | Packages and locked versions |
|---|---|
| Apache-2.0 | `playwright 1.63.0`, `playwright-core 1.63.0`, `sumchecker 3.0.1` |
| BSD-2-Clause | `@electron-internal/extract-zip 1.0.5` |
| ISC | `graceful-fs 4.2.11`, `semver 7.8.5` |
| MIT | `@electron/get 5.1.0`, `@types/node 24.13.3`, `debug 4.4.3`, `electron 44.2.0`, `env-paths 3.0.0`, `ms 2.1.3`, `progress 2.0.3`, `undici 7.29.1`, `undici-types 7.18.2` |

Development tooling also uses [uv 0.12.9](https://github.com/astral-sh/uv)
(MIT OR Apache-2.0) and CPython 3.13 (PSF-2.0). These runtimes are not vendored.
Electron's npm license does not replace the notices for its bundled Chromium and
other binary components. Anyone distributing an application bundle must preserve
the actual dependencies' licenses/notices and review that bundle's contents;
this source inventory is not a binary-distribution inventory.

## Optional independent SearXNG service

`providers/searxng_search.py` is an original HTTP JSON adapter for a separately
operated SearXNG instance. This repository includes neither SearXNG application
or engine code nor its container, deployment configuration, or private data.
The protocol was checked against [SearXNG commit
ad46ebd7d87666edb5f116c5b274b47f689ac6cd](https://github.com/searxng/searxng/tree/ad46ebd7d87666edb5f116c5b274b47f689ac6cd).
SearXNG remains licensed under
[AGPL-3.0-or-later](https://github.com/searxng/searxng/blob/ad46ebd7d87666edb5f116c5b274b47f689ac6cd/LICENSE);
it is not relicensed by this project's MIT license. Deployment, modification or
redistribution of that service must respect its own applicable license and
corresponding-source requirements. An HTTP integration is not a substitute for
reviewing those requirements if the service or its code is redistributed.

## Full MIT notices

### Cyrene-Agent

```text
MIT License

Copyright (c) 2026 Playa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### DeepSeek recipe

```text
MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### pi-web-access

```text
MIT License

Copyright (c) 2025 Nico Bailon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Tavily Python SDK

```text
MIT License

Copyright (c) 2024 Alpha AI Technologies Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
