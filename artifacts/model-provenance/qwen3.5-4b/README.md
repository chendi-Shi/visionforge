# Qwen3.5:4b model provenance

Public model metadata captured on 2026-10-15 from local Ollama 0.40.4. This appendix includes no weights, request data, private filesystem paths, credentials, or signed download URLs.

## Original bytes and the API digest

`registry-manifest.json` is the **exact 1,088-byte original registry object retained in the local Ollama blob store**, copied without JSON re-serialization. Its SHA-256 equals the `/api/tags` model digest. Its bytes were independently compared with the raw response from the [official registry manifest endpoint](https://registry.ollama.ai/v2/library/qwen3.5/manifests/4b); they were identical.

`local-wrapper.json` is the exact 1,407-byte local `manifests/registry.ollama.ai/library/qwen3.5/4b` file. Ollama 0.40.4 retains the five original runtime layers and adds references to a manifest list and the original registry object. The wrapper has a different SHA-256 and must not be described as the API-addressed registry object. Removing its added layers and serializing JSON would not preserve the original registry bytes.

Both objects have identical config, ordered runtime layers, runner (`llamacpp`), and format (`gguf`). The locally retained manifest list was separately checked against its declared length and SHA-256; its `llamacpp`/`gguf` entry selects the registry digest below. That list and the model weights are not included in this metadata-only appendix.

`provenance.json` is a generated receipt recording byte lengths, hashes, and the `local_retained_registry_blob` origin; it is not original registry output. This appendix does not change the frozen temporal experiment model or inference protocol.

## Checksums

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `registry-manifest.json` | 1088 | `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` |
| `local-wrapper.json` | 1407 | `e0b8c85c75aad2573ff49bec177ebb1013772fba665a53d64c04266439a2222d` |
| `provenance.json` | 446 | `73246ae75fe39413d605228b6bf10e1b2af1d6e41b0d5533ff9c8de134d5416d` |

`SHA256SUMS` additionally covers this README. For example, run `Get-FileHash registry-manifest.json -Algorithm SHA256` in PowerShell from this directory. Checksums demonstrate file consistency; authenticity also depends on the independently recorded official model source and experiment pins.

The repaired official model layer has SHA-256 `a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50` and length 2,648,593,216 bytes. The downloader verified the assembled object and re-read it from disk for a second SHA-256 check before import. All imported bytes came from the official registry/CDN. Selecting a currently DNS-advertised CDN address preserved the original TLS hostname and certificate verification. No alternate quantization or mirror weight was substituted. The projector and other layers were finalized with the normal Ollama pull operation.
