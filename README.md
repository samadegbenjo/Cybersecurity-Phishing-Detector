# Phishing Detection Research Prototype

A reproducible phishing ML study with **user-triggered active-page URL evaluation** in a Manifest V3 extension, plus the original richer offline feature-record model. The live model uses **39 measured lexical URL features** aligned to the dataset author's extractor; it never fills unavailable DNS/WHOIS fields or reuses the offline model's score. It assesses the current page's URL, **not HTML or page behavior**. A below-threshold result does not guarantee safety.

## Setup and reproduce

Tested on Python 3.12.14 and Node 24.19.0. Chrome is needed for native extension checks. No paid API, remote lookups or cloud service is required.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python model_training.py audit
python model_training.py train --seed 42 --precision-target 0.95
python model_training.py train --mode live_url --output artifacts/live --seed 42 --precision-target 0.95
python model_training.py evaluate
python model_training.py evaluate --output artifacts/live
python -m pytest -q
node --test tests/extension.test.cjs
python scripts/check_integration.py
python scripts/export_record.py
python app.py
```

The JavaScript suite without a server skips its two real HTTP tests. The integration script starts/stops the actual API on port 5000, runs all tests, and verifies real failed-request handling. Train both models first and keep port 5000 free. Training produces model.joblib, metrics.json and splits.json in each output directory. Generated model binaries are not committed; regenerate them locally. Deserialize only trusted, locally trained joblib files. `dataset_preprocessing.py` audits only; no learned preprocessing is fitted on the whole dataset.

## Evaluate the active page

1. Load `extension/` unpacked in Chrome Developer mode at `chrome://extensions`.
2. Copy the extension ID and run `python app.py --extension-id YOUR_EXTENSION_ID`.
3. Open a supported HTTP(S) page and click **Evaluate active page** in the popup.
4. The popup sends that tab's actual URL to the loopback API, which extracts 39 real lexical measurements, applies the separate live pipeline and displays a risk score, threshold, evaluated URL and limitation notice.
5. Below threshold, suspicious and Unable to evaluate are distinct. Stop the API and repeat: expect Unable to evaluate. Native Chrome permissions still need the checklist below.
6. You can also load `artifacts/record.json` in the popup to score an offline 111-feature record. That score is explicitly separate from the active URL.

No content script, DOM scraping, background worker or all-sites permissions are required. Permissions are only `activeTab` and loopback host access; Chrome host patterns cannot limit a host to port 5000, so the application hardcodes that port. Popup-context requests use exact configured extension-origin CORS. URLs are sent only to your local service and are not fetched/resolved or persisted. Text is rendered with `textContent`; timeouts, malformed responses, invalid inputs and unsupported schemes never become benign verdicts. Requests are click-triggered; no continuous monitoring is claimed.

Native Chrome was unavailable in this execution environment. Mocked tab/DOM checks and real Node client → HTTP API → pipeline checks passed, but native loading, activeTab grants, CORS and Local Network Access permission behavior remain unverified. See [docs/BROWSER_CHECKS.md](docs/BROWSER_CHECKS.md). Do not disable browser security protections to claim compatibility.

## Dataset and source evidence

Included dataset.csv: 88,647 rows, 111 numeric predictors, 58,000 legitimate (0), 30,647 phishing (1). No null cells, but many -1 sentinels; 1,438 duplicate rows, six conflicting full vectors and 13 constant predictors. No raw URL, domain identity or timestamp is present. ASN is an infrastructure identifier, and correlations with it may limit generalization.

Every parsed value, row and ordered column matches the author's dataset_full.csv at upstream commit `19674806813c1205d8c12bae99c15c5a735a4ac9`; bytes differ. [reports/provenance.json](reports/provenance.json) records source links and both checksums. Original acquisition history by this project's author remains unrecorded.

Attribution: Grega Vrbančič (2020), *Phishing Websites Dataset*, Mendeley Data V1, DOI [10.17632/72ptz43s9v.1](https://data.mendeley.com/datasets/72ptz43s9v/1), listed as **CC BY 4.0**. Author's [dataset repository](https://github.com/gregavrbancic/Phishing-Dataset); paper DOI [10.1016/j.dib.2020.106438](https://doi.org/10.1016/j.dib.2020.106438). The extension's MIT license does not license third-party data. Dataset values were not changed by this revision.

The author's [URL extractor](https://github.com/gregavrbancic/url-feature-extractor/tree/15545d8186e0efd02fd879aee394dcffb9296933) supplies a computable lexical contract for the separate live model. Eight controlled URLs × 39 values matched isolated upstream pure functions. One important detail: URL counts operate on `netloc + path + params + query + fragment`, excluding scheme and separators, not on the entire visible URL. This verifies selected code semantics; it does not prove extraction revision/procedure for every historical dataset row. Browser normalization may produce distribution shifts.

[docs/LIVE_FEATURES.md](docs/LIVE_FEATURES.md) documents exact rules, parity reproduction, selected columns and live evaluation; [phishing/live_schema.json](phishing/live_schema.json) is the machine-readable ordered live contract. TLD lists, component-absence encodings, email regex and all external lookup features are excluded from the live model. The source clarifies that `qty_tld_url` counts TLD occurrences despite an upstream README “length” description; it remains excluded.

[docs/FEATURES.md](docs/FEATURES.md) and [phishing/feature_schema.json](phishing/feature_schema.json) retain the offline 111-field contract. Offline nulls are explicit and imputed with training medians; external -1 sentinels are assumed missing, while simultaneous directory/file/query -1 blocks are assumed absent components and retained. These are documented assumptions. Missing keys, invalid types/ranges, nonfinite values and all-null records are rejected; all-null training columns fail rather than receive arbitrary zeros. Live requests have **no missing feature values**; zero counts are actual measurements.

## Defensible evaluation

Both modes remove original full-vector conflicting labels and deduplicate remaining full vectors, leaving 87,197 records. Offline: seeded stratified 60/20/20 raw split (52,318/17,439/17,440). Live: retain contradictory labels for selected-feature collisions (1,069 groups), keep equal 39-feature vectors in the same partition via five-fold StratifiedGroupKFold, and designate fold 0 test, fold 1 validation, other folds training (52,319/17,438/17,440). This avoids selected-vector overlap; **domain/campaign and temporal isolation remain unverified**. Raw indices and group rules are saved.

Compare logistic regression and RF using training-only median imputation/missingness indicators; the baseline adds StandardScaler, RF does not need scaling. No PCA or hyperparameter search. Select thresholds using validation only: maximize recall subject to precision >=95% with nonzero positive predictions; report failure and fall back to 0.5 if infeasible. Select the model by validation feasibility, then recall (fallback F1), then average precision. Do not refit on validation. Test only the frozen selected model. Neither mode changed thresholds after test evaluation.

| Mode / candidate | Validation precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|
| Offline logistic regression | .950040 | .773368 | .852649 | .780356 |
| Offline RF (selected) | .950008 | .953591 | .951796 | .495604 |
| Live URL logistic regression | .958964 | .571101 | .715872 | .872246 |
| Live URL RF (selected) | .950148 | .787764 | .861370 | .712133 |

| Test metric | Offline 111 features | Active URL 39 features |
|---|---:|---:|
| Precision | .948626 | .948165 |
| Recall | .956871 | .800787 |
| F1 | .952731 | .868267 |
| Accuracy | .966800 | .915023 |
| PR-AUC (average precision) | .989714 | .959012 |
| ROC-AUC | .994053 | .974528 |
| False-positive rate | .027861 | .023543 |
| Frozen threshold | .4956042402315646 | .712132751457169 |
| Test legitimate / phishing | 11,342 / 6,098 | 11,341 / 6,099 |
| Confusion matrix (actual rows, predicted columns 0/1) | [[11026,316],[263,5835]] | [[11074,267],[1215,4884]] |

All candidates reached the validation precision objective; **both selected models missed 95% precision on test**. Active URL false negatives are materially higher; never use offline metrics to advertise live performance. This is one historical dataset, one seed, no independent contemporary cohort and no confidence intervals. Probabilities are uncalibrated, not guaranteed confidence. The two modes use different splits and input information, so the table is not a controlled feature ablation.

Actual results/versions/seeds/parameters/checksums are in [reports/metrics.json](reports/metrics.json) and [reports/live/metrics.json](reports/live/metrics.json); corresponding splits are in each reports directory. Evaluation verifies dataset, split and model checksums. Models are regenerated, not downloaded; tiny last-digit arithmetic variation is possible. [reports/VALIDATION.md](reports/VALIDATION.md) records what actually ran.

## API and architecture

Offline record → `/predict` → 111-field validation → sentinel handling → saved preprocessing/classifier → phishing class-label lookup → offline threshold.

Active tab URL → `/predict-url` → validated URL parsing → 39 measured lexical features → separate saved pipeline → explicit phishing probability → live threshold → popup score/verdict.

One reliable Python sklearn Pipeline path replaces ONNX. Each artifact bundles preprocessing, schema, threshold and metadata. Phishing probability uses class label **1**, not an assumed output index. Old artifacts are retained under `legacy/` and are not loaded.

- `GET /health`: independent offline/live model readiness; process health alone does not mean a model is ready.
- `GET /metadata`: offline and live schema/model/threshold/limitations.
- `POST /predict`: exactly `{"schema_version":"offline-v1","features":{...111 named fields...}}`.
- `POST /predict-url`: exactly `{"schema_version":"live-url-v1","url":"https://example.invalid/"}`; API extracts features itself.
- `POST /observe`: retained nonclassifying URL observation endpoint (`url-observation-v1`). The extension's main button now uses `/predict-url`.

Errors have `state: unable` and appropriate status, with no benign classification: 422 invalid contract, 503 missing model, 500 prediction failure. Model/schema/feature-order mismatches disable the affected inference path. Bind is loopback-only, debug off; bodies are size limited and arbitrary website Origins rejected. Exact extension-origin CORS is enabled with `--extension-id`; no-Origin CLI clients work. This is a local research API, not an authenticated public service. Use `--live-model` to change the live artifact location.

Official Chrome references: [MV3](https://developer.chrome.com/docs/extensions/develop/migrate/what-is-mv3), [permissions](https://developer.chrome.com/docs/extensions/develop/concepts/declare-permissions), [networking](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests), [Local Network Access](https://developer.chrome.com/blog/local-network-access).

## Contribution and limitations

Samuel Adegbenjo is the project author and maintainer. AI coding assistance was used for this revision's leakage fixes, reproducible evaluation, contracts, local inference, tests and active-URL extension integration. Dataset collection and upstream feature design belong to the cited authors; sklearn classifiers are established algorithms. No novel dataset/algorithm is claimed.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md). The current implementation cannot inspect HTML/forms or visual impersonation and can miss clean-looking malicious URLs, compromised legitimate domains and novel attacks. No domain/time-held-out, calibration, adversarial, independent-source or multi-seed evaluation was conducted. Historical extraction lineage and native browser permissions remain limitations.

Two-sentence application description: “I developed a reproducible phishing-classification prototype with separate offline and active-URL models, training-only preprocessing and validation-selected thresholds. I implemented a versioned lexical feature extractor and Manifest V3 extension integration, tested local inference and failure handling, and documented dataset provenance, benchmark results and generalization limits.”
