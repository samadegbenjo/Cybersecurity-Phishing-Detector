# Phishing Detection Research Prototype

A reproducible **offline** phishing classification study, with a Manifest V3 browser demonstration. This project does **not** currently classify the active webpage: the supplied CSV contains precomputed features without raw URLs, and exact extraction equivalence has not been established. The demonstration reports real URL observations and an explicit **Unable to evaluate** state. Users can separately load an offline feature-record JSON into the popup to exercise API → preprocessing → model → display.

## Reproduce

Tested on Python 3.12.14 and Node 24.19.0. Node is only needed for JavaScript tests; Chrome is needed for manual extension checks. No paid services or remote lookups are required.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python model_training.py audit --dataset dataset.csv
python model_training.py train --dataset dataset.csv --output artifacts --seed 42 --precision-target 0.95
python model_training.py evaluate --dataset dataset.csv --output artifacts
python -m pytest -q
node --test tests/extension.test.cjs
python scripts/check_integration.py       # starts/stops local API; checks real HTTP and failure
python scripts/export_record.py --row 0 --output artifacts/record.json
python app.py                           # loopback 127.0.0.1:5000; debug disabled
```

The JavaScript suite without a running server intentionally skips its one live HTTP test; the integration command runs all seven tests. It requires port 5000 to be free. The CSV already included here is sufficient for training. Training regenerates `artifacts/model.joblib`, `metrics.json`, and `splits.json`; generated models are not committed. Load only locally trained/trusted joblib files, which can execute code during deserialization. `dataset_preprocessing.py` now audits only and cannot fit preprocessing on the whole dataset.

## Dataset and feature contract

The included CSV has 88,647 rows, 111 numeric predictors, and labels 0=legitimate (58,000), 1=phishing (30,647). It has no null cells, but many -1 sentinels, 1,438 duplicate rows, six conflicting feature vectors, and 13 constant predictors. No URL, domain identifier or observation timestamp is present. ASN is a numeric infrastructure identifier; its treatment as a number and shared infrastructure can limit generalization.

Parsed values, ordered columns and rows **exactly match** the author's `dataset_full.csv` at upstream commit `19674806813c1205d8c12bae99c15c5a735a4ac9`; raw file bytes differ. Both SHA-256 hashes and source evidence are in [reports/provenance.json](reports/provenance.json). Original acquisition history by this project's author is not recorded.

Dataset attribution: Grega Vrbančič (2020), *Phishing Websites Dataset*, Mendeley Data V1, DOI [10.17632/72ptz43s9v.1](https://data.mendeley.com/datasets/72ptz43s9v/1), listed as **CC BY 4.0**. Upstream [repository](https://github.com/gregavrbancic/Phishing-Dataset) and associated paper DOI [10.1016/j.dib.2020.106438](https://doi.org/10.1016/j.dib.2020.106438). The original extension's MIT license is not a license for this third-party dataset. No dataset values were changed by this revision.

[docs/FEATURES.md](docs/FEATURES.md) lists every predictor in order, numeric types, upstream descriptions, unavailable extraction details and missing-value policy. The shared machine-readable contract is [phishing/feature_schema.json](phishing/feature_schema.json). Upstream descriptions are available, but exact parsing, URL normalization, external lookup procedures and sentinel rules are not verified. In particular, `qty_tld_url` is described upstream as TLD length but its observed values merit further investigation. Lexical features appear computable in principle; reconstructing them without raw URLs cannot validate training/inference parity. DNS, WHOIS, certificate, index and timing fields require external collection unavailable to a simple page script. Therefore **no fabricated browser feature vector is sent to the classifier**.

Offline null values are explicitly marked and imputed using training medians, with training-learned missingness indicators. External -1 sentinels are assumed unavailable and normalized to missing. Simultaneous -1 blocks in directory/file/query features are assumed absent components and retained; these interpretations are assumptions, not upstream extractor verification. Absent keys, nonnumeric values, nonfinite values, unsupported negative values and all-missing records are rejected. All-missing training columns also cause an error rather than arbitrary zero filling.

## Evaluation protocol and actual results

Remove all vectors with conflicting labels, then deduplicate remaining feature vectors before a seeded stratified 60/20/20 raw split. This excludes 1,450 rows and leaves 87,197: training 52,318, validation 17,439, test 17,440. Exact feature duplicates cannot cross splits. **Domain/campaign overlap and temporal leakage remain unmeasurable without identifiers.** These are random-record benchmark results, not deployment estimates.

A logistic-regression baseline uses training-only median imputation, missingness indicators and StandardScaler. Random Forest uses the same imputation without scaling. PCA is omitted: no demonstrated benefit supports its use, and RF does not need scaling. Fixed model configurations are reported; there is no hyperparameter search. Select each candidate's threshold to maximize validation recall subject to precision >=95% and at least one positive prediction; if infeasible, explicitly report failure and use 0.5. Select the model using validation feasibility, then recall (fallback F1), then average precision. Do not refit on validation after selection. Evaluate only the selected frozen model on test.

| Candidate | Validation precision | Validation recall | Validation F1 | Threshold |
|---|---:|---:|---:|---:|
| Logistic regression | 0.95004 | 0.77337 | 0.85265 | 0.780356 |
| Random Forest (selected) | 0.95001 | 0.95359 | 0.95180 | 0.495604 |

| Selected-model test metric | Value |
|---|---:|
| Precision | 0.948626 |
| Recall | 0.956871 |
| F1 | 0.952731 |
| Accuracy | 0.966800 |
| PR-AUC (average precision; not trapezoidal area) | 0.989714 |
| ROC-AUC | 0.994053 |
| False-positive rate | 0.027861 |
| Threshold | 0.4956042402315646 |
| Test legitimate / phishing | 11,342 / 6,098 |

Confusion matrix (rows actual 0/1, columns predicted 0/1): `[[11026, 316], [263, 5835]]`. Both candidates achieved the precision target on validation. **The target was not achieved on test (94.86% precision)**; no adjustment was made using test data. Scores are uncalibrated model probabilities, not guaranteed confidence. These measurements cover one seed, one dataset and no independent contemporary, domain-held-out or time-held-out cohort.

[reports/metrics.json](reports/metrics.json) contains actual run outputs, seed, dependency versions, checksum, candidate parameters, class counts, split details and all required metrics. [reports/splits.json](reports/splits.json) records original CSV indices. Re-evaluation checks dataset, model and split-manifest checksums; floating-point ranking metrics may vary at insignificant last digits across executions/platforms. Model hash refers to the generated local artifact, which must be retrained rather than downloaded from this repository. [reports/VALIDATION.md](reports/VALIDATION.md) distinguishes execution from assumptions.

## API and architecture

`dataset.csv → raw split → training-only sklearn Pipeline → validation selection → frozen model.joblib → test metrics`.

`offline record → /predict → ordered schema validation → sentinel handling → saved Pipeline → class-label lookup → thresholded score`.

`activeTab URL → /observe → URL measurements → Unable to evaluate (no model prediction)`.

One Python inference path replaces ONNX to eliminate an unverified conversion boundary. Preprocessing, schema and threshold are saved together. Phishing probability is indexed by the explicit class label **1**, not an assumed output position. Old PCA/scaler/ONNX artifacts are retained in `legacy/` for historical inspection only.

- `GET /health`: process status and model readiness (health alone does not mean inference is ready).
- `GET /metadata`: full schema, model name, threshold and limitations.
- `POST /predict`: `{"schema_version":"offline-v1","features":{"qty_dot_url":3,...}}`; exactly 111 named fields, values numeric or null. Unknown fields and old positional arrays are rejected. Below-threshold results mean safety is unverified.
- `POST /observe`: `{"schema_version":"url-observation-v1","url":"http://127.0.0.1:8000/page"}`. No outgoing fetch, DNS access or prediction.

Errors return an explicit `state: unable`, an appropriate HTTP status and no benign classification. Inference returns 503 if training has not produced a valid model, 422 for contract violations and 500 for prediction failures. Request bodies are size limited. Arbitrary website Origins are rejected; exact extension-origin CORS can be enabled with `python app.py --extension-id YOUR_EXTENSION_ID`. Command-line clients without an Origin are supported. This is a local research service, not an authenticated public API.

## Browser demonstration

1. Load `extension/` unpacked from `chrome://extensions` in Developer mode.
2. Copy its ID and run `python app.py --extension-id YOUR_EXTENSION_ID`.
3. Visit a controlled HTTP(S) page; click **Observe current URL**. Expect real length/dot/@/HTTPS observations, score **unavailable**, the offline threshold if a model is ready, and **Unable to evaluate**.
4. Export `artifacts/record.json` using the command above and choose it in the popup. This explicitly evaluates an **offline record**, never the active page.
5. Stop the API and repeat either operation. Expect **Unable to evaluate**, never a below-threshold result.

MV3 permissions are only `activeTab` and loopback `host_permissions`. Chrome host match patterns cannot restrict a host to port 5000; application requests are hardcoded to that port. No all-sites content scripts, scripting permission or background worker is needed. All networking occurs in the popup extension context, not in page scripts. UI renders external strings with `textContent`. Timeouts, bad HTTP, invalid JSON/responses and unsupported schemes all produce error states. Safe/suspicious/error are represented separately; the apparent safe state is deliberately labeled **below threshold (safety unverified)**.

Migration references: official Chrome [MV3 migration](https://developer.chrome.com/docs/extensions/develop/migrate/what-is-mv3), [host permissions](https://developer.chrome.com/docs/extensions/develop/concepts/declare-permissions), [extension networking](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests), and [Local Network Access](https://developer.chrome.com/blog/local-network-access). Extension-context host permission and the backend's exact Origin allowlist address cross-origin access; they do not establish that a given Chrome version/policy will permit local-network access. Native Chrome and its permission behavior were **not executed** in this environment. Keep any blocked request as an error; do not disable browser security flags to claim successful validation. See the controlled-page checklist in [docs/BROWSER_CHECKS.md](docs/BROWSER_CHECKS.md).

## Contribution and limits

Samuel Adegbenjo is the project author and maintainer. This revision was implemented with AI coding assistance: it repairs leakage, formalizes the feature contract, adds reproducible comparison/evaluation, robust inference and tests, and replaces unsupported live-detection claims with an honest demonstration. Dataset collection and original feature engineering belong to the cited dataset authors; the classifiers are established sklearn algorithms. This is engineering and evaluation work, not a claim of a novel algorithm or dataset.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Generalization to new campaigns, prevalence shifts and contemporary browsing remains unknown. False positives and false negatives are expected. Null-heavy records may be out of distribution even when syntactically valid. No calibrated uncertainty, explanation method, production robustness, continuous monitoring or autonomous protection is claimed.

Two-sentence application description: “I developed a reproducible offline phishing-classification prototype that compares logistic regression and Random Forest using training-only preprocessing and validation-selected thresholds. I added a versioned feature contract, robust local inference, automated tests and a Manifest V3 browser demonstration, while documenting dataset provenance and the limits of live detection and generalization.”
