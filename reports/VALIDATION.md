# Executed validation

Environment: Python 3.12.14, dependencies pinned in requirements.txt; Node version recorded below. Clean clone of base commit 0acc412; local branch research-prototype. No pushes, PRs, deployments or live phishing-site visits.

- Full original dataset audit completed; compared every parsed value and row against upstream snapshot. File hashes differ, parsed data matches exactly. Mendeley source and license listing inspected.
- Full-data training completed: logistic regression and RF fit using training data only; thresholds and winner chosen on validation only. RF test results saved in metrics.json. Precision target reached on validation, missed on test. Frozen evaluation command completed and reproduced the recorded values.
- Python: 19 tests passed (schema ordering/types/nulls, sentinels, counts, reversed probability labels, threshold equality, infeasible objective, duplicate/conflict exclusion, train-only imputer/scaler fitting, validation-only threshold calls, frozen evaluation, API error/status/metadata/CORS handling, unavailable model, prediction exception, browser-demo abstention).
- JavaScript: seven tests passed with integration enabled (three verdict states, failed requests/invalid JSON/HTTP/responses, real timeout abort, unsupported schemes, textContent rendering, MV3 permission inventory, shared popup client → real HTTP API → saved Pipeline → classification result).
- Real backend stopped: shared popup client returned `Unable to evaluate: fetch failed`; never returned a benign classification.
- Real URL observation request used only a controlled loopback URL. API did not fetch or resolve that URL and returned abstention plus measured URL properties. Offline-record integration classified an existing CSV record, explicitly separate from the page.

Not executed: native Chrome installation/loading, actual activeTab permission grant, browser UI rendering, native extension CORS behavior or Chrome Local Network Access policies. No native Chrome binary was found. The Node DOM test is a mock, not a browser execution. Manual checklist is in docs/BROWSER_CHECKS.md. Also not verified: historical extractor parity, domain/temporal independence, probability calibration, independent dataset performance, adversarial robustness, multi-seed stability or confidence intervals.
v24.19.0
