# Model card — offline and active-URL phishing classifiers

**Intended use:** reproducible cybersecurity ML research, offline feature-record scoring, and experimental user-triggered active-page **URL** assessment. Not autonomous browsing protection or a guarantee of site safety. The live path does not inspect HTML, forms, screenshots, credentials or page behavior.

**Models:** separate 200-tree Random Forests, min_samples_leaf=2, seed 42, compared with logistic regression. Training-only preprocessing; no PCA. Positive label 1=phishing. Offline threshold .4956042402315646; live threshold .712132751457169, each selected on validation only. Scores uncalibrated.

**Data:** Vrbančič's 2020 full dataset, 88,647 supplied rows, 111 precomputed predictors; verified parsed equality to author snapshot. CC BY 4.0 listed on Mendeley. No raw URL/domain identity/timestamp; original full conflicts/duplicates excluded, leaving 87,197 rows. Offline sentinel meanings remain assumptions. Live uses only 39 complete lexical predictors; no unavailable browser value is substituted or imputed.

**Extraction evidence:** selected normalization/counting semantics match the author's extractor at commit 15545d8186e0efd02fd879aee394dcffb9296933 on eight controlled synthetic URLs × 39 values. API extracts from actual activeTab URL without outgoing network requests. This is source-code parity, not proof of every historical row's extraction lineage. Browser canonicalization can shift feature distributions; unsupported inputs fail explicitly. See LIVE_FEATURES.md.

**Evaluation:** offline stratified record 60/20/20; live five-fold stratified lexical-vector groups, with equal vectors isolated across partitions and 1,069 contradictory-label groups retained. Related domains and campaigns cannot be fully isolated. Model/threshold selection uses validation only; test is frozen. Both validation precision targets reached, both test precision targets missed.

| Test metric | Offline | Active URL |
|---|---:|---:|
| Precision | .948626 | .948165 |
| Recall | .956871 | .800787 |
| F1 | .952731 | .868267 |
| Accuracy | .966800 | .915023 |
| Average precision | .989714 | .959012 |
| ROC-AUC | .994053 | .974528 |
| FPR | .027861 | .023543 |
| Confusion matrix | [[11026,316],[263,5835]] | [[11074,267],[1215,4884]] |

**Errors and impacts:** live model has 267 false positives (legitimate URLs may be flagged) and 1,215 false negatives (malicious URLs may be below threshold). Clean-looking URLs, compromised domains, encoded attacks and concept drift remain concerns. Neither HTTPS nor a low score proves safety. The popup labels low scores “below threshold (safety unverified)” and service/input errors “Unable to evaluate.”

**Generalization:** unknown on contemporary, novel-domain/campaign, non-English or adversarial cohorts, different prevalence or browser canonicalization patterns. No calibration, confidence intervals, multi-seed stability or independent-source validation. Full-feature offline metrics are not evidence of live detection performance; split protocols differ.

**Execution:** complete dataset training and frozen evaluation for both modes ran; controlled extractor parity and real loopback client/API/pipeline flow passed, including actual backend failure. Native Chrome loading, permission grants, CORS/LNA policies and browser rendering were not executed; tab/DOM tests are mocks. See BROWSER_CHECKS.md and reports/VALIDATION.md.

**Authorship:** Samuel Adegbenjo's project, revised with AI coding assistance. Dataset and feature design attributed upstream; established sklearn algorithms used without novelty claims.
