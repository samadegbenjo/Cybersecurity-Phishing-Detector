# Model card — offline phishing classifier

**Intended use:** reproducible cybersecurity ML research, feature-record scoring and pipeline demonstrations. Not browser protection, a phishing-site scanner, or evidence that a URL is safe.

**Model:** selected 200-tree Random Forest, min_samples_leaf=2, seed=42; training-only median imputation with missingness indicators; no PCA or scaling. Positive class 1=phishing, negative class 0=legitimate. Threshold 0.4956042402315646 selected on validation. Classifier scores are uncalibrated.

**Data:** Vrbančič's 2020 full dataset, 88,647 supplied rows and 111 precomputed numeric predictors, verified parsed equality to the author's repository snapshot. Mendeley lists CC BY 4.0. No raw URLs, timestamps or domain identities. All conflicting-label feature vectors removed and remaining vectors deduplicated; 87,197 retained. Sentinel interpretation and exact extraction details remain limitations. See FEATURES.md and provenance report.

**Conditions:** stratified random-record train/validation/test (52,318/17,439/17,440), one seed; duplicates isolated but related domains/campaigns cannot be identified. Logistic regression baseline compared using validation only. No test-based threshold adjustment or model selection. Validation precision reached 95%; test precision did not.

**Test:** precision .948626, recall .956871, F1 .952731, accuracy .966800; average precision .989714, ROC-AUC .994053; FPR .027861. Confusion matrix [[11026,316],[263,5835]]. 316 false positives could disrupt legitimate use; 263 false negatives could expose users to phishing if mistaken for safety. Inference labels low scores “below threshold,” not “safe.”

**Generalization:** unknown on contemporary sites, previously unseen domains, targeted phishing, non-English campaigns or different phishing prevalence. No domain-held-out, temporal, independent-source or adversarial evaluation. Numeric ASN can encode infrastructure correlations; null-heavy inputs can differ from the benchmark. Missing-field syntax checks cannot verify semantic compatibility.

**Browser:** URL length and literal punctuation counts/HTTPS are observations only. No verified mapping of live measurements to all historical dataset features. No active-page classification; offline JSON input is distinct. Native Chrome permission and loading behavior still require manual verification.

**Reproducibility:** pinned Python requirements, saved checksums, split indices and versions, CLI evaluation and automated Python/JS tests. Repeated floating arithmetic may affect insignificant final digits; reported discrimination metrics are specific to this cohort. No confidence intervals or multi-seed stability study was performed.

**Authorship:** Samuel Adegbenjo's prototype, revised with AI coding assistance. Dataset and extraction design attributed to upstream authors; existing sklearn algorithms used without algorithm-novelty claims.
