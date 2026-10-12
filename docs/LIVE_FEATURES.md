# Active-page URL model contract — live-url-v1

The popup reads the current tab's exact URL under `activeTab` on a user click, then sends it to `/predict-url` on loopback. The API uses the same ordered 39-column subset used during training. No vector is fabricated, no unavailable feature is imputed for the live request, no page is fetched, and no DNS/WHOIS/API call is made. HTTP(S) ASCII-encoded browser URLs are supported; unsupported schemes, non-ASCII inputs, whitespace/control/backslash inputs, malformed hosts/ports and URLs over 8,192 characters fail explicitly.

## Evidence and scope

The author's [URL feature extractor](https://github.com/gregavrbancic/url-feature-extractor/tree/15545d8186e0efd02fd879aee394dcffb9296933) was inspected at commit `15545d8186e0efd02fd879aee394dcffb9296933`, especially `lib/functions.py:start_url` and lexical code in `extract.py`. The new independent implementation reproduces its selected parsing/counting semantics. Eight controlled synthetic URLs × 39 feature values were compared against isolated upstream pure functions, including IPv4, IPv6, ports, userinfo, semicolon parameters, nested question marks, fragments and encoded characters. Expected values, source checksum and upstream commit are in `tests/fixtures/lexical_reference.json`; upstream code is not vendored. No phishing website was visited.

Reproduce the reference comparison (optional; requires a trusted clone of the inspected source):

```bash
git clone https://github.com/gregavrbancic/url-feature-extractor.git ../url-feature-extractor
git -C ../url-feature-extractor checkout 15545d8186e0efd02fd879aee394dcffb9296933
python scripts/verify_lexical_reference.py ../url-feature-extractor
```

This verifies code semantics for the selected subset, **not the complete historical dataset collection chain**. The source is the dataset author's extractor, but its use/revision for every supplied historical row cannot be established from the available records. Browser canonicalization (punycode, encoding, lowercase host, inserted root slash) may also cause distribution shifts relative to historical URLs. Independent browser/time/domain-held-out validation is still required. Results assess lexical URL patterns, not HTML, forms, behavior, visual impersonation or a site's guarantee of safety.

## Exact representation

Use Python `urllib.parse.urlparse` on the received URL, then:

- `lexical = netloc + path + params + query + fragment`. Scheme and separator `?`, `;`, `#` are excluded; separator characters embedded inside a component remain. No percent-decoding is performed. Slashes in the path remain.
- `host = netloc`. Userinfo and port are included, matching upstream; this is intentionally not `hostname`.
- Counts are literal character counts; lengths are Python string lengths (inputs restricted to ASCII). Vowels count a/e/i/o/u case-insensitively. `domain_in_ip` tests the complete netloc with `ipaddress.ip_address`, so bracketed IPv6 and IPv4 with a port are false just as upstream. `server_client_domain` tests either substring case-insensitively.

Ordered names/types/metadata are in `phishing/live_schema.json`. There are 17 character counts for lexical (`. - _ / ? = @ & ! space ~ , + * # $ %`), lexical length, the same 17 netloc counts, vowel count, netloc length, IP-format flag and server/client flag. All 39 values are integer measurements; none may be unavailable or null. Constants are retained to preserve the documented ordered subset; learned sklearn transformations fit only on training data. The API does not accept client-supplied live vectors.

Excluded fields: TLD-reference-list features, directory/file/query absent-component features, email regex, DNS, WHOIS, ASN, timing, SSL certificate, redirects, search indexing and shortener lists. No historical unavailable value is replaced with a fixed zero. A computed zero simply means a counted character or flag was absent.

## Training and evaluation

Train with `python model_training.py train --mode live_url --output artifacts/live`. First remove the original full-vector conflicts and duplicates as for the offline study (87,197 records remain). **Retain** contradictory labels for equal selected lexical vectors: these are real ambiguity, not errors that can be deleted to improve scores. There are 1,069 such groups. Group equal 39-feature vectors together using deterministic pandas hashes and five-fold StratifiedGroupKFold (seed 42); fold 0 is test, fold 1 validation, remaining folds training. Equal lexical vectors cannot cross partitions. This grouping is not proof of domain independence; related URL/domain variants may still occur in multiple partitions.

Training has 52,319 records, validation 17,438, test 17,440. Compare logistic regression with RF, training-only imputation/scaling as applicable, validation-only precision-constrained threshold selection, and no refit on validation or test. Live requests contain complete measurements and need no imputation. Both candidates reached precision >=95% on validation; RF offered greater validation recall and was selected. No other seed, threshold objective or feature subset was tested against this test set before selection.

| Candidate | Validation precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|
| Logistic regression | .958964 | .571101 | .715872 | .872246 |
| RF (selected) | .950148 | .787764 | .861370 | .712133 |

Selected RF test: precision .948165, recall .800787, F1 .868267, accuracy .915023, average precision .959012, ROC-AUC .974528, FPR .023543, threshold .712132751457169. Confusion matrix [[11074,267],[1215,4884]], classes 11,341 legitimate / 6,099 phishing. The 95% precision target was **missed on test**, and the threshold was not changed afterward. More false negatives occur than in the full-feature offline model; its stronger metrics must not be advertised for active browsing. Saved outputs are in `reports/live/`.
