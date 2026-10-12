# Superseded artifacts — retained only for historical inspection

These unchanged files came from the original repository. The old training scripts fitted scaler/PCA before splitting and chose a threshold using test labels. Their metrics, provenance and serialization compatibility were not validated. No current script imports these artifacts; regenerate `artifacts/model.joblib` with the new training command. ONNX is superseded by one Python sklearn Pipeline so no unverified conversion path remains.
