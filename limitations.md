# Limitations and safety

AURELIS is an educational research prototype, not a medical device, diagnostic system, or substitute for a qualified healthcare professional. Its outputs must never be used for diagnosis, treatment, patient prioritization, or triage.

The public benchmark may contain label noise, duplicates, source-specific shortcuts, uncertain patient-level separation, and distribution shift. A model trained on this dataset may fail on images from different hospitals, scanners, age groups, or populations. Softmax confidence is not a calibrated probability of disease. The current interface does not implement Grad-CAM, external validation, robust image-quality checks, or a clinically validated abstention policy.

Any future clinical investigation would require independent external validation, subgroup analysis, calibration, prospective evaluation, ethics review, and applicable regulatory oversight. Do not upload identifiable patient data, credentials, or secrets to this repository.
