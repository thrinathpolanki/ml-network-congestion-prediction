# 🚀 Phase 17: Logging and Error Handling

**Project:** Network Congestion Prediction & Alert System

---

## 📁 What Was Added or Changed

| File | Change |
| --- | --- |
| `src/utils/logging_config.py` | New `redact()`; the filter now also covers tracebacks |
| `src/api/main.py` | Calls `setup_logging` at startup; logs startup, validation errors and HTTP errors; startup summary |
| `tests/test_logging.py` | 10 tests |
