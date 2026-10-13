# Feedback contracts

`FeedbackEvent v1`, `FeedbackExport v1` and `FeedbackAnalysis v1` are the versioned contracts for feedback capture, explicit export and derived analysis.

- `FeedbackEvent` is the local raw user reaction. `quick_tags` remains optional and `freeform_reaction` is the first-class free-text field; at least one must be present.
- `FeedbackExport` is a separate, user-selected local artifact. Its revision list is frozen in one repository transaction and it requires preview, redaction and external-processing confirmation.
- `FeedbackAnalysis` is a separate derived artifact. It binds the export hash, model and prompt hash, cites feedback IDs, and always requires human confirmation.

Feedback capture and export do not upload an artifact or invoke a model provider. The JSON schemas are data contracts; the Python domain validates the same invariants at runtime.
