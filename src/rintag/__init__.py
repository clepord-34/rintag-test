"""RinTag — POS tagger for the Rinconada Bikol Language."""

from rintag.tagger import tag, TagResult, TaggedToken, reload_model
from rintag.feedback import (
    flag_low_confidence,
    export_feedback,
    FeedbackCollector,
    get_feedback_collector,
    enable_passive_feedback,
    disable_passive_feedback,
    get_passive_feedback,
    export_passive_feedback,
    clear_passive_feedback,
    collect_feedback,
)

__version__ = "0.1.0"
__all__ = [
    "tag",
    "TagResult",
    "TaggedToken",
    "reload_model",
    "flag_low_confidence",
    "export_feedback",
    "FeedbackCollector",
    "get_feedback_collector",
    "enable_passive_feedback",
    "disable_passive_feedback",
    "get_passive_feedback",
    "export_passive_feedback",
    "clear_passive_feedback",
    "collect_feedback",
]
