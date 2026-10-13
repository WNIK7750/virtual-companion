"""Host-side risk routing for observed text units; it never approves or refuses.

An ordinary no-tool turn should not pay a fixed model review, but the host cannot
decide semantic self-containment by itself either. This scan therefore answers one
narrow question: does this candidate carry a risk *form* the host can point at, so
that a model opinion has to be spent before anything is displayed?

Bounded by design:

* only forms the host can verify are recognized -- an unclosed group or quote, a
  dangling separator or connective, a condition or enumeration promise left open, a
  leading reference with no antecedent in the user text or the displayed prefix, a
  quote attributed to the user that is not verbatim, a polarity flip on a shared
  span, and the explicit user output constraints from ``user_constraints``;
* the result is not a presentation decision: publication still needs its own
  receipt, so this module can never become punctuation-based release;
* unflagged only means "no recognized risk form", not "verified": the policy that
  would publish unflagged text without a model opinion needs its own evidence and
  approval. Flagged is the conservative direction -- it spends an opinion that may
  not have been strictly necessary.
"""

import re
from dataclasses import dataclass

from .user_constraints import NUMERALS, constraint_issues, recognize_user_constraints

VERSION = "unit-risk-scan.v1"
MAX_SCAN_CHARS = 64000
RISK_CODES = frozenset({
    "dangling_ending",
    "dangling_connective",
    "unclosed_condition",
    "open_group",
    "unbound_reference",
    "enumeration_promise",
    "requested_count_unmet",
    "non_verbatim_user_quote",
    "polarity_conflict",
    "constraint_violation",
    "unparsed_constraint",
})
SOURCES = ("candidate", "shown_text", "user_text")
TRAILING_BLANKS = " \t\r\n"
SENTENCE_TAIL = "。！？!?…"
DANGLING_ENDINGS = (",", "，", ":", "：", ";", "；", "、", "—")
CONNECTIVES = (
    "不过", "但是", "可是", "而且", "所以", "因此", "因为", "由于", "如果", "要是",
    "假如", "倘若", "即使", "虽然", "只是", "就是", "只有", "只要", "还有", "另外",
    "首先", "其次", "然后", "接着", "以及", "并且", "或者", "而是", "至于", "例如",
    "比如", "不过如此", "但", "而", "却",
)
CONDITION_MARKERS = (
    "如果", "要是", "假如", "倘若", "即使", "虽然", "因为", "由于", "只要", "只有",
    "一旦", "除非", "为了",
)
CLAUSE_SEPARATORS = "，,、；;：:"
PROMISE_ENDINGS = (
    re.compile(r"(?:我)?先(?:说|讲|提|聊|来)(?:一下|说)?"),
    re.compile(r"(?:下面|接下来)(?:我)?(?:说|讲|来|介绍|聊)"),
    re.compile(r"(?:从)?第[一二两三四五六七八九十\d]{1,2}(?:点|个|条|方面)$"),
    re.compile(r"首先$"),
)
REFERENCE_HEADS = (
    "他们", "她们", "它们", "这个", "那个", "这些", "那些", "这样", "那样",
    "这件事", "这个人", "它", "他", "她", "该", "此", "其",
)
NEGATIONS = ("不", "没", "别", "无", "未", "非", "莫", "勿", "甭")
NEGATION_HEADS = ("没有", "不是", "不会", "不能", "不用", "不要", "不再", "并没")
COUNT_PATTERN = re.compile(
    r"([一二两三四五六七八九十\d]{1,2})\s*(?:个|条|点|种|件|方面)\s*(?!字|数)")
# "三点" is a clock time far more often than a three-item request, so a time word
# before the number or a clock word after it disqualifies the count.
TIME_BEFORE = ("上午", "下午", "早上", "晚上", "中午", "凌晨", "夜里", "傍晚", "次日")
TIME_AFTER = ("钟", "半", "整", "多", "左右", "以前", "之后", "前后")
NUMBERED_ITEM = re.compile(r"(?:^|\n)\s*(?:\d{1,2}[.、)）]|[（(]\d{1,2}[）)])")
ENUMERATORS = (
    "首先", "其次", "再次", "最后", "另外", "再者", "接下来",
    "第一", "第二", "第三", "第四", "第五", "第六",
    "一是", "二是", "三是", "其一", "其二", "其三",
)
QUOTE_ATTRIBUTION = re.compile(
    r"你(?:刚才|刚刚)?说(?:的|过|的是)?[：:]?\s*[“\"']([^”\"']{2,64})[”\"']")
SHARED_MIN, SHARED_MAX, SHARED_CAP = 2, 4, 64
# A short shared span with the user's own text is usually a topic word ("喝茶" in
# "你平时喝茶吗？"), so user-text polarity needs a longer span than the displayed
# prefix, where a flipped 2-character span already means contradicting what was shown.
SHARED_MIN_USER = 3
GROUP_PAIRS = {
    "“": "”", "‘": "’", "「": "」", "『": "』", "《": "》",
    "（": "）", "(": ")", "【": "】", "[": "]",
}
GROUP_CLOSES = frozenset(GROUP_PAIRS.values())


@dataclass(frozen=True)
class RiskSignal:
    code: str
    span: str
    source: str = "candidate"

    def __post_init__(self):
        if self.code not in RISK_CODES:
            raise ValueError("Unknown risk code")
        if self.source not in SOURCES:
            raise ValueError("Unknown risk source")
        if not isinstance(self.span, str) or not self.span.strip() or len(self.span) > 64:
            raise ValueError("Risk provenance required")


@dataclass(frozen=True)
class RiskScan:
    flagged: bool
    signals: tuple[RiskSignal, ...]
    version: str = VERSION

    def __post_init__(self):
        if type(self.flagged) is not bool or self.flagged != bool(self.signals):
            raise ValueError("Flag must match the reported signals")

    @property
    def codes(self):
        return tuple(dict.fromkeys(signal.code for signal in self.signals))


def has_open_group(text):
    """True when quotes/brackets are unbalanced or closed in the wrong order."""
    if not isinstance(text, str):
        raise ValueError("Text required")
    stack = []
    for char in text:
        if char in GROUP_PAIRS:
            stack.append(GROUP_PAIRS[char])
        elif char in GROUP_CLOSES and (not stack or stack.pop() != char):
            return True
    return bool(stack)


def _is_cjk(char):
    return "\u4e00" <= char <= "\u9fff"


def _signal(codes, code, span, source="candidate"):
    text = span.strip()[:64]
    if text:
        codes.append(RiskSignal(code, text, source))


def _trailing_signals(text, signals):
    tail = text.rstrip(TRAILING_BLANKS)
    if not tail:
        return
    if tail.endswith(DANGLING_ENDINGS):
        _signal(signals, "dangling_ending", tail[-1])
    for word in sorted(CONNECTIVES, key=len, reverse=True):
        if tail.endswith(word) and not tail.endswith(SENTENCE_TAIL):
            _signal(signals, "dangling_connective", word)
            return


def _condition_signals(text, signals):
    for marker in CONDITION_MARKERS:
        start = text.rfind(marker)
        if start < 0:
            continue
        rest = text[start + len(marker):]
        if any(char in rest for char in SENTENCE_TAIL):
            continue
        closed = any(
            rest[index] in CLAUSE_SEPARATORS
            and rest[index + 1:].strip(CLAUSE_SEPARATORS + TRAILING_BLANKS)
            for index in range(len(rest))
        )
        if not closed:
            _signal(signals, "unclosed_condition", text[start:start + 64])
        return


def _reference_signals(text, user_text, shown_text, signals):
    stripped = text.lstrip(TRAILING_BLANKS + CLAUSE_SEPARATORS)
    for head in REFERENCE_HEADS:
        if not stripped.startswith(head):
            continue
        if head[0] in user_text or (shown_text and head[0] in shown_text):
            return
        _signal(signals, "unbound_reference", head)
        return


def _promise_signals(text, signals):
    tail = text.rstrip(TRAILING_BLANKS + SENTENCE_TAIL + "”’")
    for pattern in PROMISE_ENDINGS:
        found = pattern.search(tail)
        if found and found.end() == len(tail):
            _signal(signals, "enumeration_promise", found.group(0))
            return


def _number(value):
    if value.isdigit():
        return int(value)
    return sum(NUMERALS.get(char, 0) for char in value)


def _delivered_items(text):
    """Distinct delivered items: numbered markers or enumeration words, never both.

    Taking the larger count keeps the check conservative: a promise word plus a
    numbered list never inflates how much the answer actually delivered.
    """
    return max(len(NUMBERED_ITEM.findall(text)),
               sum(1 for enumerator in ENUMERATORS if enumerator in text))


def _count_signals(user_text, text, signals):
    requested = []
    for match in COUNT_PATTERN.finditer(user_text):
        if not 2 <= _number(match.group(1)) <= 9:
            continue
        before = user_text[max(0, match.start() - 2):match.start()]
        after = user_text[match.end():match.end() + 2]
        if any(before.endswith(word) for word in TIME_BEFORE):
            continue
        if any(after.startswith(word) for word in TIME_AFTER):
            continue
        requested.append(match)
    if requested:
        wanted = _number(requested[-1].group(1))
        if _delivered_items(text) < wanted:
            _signal(signals, "requested_count_unmet", requested[-1].group(0), "user_text")


def _quote_signals(user_text, text, signals):
    for found in QUOTE_ATTRIBUTION.finditer(text):
        quoted = found.group(1)
        if quoted not in user_text:
            _signal(signals, "non_verbatim_user_quote", quoted)
            return


def _negated_before(text, index):
    if index <= 0:
        return False
    if text[index - 1] in NEGATIONS:
        return True
    return index >= 2 and text[index - 2:index] in NEGATION_HEADS


def _shared_spans(text, other, minimum):
    spans = []
    for size in range(SHARED_MAX, minimum - 1, -1):
        for index in range(len(text) - size + 1):
            piece = text[index:index + size]
            if not any(_is_cjk(char) for char in piece):
                continue
            if any(index < end and start < index + size for start, end in spans):
                continue
            if other.find(piece) < 0:
                continue
            spans.append((index, index + size))
            if len(spans) >= SHARED_CAP:
                return spans
    return spans


def _polarity_signals(text, other, source, signals, minimum):
    for start, end in _shared_spans(text, other, minimum):
        found = other.find(text[start:end])
        if _negated_before(text, start) != _negated_before(other, found):
            _signal(signals, "polarity_conflict", text[start:end], source)
            return


def _constraint_signals(user_text, text, signals):
    constraints, unparsed = recognize_user_constraints(user_text)
    issues = constraint_issues(text, constraints)
    if issues:
        _signal(signals, "constraint_violation",
                " ".join(item.span for item in constraints), "user_text")
    for phrase in unparsed:
        _signal(signals, "unparsed_constraint", phrase, "user_text")
        return


def scan_unit_risk(user_text, candidate_text, *, shown_text=""):
    """Bounded risk forms in one observed candidate, with the text that triggered it."""
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        raise ValueError("Nonempty observed candidate required")
    if len(candidate_text) > MAX_SCAN_CHARS:
        raise ValueError("Observed candidate exceeds the scan bound")
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("Nonempty user text required")
    if not isinstance(shown_text, str):
        raise ValueError("Displayed prefix must be text")
    signals = []
    if has_open_group(candidate_text):
        _signal(signals, "open_group", candidate_text[:64])
    _trailing_signals(candidate_text, signals)
    _condition_signals(candidate_text, signals)
    _reference_signals(candidate_text, user_text, shown_text, signals)
    _promise_signals(candidate_text, signals)
    _count_signals(user_text, candidate_text, signals)
    _quote_signals(user_text, candidate_text, signals)
    _polarity_signals(candidate_text, user_text, "user_text", signals, SHARED_MIN_USER)
    if shown_text:
        _polarity_signals(candidate_text, shown_text, "shown_text", signals, SHARED_MIN)
    _constraint_signals(user_text, candidate_text, signals)
    return RiskScan(bool(signals), tuple(signals))
