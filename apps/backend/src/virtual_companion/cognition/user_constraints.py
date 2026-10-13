"""Host-verifiable user output constraints; recognition and enforcement are separate.

Only explicitly bounded character counts and narrow request shapes are
recognized. Task contracts bind requested work, never inferred user facts.
Recognition never claims to cover arbitrary natural language: constraint-like
phrases that do not parse are reported as unparsed so the caller stays
conservative. Enforcement always happens on the actual display surface, never on
the model's self-reported ``constraints_satisfied``.
"""

import re
from dataclasses import dataclass

VERSION = "user-constraints.v2"
TASK_VERSION = "response-task-contract.v1"

NUMERALS = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
NUMBER = r"(?:\d{1,2}|[一两二三四五六七八九十])"
# Every exact-count form needs an explicit cue; a bare "N个字" is not a constraint.
EXACT_PATTERNS = (
    re.compile(rf"(?:只|仅)(?:需要|要)?(?:回答|回|说|答|写)?\s*({NUMBER})\s*个字"),
    re.compile(rf"(?:回答|回|说|答|写)\s*({NUMBER})\s*个字"),
    re.compile(rf"({NUMBER})\s*个字(?:就行|即可|就好)"),
)
MAX_PATTERNS = (
    re.compile(rf"(?:不要超过|不超过|最多|至多)\s*({NUMBER})\s*个字"),
    re.compile(rf"({NUMBER})\s*个字(?:以内|以下)"),
)
UNPARSED_PATTERNS = (
    re.compile(r"一句话"),
    re.compile(r"(?:尽量|要|请)?简短"),
    re.compile(r"只(?:回答|回|说|答)"),
    re.compile(r"(?:不要超过|不超过|最多|至多)"),
    re.compile(r"个字(?:以内|以下)"),
)
KINDS = ("exact_characters", "max_characters")
TRAILING_MARKS = "。！？!?… \t\r\n"

# Full matches deliberately exclude reported, conditional and compound requests.
ITEM_REQUEST = re.compile(
    rf"(?:请)?(?:给我?|给出|提供|列出)\s*({NUMBER})\s*条"
    r"[^，,。！？?；;：“”‘’\"'（）()]{0,48}建议[。.!！]?"
)
ACTIVITY_REQUEST = re.compile(
    r"我想(?:去)?(?P<activity>散步|跑步|骑车)但(?:在|正在)?下雨[，,]"
    r"\s*请给我(?:一个|一条)简单安排[。.!！]?"
)
GENERAL_REQUEST = re.compile(
    r"(?:我在想[，,]\s*)?(?:为什么|为何)"
    r"(?P<topic>[\u4e00-\u9fff]{1,24}?)(?:会)?(?:让人|令人)同时"
    r"(?P<first>[^又，,。！？?；;：“”‘’\"'（）()]{1,8})又"
    r"(?P<second>[^又，,。！？?；;：“”‘’\"'（）()]{1,8})[？?]?"
)
ONLY_SENTENCE_CLAUSE = re.compile(
    rf"(?:^|[。！？!?，,；;])\s*(?:只|仅)\s*(?:写|输出|给出)\s*"
    rf"(?:开头|前)?\s*({NUMBER})\s*句(?=$|[，,。！？!?；;])"
)
ITEM_MARKER = re.compile(
    rf"({NUMBER})(?:[、)]|\.(?!\d))\s*(?=\S)"
)


@dataclass(frozen=True)
class ResponseTaskContract:
    """User-provenanced shape; semantic checks remain an explicit model opinion."""

    kind: str
    span: str
    count: int | None = None
    activity: str | None = None
    topic: str | None = None
    aspects: tuple[str, str] | None = None

    def __post_init__(self):
        if not isinstance(self.span, str) or not self.span.strip() or len(self.span) > 256:
            raise ValueError("Bounded task provenance required")
        if self.kind in {"exact_items", "exact_sentences"}:
            valid = (type(self.count) is int and 1 <= self.count <= 99
                     and self.activity is None and self.topic is None and self.aspects is None)
        elif self.kind == "activity_plan":
            valid = (self.activity in {"散步", "跑步", "骑车"} and self.activity in self.span
                     and self.count is None and self.topic is None and self.aspects is None)
        elif self.kind == "general_explanation":
            valid = (isinstance(self.topic, str) and bool(self.topic) and self.topic in self.span
                     and isinstance(self.aspects, tuple) and len(self.aspects) == 2
                     and all(isinstance(item, str) and item and item in self.span
                             for item in self.aspects)
                     and self.count is None and self.activity is None)
        else:
            valid = False
        if not valid:
            raise ValueError("Typed response task shape required")

    @property
    def requires_complete_answer(self):
        return True

    def to_payload(self):
        payload = {"version": TASK_VERSION, "kind": self.kind, "source": "user_text",
                   "span": self.span, "requires_complete_answer": True}
        if self.kind == "exact_items":
            payload.update(count=self.count, numeric_claims="supported_or_explicit_plan_choice")
        elif self.kind == "exact_sentences":
            payload.update(count=self.count, content_only=True)
        elif self.kind == "activity_plan":
            payload.update(activity=self.activity, waiting="condition_of_same_path",
                           substitute_activities=False,
                           numeric_claims="supported_or_explicit_plan_choice")
        else:
            payload.update(topic=self.topic, aspects=list(self.aspects),
                           attribution="general_not_user_personal")
        return payload


def recognize_response_task(text):
    """Recognize only explicit full request shapes; ambiguity yields no contract."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty user text required")
    span = text.strip()
    if len(span) > 256:
        return None
    sentence_matches = [match for match in ONLY_SENTENCE_CLAUSE.finditer(span)
                        if _outside_quoted_scope(span[:match.start()])]
    if len(sentence_matches) == 1 and 1 <= (count := _number(sentence_matches[0].group(1))) <= 99:
        return ResponseTaskContract("exact_sentences", span, count=count)
    match = ITEM_REQUEST.fullmatch(span)
    if match and 1 <= (count := _number(match.group(1))) <= 99:
        return ResponseTaskContract("exact_items", span, count=count)
    match = ACTIVITY_REQUEST.fullmatch(span)
    if match:
        return ResponseTaskContract("activity_plan", span, activity=match["activity"])
    match = GENERAL_REQUEST.fullmatch(span)
    if match and not any(char in match["topic"] for char in "我你"):
        return ResponseTaskContract("general_explanation", span, topic=match["topic"],
                                    aspects=(match["first"], match["second"]))
    return None


def _outside_quoted_scope(prefix):
    pairs = {"“": "”", "‘": "’", '"': '"', "（": "）", "(": ")"}
    stack = []
    for char in prefix:
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs:
            stack.append(pairs[char])
    return not stack


def task_contract_issues(display_text, contract):
    """Check explicit sentence punctuation and plain numbered rows.

    This checks the unmodified display surface. It never guesses actions or
    rewrites output; semantic content remains the reviewer's responsibility.
    """
    if not isinstance(display_text, str):
        raise ValueError("Display text required")
    if contract is None:
        return ()
    if contract.kind == "exact_sentences":
        surface = display_text.strip()
        if (len(surface) >= 2 and surface[0] in {'“', '‘', '"'}
                and surface[-1] == {'“': '”', '‘': '’', '"': '"'}[surface[0]]):
            surface = surface[1:-1].strip()
        pieces = re.findall(r"[^。！？!?]+[。！？!?]+", surface)
        if len(pieces) != contract.count or ''.join(pieces) != surface:
            return ("exact_sentences",)
        return ()
    if contract.kind != "exact_items":
        return ()
    if any(char in display_text for char in '“”‘’"\''):
        return ()
    rows = [row for row in display_text.splitlines() if row.strip()]
    matches = [ITEM_MARKER.match(row) for row in rows]
    if not matches or not all(matches):
        return ()
    if any(ITEM_MARKER.search(row[match.end():]) for row, match in zip(rows, matches, strict=True)):
        return ()
    if len(matches) != contract.count:
        return ("exact_items",)
    return ()


@dataclass(frozen=True)
class UserConstraint:
    kind: str
    limit: int
    span: str
    source: str = "user_text"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("Unknown user output constraint kind")
        if type(self.limit) is not int or not 1 <= self.limit <= 99:
            raise ValueError("Bounded user output constraint limit required")
        if not isinstance(self.span, str) or not self.span.strip() or len(self.span) > 64:
            raise ValueError("Constraint provenance required")
        if self.source != "user_text":
            raise ValueError("Only explicit user text constraints are recognized")


def _number(value):
    if value.isdigit():
        return int(value)
    total = 0
    for char in value:
        total += NUMERALS.get(char, 0)
    return total


def _accepted_matches(text, patterns):
    """Non-overlapping matches, earliest and then longest first, with their limit."""
    found = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            limit = _number(match.group(1))
            if 1 <= limit <= 99:
                found.append((match.start(), match.end(), limit))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    accepted = []
    for start, end, limit in found:
        if any(start < taken_end and taken_start < end for taken_start, taken_end, _ in accepted):
            continue
        accepted.append((start, end, limit))
    return accepted


# Count only a complete length clause with an unambiguous whole-reply carrier.
# Other grammatical carriers keep their original words for the semantic reviewer.
_LENGTH_CLAUSE_BREAKS = frozenset("，,。！？!?；;\r\n")
_WHOLE_REPLY_PREFIX = re.compile(
    r"(?:请\s*)?(?:(?:全文|整段回复|整段回答|整个回复|整个回答|"
    r"全部回复|全部回答|回复|回答|总字数)\s*)?"
)
_LENGTH_CLAUSE_END = re.compile(r"(?:就行|即可|就好)?\s*")


def _length_clause(text, start, end):
    left, right = start, end
    while left and text[left - 1] not in _LENGTH_CLAUSE_BREAKS:
        left -= 1
    while right < len(text) and text[right] not in _LENGTH_CLAUSE_BREAKS:
        right += 1
    whole_reply = (_WHOLE_REPLY_PREFIX.fullmatch(text[left:start].strip()) is not None
                   and _LENGTH_CLAUSE_END.fullmatch(text[end:right].strip()) is not None)
    return whole_reply, text[left:right].strip()


def recognize_user_constraints(text):
    """Return whole-reply counts and original clauses needing semantic scope.

    A substring such as ``不超过十个字`` does not establish what it limits.
    Only the narrow whole-reply grammar owns a host character count; scoped,
    quoted or otherwise unparsed clauses remain current-user evidence for the
    existing opinion. No item, title or sentence is inferred from the reply.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty user text required")
    constraints, consumed, unparsed = [], [], []
    for kind, patterns in (("exact_characters", EXACT_PATTERNS),
                           ("max_characters", MAX_PATTERNS)):
        for start, end, limit in _accepted_matches(text, patterns):
            whole_reply, clause = _length_clause(text, start, end)
            if whole_reply:
                constraints.append(UserConstraint(kind, limit, text[start:end].strip()))
                consumed.append((start, end))
            else:
                unparsed.append(clause)
    for pattern in UNPARSED_PATTERNS:
        for match in pattern.finditer(text):
            if not any(start <= match.start() and match.end() <= end for start, end in consumed):
                _, clause = _length_clause(text, match.start(), match.end())
                unparsed.append(clause)
    return tuple(constraints), tuple(dict.fromkeys(unparsed))


def character_count(display_text):
    """Explicit counting rule for user character counts.

    Count Unicode code points of the display surface after dropping trailing
    whitespace and trailing sentence-final marks (``。！？!?…``). Those marks carry
    no content, so "好。" satisfies a one-character request while "好的" does not.
    """
    if not isinstance(display_text, str):
        raise ValueError("Display text required")
    return len(display_text.rstrip(TRAILING_MARKS))


def constraint_issues(display_text, constraints):
    """Host-verifiable violations of the given constraints on the display surface."""
    count = character_count(display_text)
    issues = []
    for constraint in constraints:
        if constraint.kind == "exact_characters" and count != constraint.limit:
            issues.append("exact_characters")
        elif constraint.kind == "max_characters" and count > constraint.limit:
            issues.append("max_characters")
    return tuple(dict.fromkeys(issues))
