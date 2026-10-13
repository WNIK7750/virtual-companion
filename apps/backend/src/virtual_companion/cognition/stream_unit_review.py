"""Optional bounded model opinion for observed, complete text-unit candidates.

The assessor is a controlled, budgeted Port. A structured opinion is not proof of
semantic quality; real-model counterexamples are required before default use.

Candidates are faithful prefixes of the observed stream: every display unit owns a
contiguous range of the immutable original text, so nothing is inserted, trimmed
or rejoined. A boundary (blank line, line break, two-space line break or sentence
end) only schedules a review; it never approves content.
"""

import json
from copy import deepcopy
from uuid import uuid4

from jsonschema import Draft202012Validator

from virtual_companion.boundary.contracts import Scope, canonical
from virtual_companion.boundary.history_readset import validate_context_time
from virtual_companion.boundary.presentation import (
    SPANS,
    MessageUnit,
    Presentation,
    PresentationPolicy,
    ReviewedPresentation,
    digest,
    review,
    review_by_policy,
)
from virtual_companion.boundary.turns import RunClaim

from .short_dialogue import (
    CONTEXT_RESPONSE_RESPONSIBILITY,
    CONTEXT_TIME_SYSTEM,
    CONVERSATION_USE_RESPONSIBILITY,
    CONVERSATION_VERDICT_RESPONSIBILITY,
    GROUNDED_CONVERSATION_RESPONSIBILITY,
    RESPONSE_TASK_RESPONSIBILITY,
    ShortRequest,
    conversation_frame,
)
from .unit_risk import VERSION as RISK_SCAN_VERSION
from .unit_risk import has_open_group, scan_unit_risk
from .user_constraints import (
    constraint_issues,
    recognize_response_task,
    recognize_user_constraints,
    task_contract_issues,
)

VERSION = "stream-unit-review.v7"
AUDIT_VERSION = "candidate-propositions.v3"
UNIT_SCOPE_AUDIT_VERSION = "candidate-unit-scope.v4"
COMPACT_UNIT_SCOPE_AUDIT_VERSION = "candidate-unit-scope.compact.v5"
GROUNDED_UNIT_AUDIT_VERSION = "candidate-unit-grounding.v6"
USE_UNIT_AUDIT_VERSION = "candidate-unit-meaning.v7"
VERDICT_UNIT_AUDIT_VERSION = "candidate-unit-verdict.v8"
MAX_UNIT_USE_CHARS = 32
SUPPORTED_AUDIT_VERSIONS = (
    AUDIT_VERSION, UNIT_SCOPE_AUDIT_VERSION, COMPACT_UNIT_SCOPE_AUDIT_VERSION,
    GROUNDED_UNIT_AUDIT_VERSION, USE_UNIT_AUDIT_VERSION, VERDICT_UNIT_AUDIT_VERSION,
)
COMPACT_CHECKS = (
    "self_contained", "no_open_dependency", "consistent_with_displayed", "constraints_satisfied",
)
MAX_REVIEW_SEGMENTS = 8
MAX_UNIT_MEANING_CHARS = 128
HOST_RISK_POLICY_REVISION = "host-risk-policy.v1"
REVIEW_POLICIES = ("model", "host-risk")
DECISION_FIELDS = {
    "request_sha256",
    "decision",
    "self_contained",
    "no_open_dependency",
    "consistent_with_displayed",
    "constraints_satisfied",
}
CHECKS = DECISION_FIELDS - {"request_sha256", "decision"}
BASE_MODEL_FIELDS = DECISION_FIELDS - {"request_sha256"}
MODEL_DECISION_FIELDS = BASE_MODEL_FIELDS | {"audit_version", "segments"}
HOST_AUDIT_FIELDS = {"audit_version", "propositions", "model_opinion"}
UNIT_SCOPE_HOST_FIELDS = {"audit_version", "unit_scopes", "model_opinion"}
VERDICT_HOST_FIELDS = UNIT_SCOPE_HOST_FIELDS | {"checks", "failure"}
REPAIR_REVIEW_PROJECTION_VERSION = "repair-unit-verdict-projection.v1"
SOURCE_IDENTITY = ("id", "owner", "source_id", "role", "revision")
LEGACY_SOURCE_REFERENCE_WIRE = "legacy"
CATALOG_SOURCE_REFERENCE_WIRE = "catalog-quote.v1"
DANGLING_ENDINGS = (",", "，", ":", "：", ";", "；", "、", "—")
SENTENCE_ENDS = "。！？!?…"
CLOSING_MARKS = "”’」』》】）)]"
SEPARATOR_SPACE = " \t\r"
SPEECH_SCOPE_RESPONSIBILITY = (
    "\n实际所指与言语作用：\n"
    "先在完整候选和当前交流中确定每句话谈的是一般对象、这位用户的具体情况，"
    "还是在提出建议或计划、作当下回应、写当前授权的作品，再判断它实际承担什么。"
    "所指与作用分开：建议可以面向用户，提出的动作不等于用户已有条件或实际做过；"
    "任务请求只授权交付，不证明用户已有条件、经历或情绪原因。"
    "一般内容不要求个人quote，但仍核对准确性、适用范围和当前所求，不能仅因无个人事实就批准。"
    "自身承诺或ack是助手此刻的回应，不是用户过去事实；仍须符合当前意图和实际操作状态。"
    "所指由上下文、范围和前后句共同决定：泛指解释中的听者称呼不自动成为个人归因，"
    "条件、隐喻或不确定措辞也不能掩去这位用户实际承担的前提。"
)
SYSTEM = (
    "你只审查候选文字，不重写、不生成替代回复。"
    "同一次审核先独立判断实际所指与言语作用，再核对事实来源、回应和表达是否成立。"
    + SPEECH_SCOPE_RESPONSIBILITY
    + CONTEXT_RESPONSE_RESPONSIBILITY
    + "\n批准职责：\n"
    "候选须尊重最新有效请求，接住仍有关的材料；有据的不同看法可以通过，"
    "仅主题相关或文笔顺畅不能替代当前所求。一般隐喻、一般关系和明确虚构按实际范围判断。"
    "用户个人事实（含感官细节、情绪归因）只据user_text或context中有效的准入来源；"
    "操作声明只据宿主实际操作证据。无法支持的实际个人断言、问题前提、操作声明，"
    "或违背当前意图与明确限制时，constraints_satisfied=false且decision=reject。"
    "自足与开放依赖只检查待审候选的表达是否闭合：条件、否定、引用、范围和数值说明"
    "尚未说完则hold；完整的询问可以等待用户回答，不因此hold。"
    "核对与已显示内容的连续性，不能改写已显示部分来掩盖矛盾；final不改变事实责任。"
)
TASK_SYSTEM = RESPONSE_TASK_RESPONSIBILITY
AUDIT_SYSTEM = (
    "\n来源审计（candidate-propositions.v3）：\n"
    "review_segments按顺序完整覆盖待审候选，包括空白与已显示部分。"
    "每个segment_id恰好返回一次，只写commitments和questions；questions是原段中精确的问句。"
    "commitments只列实际归给这位用户的个人断言、问句已承担的个人前提，及当前作品的虚构授权关系。"
    "这里的assertion不是所有陈述句：一般关系、泛指解释、意见、建议、计划选择和助手当下回应"
    "不编码为个人assertion，也不借用户的话题或任务请求伪造来源支持。"
    "询问未知是否发生不等于预设已经发生；问句包含已成立的个人属性或原因时仍须列其前提。"
    "没有这些个人或虚构承诺时commitments可为空；这不免除前述回应、准确性和操作核对。"
    "每项kind为assertion、presupposition或fiction，claim取原段中最短但语义完整的片段，"
    "保留其否定、引用、范围、频率、程度和因果；有据部分与新增细节不能合并成已支持。"
    "support为supported、unsupported或uncertain；sources来自personal_sources的实际身份及精确quote。"
    "supported要求引用支持完整个人claim，而非只同主题或要求解释它；一般提问不证明个人原因。"
    "来源的id、owner、source_id、role、revision原样对应；助手文字、候选本身、操作收据、"
    "失效或旧修订都不能充当个人事实来源。fiction的supported必须有当前用户授权引用，"
    "只支持本轮作品范围，不证明用户实际经历。实际个人或虚构承诺的unsupported或uncertain"
    "必须reject，不能通过删去commitments避开来源审计。宿主只绑定原文、身份与结构；"
    "含义、完整蕴含与本轮回应由此次审核承担，decision必须与这些判断一致。"
    "\n所指对照（仅示意，不是本轮事实）：\n"
    "只给房间安静时，‘安静时，冰箱、水管等小声音可能更明显’可以是一般条件和声音举例；"
    "‘不是你的房间变了，是分走你注意的东西少了’却增加这位用户的原因，不能借安静原话支持。"
    "要求安排散步时，‘雨小且路面安全就散步十分钟’是条件计划选择；"
    "‘你平常散步十分钟就累’才是个人习惯和体力断言，需要来源。"
)

UNIT_SCOPE_SYSTEM = (
    "你只审查候选文字，不重写、不生成替代回复。"
    "在同一次审核中判断完整候选的实际含义、当前回应、事实范围和表达是否成立。"
    "尊重最新有效请求及仍有关的完整材料；一般说明、建议、作品或接话须符合当前所求，"
    "不能只凭话题相关或措辞流畅批准。已停止或失效的旧材料不复活，已显示内容不能改写。"
    "助手不能虚构自己的生活或双方关系；操作声明只据宿主实际操作证据。"
    "一般内容仍须准确并适合当前范围，个人来源不足或违背意图、限制、操作真相时reject。"
    "自足和开放依赖检查表达是否闭合：条件、否定、引用、范围、数值尚未说完则hold；"
    "完整提问可以等待回答，不因此hold。final只说明生成结束，不改变审核责任。"
)
UNIT_SCOPE_SUPPORT_RESPONSIBILITY = (
    "先核所指，再核完整支持，最后汇总单元："
    "从整段交流确定单元每项陈述和问句前提的对象、时间、范围与言语作用。"
    "省略主语也可能指用户刚提到的实物、事件或经历；泛指人和一般事物不自动指这位用户。"
    "一般开头、比喻或建议不能替同单元后面的具体个人前提改换所指。"
    "对实际归于用户及其具体对象的含义，核对来源是否覆盖全部谓词、程度、数量和因果。"
    "来源提到对象或话题，只支持它实际提供的内容；类别经验不能补成这件实物的状态、"
    "某次发生的动作或某段个人经历。已有部分不能为新增部分提供完整支持。"
    "保留候选自己的模态与范围：一般观察、类比和明示未来条件方案不要求个人证据；"
    "有据且明确标为推测的当时解释，按其依据和有限范围判断，不改判为确定事实。"
    "可能性措辞本身不提供依据；核对已知内容是否足以支撑这个有范围的推测，"
    "或它是否仍预设了未提供的具体属性、数值、经历或原因。用户更正优先于旧推测。"
    "personal_support=none只在整单元没有实际个人断言或个人问句前提时使用。"
    "supported要求所列有效personal_source_ids支持其中全部个人含义及其实际模态范围；"
    "明确存在无据部分为unsupported，完整支持无法确定为uncertain。"
    "混合单元按全部含义汇总，不用一般部分或已知分句替无据个人部分获批；"
    "unsupported和uncertain都reject。none仍须通过回应、准确性、任务和操作检查。"
)
UNIT_SCOPE_AUDIT_SYSTEM = (
    "\n完整单元审核（candidate-unit-scope.v4）：\n"
    "units按顺序拼成原始待审候选；每个segment_id恰好一项结果，不抽claim、quote或重述原句。"
    + UNIT_SCOPE_SUPPORT_RESPONSIBILITY
    + "fiction_support=none表示没有需要用户授权的虚构或角色代入；"
    "authorized表示fiction_source_ids含本轮current来源且明确授权该单元的作品范围，"
    "旧授权不代替当前授权，作品内容不证明用户真实经历；缺授权unauthorized或uncertain都reject。"
    "两组source_ids只能取sources里的id，不重复owner、role、revision或来源原文；"
    "它们由宿主绑定同一封包内的真实来源版本。none时对应IDs为空，"
    "supported或authorized时对应IDs非空；拒绝时可列部分有关来源，也可为空。"
    "整段任务、实际操作及与已显示内容的连续性仍由同一次审核负责。"
    "ack只接住话语，不冒充已答或完成操作。decision须与全部判断一致："
    "个人unsupported/uncertain、虚构unauthorized/uncertain、限制或连续性失败必须reject；"
    "表达未闭合且无这些失败才hold；全部检查成立才approve。拒绝时constraints_satisfied=false。"
)


COMPACT_UNIT_SCOPE_AUDIT_SYSTEM = (
    "\n紧凑完整单元审核（candidate-unit-scope.compact.v5）：\n"
    "units按顺序拼成原始待审候选；每个u别名恰好一项结果，不抽claim、quote或重述原句。"
    + UNIT_SCOPE_SUPPORT_RESPONSIBILITY
    + "fiction_support=none表示没有需要用户授权的虚构或角色代入；"
    "authorized要求所列s别名含本轮current来源且明确授权该单元的作品范围，"
    "旧授权不代替当前授权，作品内容不证明用户真实经历；unauthorized或uncertain都reject。"
    "u和s仅是本请求内单元与有效来源的一一别名，不能引用别的请求、候选或来源。"
    "units结果每项严格为[u,personal_support,[s],fiction_support,[s]]，"
    "每个完整单元都须给出两种判断；none时对应来源列表为空，supported或authorized时非空。"
    "拒绝时可列部分有关来源，也可为空；不得漏掉单元、重述或缩窄其完整含义。"
    "checks依次为自足、无开放依赖、与已显示内容连续、符合当前请求及限制四个布尔值。"
    "整段任务、实际操作及已显示内容仍由同一次审核负责，ack不冒充已答或完成操作。"
    "个人unsupported/uncertain、虚构unauthorized/uncertain、限制或连续性失败必须reject；"
    "表达未闭合且无这些失败才hold；全部检查成立才approve。拒绝时第四项checks=false。"
)


GROUNDED_UNIT_SYSTEM = (
    "你只审查候选文字，不改写、不生成替代回复。在同一次审核内，"
    "先忠实读完整单元实际承担的含义，再判断完整来源支持及当前回应是否成立。"
    + GROUNDED_CONVERSATION_RESPONSIBILITY
    + "\n批准责任：仍核对所求成果、一般内容的准确性和适用范围、明确限制、"
    "实际操作及已显示内容的连续性；无个人断言不等于自动批准。"
    "final只说明生成结束；否定、条件、范围或数值表达尚未闭合才hold，"
    "完整提问或接话可成立。ack不要求结清待答，也不得冒充操作完成。"
)
GROUNDED_AUDIT_SYSTEM = (
    "\n完整单元语义依据（candidate-unit-grounding.v6）：\n"
    "units按顺序拼成完整原候选；每个segment_id恰好一项，不抽取较小claim或quote。"
    "先写meaning：用简短的忠实语义依据说明整个单元实际谈的对象、言语作用、"
    "已承担的个人前提及其模态；混合内容保留全部谓词和因果，不能漏掉后半句。"
    "meaning解释原话，不补事实、不改写成较容易获批的含义，也不是新的来源。"
    "然后给personal_support及personal_source_ids：整单元无个人断言或既知问题前提才none；"
    "supported要求同封包来源覆盖全部个人含义与模态，相关对象或已知分句不支持新增因果。"
    "明确无据部分为unsupported，归属或完整支持无法确定为uncertain；二者均reject。"
    "开放未知或感觉选项保留为询问，不把选项当个人既成事实；"
    "问题已承担的具体前提仍须支持。一般联想、意见和条件建议仍核对实际范围与所求。"
    "fiction_support为none、authorized、unauthorized或uncertain；authorized须由本轮current"
    "来源明确授权该单元作品范围，旧授权不代替本轮，作品不证明真实经历。"
    "source_ids只用sources的真实ID；none时为空，supported或authorized时非空。"
    "语义依据和支持结论均由此次审核承担，宿主只绑定完整原文、来源身份和结构。"
    "全部检查成立才approve；个人unsupported/uncertain、虚构unauthorized/uncertain、"
    "限制或连续性失败必须reject且constraints_satisfied=false；无这些失败但未闭合才hold。"
    "每项meaning只用一句简短完整的语义依据，不逐句复述原文或展开判断过程；"
    "支持ID只列共同覆盖个人含义所需的有效来源，按schema只输出一个JSON对象。"
)


USE_UNIT_SYSTEM = (
    "你只审查候选文字，不改写或生成替代回复。"
    + CONVERSATION_USE_RESPONSIBILITY
    + "\n同一次审核分别读完整含义和当前使用关系，再核完整来源、一般准确性、任务、操作及连续性。"
    "表达未闭合才hold；完整问题与ack不因待用户回答而未闭合。final不免除责任。"
)
USE_AUDIT_SYSTEM = (
    "\n完整单元意见（candidate-unit-meaning.v7）：units顺序保留全部原文，每个segment_id恰好一项。"
    "meaning简短说明完整单元的实际actor、所指、重要谓词与因果及模态；不把具体范围改成泛指，"
    "不把助手贡献归给source，不省去改变承诺强弱的模态。"
    "use用一句短说明记录这句对最新原话的言语作用及适用对象/时点，不复述全部材料。"
    "response_fit为fits、incompatible或uncertain；包括personal_support=none的每项都核当前关系。"
    "来源支持与当前适用分别判断：personal_support=none只指没有个人断言或已知问句前提；"
    "supported须来源共同覆盖全部个人含义，未知个人前提为unsupported，归属或支持无法确定为uncertain。"
    "开放选项保留未知，不当既成事实；一般内容仍核准确性和所求。fiction授权只据current来源的本轮范围。"
    "source_ids只绑定同表身份，none为空，supported/authorized非空。meaning/use/fit是模型意见，不是来源或机械证明。"
    "任一personal unsupported/uncertain、fiction unauthorized/uncertain或"
    "fit incompatible/uncertain都reject，"
    "constraints_satisfied=false；其他现有检查全成立且闭合才approve。"
    "meaning最多128字符，use最多32字符；简短保留责任，只输出schema的一个JSON对象。"
)


VERDICT_UNIT_SYSTEM = (
    "你只审查候选文字，不改写或生成替代回复。"
    + CONVERSATION_VERDICT_RESPONSIBILITY
    + "\n同一次审核完成全部单元和整体核对，再给原样可查的裁决声明。"
)
VERDICT_AUDIT_SYSTEM = (
    "\n完整裁决（candidate-unit-verdict.v8）：每个segment_id恰好一项，覆盖完整原单元。"
    "meaning简短保留实际actor、全部重要谓词/因果/模态；use记录当前对象、时点和言语作用；"
    "response_fit独立为fits/incompatible/uncertain。个人断言及已知问句前提按全部来源支持，"
    "none只表示无个人承诺；未知开放选项不当事实。虚构授权只据current本轮范围。"
    "source_ids绑定原表；none为空，supported/authorized非空。解释与依据都是意见，不是事实来源。"
    "checks按固定顺序声明完整核对：自足、无开放依赖、已显示连续性、当前请求/限制/准确性/实际操作。"
    "每项pass/fail/uncertain；不得省项。最后一项包括一般准确性、当前限制和实际操作。"
    "final=false表示只观察到生成中前缀，核当前含义与请求的兼容性；"
    "尚未生成的后续任务内容本身不能判task失败或fit incompatible/uncertain。"
    "当前含义闭合且其余检查通过可approve；否定/条件等当前含义未闭合才影响前两项并hold。"
    "final=true时还须核完整任务交付，缺少要求的内容不能当成尚待后续。"
    "不能仅因所有units为fits或none跳过第四项；完整问题和ack可闭合。"
    "任一个人unsupported/uncertain、虚构unauthorized/uncertain或fit incompatible/uncertain，"
    "第四项必须非pass；第四项或连续性非pass则reject。仅前两项非pass且无这些失败才hold；"
    "全pass且无单元失败才approve。decision和checks须一致，不在最后另给未说明的拒绝。"
    "approve的failure为null；否则给一个主要失败对象和简短basis，不推测其他隐藏理由。"
    "个人/虚构/适用失败scope分别personal/fiction/fit，segment_id必须定位原unit；"
    "一般准确性/任务/操作scope分别accuracy/task/operation，可定位unit或用null指整体；"
    "continuity定位第三项，closure定位前两项。basis说明该范围为何失败或未核定，不改写候选。"
    "meaning最多128字符、use32、basis96；按schema输出一个JSON，不展开判断过程。"
)


def _verdict_unit_model_schema():
    unit = _use_unit_model_schema()["properties"]["units"]
    failure = {
        "type": "object", "additionalProperties": False,
        "required": ["scope", "segment_id", "basis"],
        "properties": {
            "scope": {"enum": ["personal", "fiction", "fit", "accuracy", "task", "operation",
                               "continuity", "closure"]},
            "segment_id": {"type": ["string", "null"], "minLength": 1},
            "basis": {"type": "string", "minLength": 1, "maxLength": 96, "pattern": r"\S"},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["audit_version", "decision", "checks", "failure", "units"],
        "properties": {
            "audit_version": {"const": VERDICT_UNIT_AUDIT_VERSION},
            "decision": {"enum": ["approve", "hold", "reject"]},
            "checks": {"type": "array", "minItems": 4, "maxItems": 4,
                       "items": {"enum": ["pass", "fail", "uncertain"]}},
            "failure": {"anyOf": [{"type": "null"}, failure]}, "units": unit,
        },
    }


class InvalidUnitVerdict(ValueError):
    """The model's declarations cannot form a coherent opinion."""

    def __init__(self, message, opinion):
        super().__init__(message)
        self.opinion = deepcopy(opinion)


def _derive_verdict_decision(result):
    try:
        return _validate_verdict_decision(result)
    except ValueError as error:
        raise InvalidUnitVerdict(str(error), result) from error


def _validate_verdict_decision(result):
    """Validate explicit declarations and derive rejection, never infer semantic truth."""
    schema = _verdict_unit_model_schema()
    schema["properties"]["units"]["items"]["properties"]["use"][
        "maxLength"] = MAX_UNIT_MEANING_CHARS
    if not Draft202012Validator(schema).is_valid(result):
        raise ValueError("Complete verdict response violates bound schema")
    checks, units, failure = result["checks"], result["units"], result["failure"]
    by_id = {item["segment_id"]: item for item in units}
    if len(by_id) != len(units):
        raise ValueError("Verdict unit IDs must be unique")
    closure_only = (
        result["decision"] == "hold" and failure is not None and failure["scope"] == "closure"
        and (checks[0] != "pass" or checks[1] != "pass") and checks[2:] == ["pass", "pass"]
        and all(item["personal_support"] not in {"unsupported", "uncertain"}
                and item["fiction_support"] not in {"unauthorized", "uncertain"}
                and item["response_fit"] != "incompatible" for item in units)
    )
    adverse = any(item["personal_support"] in {"unsupported", "uncertain"}
                  or item["fiction_support"] in {"unauthorized", "uncertain"}
                  or (item["response_fit"] in {"incompatible", "uncertain"}
                      and not closure_only) for item in units)
    if adverse and checks[3] == "pass":
        raise ValueError("Adverse unit contradicts the complete constraint declaration")
    expected = ("reject" if adverse or checks[2] != "pass" or checks[3] != "pass"
                else "hold" if checks[0] != "pass" or checks[1] != "pass" else "approve")
    if expected != "reject" and result["decision"] != expected:
        raise ValueError("Decision contradicts explicit complete checks; cannot upgrade rejection")
    if expected == "approve":
        if failure is not None:
            raise ValueError("Approval cannot declare a failure")
        return expected
    if failure is None:
        raise ValueError("Unresolved verdict requires an observable failure object")
    segment_id, scope = failure["segment_id"], failure["scope"]
    if segment_id is not None and segment_id not in by_id:
        raise ValueError("Failure refers to a foreign unit")
    if scope in {"personal", "fiction", "fit"}:
        if segment_id is None:
            raise ValueError("Unit failure requires an exact original unit identity")
        field, values = {
            "personal": ("personal_support", {"unsupported", "uncertain"}),
            "fiction": ("fiction_support", {"unauthorized", "uncertain"}),
            "fit": ("response_fit", {"incompatible", "uncertain"}),
        }[scope]
        valid = by_id[segment_id][field] in values and checks[3] != "pass"
    elif scope in {"accuracy", "task", "operation"}:
        valid = checks[3] != "pass"
    elif scope == "continuity":
        valid = checks[2] != "pass"
    else:
        valid = checks[0] != "pass" or checks[1] != "pass"
    if not valid:
        raise ValueError("Failure scope contradicts the declared unit or complete check")
    return expected


def _bind_verdict_unit_audit(payload, result, request_sha256):
    decision = _derive_verdict_decision(result)
    if payload.get("audit_version") != VERDICT_UNIT_AUDIT_VERSION:
        raise ValueError("Verdict audit differs from sealed selector")
    _validate_conversation_frame(payload)
    normalized = {"audit_version": GROUNDED_UNIT_AUDIT_VERSION, "decision": decision,
                  "units": [{key: value for key, value in item.items()
                             if key not in {"use", "response_fit"}} for item in result["units"]],
                  **{key: status == "pass" for key, status in zip(
                      COMPACT_CHECKS, result["checks"], strict=True)}}
    bound = _bind_grounded_unit_audit(
        {**payload, "audit_version": GROUNDED_UNIT_AUDIT_VERSION}, normalized, request_sha256,
    )
    if (payload["final"] is False and result["decision"] == "hold"
            and result["checks"] == ["pass", "pass", "pass", "uncertain"]
            and result["failure"]["scope"] == "task"
            and all(item["personal_support"] in {"none", "supported"}
                    and item["fiction_support"] in {"none", "authorized"}
                    and item["response_fit"] == "fits" for item in result["units"])):
        # v8 requires a prefix-compatibility judgment, not delivery of the
        # not-yet-generated remainder. This contradictory preliminary hold
        # neither approves the prefix nor supplies an explicit task rejection.
        # Bind the original units/sources first, then let the existing bounded
        # deferral obtain the final opinion on the complete original candidate.
        raise InvalidUnitVerdict("Preliminary task uncertainty cannot reject the candidate",
                                 result)
    opinions = {item["segment_id"]: item for item in result["units"]}
    return {**bound, "audit_version": VERDICT_UNIT_AUDIT_VERSION, "decision": decision,
            "unit_scopes": [dict(item, use=opinions[item["segment_id"]]["use"],
                                 response_fit=opinions[item["segment_id"]]["response_fit"])
                            for item in bound["unit_scopes"]],
            "checks": deepcopy(result["checks"]), "failure": deepcopy(result["failure"]),
            "model_opinion": deepcopy(result)}


def project_repair_review(text, verdict):
    """Lossless outbound projection; the canonical original verdict stays in the audit."""
    if (not isinstance(verdict, dict) or set(verdict) != DECISION_FIELDS | VERDICT_HOST_FIELDS
            or verdict.get("audit_version") != VERDICT_UNIT_AUDIT_VERSION
            or verdict.get("decision") != "reject"):
        raise ValueError("Only a bound explicit rejection can prepare this repair")
    projection = {
        "projection_version": REPAIR_REVIEW_PROJECTION_VERSION,
        "original_verdict_sha256": digest(verdict), "displayed_prefix": "",
        "model_opinion": deepcopy(verdict["model_opinion"]),
        "unit_bindings": [{key: item[key] for key in ("segment_id", "start", "end")}
                          for item in verdict["unit_scopes"]],
        "bound_fields": {key: deepcopy(value) for key, value in verdict.items()
                         if key not in VERDICT_HOST_FIELDS - {"audit_version"}},
    }
    if restore_repair_review(text, projection) != verdict:
        raise ValueError("Repair projection cannot reproduce the exact original verdict")
    return projection


def restore_repair_review(text, projection):
    """Restore original full units by codepoint offsets, never from a model's meaning."""
    fields = {"projection_version", "original_verdict_sha256", "displayed_prefix",
              "model_opinion", "unit_bindings", "bound_fields"}
    if (not isinstance(projection, dict) or set(projection) != fields
            or projection["projection_version"] != REPAIR_REVIEW_PROJECTION_VERSION
            or projection["displayed_prefix"] != ""):
        raise ValueError("Exact zero-approved repair projection required")
    opinion, bound_fields = projection["model_opinion"], projection["bound_fields"]
    decision = _derive_verdict_decision(opinion)
    if (not isinstance(bound_fields, dict)
            or set(bound_fields) != DECISION_FIELDS | {"audit_version"}
            or bound_fields["audit_version"] != VERDICT_UNIT_AUDIT_VERSION
            or bound_fields["decision"] != decision or decision != "reject"
            or any(type(bound_fields[key]) is not bool or bound_fields[key] != (status == "pass")
                   for key, status in zip(COMPACT_CHECKS, opinion["checks"], strict=True))):
        raise ValueError("Projected bound fields differ from the original declarations")
    units = candidate_segments(text)
    expected = [{key: item[key] for key in ("segment_id", "start", "end")} for item in units]
    by_id = {item["segment_id"]: item for item in opinion["units"]}
    bindings = projection["unit_bindings"]
    if (not isinstance(bindings, list)
            or any(not isinstance(item, dict) or set(item) != {"segment_id", "start", "end"}
                   or type(item["start"]) is not int or type(item["end"]) is not int
                   for item in bindings)
            or bindings != expected
            or set(by_id) != {item["segment_id"] for item in units}):
        raise ValueError("Repair offsets or identities differ from the full original candidate")
    verdict = {**deepcopy(bound_fields), "checks": deepcopy(opinion["checks"]),
               "failure": deepcopy(opinion["failure"]), "model_opinion": deepcopy(opinion),
               "unit_scopes": [dict(item, **deepcopy(by_id[item["segment_id"]])) for item in units]}
    if digest(verdict) != projection["original_verdict_sha256"]:
        raise ValueError("Restored repair verdict differs from its immutable original hash")
    return verdict


def _use_unit_model_schema():
    schema = _grounded_unit_model_schema()
    schema["properties"]["audit_version"]["const"] = USE_UNIT_AUDIT_VERSION
    item = schema["properties"]["units"]["items"]
    item["required"].extend(["use", "response_fit"])
    item["properties"]["use"] = {"type": "string", "minLength": 1,
        "maxLength": MAX_UNIT_USE_CHARS, "pattern": r"\S"}
    item["properties"]["response_fit"] = {"enum": ["fits", "incompatible", "uncertain"]}
    return schema


def _validate_conversation_frame(payload):
    _unit_scope_packet_bindings(payload)
    expected = conversation_frame(payload["personal_sources"], payload["user_text"])
    if payload.get("conversation_frame") != expected:
        raise ValueError("Conversation frame differs from admitted raw roles and current identity")
    return expected


def _bind_use_unit_audit(payload, result, request_sha256):
    """Bind declared use, never classify words or prove a model interpretation."""
    _validate_conversation_frame(payload)
    if (payload.get("audit_version") != USE_UNIT_AUDIT_VERSION
            or not Draft202012Validator(_use_unit_model_schema()).is_valid(result)):
        raise ValueError("Meaning/use unit response violates bound schema")
    normalized = {**result, "audit_version": GROUNDED_UNIT_AUDIT_VERSION,
        "units": [{key: value for key, value in item.items() if key not in {"use", "response_fit"}}
                  for item in result["units"]]}
    bound = _bind_grounded_unit_audit({**payload, "audit_version": GROUNDED_UNIT_AUDIT_VERSION},
                                    normalized, request_sha256)
    opinions = {item["segment_id"]: item for item in result["units"]}
    adverse = any(item["response_fit"] in {"incompatible", "uncertain"} for item in result["units"])
    return {**bound, "audit_version": USE_UNIT_AUDIT_VERSION,
        "unit_scopes": [dict(item, use=opinions[item["segment_id"]]["use"],
                             response_fit=opinions[item["segment_id"]]["response_fit"])
                        for item in bound["unit_scopes"]],
        "model_opinion": deepcopy(result),
        **({"decision": "reject", "constraints_satisfied": False} if adverse else {})}


def _grounded_unit_model_schema():
    schema = deepcopy(_unit_scope_model_schema())
    schema["properties"]["audit_version"]["const"] = GROUNDED_UNIT_AUDIT_VERSION
    item = schema["properties"]["units"]["items"]
    item["required"].insert(1, "meaning")
    item["properties"] = {
        "segment_id": item["properties"]["segment_id"],
        "meaning": {"type": "string", "minLength": 1, "maxLength": MAX_UNIT_MEANING_CHARS,
                    "pattern": r"\S"},
        **{key: value for key, value in item["properties"].items() if key != "segment_id"},
    }
    return schema


def _compact_unit_scope_model_schema():
    source_ids = {"type": "array", "maxItems": 16, "uniqueItems": True,
                  "items": {"type": "string", "minLength": 1}}
    unit = {
        "type": "array", "minItems": 5, "maxItems": 5, "items": False,
        "prefixItems": [
            {"type": "string", "minLength": 1},
            {"enum": ["none", "supported", "unsupported", "uncertain"]},
            source_ids,
            {"enum": ["none", "authorized", "unauthorized", "uncertain"]},
            deepcopy(source_ids),
        ],
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["audit_version", "decision", "checks", "units"],
        "properties": {
            "audit_version": {"const": COMPACT_UNIT_SCOPE_AUDIT_VERSION},
            "decision": {"enum": ["approve", "hold", "reject"]},
            "checks": {"type": "array", "minItems": 4, "maxItems": 4, "items": False,
                       "prefixItems": [{"type": "boolean"} for _ in COMPACT_CHECKS]},
            "units": {"type": "array", "minItems": 1, "maxItems": MAX_REVIEW_SEGMENTS,
                      "items": unit},
        },
    }


def _unit_scope_model_schema():
    source_ids = {"type": "array", "maxItems": 16, "uniqueItems": True,
                  "items": {"type": "string", "minLength": 1}}
    unit = {
        "type": "object", "additionalProperties": False,
        "required": ["segment_id", "personal_support", "personal_source_ids",
                     "fiction_support", "fiction_source_ids"],
        "properties": {
            "segment_id": {"type": "string", "minLength": 1},
            "personal_support": {"enum": ["none", "supported", "unsupported", "uncertain"]},
            "personal_source_ids": source_ids,
            "fiction_support": {"enum": ["none", "authorized", "unauthorized", "uncertain"]},
            "fiction_source_ids": deepcopy(source_ids),
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(BASE_MODEL_FIELDS | {"audit_version", "units"}),
        "properties": {
            "decision": {"enum": ["approve", "hold", "reject"]},
            **{key: {"type": "boolean"} for key in CHECKS},
            "audit_version": {"const": UNIT_SCOPE_AUDIT_VERSION},
            "units": {"type": "array", "minItems": 1, "maxItems": MAX_REVIEW_SEGMENTS,
                      "items": unit},
        },
    }


def _unit_scope_packet_bindings(payload):
    """Exact original ranges and admitted source identities for whole-unit Ports."""
    text = payload.get("candidate_prefix")
    bound = candidate_segments(text)
    if payload.get("review_segments") != [
        {"segment_id": item["segment_id"], "text": item["text"]} for item in bound
    ]:
        raise ValueError("Review segments differ from bound candidate")
    sources = payload.get("personal_sources")
    if not isinstance(sources, list):
        raise ValueError("Admitted personal source catalog required")
    by_id = {}
    for source in sources:
        if (not isinstance(source, dict) or source.get("role") != "user"
                or source.get("kind") not in {"current", "history", "memory"}
                or not isinstance(source.get("text"), str) or not source["text"].strip()
                or type(source.get("revision")) is not int or source["revision"] < 1
                or any(not isinstance(source.get(key), str) or not source[key]
                       for key in SOURCE_IDENTITY if key != "revision")
                or source["id"] in by_id
                or source["owner"] != payload.get("basis", {}).get("companion_id")):
            raise ValueError("Invalid admitted personal source identity")
        by_id[source["id"]] = source
    return bound, by_id


def _compact_unit_scope_bindings(payload):
    """Request-local aliases never replace the identity in the sealed authority."""
    if payload.get("audit_version") != COMPACT_UNIT_SCOPE_AUDIT_VERSION:
        raise ValueError("Compact unit audit differs from sealed selector")
    bound, sources = _unit_scope_packet_bindings(payload)
    units = {f"u{index + 1}": item for index, item in enumerate(bound)}
    aliases = {f"s{index + 1}": item for index, item in enumerate(sources.values())}
    return units, aliases


def _bind_compact_unit_scope_audit(payload, result, request_sha256):
    """Normalize structure, retaining the complete compact opinion and its mistakes.

    No host inference of personal entailment or fiction authorization is added.
    The shared whole-unit binder restores exact original ranges and identities.
    """
    if not Draft202012Validator(_compact_unit_scope_model_schema()).is_valid(result):
        raise ValueError("Compact unit scope model response violates bound schema")
    units, sources = _compact_unit_scope_bindings(payload)
    if (len({item[0] for item in result["units"]}) != len(result["units"])
            or {item[0] for item in result["units"]} != set(units)):
        raise ValueError("Compact unit aliases must cover the bound candidate exactly once")
    normalized = {
        "audit_version": UNIT_SCOPE_AUDIT_VERSION, "decision": result["decision"],
        **dict(zip(COMPACT_CHECKS, result["checks"], strict=True)), "units": [],
    }
    for alias, personal, personal_ids, fiction, fiction_ids in result["units"]:
        if any(source_id not in sources for source_id in [*personal_ids, *fiction_ids]):
            raise ValueError("Compact source alias differs from admitted source")
        normalized["units"].append({
            "segment_id": units[alias]["segment_id"], "personal_support": personal,
            "personal_source_ids": [sources[source_id]["id"] for source_id in personal_ids],
            "fiction_support": fiction,
            "fiction_source_ids": [sources[source_id]["id"] for source_id in fiction_ids],
        })
    bound = _bind_unit_scope_audit(
        {**payload, "audit_version": UNIT_SCOPE_AUDIT_VERSION}, normalized, request_sha256,
    )
    return {**bound, "audit_version": COMPACT_UNIT_SCOPE_AUDIT_VERSION,
            "model_opinion": deepcopy(result)}


def _bind_unit_scope_audit(payload, result, request_sha256):
    """Bind whole-unit judgments to one sealed catalog, without inferring meaning.

    The full original v4 opinion is retained. It is not a v3 proposition audit:
    no extracted claim/quote can narrow the complete unit's responsibility.
    A model's false none/supported remains an observable semantic counterexample.
    """
    if (payload.get("audit_version") != UNIT_SCOPE_AUDIT_VERSION
            or not Draft202012Validator(_unit_scope_model_schema()).is_valid(result)):
        raise ValueError("Unit scope model response violates bound schema")
    bound, by_id = _unit_scope_packet_bindings(payload)
    opinions = {item["segment_id"]: item for item in result["units"]}
    if (len(opinions) != len(result["units"])
            or set(opinions) != {item["segment_id"] for item in bound}):
        raise ValueError("Unit IDs must cover the bound candidate exactly once")
    adverse = False
    for item in result["units"]:
        for prefix, supported in (("personal", "supported"), ("fiction", "authorized")):
            support, ids = item[prefix + "_support"], item[prefix + "_source_ids"]
            if (support == "none" and ids) or (support == supported and not ids):
                raise ValueError("Unit support requires its exact source relation")
            if any(source_id not in by_id for source_id in ids):
                raise ValueError("Unit reference differs from admitted source")
        if item["fiction_support"] == "authorized" and not any(
            by_id[source_id]["kind"] == "current" for source_id in item["fiction_source_ids"]
        ):
            raise ValueError("Fiction requires a current user authorization reference")
        adverse |= (item["personal_support"] in {"unsupported", "uncertain"}
                    or item["fiction_support"] in {"unauthorized", "uncertain"})
    decision = result["decision"]
    if adverse:
        decision = "reject"
    elif decision == "approve":
        if not result["constraints_satisfied"] or not result["consistent_with_displayed"]:
            decision = "reject"
        elif not result["self_contained"] or not result["no_open_dependency"]:
            decision = "hold"
    return {**{key: result[key] for key in BASE_MODEL_FIELDS},
            "request_sha256": request_sha256, "audit_version": UNIT_SCOPE_AUDIT_VERSION,
            "unit_scopes": [dict(item, **deepcopy(opinions[item["segment_id"]])) for item in bound],
            "model_opinion": deepcopy(result), "decision": decision,
            **({"constraints_satisfied": False} if adverse else {})}


def _bind_grounded_unit_audit(payload, result, request_sha256):
    """Bind a faithful-reading opinion; meaning is model evidence, never proof."""
    if (payload.get("audit_version") != GROUNDED_UNIT_AUDIT_VERSION
            or not Draft202012Validator(_grounded_unit_model_schema()).is_valid(result)):
        raise ValueError("Grounded unit model response violates bound schema")
    normalized = {**result, "audit_version": UNIT_SCOPE_AUDIT_VERSION,
                  "units": [{key: value for key, value in item.items() if key != "meaning"}
                            for item in result["units"]]}
    bound = _bind_unit_scope_audit(
        {**payload, "audit_version": UNIT_SCOPE_AUDIT_VERSION}, normalized, request_sha256,
    )
    meanings = {item["segment_id"]: item["meaning"] for item in result["units"]}
    return {**bound, "audit_version": GROUNDED_UNIT_AUDIT_VERSION,
            "unit_scopes": [dict(item, meaning=meanings[item["segment_id"]])
                            for item in bound["unit_scopes"]],
            "model_opinion": deepcopy(result)}


def _model_schema():
    reference = {
        "type": "object", "additionalProperties": False,
        "required": [*SOURCE_IDENTITY, "quote"],
        "properties": {
            **{key: {"type": "string", "minLength": 1}
               for key in SOURCE_IDENTITY if key not in {"revision", "role"}},
            "revision": {"type": "integer", "minimum": 1}, "role": {"const": "user"},
            "quote": {"type": "string", "minLength": 1, "maxLength": 64000},
        },
    }
    commitment = {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "claim", "support", "sources"],
        "properties": {
            "kind": {"enum": ["assertion", "presupposition", "fiction"]},
            "claim": {"type": "string", "minLength": 1, "maxLength": 64000},
            "support": {"enum": ["supported", "unsupported", "uncertain"]},
            "sources": {"type": "array", "maxItems": 16, "items": reference},
        },
    }
    segment = {
        "type": "object", "additionalProperties": False,
        "required": ["segment_id", "commitments", "questions"],
        "properties": {
            "segment_id": {"type": "string", "minLength": 1},
            "commitments": {"type": "array", "maxItems": 16, "items": commitment},
            "questions": {"type": "array", "maxItems": 8,
                          "items": {"type": "string", "minLength": 1, "maxLength": 64000}},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(MODEL_DECISION_FIELDS),
        "properties": {
            "decision": {"enum": ["approve", "hold", "reject"]},
            **{key: {"type": "boolean"} for key in CHECKS},
            "audit_version": {"const": AUDIT_VERSION},
            "segments": {"type": "array", "minItems": 1, "maxItems": MAX_REVIEW_SEGMENTS,
                         "items": segment},
        },
    }


def _personal_source_catalog(payload):
    sources = payload.get("personal_sources", [])
    if not isinstance(sources, list):
        raise ValueError("Admitted personal source catalog required")
    by_id = {}
    for source in sources:
        if (not isinstance(source, dict) or source.get("role") != "user"
                or source.get("kind") not in {"current", "history", "memory"}
                or not isinstance(source.get("text"), str) or not source["text"].strip()
                or type(source.get("revision")) is not int or source["revision"] < 1
                or any(not isinstance(source.get(key), str) or not source[key]
                       for key in SOURCE_IDENTITY if key != "revision")
                or source["id"] in by_id
                or source["owner"] != payload.get("basis", {}).get("companion_id")):
            raise ValueError("Invalid admitted personal source identity")
        by_id[source["id"]] = source
    return by_id


def _catalog_quote_model_schema(catalog):
    schema = _model_schema()
    sources = schema["properties"]["segments"]["items"]["properties"]["commitments"][
        "items"]["properties"]["sources"]
    source_id = {"type": "string", "minLength": 1}
    if catalog:
        source_id["enum"] = list(catalog)
    else:
        sources["maxItems"] = 0
    sources["items"] = {
        "type": "object", "additionalProperties": False, "required": ["id", "quote"],
        "properties": {
            "id": source_id,
            "quote": {"type": "string", "minLength": 1, "maxLength": 64000},
        },
    }
    return schema


def _expand_catalog_quote_result(result, schema, catalog):
    if not Draft202012Validator(schema).is_valid(result):
        # Older recordings may supply the complete v3 shape. Keep every supplied
        # identity value for the original binder; never repair a full reference.
        if Draft202012Validator(_model_schema()).is_valid(result):
            return result
        raise ValueError("Unit model response violates bound source reference schema")
    expanded = deepcopy(result)
    for segment in expanded["segments"]:
        for commitment in segment["commitments"]:
            for ref in commitment["sources"]:
                source = catalog[ref["id"]]
                ref.update({key: source[key] for key in SOURCE_IDENTITY})
    return expanded


def _bind_audit(payload, result, request_sha256):
    """Check coverage/references and expose the original model opinion.

    The host does not infer entailment or independently classify personal claims.
    A model that omits commitments or lies about support remains a semantic
    counterexample. Candidate quotes bind the diagnostic to visible words;
    they do not prove its interpretation or completeness.
    """
    if payload.get("audit_version") == VERDICT_UNIT_AUDIT_VERSION:
        return _bind_verdict_unit_audit(payload, result, request_sha256)
    if payload.get("audit_version") == USE_UNIT_AUDIT_VERSION:
        return _bind_use_unit_audit(payload, result, request_sha256)
    if payload.get("audit_version") == GROUNDED_UNIT_AUDIT_VERSION:
        return _bind_grounded_unit_audit(payload, result, request_sha256)
    if payload.get("audit_version") == COMPACT_UNIT_SCOPE_AUDIT_VERSION:
        return _bind_compact_unit_scope_audit(payload, result, request_sha256)
    if payload.get("audit_version") == UNIT_SCOPE_AUDIT_VERSION:
        return _bind_unit_scope_audit(payload, result, request_sha256)
    if not Draft202012Validator(_model_schema()).is_valid(result):
        raise ValueError("Unit model response violates bound schema")
    text = payload.get("candidate_prefix")
    if not isinstance(text, str) or not text:
        raise ValueError("Nonempty bound candidate required for proposition audit")
    by_id = _personal_source_catalog(payload)
    bound = candidate_segments(text)
    if payload.get("review_segments") != [
        {"segment_id": item["segment_id"], "text": item["text"]} for item in bound
    ]:
        raise ValueError("Review segments differ from bound candidate")
    opinions = {item["segment_id"]: item for item in result["segments"]}
    if (len(opinions) != len(result["segments"])
            or set(opinions) != {item["segment_id"] for item in bound}):
        raise ValueError("Segment IDs must cover the bound candidate exactly once")
    propositions = [dict(item, **{
        key: deepcopy(opinions[item["segment_id"]][key])
        for key in ("commitments", "questions")
    }) for item in bound]
    unsupported = False
    for item in propositions:
        for question in item["questions"]:
            if not question.strip() or question not in item["text"]:
                raise ValueError("Question differs from bound segment")
        for commitment in item["commitments"]:
            if not commitment["claim"].strip() or commitment["claim"] not in item["text"]:
                raise ValueError("Commitment differs from bound segment")
            support, refs = commitment["support"], commitment["sources"]
            if support == "supported" and not refs:
                raise ValueError("Personal commitment requires an explicit source relation")
            referenced = []
            for ref in refs:
                source = by_id.get(ref["id"])
                if (source is None or any(ref[key] != source[key] for key in SOURCE_IDENTITY)
                        or not ref["quote"].strip() or ref["quote"] not in source["text"]):
                    raise ValueError("Commitment reference differs from admitted source")
                referenced.append(source)
            if commitment["kind"] == "fiction" and support == "supported" and not any(
                source["kind"] == "current" for source in referenced
            ):
                raise ValueError("Fiction requires a current user authorization reference")
            unsupported |= support in {"unsupported", "uncertain"}
    model_opinion = {key: result[key] for key in BASE_MODEL_FIELDS}
    decision = model_opinion['decision']
    if unsupported:
        decision = 'reject'
    elif decision == 'approve':
        if not result['constraints_satisfied'] or not result['consistent_with_displayed']:
            decision = 'reject'
        elif not result['self_contained'] or not result['no_open_dependency']:
            decision = 'hold'
    return {**model_opinion, "audit_version": AUDIT_VERSION, "propositions": propositions,
            "model_opinion": model_opinion,
            "decision": decision,
            **({"constraints_satisfied": False} if unsupported else {}),
            "request_sha256": request_sha256}


def _paragraph_end(text, index):
    """End of a blank-line run starting at a newline, or None for a single break."""
    cursor = index + 1
    while cursor < len(text) and text[cursor] in SEPARATOR_SPACE:
        cursor += 1
    if cursor >= len(text) or text[cursor] != "\n":
        return None
    while cursor < len(text) and (text[cursor] == "\n" or text[cursor] in SEPARATOR_SPACE):
        cursor += 1
    return cursor


def first_faithful_boundary(text, start):
    """Earliest ``(kind, end)`` display boundary at or after ``start``, else None.

    Kinds: ``paragraph`` (blank line), ``line`` (single line break, including the
    two-space line break), ``sentence`` (sentence-final punctuation plus closing
    marks). The returned end offset keeps the model's own separator characters in
    the unit that precedes them.
    """
    if not isinstance(text, str) or type(start) is not int or not 0 <= start <= len(text):
        raise ValueError("Bounded observed text and offset required")
    index = start
    while index < len(text):
        char = text[index]
        if char == "\n":
            paragraph = _paragraph_end(text, index)
            return ("paragraph", paragraph) if paragraph is not None else ("line", index + 1)
        if char in SENTENCE_ENDS:
            end = index + 1
            while end < len(text) and text[end] in CLOSING_MARKS:
                end += 1
            return ("sentence", end)
        index += 1
    return None


def candidate_segments(text):
    """Host-owned exact ranges for a small number of whole sentence/paragraph units.

    Single line breaks do not fragment the audit. Separators stay with the preceding
    unit and a bounded final unit owns any remaining text. IDs bind the ordinal,
    codepoint range and original UTF-8 text, rather than a model's character count.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Nonempty candidate required for review segments")
    segments, start = [], 0
    while start < len(text):
        end = len(text)
        if len(segments) < MAX_REVIEW_SEGMENTS - 1:
            cursor = start
            while found := first_faithful_boundary(text, cursor):
                kind, boundary = found
                if kind != "line":
                    end = boundary
                    break
                cursor = boundary
            while end < len(text) and text[end].isspace():
                end += 1
        span = {"start": start, "end": end, "text": text[start:end]}
        segments.append({
            "segment_id": f"segment-{len(segments) + 1}-{digest(span)[:16]}", **span,
        })
        start = end
    return segments


def semantic_review_view(payload):
    """One semantic reading of a sealed host packet, without new evidence.

    The complete packet remains the audit/repair authority. The model needs its
    actual words and responsibilities, not host fingerprints or three copies of
    the candidate. Source references only replace identical admitted text; they
    never turn assistant history, exchanges or operation receipts into proof.
    """
    if payload.get("audit_version") == VERDICT_UNIT_AUDIT_VERSION:
        view = semantic_review_view({**payload, "audit_version": USE_UNIT_AUDIT_VERSION})
        return {**view, "audit_version": VERDICT_UNIT_AUDIT_VERSION,
                "view_version": "unit-verdict-semantics.v1"}
    if payload.get("audit_version") == USE_UNIT_AUDIT_VERSION:
        frame = _validate_conversation_frame(payload)
        view = semantic_review_view({**payload, "audit_version": UNIT_SCOPE_AUDIT_VERSION})
        return {**view, "audit_version": USE_UNIT_AUDIT_VERSION,
                "view_version": "unit-meaning-use-semantics.v1", "conversation_frame": frame,
                "sources": [{key: source[key] for key in ("id", "kind", "role", "text")}
                            for source in payload["personal_sources"]]}
    if payload.get("audit_version") == GROUNDED_UNIT_AUDIT_VERSION:
        view = semantic_review_view({**payload, "audit_version": UNIT_SCOPE_AUDIT_VERSION})
        return {**view, "audit_version": GROUNDED_UNIT_AUDIT_VERSION,
                "view_version": "unit-grounding-semantics.v1"}
    segments = deepcopy(payload.get("review_segments", []))
    text = payload.get("candidate_prefix", "")
    if not segments or "".join(item["text"] for item in segments) != text:
        raise ValueError("Semantic view requires exact bound review segments")
    sources = deepcopy(payload.get("personal_sources", []))
    view = {"view_version": "unit-review-semantics.v1", "review_segments": segments,
            "personal_sources": sources, "final": payload.get("final", False)}
    current = [item for item in sources if item.get("kind") == "current"
               and item.get("text") == payload.get("user_text")]
    if len(current) == 1:
        view["user_source_id"] = current[0]["id"]
    else:
        # Small independent assessor fixtures can have no admitted catalog.
        # This is current request text only, never a new citable source.
        view["user_text"] = payload.get("user_text", "")
    context = payload.get("context")
    if context is not None:
        context = json.loads(context)
        if "evidence_access" in context:
            from virtual_companion.boundary.history_readset import validate_evidence_access

            validate_evidence_access(context["evidence_access"], facts=context.get("memories"),
                                     retrieved_dialogue=context.get("retrieved_dialogue"))
        if "context_time" in context:
            validate_context_time(context["context_time"], context)
        by_id = {item["id"]: item for item in sources}
        for item in context.get("history", []):
            source = by_id.get("history:" + item.get("id", ""))
            if (item.get("role") == "user" and source is not None
                    and source.get("text") == item.get("text")):
                del item["text"]
                item["personal_source_id"] = source["id"]
        for item in context.get("memories", []):
            source = by_id.get("memory:" + item.get("record_id", ""))
            if source is not None and source.get("text") == item.get("content"):
                del item["content"]
                item["personal_source_id"] = source["id"]
        for fragment in context.get('retrieved_dialogue', []):
            for item in fragment['members']:
                source = by_id.get('history:' + item['source_id'])
                if (item['role'] == 'user' and source is not None
                        and source.get('text') == item.get('text')):
                    del item['text']
                    item['personal_source_id'] = source['id']
    view["context"] = context
    if 'public_evidence' in payload:
        from virtual_companion.boundary.public_capture import validate_public_capture

        view['public_evidence'] = deepcopy(validate_public_capture(payload['public_evidence']))
    shown = payload.get("displayed_prefix", "")
    if not text.startswith(shown):
        raise ValueError("Displayed content must belong to the bound candidate")
    view["displayed_segments"] = []
    offset = 0
    for item in segments:
        end = offset + len(item["text"])
        if end <= len(shown):
            view["displayed_segments"].append(item["segment_id"])
        elif offset < len(shown):
            view["displayed_partial"] = {
                "segment_id": item["segment_id"], "text": shown[offset:end]}
        offset = end
    for key in ("forbidden_characters", "constraints", "unparsed_constraints",
                "response_task_contract", "response_type"):
        if key in payload:
            view[key] = deepcopy(payload[key])
    if payload.get("audit_version") in (
        UNIT_SCOPE_AUDIT_VERSION, COMPACT_UNIT_SCOPE_AUDIT_VERSION,
    ):
        view["audit_version"] = UNIT_SCOPE_AUDIT_VERSION
        view["view_version"] = "unit-scope-semantics.v1"
        view["units"] = view.pop("review_segments")
        view["sources"] = [{key: source[key] for key in ("id", "kind", "text")}
                           for source in view.pop("personal_sources")]
        view["displayed_units"] = view.pop("displayed_segments")
    if payload.get("audit_version") == COMPACT_UNIT_SCOPE_AUDIT_VERSION:
        units, sources = _compact_unit_scope_bindings(payload)
        unit_ids = {item["segment_id"]: alias for alias, item in units.items()}
        source_ids = {item["id"]: alias for alias, item in sources.items()}
        view["audit_version"] = COMPACT_UNIT_SCOPE_AUDIT_VERSION
        view["view_version"] = "unit-scope-compact-semantics.v1"
        view["units"] = [{"id": alias, "text": item["text"]} for alias, item in units.items()]
        view["sources"] = [{"id": alias, "kind": item["kind"], "text": item["text"]}
                           for alias, item in sources.items()]
        if "user_source_id" in view:
            view["user_source_id"] = source_ids[view["user_source_id"]]
        if context is not None:
            for item in [*context.get("history", []), *context.get("memories", []),
                         *(member for fragment in context.get('retrieved_dialogue', [])
                           for member in fragment['members'])]:
                if "personal_source_id" in item:
                    item["personal_source_id"] = source_ids[item["personal_source_id"]]
        view["displayed_units"] = [unit_ids[item] for item in view["displayed_units"]]
        if "displayed_partial" in view:
            partial = view["displayed_partial"]
            view["displayed_partial"] = {
                "unit_id": unit_ids[partial["segment_id"]], "text": partial["text"],
            }
    return view


class StructuredUnitAssessor:
    """Structured stage over the existing controlled invoker and cancellation."""

    def __init__(self, invoke, cancelled, *, audit_version=AUDIT_VERSION,
                 source_reference_wire=LEGACY_SOURCE_REFERENCE_WIRE):
        if not callable(invoke) or not callable(getattr(cancelled, "is_set", None)):
            raise ValueError("Controlled model invoker and cancellation required")
        if audit_version not in SUPPORTED_AUDIT_VERSIONS:
            raise ValueError("Unknown unit audit version")
        if source_reference_wire not in (
            LEGACY_SOURCE_REFERENCE_WIRE, CATALOG_SOURCE_REFERENCE_WIRE,
        ):
            raise ValueError("Unknown source reference wire")
        if (source_reference_wire == CATALOG_SOURCE_REFERENCE_WIRE
                and audit_version != AUDIT_VERSION):
            raise ValueError("Catalog source references require proposition audit")
        self.invoke, self.cancelled = invoke, cancelled
        self.audit_version = audit_version
        self.source_reference_wire = source_reference_wire

    def __call__(self, packet):
        if self.cancelled.is_set():
            raise InterruptedError("Unit assessment cancelled")
        if (
            not isinstance(packet, dict)
            or set(packet) != {"payload", "sha256"}
            or digest(packet["payload"]) != packet["sha256"]
        ):
            raise ValueError("Bound unit review packet required")
        data = canonical(packet["payload"])
        payload = json.loads(data)
        if payload.get("audit_version", AUDIT_VERSION) != self.audit_version:
            raise ValueError("Unit audit differs from bound selector")
        request_sha256 = packet["sha256"]
        if len(data.encode()) > 128000:
            raise ValueError("Unit assessment capacity exceeded")
        catalog = None
        if self.audit_version == VERDICT_UNIT_AUDIT_VERSION:
            schema = _verdict_unit_model_schema()
            task_system = TASK_SYSTEM if payload.get("response_task_contract") else ""
            messages = [
                {"role": "system", "content": VERDICT_UNIT_SYSTEM + task_system
                 + VERDICT_AUDIT_SYSTEM
                 + "\nunits为完整助手原文，sources保留原话和role，frame只绑定身份；"
                 "context保留当前背景/待答/实际操作，displayed_units只绑定已显示范围。"},
                {"role": "user", "content": canonical(semantic_review_view(payload))},
            ]
        elif self.audit_version == USE_UNIT_AUDIT_VERSION:
            schema = _use_unit_model_schema()
            task_system = TASK_SYSTEM if payload.get("response_task_contract") else ""
            messages = [
                {"role": "system", "content": USE_UNIT_SYSTEM + task_system + USE_AUDIT_SYSTEM
                 + "\n输入units是完整助手候选；sources保留原文/原role。"
                 "conversation_frame只给角色与身份，"
                 "context保持有效背景/待答/实际操作；displayed_units仅证明已显示。"},
                {"role": "user", "content": canonical(semantic_review_view(payload))},
            ]
        elif self.audit_version == GROUNDED_UNIT_AUDIT_VERSION:
            schema = _grounded_unit_model_schema()
            task_system = TASK_SYSTEM if payload.get("response_task_contract") else ""
            messages = [
                {"role": "system", "content": GROUNDED_UNIT_SYSTEM + task_system
                 + GROUNDED_AUDIT_SYSTEM
                 + "\n输入是unit-grounding-semantics.v1；units保留完整原单元，sources保留有效原文。"
                 "user_source_id指当前原话，context里的personal_source_id引用同表来源；"
                 "其他历史、操作、访问状态保持原角色和范围。displayed_units/displayed_partial"
                 "只标明已显示范围，不为其中断言提供依据。"},
                {"role": "user", "content": canonical(semantic_review_view(payload))},
            ]
        elif self.audit_version == COMPACT_UNIT_SCOPE_AUDIT_VERSION:
            schema = _compact_unit_scope_model_schema()
            task_system = TASK_SYSTEM if payload.get("response_task_contract") else ""
            messages = [
                {"role": "system", "content": UNIT_SCOPE_SYSTEM + task_system
                 + COMPACT_UNIT_SCOPE_AUDIT_SYSTEM
                 + "\n输入是unit-scope-compact-semantics.v1，units和sources保留全部原文。"
                 "user_source_id指当前原话；context内personal_source_id指同表来源原文，"
                 "其他历史、操作结果、待续及访问状态按其原角色和有效范围判断。"
                 "displayed_units/displayed_partial保留已显示范围，不证明个人事实。"
                 "只按schema输出，不复述输入，别名仅在本请求内有效。"},
                {"role": "user", "content": canonical(semantic_review_view(payload))},
            ]
        elif self.audit_version == UNIT_SCOPE_AUDIT_VERSION:
            schema = _unit_scope_model_schema()
            task_system = TASK_SYSTEM if payload.get("response_task_contract") else ""
            messages = [
                {"role": "system", "content": UNIT_SCOPE_SYSTEM + task_system
                 + UNIT_SCOPE_AUDIT_SYSTEM
                 + "\n输入是unit-scope-semantics.v1：units是完整原单元；sources只含同封包有效来源。"
                 "user_source_id指当前原话；context内personal_source_id指同表原文，"
                 "其他历史、操作结果、待续及访问状态按其原角色和有效范围判断。"
                 "displayed_units/displayed_partial是已显示范围，不证明个人事实。"
                 "输出只按schema，不复述输入。"},
                {"role": "user", "content": canonical(semantic_review_view(payload))},
            ]
        else:
            schema = _model_schema()
            audit_system = AUDIT_SYSTEM
            if self.source_reference_wire == CATALOG_SOURCE_REFERENCE_WIRE:
                # payload is the private, canonical snapshot taken before invoke.
                catalog = _personal_source_catalog(payload)
                schema = _catalog_quote_model_schema(catalog)
                audit_system = audit_system.replace(
                    "来源的id、owner、source_id、role、revision原样对应；",
                    "sources每个引用只输出{id,quote}，id必须取本封包来源目录；"
                    "owner、source_id、role、revision由宿主补齐；",
                )
            task_system = TASK_SYSTEM if payload.get("response_task_contract") else ""
            messages = [
                {"role": "system", "content": SYSTEM + task_system + audit_system
                 + "\n输入是unit-review-semantics.v1：review_segments按顺序拼成完整待审候选。"
                 "user_source_id指personal_sources中的当前原话；其余来源保持原身份和精确文字。"
                 "context内personal_source_id只引用同表原文，其他历史、操作结果与待续线索仍按其角色。"
                 "displayed_segments及displayed_partial标明已显示范围，不证明其中的个人事实。"
                 "final说明生成是否结束，不改变同一句话的个人事实责任。"
                 "输出结构由调用的schema规定，不复述输入。"},
                {"role": "user", "content": canonical(semantic_review_view(payload))},
            ]
        context = json.loads(payload["context"]) if payload.get("context") else {}
        if 'public_evidence' in payload:
            from virtual_companion.boundary.public_capture import PUBLIC_RESPONSIBILITY

            messages[0]['content'] += PUBLIC_RESPONSIBILITY
        from virtual_companion.text.character import character_review_messages

        messages[1:1] = character_review_messages(context.get("character_context"))
        if "context_time" in context:
            # The sealed admission owns this finite observation. Do not sample
            # a second clock or replace it with the review's arrival time.
            validate_context_time(context["context_time"], context)
            messages[0]["content"] += CONTEXT_TIME_SYSTEM
        result = self.invoke("stream_unit_review", messages, deepcopy(schema), self.cancelled)
        if self.cancelled.is_set():
            raise InterruptedError("Unit assessment cancelled")
        canonical(result)
        if catalog is not None:
            result = _expand_catalog_quote_result(result, schema, catalog)
        return _bind_audit(payload, result, request_sha256)


class UserConstraintViolation(ValueError):
    """A host-verifiable user constraint cannot be met by this answer."""


class CandidateRejected(ValueError):
    """A validated rejection stops this candidate and carries its exact feedback."""

    def __init__(self, verdict, text=None):
        super().__init__("Complete candidate explicitly rejected")
        self.verdict, self.text = verdict, text


class BoundedUnitReviewer:
    """At most one early opinion and one final opinion per attempted turn.

    Early eligibility needs an observed faithful boundary (blank line, line break,
    two-space line break or sentence end). The boundary only schedules a review; it
    never approves content. Candidates are faithful prefixes of the observed bytes,
    so the displayed text is the model's own text without inserted separators.

    Ordinary turns pay for the work they actually need: one early opinion only while
    the producer is still generating (the coordinator skips early opinions once the
    model has stopped), plus one final opinion that owns the complete text. Unclosed
    semantics, source conflicts and explicit user constraints keep the conservative
    path: buffering plus an explicit opinion before anything is shown.

    ``review_policy="host-risk"`` (opt-in, never the default) routes that work: a
    candidate the host scan does not flag is published under the host risk policy
    receipt (``approval_source="host_policy"``, ``model_approved=False``, revision
    pinned to the scan version), and a flagged candidate keeps the model opinion.
    Unflagged is not verified, so this policy needs its own evidence before any
    default entry may use it.

    Explicitly bounded user output constraints are recognized by the host and
    checked on the display surface: they withhold early units and reject a
    violating answer without consulting the model.
    """

    def __init__(self, claim, request, assess, *, early_limit=1, review_policy="model",
                 response_type="answer", audit_version=AUDIT_VERSION):
        if not isinstance(claim, RunClaim) or not isinstance(request, ShortRequest):
            raise ValueError("Typed stream claim and request required")
        if request.action != "speak" or request.risks or not callable(assess):
            raise ValueError("Explicit ordinary-text assessor required")
        if type(early_limit) is not int or not 0 <= early_limit <= 2:
            raise ValueError("Bounded early assessment limit required")
        if review_policy not in REVIEW_POLICIES:
            raise ValueError("Unknown unit review policy")
        if audit_version not in SUPPORTED_AUDIT_VERSIONS:
            raise ValueError("Unknown unit audit version")
        if audit_version != AUDIT_VERSION and review_policy != "model":
            raise ValueError("Unit scope audit requires the model opinion")
        self.audit_version = audit_version
        self.claim, self.request, self.assess = claim, request, assess
        self.early_limit = early_limit
        self.review_policy = review_policy
        if response_type not in {"answer", "ack"}:
            raise ValueError("Spoken response type required for unit review")
        self.response_type = response_type
        self.candidate_id = uuid4().hex
        self.policy = PresentationPolicy(64000, 16000, 3, False)
        self.approved = None
        self.early_calls = 0
        self.final_calls = 0
        self.last_early_candidate = None
        self.deferred_early_opinion = None
        self.rejection = None
        self.constraints, self.unparsed_constraints = recognize_user_constraints(
            request.user_text)
        self.task_contract = (recognize_response_task(request.user_text)
                              if response_type == "answer" else None)

    @property
    def approved_text(self):
        return "" if self.approved is None else self.approved.candidate.text

    def _eligible(self, observed, final):
        old = self.approved_text
        if not isinstance(observed, str) or not observed.startswith(old):
            raise ValueError("Observed output cannot revise an approved prefix")
        if final:
            return None if (observed.rstrip().endswith(DANGLING_ENDINGS)
                            or has_open_group(observed)) else observed
        if self.early_calls >= self.early_limit:
            return None
        if (self.constraints or self.unparsed_constraints
                or self.task_contract is not None):
            # A whole-answer constraint cannot be verified from a prefix.
            return None
        # A boundary is only a candidate boundary, never a semantic decision.
        start = len(old)
        while True:
            found = first_faithful_boundary(observed, start)
            if found is None:
                return None
            _, end = found
            candidate = observed[:end]
            if (candidate == old or candidate == self.last_early_candidate
                    or not candidate[len(old):].strip()
                    or candidate.rstrip().endswith(DANGLING_ENDINGS)
                    or has_open_group(candidate)):
                start = end
                continue
            return candidate

    def __call__(self, observed, final):
        if type(final) is not bool:
            raise ValueError("Explicit finality required")
        if self.rejection is not None:
            # This reviewer belongs to one candidate attempt. A bound reject
            # terminates that attempt; appending text cannot reopen its review.
            raise self.rejection
        candidate_text = self._eligible(observed, final)
        old = self.approved_text
        if final and old and candidate_text and not candidate_text[len(old):].strip():
            # Only trailing whitespace remains; the model text is already displayed.
            return self.approved
        if final and candidate_text:
            issues = constraint_issues(candidate_text, self.constraints)
            issues += task_contract_issues(candidate_text, self.task_contract)
            if issues:
                raise UserConstraintViolation(
                    "User output constraint violated on the display surface: "
                    + ",".join(issues))
        if not candidate_text or candidate_text == old:
            return self.approved if final and candidate_text == old else None
        if old and not candidate_text.startswith(old):
            raise ValueError("New unit must follow the immutable display prefix")
        suffix = candidate_text[len(old):]
        if not suffix.strip():
            raise ValueError("Empty display unit")
        units = (() if self.approved is None else self.approved.candidate.units) + (
            MessageUnit(
                f"unit-{1 if self.approved is None else len(self.approved.candidate.units) + 1}",
                "text",
                suffix,
                len(old),
                len(candidate_text),
            ),
        )
        candidate = Presentation(
            Scope(self.claim.companion_id, self.claim.session_id),
            self.claim.input_id,
            self.claim.attempt_id,
            self.candidate_id,
            1,
            candidate_text,
            units,
            "speak",
            SPANS,
            candidate_text,
            response_type=self.response_type,
        )
        if candidate.text != candidate_text:
            raise ValueError("Observed text cannot be represented as immutable units")
        if final:
            if self.final_calls:
                raise ValueError("Final opinion already spent")
            self.final_calls += 1
        else:
            self.early_calls += 1
            self.last_early_candidate = candidate_text
        packet = {
            **({"conversation_frame": self.request.conversation_frame()}
               if self.audit_version in (USE_UNIT_AUDIT_VERSION, VERDICT_UNIT_AUDIT_VERSION)
               else {}),
            "version": VERSION,
            **({"audit_version": self.audit_version}
               if self.audit_version != AUDIT_VERSION else {}),
            "basis": {
                "companion_id": self.claim.companion_id,
                "session_id": self.claim.session_id,
                "input_id": self.claim.input_id,
                "attempt_id": self.claim.attempt_id,
                "state_version": self.claim.state_version,
            },
            "user_text": self.request.user_text,
            "context": self.request.context.payload if self.request.context else None,
            "personal_sources": self.request.personal_sources(
                self.claim.companion_id, self.claim.input_id, self.claim.state_version),
            **({'public_evidence': json.loads(self.request.public_evidence)}
               if self.request.public_evidence is not None else {}),
            "forbidden_characters": self.request.forbidden_characters,
            "observed_text": observed,
            "candidate_prefix": candidate_text,
            "review_segments": [{"segment_id": item["segment_id"], "text": item["text"]}
                                for item in candidate_segments(candidate_text)],
            "displayed_prefix": old,
            "candidate_sha256": candidate.fingerprint,
            "final": final,
            "constraints": [
                {"kind": item.kind, "limit": item.limit, "span": item.span}
                for item in self.constraints
            ],
            "unparsed_constraints": list(self.unparsed_constraints),
            **({"response_task_contract": self.task_contract.to_payload()}
               if self.task_contract is not None else {}),
            **({"response_type": "ack"} if self.response_type == "ack" else {}),
        }
        envelope = {"payload": packet, "sha256": digest(packet)}
        if self.review_policy == "host-risk" and self.task_contract is None:
            scan = scan_unit_risk(self.request.user_text, candidate_text, shown_text=old)
            if not scan.flagged:
                # Nothing to ask the model: the receipt names the policy and the scan
                # revision instead of claiming an opinion that never happened.
                return self._publish(candidate, review_by_policy(
                    candidate, self.policy,
                    f"host-risk-review-v1-{RISK_SCAN_VERSION}-{uuid4().hex}",
                    policy_revision=HOST_RISK_POLICY_REVISION))
        try:
            verdict = self.assess(envelope)
        except InvalidUnitVerdict as error:
            if final or self.audit_version != VERDICT_UNIT_AUDIT_VERSION:
                raise
            # An unusable preliminary opinion grants no display permission.
            # Continue the same producer and spend the existing final opinion
            # on its complete original text; do not normalize or regenerate it.
            self.deferred_early_opinion = {
                "request_sha256": envelope["sha256"], "error": str(error),
                "model_opinion": deepcopy(error.opinion),
            }
            return None
        except CandidateRejected as error:
            verdict = error.verdict
        if (self.audit_version != AUDIT_VERSION
                and (not isinstance(verdict, dict)
                     or verdict.get("audit_version") != self.audit_version)):
            raise ValueError("Whole unit scope opinion required")
        self._validate_opinion(verdict, envelope["sha256"], packet)
        if verdict["decision"] == "reject":
            self.rejection = CandidateRejected(verdict, candidate_text)
            raise self.rejection
        if verdict["decision"] == "hold":
            return None
        if not all(verdict[key] for key in CHECKS):
            raise ValueError("Approval conflicts with unresolved semantic checks")
        return self._publish(candidate, review(
            candidate, self.policy, f"unit-review-v1-{uuid4().hex}", model_approved=True))

    @staticmethod
    def _validate_opinion(verdict, request_sha256, payload):
        # Trusted Python assessor Ports retain their existing typed contract.
        # The actual model wire must pass StructuredUnitAssessor's mandatory audit.
        if not isinstance(verdict, dict) or set(verdict) not in (
            DECISION_FIELDS, DECISION_FIELDS | HOST_AUDIT_FIELDS,
            DECISION_FIELDS | UNIT_SCOPE_HOST_FIELDS,
            DECISION_FIELDS | VERDICT_HOST_FIELDS,
        ):
            raise ValueError("Exact unit opinion required")
        if verdict["request_sha256"] != request_sha256:
            raise ValueError("Foreign unit opinion")
        if (not isinstance(verdict["decision"], str)
                or verdict["decision"] not in {"approve", "hold", "reject"}):
            raise ValueError("Unknown unit decision")
        if any(type(verdict[key]) is not bool for key in CHECKS):
            raise ValueError("Boolean unit checks required")
        if verdict.get("audit_version") == VERDICT_UNIT_AUDIT_VERSION:
            if (set(verdict) != DECISION_FIELDS | VERDICT_HOST_FIELDS
                    or verdict != _bind_verdict_unit_audit(
                        payload, verdict["model_opinion"], request_sha256)):
                raise ValueError("Unit opinion differs from bound complete verdict declarations")
            return
        if verdict.get("audit_version") == USE_UNIT_AUDIT_VERSION:
            if (set(verdict) != DECISION_FIELDS | UNIT_SCOPE_HOST_FIELDS
                    or verdict != _bind_use_unit_audit(
                        payload, verdict["model_opinion"], request_sha256)):
                raise ValueError("Unit opinion differs from bound meaning/use audit")
            return
        if verdict.get("audit_version") == GROUNDED_UNIT_AUDIT_VERSION:
            if (set(verdict) != DECISION_FIELDS | UNIT_SCOPE_HOST_FIELDS
                    or verdict != _bind_grounded_unit_audit(
                        payload, verdict["model_opinion"], request_sha256)):
                raise ValueError("Unit opinion differs from bound grounded unit audit")
            return
        if verdict.get("audit_version") == COMPACT_UNIT_SCOPE_AUDIT_VERSION:
            if (set(verdict) != DECISION_FIELDS | UNIT_SCOPE_HOST_FIELDS
                    or verdict != _bind_compact_unit_scope_audit(
                        payload, verdict["model_opinion"], request_sha256)):
                raise ValueError("Unit opinion differs from bound compact whole unit audit")
            return
        if verdict.get("audit_version") == UNIT_SCOPE_AUDIT_VERSION:
            if (set(verdict) != DECISION_FIELDS | UNIT_SCOPE_HOST_FIELDS
                    or verdict != _bind_unit_scope_audit(
                        payload, verdict["model_opinion"], request_sha256)):
                raise ValueError("Unit opinion differs from bound whole unit audit")
            return
        if "audit_version" in verdict:
            model_opinion = verdict["model_opinion"]
            if not isinstance(model_opinion, dict) or set(model_opinion) != BASE_MODEL_FIELDS:
                raise ValueError("Original model opinion required for proposition audit")
            original = {**model_opinion, "audit_version": verdict["audit_version"],
                        "segments": [{key: item[key] for key in (
                            "segment_id", "commitments", "questions",
                        )} for item in verdict["propositions"]]}
            if verdict != _bind_audit(payload, original, request_sha256):
                raise ValueError("Unit opinion differs from bound proposition audit")

    def _publish(self, candidate, verdict):
        result = ReviewedPresentation(candidate, self.policy, verdict)
        self.approved = result
        return result
