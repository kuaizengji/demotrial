from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "demotrial.html"
NOTES_DATA_FILE = BASE_DIR / "notes-data.js"
PROMPT_FILE = BASE_DIR / "prompt_1.md"
logger = logging.getLogger("taoran.demo")

app = FastAPI(title="Taoran AI Demo API", version="0.2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    sender: Literal["user", "ai"]
    content: str = Field(..., min_length=1, max_length=12000)


class ImagePayload(BaseModel):
    file_name: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(..., min_length=1, max_length=100)
    data_url: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1, max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list)
    latest_image: ImagePayload | None = None
    preferred_text_model: str | None = None
    preferred_vision_model: str | None = None
    preferred_ocr_model: str | None = None
    use_ocr_first: bool = True


class ModelMeta(BaseModel):
    route: Literal["demo", "text", "vision", "ocr_plus_vision"]
    provider: str
    text_model: str | None = None
    vision_model: str | None = None
    ocr_model: str | None = None
    used_demo_fallback: bool = False


class ChatResponse(BaseModel):
    reply: str
    meta: ModelMeta
    ocr_text: str | None = None
    ocr_stem: str | None = None


class ModelSettings(BaseModel):
    provider_name: str = "dashscope-compatible"
    api_key: str | None = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    text_model: str = "qwen-flash"
    backup_text_model: str = "qwen-plus"
    vision_model: str = "qwen3-vl-flash"
    ocr_model: str = "qwen-vl-ocr-latest"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_model_settings() -> ModelSettings:
    return ModelSettings(
        api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        text_model=os.getenv("QWEN_TEXT_MODEL", "qwen-flash"),
        backup_text_model=os.getenv("QWEN_BACKUP_TEXT_MODEL", "qwen-plus"),
        vision_model=os.getenv("QWEN_VISION_MODEL", "qwen3-vl-flash"),
        ocr_model=os.getenv("QWEN_OCR_MODEL", "qwen-vl-ocr-latest"),
    )


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI | None:
    settings = get_model_settings()
    if not settings.enabled:
        return None
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def get_teaching_prompt() -> str:
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8").strip()
    logger.warning("Teaching prompt file not found: %s", PROMPT_FILE)
    return (
        "你是陶然，高考英语老师。讲题、答疑、陪学生练英语。听懂用户在说什么，再自然作答。"
        "阅读、完形、语法、七选五都直接讲。有材料就给答案和依据；材料不够就说明还缺什么。不要说“不支持这种题”。"
    )


def build_demo_reply(message: str, history: list[ChatMessage], latest_image: ImagePayload | None = None) -> str:
    text = message.strip()
    history_size = len(history)

    if not text:
        raise HTTPException(status_code=400, detail="message 不能为空")

    if latest_image is not None:
        return (
            f"图片 `{latest_image.file_name}` 已收到，但当前没连上模型，所以还不能讲这道题。"
            "请检查接口密钥后再发一次。"
        )

    preview = text if len(text) <= 80 else f"{text[:80]}…"
    return (
        f"我收到了：{preview} 当前没连上模型，所以还不能具体讲题。"
        f"这是第 {max(history_size // 2, 0) + 1} 轮，连上之后把原文和题目一起发过来即可。"
    )


def build_structured_reply_json(
    *,
    supported: bool,
    question_type: str,
    subtype: str,
    answer: str,
    confidence: str,
    need_more_context: bool,
    unsupported_reason: str,
    stem_understanding: str,
    reasoning_steps: list[dict[str, object]],
    distractor_analysis: dict[str, str] | None = None,
    knowledge_methodology: list[str] | None = None,
    knowledge_cards: list[str] | None = None,
    follow_up: str = "",
) -> str:
    payload = {
        "supported": supported,
        "question_type": question_type,
        "subtype": subtype,
        "answer": answer,
        "confidence": confidence,
        "need_more_context": need_more_context,
        "unsupported_reason": unsupported_reason,
        "stem_understanding": stem_understanding,
        "reasoning_steps": reasoning_steps,
        "distractor_analysis": distractor_analysis or {"A": "", "B": "", "C": "", "D": ""},
        "knowledge_methodology": knowledge_methodology or [],
        "knowledge_cards": knowledge_cards or [],
        "follow_up": follow_up,
    }
    return json.dumps(payload, ensure_ascii=False)


SPECIFIED_SINGLE_TARGET_RE = re.compile(
    r"只要讲这一空|只要这一空|只讲这一空|只看这一空|"
    r"只要(?:讲|看|说)?第\s*\d+\s*[空题]|"
    r"先(?:讲|看|说)第\s*\d+\s*[空题]?|"
    r"先(?:讲|看|说)第[一二三四五六七八九十]+题|"
    r"只要第[一二三四五六七八九十]+题|"
    r"先讲第一题|先讲第二题|先看第一题|先看第二题",
    re.IGNORECASE,
)
TEACH_ALL_RE = re.compile(
    r"都讲完|一起讲|全部讲|挨个讲|分别讲完|三空都|两空都|所有空|都讲一下",
)
INDEPENDENT_ITEM_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"第[一二三四五六七八九十]+\s*题|"
    r"第\s*\d+\s*题|"
    r"[1-9]\s*[.．、:：)）]\s+(?!A\s*[.．、:：)])"
    r")",
)
ESSAY_REQUEST_RE = re.compile(
    r"(帮我写|代写|写一篇).{0,24}(作文|essay|article)|(作文|essay).{0,12}(帮我写|代写)",
    re.IGNORECASE,
)
RULE_INDEX_CHUNK_RE = re.compile(
    r"(?:根据)?"
    r"(?:教学体系|语法方法论|完型方法论|完形方法论|阅读方法论|七选五方法论|方法论|知识点)?"
    r"(?:中的)?"
    r"(?:"
    r"第\s*\d+\s*(?:条|点)|"
    r"规则\s*\d+"
    r")"
    r"(?:提到了|提到|规定|指出|说)?"
    r"\s*[：:、.]?",
)
RULE_INDEX_SHORT_RE = re.compile(
    r"(?:方法论|知识点)\s*\d+\s*[：:、.]|"
    r"根据(?:知识点|方法论)\s*\d+",
)
TEACHING_SYS_DUMP_RE = re.compile(
    r"在?(?:教学体系|语法方法论|完型方法论|完形方法论|阅读方法论)(?:的(?:语法|完型|完形|阅读)?方法论)?中[，,]?",
)
METHODOLOGY_LIST_DUMP_RE = re.compile(
    r"(?:从|根据|回顾)?(?:语法|完型|完形|阅读)?方法论[ \t]*(?:\n+\s*\d+\.\s+[^\n]*)+",
)
HIGH_TEACHING_LINE_RE = re.compile(
    r"(?m)^\s*(?:[5-9]|[1-9]\d+)\.\s+(?=.*(?:被动|非谓语|介词|完成时|提示词|动词后面|是否.?只用))[^\n]+",
)


def user_specified_single_target(message: str) -> bool:
    return bool(SPECIFIED_SINGLE_TARGET_RE.search(message or ""))


def user_asked_to_teach_all(message: str) -> bool:
    return bool(TEACH_ALL_RE.search(message or ""))


def looks_like_unspecified_independent_questions(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if is_blank_followup(text) or user_specified_single_target(text):
        return False
    if user_asked_to_teach_all(text) and (looks_like_grammar_fill(text) or looks_like_cloze_blanks(text)):
        return False
    if looks_like_cloze_blanks(text) or looks_like_seven_choose_five(text):
        return False
    if re.search(r"改错", text):
        return False
    if len(INDEPENDENT_ITEM_RE.findall(text)) < 2:
        return False
    if looks_like_reading(text) and has_explicit_options(text) and not looks_like_grammar_fill(text):
        return False
    return True


def has_multiple_question_targets(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    return looks_like_unspecified_independent_questions(combined)


def is_essay_request(message: str) -> bool:
    return bool(ESSAY_REQUEST_RE.search((message or "").strip()))


def build_essay_unsupported_reply() -> str:
    return build_structured_reply_json(
        supported=False,
        question_type="综合",
        subtype="代写",
        answer="",
        confidence="low",
        need_more_context=False,
        unsupported_reason="当前只讲题，不代写作文。把题目或要求发来，可以帮你审题、列提纲或改句子。",
        stem_understanding="这是代写整篇作文的请求，不是讲题。",
        reasoning_steps=[
            {
                "step": 1,
                "focus": "任务类型",
                "basis": "用户要的是直接写一篇作文。",
                "conclusion": "这里只讲题，不代写全文。",
            }
        ],
        follow_up="你可以把题目、要点或已经写好的段落发来，我帮你看结构和句子。",
    )


GREETING_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|在吗|hi+|hello)[呀啊吗么嘛！!。.～~\s]*$",
    re.IGNORECASE,
)


def is_simple_greeting(message: str) -> bool:
    return bool(GREETING_RE.match((message or "").strip()))


def build_greeting_reply() -> str:
    return build_structured_reply_json(
        supported=True,
        question_type="综合",
        subtype="寒暄",
        answer="需要确认",
        confidence="high",
        need_more_context=True,
        unsupported_reason="",
        stem_understanding="这是打招呼，还没有题目。",
        reasoning_steps=[
            {
                "step": 1,
                "focus": "先收题目",
                "basis": "目前只有问候，没有原文、空格或选项。",
                "conclusion": "把要讲的题或截图发来即可。",
            }
        ],
        follow_up="语法填空、阅读、完形或截图都可以直接发。",
    )


def deterministic_structured_reply(message: str, image_context: str | None = None) -> str | None:
    if is_simple_greeting(message):
        return build_greeting_reply()
    if is_essay_request(message):
        return build_essay_unsupported_reply()
    combined = "\n".join(part for part in (message, image_context) if part).strip()
    if looks_like_unspecified_independent_questions(combined) and not user_specified_single_target(message):
        return build_multi_question_focus_reply()
    return None


def scrub_rule_index_text(text: str) -> str:
    value = RULE_INDEX_CHUNK_RE.sub("", text or "")
    value = METHODOLOGY_LIST_DUMP_RE.sub("", value)
    value = HIGH_TEACHING_LINE_RE.sub("", value)
    value = re.sub(r'"\s*(?:[5-9]|[1-9]\d+)\.\s+', '"', value)
    value = RULE_INDEX_SHORT_RE.sub("", value)
    value = TEACHING_SYS_DUMP_RE.sub("", value)
    value = re.sub(r"调用教学体系[：:]\s*", "", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip(" \t：:、.-")


def scrub_parsed_rule_indexes(parsed: dict[str, object]) -> None:
    for key in ("stem_understanding", "follow_up", "answer", "subtype", "unsupported_reason"):
        parsed[key] = scrub_rule_index_text(str(parsed.get(key) or ""))
    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for key in ("focus", "basis", "conclusion"):
                step[key] = scrub_rule_index_text(str(step.get(key) or ""))
    methods = parsed.get("knowledge_methodology")
    if isinstance(methods, list):
        parsed["knowledge_methodology"] = [
            scrub_rule_index_text(str(item))
            for item in methods
            if scrub_rule_index_text(str(item))
        ]


def build_multi_question_focus_reply() -> str:
    return build_structured_reply_json(
        supported=True,
        question_type="综合",
        subtype="多题待选择",
        answer="需要确认",
        confidence="high",
        need_more_context=True,
        unsupported_reason="",
        stem_understanding="这条消息里有不止一道待讲的题。为了讲清楚，需要先确定先看哪一道。",
        reasoning_steps=[
            {
                "step": 1,
                "focus": "先选定一题",
                "basis": "同时展开几道独立的题，容易把题号、空格和依据混在一起。",
                "conclusion": "请先指定要讲的题号、空格、截图位置或段落。"
            },
        ],
        knowledge_cards=[],
        follow_up="你说一下先讲哪一题就行，比如“先讲第12空”或“先看图片里第二题”。",
    )


def extract_json_object(text: str) -> str:
    start = -1
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start : index + 1].strip()
                    if candidate.startswith("{") and candidate.endswith("}"):
                        return candidate
                    start = -1
    return ""


def repair_json_candidate(candidate: str) -> str:
    repaired = (candidate or "").strip()
    if repaired.startswith("```"):
        repaired = re.sub(r"^```(?:json|JSON)?\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r'(:\s*\d+)"(?=\s*[,}])', r"\1", repaired)
    repaired = re.sub(r'(:\s*true|:\s*false)"(?=\s*[,}])', r"\1", repaired, flags=re.IGNORECASE)
    return repaired


def _load_structured_dict(candidate: str) -> dict[str, object] | None:
    if not candidate or not candidate.strip():
        return None
    attempts = [candidate, repair_json_candidate(candidate)]
    for attempt in attempts:
        if not attempt:
            continue
        payload: dict[str, object] | None
        try:
            loaded = json.loads(attempt)
            payload = loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            payload = None
            if attempt.lstrip().startswith("{") and "'" in attempt:
                try:
                    import ast

                    lit = ast.literal_eval(attempt)
                    payload = lit if isinstance(lit, dict) else None
                except (SyntaxError, ValueError, MemoryError):
                    payload = None
        if isinstance(payload, dict) and "supported" in payload:
            return payload
    return None


def parse_structured_reply(reply: str) -> dict[str, object] | None:
    candidates = [reply.strip()]
    fence = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", reply or "", flags=re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    extracted = extract_json_object(reply)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in candidates:
        payload = _load_structured_dict(candidate)
        if payload is not None:
            return payload
    return None


INLINE_ABCD_RE = re.compile(
    r"A[\.．、:：)\s][A-Za-z].{0,80}?B[\.．、:：)\s][A-Za-z].{0,80}?"
    r"C[\.．、:：)\s][A-Za-z].{0,80}?D[\.．、:：)\s][A-Za-z]",
    re.IGNORECASE,
)
GRAMMAR_HINT_RE = re.compile(
    r"\((?![A-Da-d]\))([A-Za-z][A-Za-z\s\-']{1,24})\)|"
    r"（(?![A-Da-d]）)([A-Za-z][A-Za-z\s\-']{1,24})）"
)
CLOZE_OPTION_GROUP_RE = re.compile(
    r"(?:^|\n)\s*(?:\(\s*)?(\d{1,2})(?:\s*\))?\s*[.．、:：]?\s*A[\.．、:：)\s]",
    re.MULTILINE,
)
NUMBERED_UNDERSCORE_BLANK_RE = re.compile(r"\(\s*\d{1,2}\s*\)\s*[_.＿—–-]{2,}")
CLOZE_LETTER_ANSWER_RE = re.compile(r"\(\s*\d{1,2}\s*\)\s*[A-D]\b")
BLANK_TEACH_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*\**\s*(空\s*[（(]?\d{1,2}[)）]?(?:\s*[-~—–到至]\s*[（(]?\d{1,2}[)）]?)?|第\s*\d{1,2}(?:\s*[-~—–到至]\s*\d{1,2})?\s*空)\s*[:：]\s*"
)
NUMBERED_TEACH_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*\**\s*((?:空\s*)?\d{1,2}|第\s*\d{1,2}\s*空)\s*[.．、:：]\s*"
)
QUESTION_TEACH_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(第\s*\d{1,2}\s*题)\s*[:：.．、]\s*"
)
LEADING_STEM_LINE_RE = re.compile(
    r"(?i)^(?:\d+\s*[.．、]\s*)?(?:what|why|which|how|when|where|who|according to|"
    r"the (?:passage|author|notice|text|main(?:\s+idea)?)|it can be inferred)\b"
)
LEADING_OPTION_LINE_RE = re.compile(r"(?i)^[A-G][.．、\)]\s+\S")
BLANK_NUM_RE = re.compile(r"(?:空\s*|第\s*)(\d{1,2})(?:\s*空)?|\(\s*(\d{1,2})\s*\)")


def has_explicit_options(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False

    option_markers = [
        r"\bA[\.\)．、:：]\s*",
        r"\bB[\.\)．、:：]\s*",
        r"\bC[\.\)．、:：]\s*",
        r"\bD[\.\)．、:：]\s*",
    ]
    if all(re.search(marker, combined) for marker in option_markers):
        return True
    return bool(INLINE_ABCD_RE.search(combined))


def looks_like_grammar_hint(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False
    return bool(GRAMMAR_HINT_RE.search(combined))


def looks_like_cloze_option_groups(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False
    return len(CLOZE_OPTION_GROUP_RE.findall(combined)) >= 2


def looks_like_cloze_letter_answers(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False
    return len(CLOZE_LETTER_ANSWER_RE.findall(combined)) >= 2


def looks_like_cloze_blanks(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False
    if looks_like_cloze_option_groups(combined) or looks_like_cloze_letter_answers(combined):
        return True
    if NUMBERED_UNDERSCORE_BLANK_RE.search(combined) and has_explicit_options(combined):
        return True
    if re.search(r"第\s*\d+\s*空", combined) and has_explicit_options(combined):
        return True
    if re.search(r"完形|完型", combined) and (
        has_explicit_options(combined) or re.search(r"_{2,}", combined)
    ):
        return True
    if len(re.findall(r"_{2,}", combined)) >= 2 and has_explicit_options(combined):
        return True
    return False


def looks_like_grammar_fill(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False
    if looks_like_grammar_hint(combined):
        return True
    if re.search(r"_{2,}", combined) and not looks_like_cloze_blanks(combined) and not has_explicit_options(combined):
        return True
    return False


SEVEN_OPTION_RE = re.compile(r"(?m)^\s*([A-G])[\.．、\)]\s+(\S.{0,180})$")
PLACE_RELATIVE_RE = re.compile(
    r"\b(?:village|city|town|place|house|home|room|school|park|country|factory|hospital|"
    r"library|museum|office|restaurant|hotel|garden|farm|station)\b"
    r"[^.\n]{0,48}_{2,}[^.\n]{0,20}\b(?:I|he|she|we|they|you)\s+"
    r"(?:spent|lived|stayed|worked|grew|was born|were born)\b",
    re.IGNORECASE,
)
TIME_RELATIVE_RE = re.compile(
    r"\b(?:day|year|morning|afternoon|evening|night|moment|period|season|month|week)\b"
    r"[^.\n]{0,48}_{2,}[^.\n]{0,20}\b(?:I|he|she|we|they|you)\s+"
    r"(?:met|arrived|left|happened|started|finished|was born|were born)\b",
    re.IGNORECASE,
)


def looks_like_seven_choose_five(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False
    if re.search(r"七选五", combined):
        return True
    options = SEVEN_OPTION_RE.findall(combined)
    letters = {letter.upper() for letter, _body in options}
    if not {"E", "F", "G"} <= letters:
        return False
    sentence_options = [
        body for _letter, body in options if len(re.findall(r"[A-Za-z]{2,}", body)) >= 4
    ]
    return len(sentence_options) >= 4


def looks_like_reading(*segments: str | None) -> bool:
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return False
    return bool(
        re.search(
            r"阅读理解|细节题|主旨|推断题|标题题|根据(?:短文|原文|文章|passage)|"
            r"Which of the following|According to (?:the )?(?:passage|text|author)|"
            r"Why (?:did|does|is|was|would)|What (?:does|did|is|can|do) (?:the|we|you)|"
            r"The (?:passage|author|text) (?:mainly|suggests|implies|is)",
            combined,
            flags=re.IGNORECASE,
        )
    )


def extract_candidate_challenge(message: str) -> str:
    patterns = [
        r"为什么不能填\s*[\"“'`]?([^，。；！？\s\"”'`]+)",
        r"为什么不是\s*[\"“'`]?([^，。；！？\s\"”'`]+)",
        r"为什么不选\s*[\"“'`]?([^，。；！？\s\"”'`]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_candidate_explanation(reasoning_steps: object, candidate: str) -> str:
    if not isinstance(reasoning_steps, list):
        return ""

    prioritized_signals = [candidate.lower()] if candidate else []
    fallback_signals = ["不能填", "不成立", "不符合", "缺少", "句子不完整", "非谓语", "不能单独作谓语"]

    for step in reasoning_steps:
        if not isinstance(step, dict):
            continue
        basis = str(step.get("basis", "")).strip()
        conclusion = str(step.get("conclusion", "")).strip()
        focus = str(step.get("focus", "")).strip()
        combined = " ".join(part for part in [focus, basis, conclusion] if part).strip()
        lowered = combined.lower()
        if prioritized_signals and any(signal in lowered for signal in prioritized_signals):
            return conclusion or basis or combined

    for step in reasoning_steps:
        if not isinstance(step, dict):
            continue
        basis = str(step.get("basis", "")).strip()
        conclusion = str(step.get("conclusion", "")).strip()
        focus = str(step.get("focus", "")).strip()
        combined = " ".join(part for part in [focus, basis, conclusion] if part).strip()
        if any(signal in combined for signal in fallback_signals):
            return conclusion or basis or combined
    return ""


def english_word_count(*segments: str | None) -> int:
    combined = "\n".join(segment for segment in segments if segment)
    return len(re.findall(r"[A-Za-z]{3,}", combined))


def chinese_char_count(*segments: str | None) -> int:
    combined = "\n".join(segment for segment in segments if segment)
    return len(re.findall(r"[\u4e00-\u9fff]", combined))


def has_translatable_or_correctable_sentence(message: str) -> bool:
    text = (message or "").strip()
    if not re.search(r"翻译|改错", text):
        return False
    if re.search(r"(翻译|改错)[:：]", text) and english_word_count(text) >= 3:
        return True
    if re.search(r"(翻译|改错)[:：]\s*\S{6,}", text):
        return True
    return chinese_char_count(text) >= 10 or english_word_count(text) >= 5


def is_label_only_prompt(text: str) -> bool:
    cleaned = re.sub(r"[\s，。！？.、：:（）()]+", "", text or "")
    return bool(re.fullmatch(r"(语法填空|完形填空|完型填空|阅读理解|七选五|改错|翻译|语法|完型|完形|阅读)", cleaned))


def has_enough_question_material(*segments: str | None) -> bool:
    text = "\n".join(segment for segment in segments if segment)
    if not text.strip():
        return False
    if is_label_only_prompt(text):
        return False
    if looks_like_grammar_fill(text) or looks_like_cloze_blanks(text):
        return True
    if has_explicit_options(text) and english_word_count(text) >= 20:
        return True
    if has_translatable_or_correctable_sentence(text):
        return True
    return english_word_count(text) >= 18


def is_bare_explain_request(message: str) -> bool:
    text = (message or "").strip()
    if not text or len(text) > 40 or is_blank_followup(text):
        return False
    if has_translatable_or_correctable_sentence(text) or looks_like_grammar_fill(text):
        return False
    return bool(re.search(r"阅读|完形|完型|语法|七选五|改错|翻译|讲题|讲一下", text))


def asked_for_knowledge_cards(message: str) -> bool:
    return bool(re.search(r"闪卡|错题卡|知识点卡片|整理关键词|做成卡片|做成闪卡", message or ""))


_QUESTION_STEM_START = re.compile(
    r"(?i)^(?:(?:第\s*)?\d+\s*[.．、)）]\s+|"
    r"which of the following|according to|"
    r"why (?:did|does|is|was|would|do)|"
    r"what (?:does|did|is|can|do|would)|"
    r"the (?:passage|author|text|best title)|"
    r"it can be inferred|the main idea)"
)


def preferred_question_number(message: str) -> str:
    match = re.search(r"第\s*(\d+)\s*[题空]", message or "")
    return match.group(1) if match else ""


def clip_question_stem(text: str, limit: int = 72) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}…"


def extract_question_stem(ocr_text: str, message: str = "") -> str:
    source = (ocr_text or "").replace("\r", "")
    if not source.strip():
        return ""

    preferred = preferred_question_number(message)
    numbered: list[tuple[str, str]] = []
    question_like: list[str] = []
    for raw_line in re.split(r"\n+", source):
        line = raw_line.strip()
        if not line:
            continue
        numbered_match = re.match(r"^(?:第\s*)?(\d+)\s*[.．、)）]\s+(.+)$", line)
        if numbered_match:
            numbered.append((numbered_match.group(1), line))
        if _QUESTION_STEM_START.search(line):
            question_like.append(line)

    if preferred:
        for number, line in numbered:
            if number == preferred:
                return clip_question_stem(line)
    if question_like:
        return clip_question_stem(question_like[0])
    if numbered:
        return clip_question_stem(numbered[0][1])

    marked = re.search(r"((?:第\s*)?\d+\s*[.．、)）]\s+[^\n]{8,90})", source)
    if marked:
        return clip_question_stem(marked.group(1))
    asked = re.search(r"([A-Z][^?\n]{12,80}\?)", source)
    if asked:
        return clip_question_stem(asked.group(1))
    return ""


def is_hollow_structured(parsed: dict[str, object]) -> bool:
    answer = str(parsed.get("answer") or "").strip()
    stem = str(parsed.get("stem_understanding") or "").strip()
    follow = str(parsed.get("follow_up") or "").strip()
    steps = parsed.get("reasoning_steps")
    useful_steps = False
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("basis") or "").strip() or str(step.get("conclusion") or "").strip() or str(step.get("focus") or "").strip():
                useful_steps = True
                break
    return not answer and not stem and not follow and not useful_steps


def sentence_containing_blank(message: str, blank: str) -> str:
    match = re.search(rf"[^.?\n]*\(\s*{re.escape(blank)}\s*\)[^.?\n]*", message or "")
    return match.group(0) if match else ""


def revise_unjustified_past_perfect(parsed: dict[str, object], message: str) -> None:
    """Only rewrite grammar-fill mistakes like (12) ______ (struggle) ... for years → had struggled.

    Do not touch cloze/discourse past perfect (e.g. reunion 场景的 had been), where there is
    no parenthetical hint word. Cross-sentence timelines are the model's job.
    """
    answer = str(parsed.get("answer") or "")
    if not answer or not re.search(r"\bhad\s+[A-Za-z]+", answer, flags=re.IGNORECASE):
        return

    rewritten: list[tuple[str, str]] = []

    def replace_had(match: re.Match) -> str:
        blank, verb = match.group(1), match.group(2)
        sentence = sentence_containing_blank(message, blank)
        if not sentence:
            return match.group(0)
        hinted = bool(GRAMMAR_HINT_RE.search(sentence)) or bool(
            re.search(rf"\(\s*{re.escape(verb)}\s*\)", sentence, flags=re.IGNORECASE)
        )
        if not hinted:
            return match.group(0)
        if re.search(r"\bby\s+(the time|then|\d{4})\b|\balready\b", sentence, flags=re.IGNORECASE):
            return match.group(0)
        duration_only = bool(
            re.search(
                r"\bfor\s+(?:\d+\s+)?(?:years?|months?|weeks?|days?|hours?|a long time)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        )
        other_pasts = re.findall(r"\b(?:was|were|did|[A-Za-z]{3,}ed)\b", sentence, flags=re.IGNORECASE)
        if duration_only and len(other_pasts) < 2:
            rewritten.append((verb, blank))
            return f"({blank}) {verb}"
        return match.group(0)

    parsed["answer"] = re.sub(r"\((\d{1,2})\)\s*had\s+([A-Za-z]+)", replace_had, answer, flags=re.IGNORECASE)
    if not rewritten:
        return

    def scrub(text: object) -> str:
        value = str(text or "")
        for verb, _blank in rewritten:
            value = re.sub(rf"(?<![A-Za-z])had\s+{re.escape(verb)}\b", verb, value, flags=re.IGNORECASE)
        return value

    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for key in ("focus", "basis", "conclusion"):
                step[key] = scrub(step.get(key))
            combined = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
            if re.search(r"过去完成|had done", combined) and re.search(r"for years|一段时间|先后", combined, flags=re.IGNORECASE):
                step["basis"] = "同一句里只有一个过去动作，for years 不是过去完成的标志。"
                step["conclusion"] = re.sub(r"过去完成时", "一般过去时", str(step.get("conclusion") or ""))

    methods = parsed.get("knowledge_methodology")
    if isinstance(methods, list):
        parsed["knowledge_methodology"] = [
            "同一句只有一个过去动作，且只有 for years → 用一般过去，不填 had done"
            if isinstance(item, str) and re.search(r"过去完成|had done|先后", item)
            else item
            for item in methods
        ]


def revise_complete_relative_clause(parsed: dict[str, object], message: str) -> None:
    """Place/time + a clause that already has its object takes where/when, not which.

    Only the single-blank pattern is rewritten. "in which" / "on which" stays.
    """
    text = message or ""
    if looks_like_cloze_blanks(text) or len(re.findall(r"_{2,}", text)) != 1:
        return
    place = bool(PLACE_RELATIVE_RE.search(text))
    time_clause = bool(TIME_RELATIVE_RE.search(text))
    if place == time_clause:
        return
    target = "where" if place else "when"
    answer = str(parsed.get("answer") or "")
    if re.search(rf"\b{target}\b", answer, flags=re.IGNORECASE):
        return
    if re.search(r"\b(?:in|on|at|to|for)\s+which\b", answer, flags=re.IGNORECASE):
        return
    if not re.search(r"\b(?:which|that|who|whom)\b", answer, flags=re.IGNORECASE):
        return
    parsed["answer"] = re.sub(
        r"\b(?:which|that|who|whom)\b",
        target,
        answer,
        count=1,
        flags=re.IGNORECASE,
    )
    basis = (
        "空格前是地点，从句里的宾语已经齐全，不缺成分，所以填 where，不填 which。"
        if place
        else "空格前是时间，从句已经完整，所以填 when，不填 which。"
    )
    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
            if re.search(rf"\b{target}\b", blob, flags=re.IGNORECASE):
                continue
            if re.search(r"\b(?:which|that|who|whom)\b|关系代词", blob, flags=re.IGNORECASE):
                step["basis"] = basis
                step["conclusion"] = target
    methods = parsed.get("knowledge_methodology")
    if isinstance(methods, list):
        parsed["knowledge_methodology"] = [
            basis
            if isinstance(item, str)
            and re.search(r"\b(?:which|that)\b|关系代词", item, flags=re.IGNORECASE)
            and not re.search(rf"\b{target}\b", item, flags=re.IGNORECASE)
            else item
            for item in methods
        ]


def infer_requested_question_type(message: str) -> str:
    text = message or ""
    if re.search(r"七选五", text):
        return "七选五"
    if re.search(r"阅读", text):
        return "阅读"
    if re.search(r"完型|完形", text):
        return "完型"
    if re.search(r"语法", text):
        return "语法"
    if re.search(r"改错", text):
        return "改错"
    if re.search(r"翻译", text):
        return "翻译"
    return "综合"


def apply_need_material_defaults(parsed: dict[str, object], message: str) -> dict[str, object]:
    question_type = infer_requested_question_type(message)
    ask = {
        "阅读": "把阅读原文、题干和选项发过来，或直接传截图。",
        "完型": "把完形短文、空号和选项发过来，或直接传截图。",
        "语法": "把带空的句子或短文发过来，有提示词也一并写上。",
        "七选五": "把七选五原文和选项 A-G 发过来，或直接传截图。",
        "改错": "把短文改错原文发过来。",
        "翻译": "把要讲的句子或段落发过来。",
    }.get(question_type, "把原文、题干或截图发过来。")
    parsed["supported"] = True
    parsed["question_type"] = question_type
    parsed["answer"] = "需要确认"
    parsed["confidence"] = "high"
    parsed["need_more_context"] = True
    parsed["unsupported_reason"] = ""
    parsed["stem_understanding"] = "还没有原文、题干或选项，没法讲具体哪一题。"
    parsed["reasoning_steps"] = [
        {
            "step": 1,
            "focus": "先补材料",
            "basis": "只有题型、没有题目，无法判断空格、定位句或选项。",
            "conclusion": ask,
        }
    ]
    parsed["distractor_analysis"] = {"A": "", "B": "", "C": "", "D": ""}
    parsed["knowledge_methodology"] = []
    parsed["knowledge_cards"] = []
    parsed["follow_up"] = ask
    return parsed


def normalize_structured_reply(
    reply: str,
    *,
    message: str,
    image_context: str | None = None,
    prior_context: str = "",
) -> str:
    forced = deterministic_structured_reply(message, image_context)
    if forced:
        return forced

    parsed = parse_structured_reply(reply)
    if not parsed:
        if is_bare_explain_request(message) and not has_enough_question_material(message, image_context, prior_context):
            return json.dumps(apply_need_material_defaults({}, message), ensure_ascii=False)
        return reply

    no_material = not has_enough_question_material(message, image_context, prior_context) and not is_blank_followup(message)
    if no_material and (is_bare_explain_request(message) or is_hollow_structured(parsed)):
        parsed = apply_need_material_defaults(parsed, message)
    else:
        parsed["question_type"] = classify_question_type(message, parsed, image_context)
        revise_unjustified_past_perfect(parsed, message)
        revise_complete_relative_clause(parsed, message)
        scrub_fabricated_distractors(parsed, message, image_context)
        if not asked_for_knowledge_cards(message):
            parsed["knowledge_cards"] = []
        scrub_parsed_rule_indexes(parsed)

    return json.dumps(parsed, ensure_ascii=False)


def classify_question_type(
    message: str,
    parsed: dict[str, object],
    image_context: str | None = None,
) -> str:
    claimed = str(parsed.get("question_type") or "").strip().replace("完形", "完型")
    subtype = str(parsed.get("subtype") or "")
    stem = str(parsed.get("stem_understanding") or "")
    answer = str(parsed.get("answer") or "")
    methods = parsed.get("knowledge_methodology")
    method_text = "\n".join(str(item) for item in methods) if isinstance(methods, list) else ""
    combined = "\n".join(
        part for part in (message, image_context, claimed, subtype, stem, answer, method_text) if part
    )

    user_side = "\n".join(part for part in (message, image_context) if part)
    # 题型以学生材料上的题头为准。模型答案里的“(1) B”和解析里的“翻译”不能把整题改判。
    if looks_like_seven_choose_five(user_side):
        return "七选五"
    if re.search(r"语法填空", user_side):
        return "语法"
    if re.search(r"完形填空|完型填空", user_side):
        return "完型"
    if re.search(r"阅读理解", user_side):
        return "阅读"
    if re.search(r"改错", user_side):
        return "改错"

    has_options = has_explicit_options(combined)
    grammar_hint = looks_like_grammar_hint(user_side)
    grammar_fill = looks_like_grammar_fill(user_side)
    cloze_blanks = looks_like_cloze_blanks(user_side)
    cloze_named = bool(re.search(r"完型|完形", user_side))
    reading_like = looks_like_reading(user_side) or claimed == "阅读"

    if re.search(r"七选五", combined):
        return "七选五"
    if has_translatable_or_correctable_sentence(message):
        if re.search(r"改错", message):
            return "改错"
        if re.search(r"翻译", message):
            return "翻译"
    if reading_like and not cloze_blanks and not looks_like_cloze_letter_answers(user_side):
        return "阅读"
    if grammar_hint and not has_options and not cloze_blanks:
        return "语法"
    if cloze_blanks or (cloze_named and not grammar_hint):
        return "完型"
    if grammar_fill:
        return "语法"
    if has_options and not reading_like:
        return "完型"
    if re.search(r"语法", claimed):
        return "语法"
    if re.search(r"完型|完形", claimed):
        return "完型"
    if re.search(r"阅读", claimed):
        return "阅读"
    if re.search(r"改错", claimed):
        return "改错"
    if re.search(r"翻译", claimed):
        return "翻译"
    return claimed or "语法"


def scrub_fabricated_distractors(
    parsed: dict[str, object],
    message: str,
    image_context: str | None = None,
) -> None:
    combined = "\n".join(part for part in (message, image_context) if part)
    if has_explicit_options(combined):
        return
    raw = parsed.get("distractor_analysis")
    if not isinstance(raw, dict) or any(str(raw.get(key) or "").strip() for key in ("A", "B", "C", "D")):
        parsed["distractor_analysis"] = {"A": "", "B": "", "C": "", "D": ""}


CONVERSATION_RULES = """思考时只用自然语言看句子：空在哪、前后是什么、为什么排除、填什么。
禁止在思考中出现 JSON、字段名、schema、输出格式、Markdown、代码块。
禁止写 supported、question_type、reasoning_steps、knowledge_methodology、distractor_analysis、stem_understanding、knowledge_cards、need_more_context。
禁止说“现在写JSON / 构造JSON / 检查字段 / 字段怎么填”。想清楚后直接输出 JSON，思考里不要谈格式。

对话记忆：
- 上文已有题干、原文、图片或已讲空格时，后续追问必须接着用，不得装作没看到。
- 学生只发空号或题号（如 13、第12空）时，视为同一套题继续讲，need_more_context 必须为 false。
- 材料里有多个空时，answer 按空号一次列全，例如 (11) a；(12) struggled；(13) to。完型同样：有几空就列几空，禁止只写前几空。
- 完型/语法同一套题有多个空时，除非学生只要其中几空，否则每一空都要讲。reasoning_steps 一空一步，focus 写空号。禁止只用一段故事复述代替逐空讲解。
- 思考要把每一空都想完再停，不要写到一半说“现在输出”。正式 JSON 的 reasoning_steps 必须覆盖同样的空，依据、原句、排除项都放在正式回复里。
- 讲解用中文。原句和选项可以留英文，判断过程不要写成英文段落。
- 讲解不是翻译。禁止把原句译成中文再贴选项字母。学生附了答案、解析或五三时，只用来核对，不要复述解析。完形按名词复现/概括、动词动作链、形容词副词正负态度、连词逻辑、结尾抽象名词为核心概念来讲。不要用“紧张的表现”代替冲突点。
- 每一空必须引用该空前后原句里的关键信息。排除项要说到和原句哪里冲突。禁止编造原文没有的物品或情节，也不要用“常见反应 / 不自然 / 不匹配 / 紧张的表现”代替理由。
- knowledge_methodology 给 1 到 4 条写完的判断句，“什么条件 → 填/用什么”。不要写半句，不要照抄带错字的原句。完型至少 1 条可迁移规则。禁止“与理解能力相关”这类空话。
- knowledge_cards 默认 []，除非学生明确要闪卡或关键词。
- 没有原文、题干或选项时，不要吐空壳：answer 写“需要确认”，need_more_context 为 true，follow_up 明确要材料。
- 文章后面跟理解题（Why/What/Which/细节/主旨）是阅读，即使有 A/B/C/D 也不要判完型。完型：短文中有空号，且每空有 A/B/C/D 或单词选项。不要因为有下划线空格就把完型改成语法。语法填空才是括号里有提示词、或没有选项的变形填空。
- 阅读/七选五：distractor_analysis 至少写清两个干扰项错在哪，并落到原文冲突点；不要只给正确选项。
- 过去完成需要能定位过去参照点（可在同句或前后句）。不要只因为本句只有一个过去就排除 had done；也不要只因为 for years / for a long time 就填 had/have done。
- 给了要翻译或改错的句子时直接处理，不要因为只有一句就说缺材料。
- 学生说「下一题 / 换题 / 新题」时，只根据本条消息作答，不要沿用上一题的原文、OCR 或答案。
- 两条完全独立的题且没说先讲哪道时，不要挑一题开讲。answer 写“需要确认”，follow_up 问先看哪一道。同一套题的多个空、用户说“先讲第X题/只要这一空/都讲完”时不要拦。
- 思考里禁止写第X条、知识点X、方法论X、教学体系编号。直接说判断规则。
- 括号里是动词、同一句里已经另有谓语时，空格填非谓语，不要再加 was/were。
- 代写整篇作文时 supported 为 false，不写全文。"""

THINKING_NOISE_RE = re.compile(
    r"```|"
    r"(?:构建|构造|组装|生成|填写|填充|输出|按照|遵循|符合)\s*(?:这个|最终|固定|以下|下面)?\s*(?:json|JSON|格式|schema|字段)|"
    r"reasoning_steps|stem_understanding|distractor_analysis|knowledge_cards|"
    r"knowledge_methodology|need_more_context|unsupported_reason|question_type|follow_up|"
    r'"supported"\s*:|固定\s*JSON|合法\s*JSON|字段名|按\s*schema|输出格式|'
    r"JSON\s*对象|json\s*对象|键值对|根据(?:我的)?(?:角色设定|系统规则)|回顾规则|根据规则|查看规则|系统规则|"
    r"检查规则|对照规则|我需要输出|输出必须是|正式回答只输出|字段必须正确|"
    r"现在(?:开始)?(?:写|构造|组装|输出|填写)\s*JSON|写JSON|构造JSON|组装JSON|"
    r"检查字段|字段怎么填|确保不添加任何额外内容|不要\s*Markdown|不要代码块|不要前言|"
    r'"focus"\s*:|"basis"\s*:|"conclusion"\s*:|"step"\s*:|'
    r"列出推理步骤|干扰项剖析|知识点与方法论|采用以下格式|思考结束后|schema|"
    r"第\s*\d+\s*(?:条|点)|规则\s*\d+|知识点\s*\d+|方法论\s*\d+|根据知识点|"
    r"\b(?:basis|conclusion|confidence)\b",
    re.IGNORECASE,
)


THINKING_HANDOFF_RE = re.compile(
    r"(?:现在|接下来|然后|最后|下面)(?:开始)?(?:构造|组装|生成|输出|填写|按照)\s*(?:这个|最终|固定|以下)?\s*(?:JSON|json|字段|格式)|"
    r"思考结束后|正式回答只输出|输出一个 JSON|输出一个JSON|"
    r"现在[，,]根据要求|现在我需要按照要求|现在[，,]\s*(?:输出|写)\s*(?:JSON|json)?|"
    r"(?:question_type|subtype|stem_understanding|reasoning_steps|knowledge_methodology|distractor_analysis)\s*[:：]",
    re.IGNORECASE,
)
THINKING_KEEP_RE = re.compile(
    r"空\s*[（(]?\d|第\s*\d+\s*[空题]|排除|原句|选项|所以|因此|填|改为|冲突|矛盾|发音|时态|因为|不能|不对|正确"
)


def salvage_thinking_sentence(sentence: str) -> str:
    item = scrub_rule_index_text(sentence)
    if not item:
        return ""
    if THINKING_HANDOFF_RE.search(item) and not re.search(r"空\s*[（(]?\d|第\s*\d+\s*空", item):
        return ""
    if re.match(r"^\s*[{[]", item) or re.match(r'^"[a-z_]+"\s*:', item):
        return ""
    noisy = bool(
        THINKING_NOISE_RE.search(item)
        or re.search(r"JSON|字段名|输出格式|schema|知识卡片|输出要求", item, flags=re.IGNORECASE)
    )
    if not noisy:
        return item
    kept = THINKING_NOISE_RE.sub("", item)
    kept = re.sub(r"JSON|字段名|输出格式|schema|知识卡片|输出要求", "", kept, flags=re.IGNORECASE)
    kept = re.sub(r"在里[，,]?", "", kept)
    kept = re.sub(r"[ \t]{2,}", " ", kept).strip(" ，,；;、:：")
    cjk = len(re.findall(r"[\u4e00-\u9fff]", kept))
    if cjk >= 16 and THINKING_KEEP_RE.search(kept):
        return kept
    return ""


def sanitize_thinking_text(text: str, live: bool = False) -> str:
    source = re.sub(r"```(?:json)?[\s\S]*?```", "\n", text or "", flags=re.IGNORECASE)
    source = re.sub(r'\{[\s\S]*?"supported"\s*:[\s\S]*?\}\s*', "\n", source)
    source = RULE_INDEX_CHUNK_RE.sub("", source)
    source = METHODOLOGY_LIST_DUMP_RE.sub("", source)
    source = HIGH_TEACHING_LINE_RE.sub("", source)
    source = re.sub(r'"\s*(?:[5-9]|[1-9]\d+)\.\s+', '"', source)
    source = RULE_INDEX_SHORT_RE.sub("", source)
    source = TEACHING_SYS_DUMP_RE.sub("", source)
    source = re.sub(r"调用教学体系[：:]\s*", "", source)
    json_start = re.search(r'```|\{\s*"(?:supported|question_type|reasoning_steps|stem_understanding)"', source)
    if json_start:
        source = source[: json_start.start()]
    handoffs = list(THINKING_HANDOFF_RE.finditer(source))
    if handoffs:
        cut_at = handoffs[-1].start()
        tail = source[cut_at:]
        if not re.search(r"空\s*[（(]?\d|第\s*\d+\s*[空题]|排除|原句|原文", tail):
            source = source[:cut_at]
    source = scrub_meta_teaching_lines(source)

    cleaned: list[str] = []
    for block in re.split(r"\n+", source):
        kept: list[str] = []
        for sentence in re.split(r"(?<=[。！？!?\n])", block):
            item = salvage_thinking_sentence(sentence)
            if not item:
                continue
            kept.append(item)
        if kept:
            cleaned.append("".join(kept))

    if live and cleaned:
        last = cleaned[-1]
        if last and not re.search(r"[。！？!?\n]$", last) and re.search(
            r'JSON|字段|schema|reasoning_|knowledge_|supported|输出格式|格式要求|"[a-z_]+"\s*:|第\s*\d+\s*条',
            last,
            flags=re.IGNORECASE,
        ):
            cleaned.pop()
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(cleaned)).strip()


def thinking_from_structured_reply(reply: str) -> str:
    parsed = parse_structured_reply(reply)
    if not parsed:
        return ""
    parts: list[str] = []
    stem = str(parsed.get("stem_understanding") or "").strip()
    if stem:
        parts.append(stem)
    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            focus = str(step.get("focus") or "").strip()
            basis = str(step.get("basis") or "").strip()
            conclusion = str(step.get("conclusion") or "").strip()
            if re.match(r"^(判断题型|调用方法|本题考查|这是一道)", focus):
                focus = ""
            pieces = []
            if focus:
                pieces.append(f"{focus}：")
            if basis:
                pieces.append(basis if basis.endswith("。") else f"{basis}。")
            if conclusion and conclusion != basis:
                pieces.append(conclusion if conclusion.endswith("。") else f"{conclusion}。")
            sentence = " ".join(pieces).strip()
            if sentence:
                parts.append(sentence)
    return sanitize_thinking_text("\n\n".join(parts))


def blank_nums_in_text(text: str) -> set[int]:
    nums: set[int] = set()
    for match in BLANK_NUM_RE.finditer(text or ""):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue
        value = int(raw)
        if 1 <= value <= 20:
            nums.add(value)
    return nums


def _teach_block_is_useful(body: str) -> bool:
    text = (body or "").strip()
    if len(text) < 24 or len(text) > 1200:
        return False
    if re.search(r"现在我需要|按照要求输出|输出完整|写JSON|构造JSON", text) and len(text) < 48:
        return False
    if re.search(r"完型知识点|名词题[:：]|动词题[:：]|方法论包括", text[:80]):
        return False
    head = text[:180]
    if not re.search(r"_{2,}|原句|原文|排除|冲突|正确|[A-D]\.", head):
        return False
    return bool(re.search(r"[A-Za-z]{3,}|选项|选\s*[A-D]|排除|原句|因为|所以", text))


def _normalize_teach_label(label: str) -> str:
    compact = re.sub(r"\s+", "", label or "")
    if compact.isdigit():
        return f"空{int(compact)}"
    return compact


META_TEACH_LINE_RE = re.compile(
    r"用户提示|教学体系|不要编造|第\s*\d+\s*条|规则\s*\d+|推理步骤|"
    r"^(?:focus|basis|conclusion|answer|question_type|stem_understanding)\s*[:：]|"
    r"^\s*(?:[1-9]\d+)\.\s*",
    re.IGNORECASE,
)


def scrub_meta_teaching_lines(text: str) -> str:
    kept: list[str] = []
    for line in (text or "").splitlines():
        if META_TEACH_LINE_RE.search(line):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _strip_leading_question_stem(body: str) -> str:
    lines = (body or "").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        cjk = len(re.findall(r"[\u4e00-\u9fff]", line))
        if LEADING_STEM_LINE_RE.match(line) and cjk < 8:
            index += 1
            continue
        if LEADING_OPTION_LINE_RE.match(line) and cjk < 4:
            index += 1
            continue
        if re.match(r"^(?:题干|题目)\s*[:：]?$", line):
            index += 1
            continue
        break
    return "\n".join(lines[index:]).strip()


def extract_blank_teach_blocks(text: str) -> list[tuple[str, str]]:
    source = text or ""
    labeled = list(BLANK_TEACH_SPLIT_RE.finditer(source))
    numbered = list(NUMBERED_TEACH_SPLIT_RE.finditer(source))
    questions = list(QUESTION_TEACH_SPLIT_RE.finditer(source))
    occupied = [(match.start(), match.end()) for match in labeled]
    matches = list(labeled)
    for match in numbered + questions:
        if any(not (match.end() <= start or match.start() >= end) for start, end in occupied):
            continue
        if any(abs(match.start() - start) < 6 for start, _end in occupied):
            continue
        matches.append(match)
        occupied.append((match.start(), match.end()))
    matches.sort(key=lambda match: match.start())
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = _normalize_teach_label(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = re.sub(r"\n{3,}", "\n\n", source[match.end():end].strip())
        body = re.sub(r"(?:现在|接下来)?我?需要按照要求输出[\s\S]*$", "", body).strip()
        body = scrub_meta_teaching_lines(body)
        body = _strip_leading_question_stem(body)
        if not _teach_block_is_useful(body):
            continue
        blocks.append((label, body))
    return blocks


def covered_blank_nums_from_steps(steps: object) -> set[int]:
    covered: set[int] = set()
    if not isinstance(steps, list):
        return covered
    for step in steps:
        if not isinstance(step, dict):
            continue
        blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
        covered.update(blank_nums_in_text(blob))
    return covered


def merge_thinking_into_structured_reply(reply: str, thinking: str) -> str:
    parsed = parse_structured_reply(reply)
    if not parsed:
        return reply
    blocks = extract_blank_teach_blocks(thinking)
    if not blocks:
        return reply

    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list):
        steps = []

    def cjk_count(text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fa5]", text or ""))

    def step_blob(step: dict[str, object]) -> str:
        return " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))

    def step_is_dump(step: dict[str, object]) -> bool:
        blob = step_blob(step)
        if len(blob) > 1200:
            return True
        return bool(re.search(r"完型知识点|名词题[:：]|方法论包括", blob[:120]))

    merged: list[dict[str, object]] = []
    used: set[int] = set()
    for step in steps:
        if isinstance(step, dict) and not step_is_dump(step):
            merged.append(step)
            used.update(blank_nums_in_text(step_blob(step)))

    for label, body in blocks:
        nums = blank_nums_in_text(label) or blank_nums_in_text(body[:80])
        if nums and nums <= used:
            target = None
            for step in merged:
                if nums & blank_nums_in_text(step_blob(step)):
                    target = step
                    break
            if target is not None and cjk_count(body) >= cjk_count(step_blob(target)) + 24:
                target["basis"] = body
                target["conclusion"] = ""
                if label:
                    target["focus"] = label
            continue
        merged.append(
            {
                "step": len(merged) + 1,
                "focus": label,
                "basis": body,
                "conclusion": "",
            }
        )
        used.update(nums)

    if not merged:
        return reply
    for index, step in enumerate(merged, start=1):
        step["step"] = index
    parsed["reasoning_steps"] = merged
    return json.dumps(parsed, ensure_ascii=False)


BLANK_FOLLOWUP_RE = re.compile(
    r"^\s*(那|然后|接着|还有|再看|继续|再讲|那再看|那再讲)?"
    r"\s*(第\s*)?[（(]?\s*\d{1,3}\s*[)）]?\s*(空|题|小题)?"
    r"\s*[。.?？!！～~]*\s*$",
    re.IGNORECASE,
)
ASK_BLANK_FOLLOWUP_RE = re.compile(
    r"^(?:可以|请|帮我|麻烦)?(?:详细)?(?:给)?(?:我)?"
    r"(?:讲|讲解|分析|看看|看下)(?:一下)?"
    r"\s*(?:第\s*)?\d{1,3}\s*(?:空|题|小题)?[吗么嘛]?"
    r"[。.?？!！～~]*$",
    re.IGNORECASE,
)


NEXT_BLANK_FOLLOWUP_RE = re.compile(
    r"^\s*(那|然后|接着|还有|再看|继续|再讲|那再看|那再讲)?"
    r"\s*(?:的)?"
    r"\s*(下一空|上一空|下一题|上一题|下空|上空)"
    r"\s*[。.?？!！～~]*\s*$",
    re.IGNORECASE,
)


CONTEXT_FOLLOWUP_RE = re.compile(
    r"(?:第\s*\d{1,3}\s*[空题小题]|上[一]?[空题]|下[一]?[空题]|这[一]?[空题])"
    r".{0,40}?"
    r"(?:为什么|为何|再讲|怎么|如何|可以|不行|不能|不选|不是|不对)"
    r"|"
    r"(?:为什么|为何|再讲一下|怎么选|怎么填)"
    r".{0,24}?"
    r"(?:第\s*\d{1,3}\s*[空题小题]|这[一]?空)"
)


def is_blank_followup(message: str) -> bool:
    text = (message or "").strip()
    if not text or len(text) > 80:
        return False
    if BLANK_FOLLOWUP_RE.match(text) or ASK_BLANK_FOLLOWUP_RE.match(text) or NEXT_BLANK_FOLLOWUP_RE.match(text):
        return True
    if len(text) <= 24:
        return False
    if looks_like_grammar_fill(text) or looks_like_cloze_blanks(text) or has_explicit_options(text):
        return False
    if english_word_count(text) > 12:
        return False
    return bool(CONTEXT_FOLLOWUP_RE.search(text))


def is_greeting_history_item(item: ChatMessage) -> bool:
    if item.sender != "ai":
        return False
    text = (item.content or "").strip()
    return text.startswith("你好，我是陶然") or "阅读、完形、语法填空都可以直接问" in text


def compact_assistant_content(content: str) -> str:
    parsed = parse_structured_reply(content)
    if not parsed:
        return (content or "").strip()[:3000]

    parts: list[str] = []
    qtype = str(parsed.get("question_type") or "").strip()
    subtype = str(parsed.get("subtype") or "").strip()
    header = " / ".join(part for part in (qtype, subtype) if part)
    if header:
        parts.append(f"题型：{header}")
    stem = str(parsed.get("stem_understanding") or "").strip()
    if stem:
        parts.append(f"题意：{stem}")
    answer = str(parsed.get("answer") or "").strip()
    if answer:
        parts.append(f"已给答案：{answer}")
    steps = parsed.get("reasoning_steps")
    briefs: list[str] = []
    if isinstance(steps, list):
        for step in steps[:12]:
            if not isinstance(step, dict):
                continue
            line = str(step.get("conclusion") or step.get("basis") or "").strip()
            if line and line not in briefs:
                briefs.append(line)
    if briefs:
        parts.append("已讲要点：" + "；".join(briefs[:12]))
    follow = str(parsed.get("follow_up") or "").strip()
    if follow:
        parts.append(f"收尾：{follow}")
    return "\n".join(parts) if parts else str(content)[:1500]


def looks_like_supplied_answer_key(message: str) -> bool:
    text = message or ""
    if re.search(r"答案|解析|参考答案|五三|正确答案|判题", text):
        return True
    marked = re.findall(r"(?:\(\s*\d{1,2}\s*\)|(?:^|\n)\s*\d{1,2}\s*[.．、])\s*[A-D]\b", text)
    return len(marked) >= 4


def decorate_current_user_message(message: str) -> str:
    text = (message or "").strip()
    if is_blank_followup(text):
        return (
            "这是对上一题的追问。必须结合上文已有题干、原文、图片和已给答案继续讲当前这个空，"
            "不要说只看到了当前这几个字，也不要让学生重发材料。need_more_context 必须为 false。\n\n"
            f"学生追问：{text}"
        )
    notes: list[str] = []
    if looks_like_reading(text) and has_explicit_options(text) and not looks_like_cloze_blanks(text):
        notes.append("这是阅读理解，不是完型。")
    if looks_like_cloze_blanks(text) and not user_specified_single_target(text):
        notes.append(
            "这是完型填空。材料里有几空，reasoning_steps 就写几步，一空一步，focus 写空号，后半篇不能省。"
            "禁止只用一段故事复述代替逐空讲解。stem_understanding 只留一句篇章走向。"
            "每一空写：原句关键词、用哪条判断规则、为什么是这个选项、另外几个选项和原句哪里冲突。"
            "禁止把整句翻译成中文来充当讲解，也不要用“紧张的表现/常见反应/不自然”代替冲突点。"
            "动词不要靠常识补原文没有的动作。keep doing 要看这个动作和后一句是否接得上，原文没写的动作不能选。"
            "思考要把每一空都想完再停，不要在半句写“现在输出”。正式 JSON 覆盖的空必须和思考一样全。"
        )
    if looks_like_grammar_fill(text) and not looks_like_cloze_blanks(text) and PLACE_RELATIVE_RE.search(text):
        notes.append(
            "空格前是地点，后面从句的宾语已经齐全（如 spent my childhood）时填 where，不填 which。"
            "which 只在从句缺主语或宾语时用。"
        )
    if looks_like_grammar_fill(text) and not looks_like_cloze_blanks(text) and TIME_RELATIVE_RE.search(text):
        notes.append("空格前是时间，后面从句已经完整时填 when，不填 which。")
    if looks_like_supplied_answer_key(text):
        notes.append(
            "学生附上了答案或解析，只用来核对，不要当讲稿复述。"
            "不要翻译原句，不要照抄五三/解析的措辞。"
            "按陶然的完形方法讲：名词看复现或概括，动词看动作链和发出者，形容词副词看正负态度，连词看逻辑，结尾抽象名词先当核心概念。"
            "学生要的是怎么判断，不是译文加选项字母。"
        )
    if looks_like_grammar_fill(text) and not looks_like_cloze_blanks(text) and not user_specified_single_target(text):
        if len(re.findall(r"\(\s*\d{1,2}\s*\)", text)) >= 2:
            notes.append("这是同一套语法填空。把能看到的每一空都讲完，answer 按空号列全。")
    if looks_like_grammar_fill(text) and re.search(r"\bfor\s+(?:years|a long time)\b", text, flags=re.IGNORECASE):
        notes.append(
            "for years / for a long time 本身不是完成时标志：先找时间参照。"
            "若参照是现在，考虑现在完成；若材料里另有过去参照点（可在前后句），过去完成仍可能成立；"
            "不要只因为本句只有一个过去动作就排除 had done。"
        )
    if looks_like_grammar_fill(text) and re.search(
        r"_{2,}\s*\([A-Za-z]+\)[^.\n]{0,120}\b(?:became|becomes|is|are|was|were)\b",
        text,
        flags=re.IGNORECASE,
    ):
        notes.append("同一句里如果已经另有谓语，空格处用非谓语（如 written），不要再填 was written / is written。")
    if is_explicit_new_question_turn(text):
        notes.append("这是新题。只根据本条消息里的原文/题干/选项作答，不要沿用上一题的答案或时态结论。")
    if notes:
        return text + "\n\n" + " ".join(notes)
    return text


def is_explicit_new_question_turn(message: str) -> bool:
    return bool(re.search(r"下一题|换题|另一题|新题|重新来一题", message or ""))


def prior_chat_history(history: list[ChatMessage], message: str, *, limit: int) -> list[ChatMessage]:
    prior = list(history)
    if prior and prior[-1].sender == "user" and prior[-1].content.strip() == message.strip():
        prior = prior[:-1]
    return prior[-limit:]


def iter_compacted_history(history: list[ChatMessage], message: str, *, limit: int) -> list[tuple[str, str]]:
    prior = prior_chat_history(history, message, limit=max(limit * 3, 18))
    if is_explicit_new_question_turn(message):
        prior = []
    compacted: list[tuple[str, str]] = []
    for item in prior:
        if is_greeting_history_item(item):
            continue
        content = compact_assistant_content(item.content) if item.sender == "ai" else (item.content or "")
        content = content.strip()
        if not content:
            continue
        role = "assistant" if item.sender == "ai" else "user"
        compacted.append((role, content[:4000]))
    return compacted[-limit:]


def build_text_messages(message: str, history: list[ChatMessage]) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": get_teaching_prompt()},
        {"role": "system", "content": CONVERSATION_RULES},
    ]
    for role, content in iter_compacted_history(history, message, limit=12):
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": decorate_current_user_message(message)})
    return messages


def build_vision_messages(
    message: str,
    history: list[ChatMessage],
    image: ImagePayload,
    ocr_text: str | None = None,
) -> list[dict]:
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                f"{get_teaching_prompt()}\n\n"
                f"{CONVERSATION_RULES}\n\n"
                "当前用户上传了题目图片。结合图片、用户问题和已有上下文作答即可。"
                "如果图里是完型填空，把能看到的每一空都讲完，一空一步写进 reasoning_steps；"
                "思考和正式讲解都要覆盖全部空，不要只写一段情节概述或整句翻译。"
                "学生如果图里带了答案或解析，只用来核对，按判断规则讲，不要复述译文。"
                "不要因为有下划线空格就判成语法填空。"
            ),
        }
    ]
    for role, content in iter_compacted_history(history, message, limit=8):
        messages.append({"role": role, "content": content})

    user_content: list[dict] = [{"type": "text", "text": decorate_current_user_message(message)}]
    if ocr_text:
        ocr_notes: list[str] = []
        if looks_like_cloze_blanks(ocr_text) and not user_specified_single_target(message):
            ocr_notes.append("图中是完型。每一空都要写进 reasoning_steps，不要只写故事概述或译文。")
        if looks_like_supplied_answer_key(ocr_text):
            ocr_notes.append("图里如果有答案或解析，只用来核对。讲解写判断规则和原句冲突，不要翻译原句，不要复述解析。")
        ocr_suffix = f"\n\n{' '.join(ocr_notes)}" if ocr_notes else ""
        user_content.append(
            {
                "type": "text",
                "text": f"下面是 OCR 识别出的参考文本，你可以结合图片一起判断：\n{ocr_text}{ocr_suffix}",
            }
        )
    user_content.append({"type": "image_url", "image_url": {"url": image.data_url}})
    messages.append({"role": "user", "content": user_content})
    return messages


def extract_ocr_text(client: OpenAI, image: ImagePayload, model: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "你是 OCR 识别助手。请只提取图片中的文字内容，保留换行，避免解释。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请提取这张图片中的全部可识别文字。"},
                    {"type": "image_url", "image_url": {"url": image.data_url}},
                ],
            },
        ],
    )
    return completion.choices[0].message.content or ""


def generate_text_reply(client: OpenAI, message: str, history: list[ChatMessage], model: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        temperature=0.5,
        max_tokens=8192,
        messages=build_text_messages(message, history),
    )
    return completion.choices[0].message.content or ""


def generate_vision_reply(
    client: OpenAI,
    message: str,
    history: list[ChatMessage],
    image: ImagePayload,
    model: str,
    ocr_text: str | None = None,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=8192,
        messages=build_vision_messages(message, history, image, ocr_text=ocr_text),
    )
    return completion.choices[0].message.content or ""


def sse_event(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def thinking_for_forced_reply(reply: str) -> str:
    prose = thinking_from_structured_reply(reply)
    if prose:
        return prose
    parsed = parse_structured_reply(reply)
    if parsed and parsed.get("supported") is False:
        return str(parsed.get("unsupported_reason") or parsed.get("stem_understanding") or "当前只讲题，不代写作文。")
    return "这条消息里有不止一道题，先确认要讲哪一道。"


def iter_forced_reply_events(reply: str, meta: ModelMeta, thinking: str | None = None):
    text = thinking if thinking is not None else thinking_for_forced_reply(reply)
    yield sse_event({"type": "status", "stage": "analyze", "text": "正在思考"})
    if text:
        yield sse_event({"type": "thinking", "text": text})
    done_payload: dict[str, object] = {"type": "done", "reply": reply, "meta": meta.model_dump()}
    if text:
        done_payload["thinking"] = text
    yield sse_event(done_payload)


def model_supports_thinking(model: str) -> bool:
    name = (model or "").lower()
    return any(token in name for token in ("qwen3", "qwen-plus", "qwen-flash", "qwen-turbo", "qwq", "qwen-vl-plus", "qwen-vl-max"))


def extract_stream_delta_text(delta: object) -> tuple[str, str]:
    content = getattr(delta, "content", None) or ""
    reasoning = getattr(delta, "reasoning_content", None) or ""
    if not reasoning:
        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
    if not isinstance(content, str):
        content = ""
    if not isinstance(reasoning, str):
        reasoning = ""
    return reasoning, content


def iter_completion_deltas(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict],
    temperature: float,
    enable_thinking: bool = False,
):
    kwargs: dict[str, object] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "stream": True,
        "max_tokens": 8192,
    }
    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True, "thinking_budget": 8192}

    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception:
        if not enable_thinking:
            raise
        kwargs["extra_body"] = {"enable_thinking": True}
        try:
            stream = client.chat.completions.create(**kwargs)
        except Exception:
            logger.exception("Thinking stream failed, retrying without thinking")
            kwargs.pop("extra_body", None)
            stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        reasoning, content = extract_stream_delta_text(delta)
        if reasoning or content:
            yield reasoning, content


def iter_streamed_reply_events(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict],
    temperature: float,
    enable_thinking: bool,
    meta: ModelMeta,
    non_stream_fallback,
    source_message: str = "",
    image_context: str | None = None,
    prior_context: str = "",
):
    content_parts: list[str] = []
    raw_thinking = ""
    emitted_thinking = ""
    try:
        deltas = iter_completion_deltas(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        for reasoning, content in deltas:
            if reasoning:
                raw_thinking += reasoning
                cleaned = sanitize_thinking_text(raw_thinking, live=True)
                if cleaned.startswith(emitted_thinking):
                    delta = cleaned[len(emitted_thinking) :]
                else:
                    delta = cleaned if not emitted_thinking else ""
                if delta:
                    emitted_thinking = cleaned
                    yield sse_event({"type": "thinking", "text": delta})
            if content:
                content_parts.append(content)
                yield sse_event({"type": "content", "text": content})
    except Exception:
        if not enable_thinking:
            raise
        logger.exception("Thinking-enabled stream failed, retrying without thinking")
        content_parts = []
        raw_thinking = ""
        emitted_thinking = ""
        for reasoning, content in iter_completion_deltas(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            enable_thinking=False,
        ):
            if content:
                content_parts.append(content)
                yield sse_event({"type": "content", "text": content})

    reply = "".join(content_parts)
    if not reply.strip():
        logger.warning("Streamed completion was empty, retrying without stream")
        reply = non_stream_fallback()
        if reply:
            yield sse_event({"type": "content", "text": reply})

    reply = normalize_structured_reply(
        reply,
        message=source_message,
        image_context=image_context,
        prior_context=prior_context,
    )
    final_thinking = sanitize_thinking_text(raw_thinking) or emitted_thinking
    if not final_thinking:
        final_thinking = thinking_from_structured_reply(reply)
    if final_thinking:
        reply = merge_thinking_into_structured_reply(reply, final_thinking)
        parsed_reply = parse_structured_reply(reply)
        if parsed_reply:
            scrub_parsed_rule_indexes(parsed_reply)
            steps = parsed_reply.get("reasoning_steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    for key in ("focus", "basis", "conclusion"):
                        step[key] = scrub_meta_teaching_lines(str(step.get(key) or ""))
            reply = json.dumps(parsed_reply, ensure_ascii=False)
    done_payload: dict[str, object] = {"type": "done", "reply": reply, "meta": meta.model_dump()}
    if final_thinking:
        done_payload["thinking"] = final_thinking
    if image_context:
        done_payload["ocr_text"] = image_context
        stem = extract_question_stem(image_context, source_message)
        if stem:
            done_payload["ocr_stem"] = stem
    yield sse_event(done_payload)


def iter_chat_sse(request: ChatRequest):
    settings = get_model_settings()
    client = get_openai_client()
    text_model = request.preferred_text_model or settings.text_model
    vision_model = request.preferred_vision_model or settings.vision_model
    ocr_model = request.preferred_ocr_model or settings.ocr_model

    def demo_done() -> str:
        meta = ModelMeta(
            route="ocr_plus_vision" if request.latest_image and request.use_ocr_first else "vision" if request.latest_image else "demo",
            provider=settings.provider_name,
            text_model=text_model,
            vision_model=vision_model,
            ocr_model=ocr_model,
            used_demo_fallback=True,
        )
        reply = build_demo_reply(request.message, request.history, request.latest_image)
        return sse_event({"type": "done", "reply": reply, "meta": meta.model_dump()})

    if client is None:
        yield sse_event({"type": "status", "stage": "analyze", "text": "正在思考"})
        yield demo_done()
        return

    try:
        if request.latest_image is not None:
            ocr_text = None
            route: Literal["vision", "ocr_plus_vision"] = "vision"
            if request.use_ocr_first:
                route = "ocr_plus_vision"
                yield sse_event({"type": "status", "stage": "ocr", "text": "先把图片里的字认出来"})
                ocr_text = extract_ocr_text(client, request.latest_image, ocr_model)
                if ocr_text:
                    yield sse_event({"type": "ocr", "text": ocr_text})
            forced = deterministic_structured_reply(request.message, ocr_text)
            meta = ModelMeta(
                route=route,
                provider=settings.provider_name,
                vision_model=vision_model,
                ocr_model=ocr_model if route == "ocr_plus_vision" else None,
            )
            if forced:
                yield from iter_forced_reply_events(forced, meta)
                return
            yield sse_event({"type": "status", "stage": "analyze", "text": "先把图里的题目看清楚"})
            yield from iter_streamed_reply_events(
                client,
                model=vision_model,
                messages=build_vision_messages(
                    request.message,
                    request.history,
                    request.latest_image,
                    ocr_text=ocr_text,
                ),
                temperature=0.7,
                enable_thinking=model_supports_thinking(vision_model),
                meta=meta,
                non_stream_fallback=lambda: generate_vision_reply(
                    client,
                    request.message,
                    request.history,
                    request.latest_image,
                    vision_model,
                    ocr_text=ocr_text,
                ),
                source_message=request.message,
                image_context=ocr_text,
                prior_context="\n".join(item.content for item in request.history if item.sender == "user"),
            )
            return

        forced = deterministic_structured_reply(request.message)
        meta = ModelMeta(
            route="text",
            provider=settings.provider_name,
            text_model=text_model,
        )
        if forced:
            yield from iter_forced_reply_events(forced, meta)
            return
        yield sse_event({"type": "status", "stage": "analyze", "text": "正在思考"})
        yield from iter_streamed_reply_events(
            client,
            model=text_model,
            messages=build_text_messages(request.message, request.history),
            temperature=0.5,
            enable_thinking=model_supports_thinking(text_model),
            meta=meta,
            non_stream_fallback=lambda: generate_text_reply(client, request.message, request.history, text_model),
            source_message=request.message,
            prior_context="\n".join(item.content for item in request.history if item.sender == "user"),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Streaming model pipeline failed, fallback to demo reply")
        yield sse_event({"type": "status", "stage": "fallback", "text": "换个方式继续想这道题"})
        yield demo_done()


def run_model_pipeline(request: ChatRequest) -> ChatResponse:
    settings = get_model_settings()
    client = get_openai_client()
    text_model = request.preferred_text_model or settings.text_model
    vision_model = request.preferred_vision_model or settings.vision_model
    ocr_model = request.preferred_ocr_model or settings.ocr_model

    if client is None:
        route = "ocr_plus_vision" if request.latest_image and request.use_ocr_first else "vision" if request.latest_image else "demo"
        return ChatResponse(
            reply=build_demo_reply(request.message, request.history, request.latest_image),
            meta=ModelMeta(
                route=route if route != "demo" else "demo",
                provider=settings.provider_name,
                text_model=text_model,
                vision_model=vision_model,
                ocr_model=ocr_model,
                used_demo_fallback=True,
            ),
        )

    try:
        if request.latest_image is not None:
            ocr_text = None
            route: Literal["vision", "ocr_plus_vision"] = "vision"
            if request.use_ocr_first:
                route = "ocr_plus_vision"
                ocr_text = extract_ocr_text(client, request.latest_image, ocr_model)
            forced = deterministic_structured_reply(request.message, ocr_text)
            if forced:
                return ChatResponse(
                    reply=forced,
                    meta=ModelMeta(
                        route=route,
                        provider=settings.provider_name,
                        vision_model=vision_model,
                        ocr_model=ocr_model if route == "ocr_plus_vision" else None,
                    ),
                    ocr_text=ocr_text,
                    ocr_stem=extract_question_stem(ocr_text or "", request.message) or None,
                )
            reply = generate_vision_reply(
                client,
                request.message,
                request.history,
                request.latest_image,
                vision_model,
                ocr_text=ocr_text,
            )
            reply = normalize_structured_reply(
                reply,
                message=request.message,
                image_context=ocr_text,
                prior_context="\n".join(item.content for item in request.history if item.sender == "user"),
            )
            return ChatResponse(
                reply=reply,
                meta=ModelMeta(
                    route=route,
                    provider=settings.provider_name,
                    vision_model=vision_model,
                    ocr_model=ocr_model if route == "ocr_plus_vision" else None,
                ),
                ocr_text=ocr_text,
                ocr_stem=extract_question_stem(ocr_text or "", request.message) or None,
            )

        forced = deterministic_structured_reply(request.message)
        if forced:
            return ChatResponse(
                reply=forced,
                meta=ModelMeta(
                    route="text",
                    provider=settings.provider_name,
                    text_model=text_model,
                ),
            )
        reply = generate_text_reply(client, request.message, request.history, text_model)
        reply = normalize_structured_reply(
            reply,
            message=request.message,
            prior_context="\n".join(item.content for item in request.history if item.sender == "user"),
        )
        return ChatResponse(
            reply=reply,
            meta=ModelMeta(
                route="text",
                provider=settings.provider_name,
                text_model=text_model,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model pipeline failed, fallback to demo reply")
        return ChatResponse(
            reply=build_demo_reply(request.message, request.history, request.latest_image),
            meta=ModelMeta(
                route="demo",
                provider=settings.provider_name,
                text_model=text_model,
                vision_model=vision_model,
                ocr_model=ocr_model,
                used_demo_fallback=True,
            ),
        )


@app.get("/health")
async def health() -> dict[str, object]:
    settings = get_model_settings()
    return {
        "status": "ok",
        "models_enabled": settings.enabled,
        "provider": settings.provider_name,
        "text_model": settings.text_model,
        "backup_text_model": settings.backup_text_model,
        "vision_model": settings.vision_model,
        "ocr_model": settings.ocr_model,
    }


@app.get("/api/model-config")
async def model_config() -> dict[str, object]:
    settings = get_model_settings()
    return {
        "provider": settings.provider_name,
        "models_enabled": settings.enabled,
        "text_model": settings.text_model,
        "backup_text_model": settings.backup_text_model,
        "vision_model": settings.vision_model,
        "ocr_model": settings.ocr_model,
        "base_url": settings.base_url,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    return run_model_pipeline(request)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    return StreamingResponse(
        iter_chat_sse(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
@app.get("/demotrial.html")
async def index() -> FileResponse:
    if not HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="demotrial.html 不存在")
    return FileResponse(HTML_FILE, headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/notes-data.js")
async def notes_data() -> FileResponse:
    if not NOTES_DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="notes-data.js 不存在")
    return FileResponse(NOTES_DATA_FILE, media_type="application/javascript")
