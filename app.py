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


AIPING_BASE_URL = "https://aiping.cn/api/v1"


class ModelSettings(BaseModel):
    provider_name: str = "aiping"
    api_key: str | None = None
    base_url: str = AIPING_BASE_URL
    text_model: str = "DeepSeek-V4-Pro"
    backup_text_model: str = "DeepSeek-V3.2"
    vision_model: str = "GLM-4.6V"
    ocr_model: str = "DeepSeek-OCR"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@lru_cache(maxsize=1)
def get_model_settings() -> ModelSettings:
    base_url = _env_value("AIPING_BASE_URL", "QWEN_BASE_URL", default=AIPING_BASE_URL) or AIPING_BASE_URL
    provider_name = "aiping" if "aiping.cn" in base_url else "openai-compatible"
    return ModelSettings(
        provider_name=provider_name,
        api_key=_env_value("AIPING_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"),
        base_url=base_url,
        text_model=_env_value("AIPING_TEXT_MODEL", "QWEN_TEXT_MODEL", default="DeepSeek-V4-Pro") or "DeepSeek-V4-Pro",
        backup_text_model=_env_value("AIPING_BACKUP_TEXT_MODEL", "QWEN_BACKUP_TEXT_MODEL", default="DeepSeek-V3.2")
        or "DeepSeek-V3.2",
        vision_model=_env_value("AIPING_VISION_MODEL", "QWEN_VISION_MODEL", default="GLM-4.6V") or "GLM-4.6V",
        ocr_model=_env_value("AIPING_OCR_MODEL", "QWEN_OCR_MODEL", default="DeepSeek-OCR") or "DeepSeek-OCR",
    )


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI | None:
    settings = get_model_settings()
    if not settings.enabled:
        return None
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


PROMPT_BLOCK_RE = re.compile(r"<!--\s*block:\s*([^\s>]+)\s*-->")
KNOWN_QUESTION_TYPES = ("语法", "完型", "阅读", "七选五", "改错", "翻译", "词汇", "长难句", "综合")
QUESTION_TYPE_PATTERN = "七选五|长难句|完型|完形|词汇|语法|阅读|改错|翻译|综合"
SOURCE_WORD_STOP = {
    "the", "and", "for", "that", "with", "from", "this", "have", "was", "were", "are",
    "you", "your", "not", "but", "she", "her", "his", "him", "they", "them", "its",
    "into", "over", "after", "before", "when", "what", "which", "there", "their",
    "been", "will", "would", "could", "should", "about", "because", "than", "then",
    "also", "just", "only", "very", "much", "more", "some", "any", "can", "did",
    "does", "has", "had", "our", "out", "who", "how", "why", "all", "one", "two",
    "is", "it", "of", "in", "on", "to", "as", "at", "be", "by", "or", "if", "so",
    "we", "he", "me", "my", "an",
}


@lru_cache(maxsize=1)
def prompt_blocks() -> dict[str, str]:
    """Split prompt_1.md into task blocks. The archive block is never sent."""
    if not PROMPT_FILE.exists():
        logger.warning("Teaching prompt file not found: %s", PROMPT_FILE)
        return {}
    text = PROMPT_FILE.read_text(encoding="utf-8")
    matches = list(PROMPT_BLOCK_RE.finditer(text))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name == "archive":
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            blocks[name] = body
    return blocks


def prompt_block(name: str) -> str:
    return prompt_blocks().get(name, "").strip()


def teach_system_prompt(question_type: str) -> str:
    """Reason stage only: one type card. The JSON contract is a later respond."""
    card_name = question_type if question_type in KNOWN_QUESTION_TYPES else "综合"
    parts = [
        prompt_block("persona"),
        prompt_block("看空"),
        f"当前题型：{card_name}。只使用下面这一题型的判断。不要套用其他题型的规则。",
        prompt_block(card_name) or prompt_block("综合"),
        prompt_block("memory"),
    ]
    return "\n\n".join(part for part in parts if part)


def format_system_prompt() -> str:
    """Format stage only. No teaching rules, so the model cannot start a new essay."""
    contract = prompt_block("output")
    return "\n\n".join(
        part
        for part in (
            contract,
            "只把已经写好的讲稿收成一个 JSON。basis 保留三句：这一空充当什么；决定它的原词；被排除项为什么接不上。不要收成翻译，不要改成空套话。括号里有提示词时，结论必须是这个词的形式。",
            "knowledge_methodology 写能用到下一题的判断：什么情况就填什么。不要复述这一题的情节。不要补充新情节，不要改答案。",
        )
        if part
    )


def get_teaching_prompt() -> str:
    """Kept for callers that have not classified a type yet. Still not the full archive."""
    return teach_system_prompt("综合")


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
    if is_bare_explain_request(message) and not has_enough_question_material(message, image_context):
        return json.dumps(apply_need_material_defaults({}, message), ensure_ascii=False)
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
# Gaokao English numbers cloze around 41–55 and grammar around 56–65.
MAX_EXAM_ITEM = 70


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


def looks_like_single_grammar_choice(*segments: str | None) -> bool:
    """One blank plus A-D is a grammar choice, not a cloze passage."""
    text = "\n".join(segment for segment in segments if segment).strip()
    if not text or re.search(r"完形|完型|七选五|阅读理解", text):
        return False
    if re.search(r"语法选择|单项选择|单项填空", text):
        return True
    if looks_like_cloze_option_groups(text) or looks_like_cloze_blanks(text):
        return False
    if len(re.findall(r"_{2,}", text)) != 1 or not has_explicit_options(text):
        return False
    return len(re.findall(r"[.!?。]", text)) <= 3 and english_word_count(text) <= 70


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
    rewritten_blanks = {blank for _verb, blank in rewritten}
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            if not any(step_targets_blank(step, blank, steps) for blank in rewritten_blanks):
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


def _relative_basis(place: bool) -> str:
    if place:
        return "空格前是地点，从句里的宾语已经齐全，不缺成分，所以填 where，不填 which。"
    return "空格前是时间，从句已经完整，所以填 when，不填 which。"


def _rewrite_relative_step(parsed: dict[str, object], blank: str, target: str, place: bool) -> None:
    basis = _relative_basis(place)
    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
            if blank and not step_targets_blank(step, blank, steps):
                continue
            if re.search(rf"\b{target}\b", blob, flags=re.IGNORECASE) and not re.search(r"\b(?:which|that)\b", blob, flags=re.IGNORECASE):
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


def revise_complete_relative_clause(parsed: dict[str, object], message: str) -> None:
    """Place/time + a clause that already has its object takes where/when, not which.

    Applies to each matching sentence, including multi-blank grammar fills.
    "in which" / "on which" stays.
    """
    text = message or ""
    if looks_like_cloze_blanks(text):
        return
    answer = str(parsed.get("answer") or "")
    if re.search(r"\b(?:in|on|at|to|for)\s+which\b", answer, flags=re.IGNORECASE):
        return
    sentences = re.split(r"(?<=[.!?。])\s+|\n+", text)
    changed = False
    for sentence in sentences:
        place = bool(PLACE_RELATIVE_RE.search(sentence))
        time_clause = bool(TIME_RELATIVE_RE.search(sentence))
        if place == time_clause:
            continue
        target = "where" if place else "when"
        numbered = re.search(r"\(\s*(\d{1,2})\s*\)\s*_{2,}", sentence)
        if numbered:
            blank = numbered.group(1)
            pattern = rf"(\(\s*{blank}\s*\)\s*)(?:which|that|who|whom)\b"
            answer, count = re.subn(pattern, rf"\1{target}", answer, count=1, flags=re.IGNORECASE)
            if count:
                changed = True
                _rewrite_relative_step(parsed, blank, target, place)
            continue
        if len(re.findall(r"_{2,}", text)) != 1:
            continue
        if re.search(rf"\b{target}\b", answer, flags=re.IGNORECASE):
            continue
        if not re.search(r"\b(?:which|that|who|whom)\b", answer, flags=re.IGNORECASE):
            continue
        answer = re.sub(r"\b(?:which|that|who|whom)\b", target, answer, count=1, flags=re.IGNORECASE)
        changed = True
        _rewrite_relative_step(parsed, "", target, place)
    if changed:
        parsed["answer"] = answer


ARTICLE_A_SOUND = {
    "university",
    "uniform",
    "useful",
    "usual",
    "european",
    "one",
    "unit",
    "union",
    "unique",
    "united",
}
ARTICLE_AN_SOUND = {"hour", "honest", "honor", "honour", "heir"}
ARTICLE_NOT_NOUN = {
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did", "done", "doing",
    "will", "would", "can", "could", "may", "might", "must", "shall", "should",
    "and", "or", "but", "nor", "so", "yet",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as", "than", "into", "over",
    "that", "which", "who", "whom", "whose", "what", "when", "where", "while", "if", "because",
    "this", "these", "those", "there", "here", "then", "also", "not", "no",
    "said", "says", "say", "adding", "added",
}
CUED_HINT_RE = re.compile(
    r"\(\s*(\d{1,2})\s*\)\s*[_.＿—–-]{2,}\s*\(\s*([A-Za-z]+)\s*\)"
)
UNCUE_VERBS = {
    "feeding": r"food|bread|milk|meal|hungry",
    "painting": r"paint|picture|brush|canvas",
    "punishing": r"punish|punishment",
}
SAFE_PERCEPTION_VERBS = {"watching", "looking", "seeing", "observing", "noticing"}
CLOZE_CHOICE_RE = re.compile(
    r"(?:^|\n)\s*(?:\(\s*)?(\d{1,2})(?:\s*\))?\s*[.．、]\s*"
    r"A[\.．、)\s]\s*([A-Za-z]+)\s+"
    r"B[\.．、)\s]\s*([A-Za-z]+)\s+"
    r"C[\.．、)\s]\s*([A-Za-z]+)\s+"
    r"D[\.．、)\s]\s*([A-Za-z]+)",
    re.IGNORECASE,
)
META_STEP_RE = re.compile(
    r"我需要判断题型|按步骤展示解题思路|对于每个空，我需要|现在，按要求|参考答案\s*[:：]|"
    r"我们需要|需要看用户|需要回答用户"
)


def expected_article(word: str) -> str:
    token = (word or "").lower()
    if token in ARTICLE_AN_SOUND:
        return "an"
    if token in ARTICLE_A_SOUND:
        return "a"
    if re.match(r"[aeiou]", token):
        return "an"
    if re.match(r"[a-z]", token):
        return "a"
    return ""


def article_basis(word: str, want: str) -> str:
    token = (word or "").lower()
    if token in ARTICLE_AN_SOUND:
        return f"{word} 的 h 不发音，开头是元音，冠词用 {want}。"
    if token in ARTICLE_A_SOUND:
        return f"{word} 开头读 /j/，按辅音选冠词，用 {want}。"
    if want == "an":
        return f"{word} 开头是元音音素，冠词用 an。"
    return f"{word} 开头是辅音音素，冠词用 a。"


def contains_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text or "", flags=re.IGNORECASE))


def article_blank_pairs(text: str) -> list[tuple[str, str]]:
    """Blanks whose next word is the noun an article would modify. A parenthetical hint is not that noun."""
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"\(\s*(\d{1,2})\s*\)\s*[_.＿—–-]{2,}(?!\s*\()", text or ""):
        word_match = re.match(r"\s*([A-Za-z]+)", (text or "")[match.end():])
        if not word_match:
            continue
        word = word_match.group(1)
        if word.lower() in ARTICLE_NOT_NOUN or expected_article(word) not in {"a", "an"}:
            continue
        pairs.append((match.group(1), word))
    return pairs


def chooses_article(text: str, article: str) -> bool:
    return bool(
        re.search(
            rf"(?<!不)(?:冠词|用|填|选|改为|改成)\s*(?:为|成|了)?\s*{article}\b|(?:所以|因此)\s*{article}\b",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def article_claim_is_wrong(blob: str, word: str, want: str) -> bool:
    """True when this text names the word but teaches the other article or the wrong sound."""
    text = blob or ""
    if not contains_word(text, word):
        return False
    other = "a" if want == "an" else "an"
    if re.search(rf"而不是\s*[\"'“]?{want}\b", text, flags=re.IGNORECASE):
        return True
    chooses_other = chooses_article(text, other)
    chooses_want = chooses_article(text, want)
    if chooses_other and not chooses_want:
        return True
    token = word.lower()
    if want == "an" and token in ARTICLE_AN_SOUND and "辅音音素" in text:
        return True
    if want == "a" and token in ARTICLE_A_SOUND and "元音音素开头" in text and "辅音" not in text:
        return True
    if want == "an" and token not in ARTICLE_A_SOUND and re.search(
        r"ju[:：']|/ju|读\s*['\"]?ju", text, flags=re.IGNORECASE
    ):
        return True
    return False


def align_article_thinking(thinking: str, message: str) -> str:
    """Replace a thinking sentence that pairs a known word with the wrong article."""
    text = message or ""
    if looks_like_cloze_blanks(text) or not looks_like_grammar_fill(text):
        return thinking
    words = [(word, expected_article(word)) for _blank, word in article_blank_pairs(text)]
    words = [(word, want) for word, want in words if want in {"a", "an"}]
    if not words:
        return thinking

    def fix_sentence(sentence: str) -> str:
        mentioned = [(word, want) for word, want in words if contains_word(sentence, word)]
        if not mentioned:
            return sentence
        if len({want for _word, want in mentioned}) > 1 and re.search(r"类似|也一样|同样", sentence):
            return "".join(article_basis(word, want) for word, want in mentioned)
        for word, want in mentioned:
            if article_claim_is_wrong(sentence, word, want):
                return article_basis(word, want)
        return sentence

    parts = re.split(r"(?<=[。！？])", thinking or "")
    return "".join(fix_sentence(part) for part in parts)


def _step_is_about_blank(step: dict[str, object], blank: str, steps: list[object] | None = None) -> bool:
    if step_targets_blank(step, blank, steps):
        return True
    conclusion = str(step.get("conclusion") or "").strip().lower()
    return not step_item_nums(step) and conclusion in {"a", "an", "the"}


def _sync_article_explanations(parsed: dict[str, object], blanks: list[tuple[str, str]]) -> None:
    words = [(blank, word, expected_article(word)) for blank, word in blanks]
    words = [(blank, word, want) for blank, word, want in words if want in {"a", "an"}]
    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
            mentioned = [
                (blank, word, want)
                for blank, word, want in words
                if contains_word(blob, word) and _step_is_about_blank(step, blank, steps)
            ]
            if len({want for _blank, _word, want in mentioned}) > 1 and re.search(r"类似|也一样|同样", blob):
                step["basis"] = "".join(article_basis(word, want) for _blank, word, want in mentioned)
                step["conclusion"] = "；".join(f"{word} {want}" for _blank, word, want in mentioned)
                continue
            for blank, word, want in mentioned:
                if not article_claim_is_wrong(blob, word, want):
                    continue
                step["basis"] = article_basis(word, want)
                step["conclusion"] = want
                break
    methods = parsed.get("knowledge_methodology")
    if isinstance(methods, list):
        rewritten: list[object] = []
        for item in methods:
            text = str(item or "")
            replacement = ""
            for _blank, word, want in words:
                if article_claim_is_wrong(text, word, want):
                    replacement = article_basis(word, want)
                    break
            rewritten.append(replacement or item)
        parsed["knowledge_methodology"] = rewritten


def revise_article_by_sound(parsed: dict[str, object], message: str) -> None:
    """a/an follows the next word's sound, including inside a multi-blank grammar fill."""
    text = message or ""
    if looks_like_cloze_blanks(text) or not looks_like_grammar_fill(text):
        return
    answer = str(parsed.get("answer") or "")
    blanks = article_blank_pairs(text)
    if not blanks:
        return
    for blank, word in blanks:
        want = expected_article(word)
        if want not in {"a", "an"}:
            continue
        pattern = rf"(\(\s*{blank}\s*\)\s*)(an|a)\b"
        match = re.search(pattern, answer, flags=re.IGNORECASE)
        if not match or match.group(2).lower() == want:
            continue
        current = match.group(2)
        replacement = want.capitalize() if current[:1].isupper() else want
        answer = re.sub(pattern, rf"\1{replacement}", answer, count=1, flags=re.IGNORECASE)
        basis = article_basis(word, want)
        steps = parsed.get("reasoning_steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
                if not _step_is_about_blank(step, blank, steps) or not contains_word(blob, word):
                    continue
                if re.search(rf"\b{current}\b", blob, flags=re.IGNORECASE) or "独一无二" in blob:
                    step["basis"] = basis
                    step["conclusion"] = replacement
    parsed["answer"] = answer
    _sync_article_explanations(parsed, blanks)


def revise_uncued_cloze_verb(parsed: dict[str, object], message: str) -> None:
    """Drop a verb option that needs an object the passage never mentions."""
    text = message or ""
    if not looks_like_cloze_blanks(text):
        return
    groups = {
        match.group(1): {
            "A": match.group(2).lower(),
            "B": match.group(3).lower(),
            "C": match.group(4).lower(),
            "D": match.group(5).lower(),
        }
        for match in CLOZE_CHOICE_RE.finditer(text)
    }
    if not groups:
        return
    answer = str(parsed.get("answer") or "")
    for blank, choices in groups.items():
        token_match = re.search(rf"\(\s*{blank}\s*\)\s*([A-Da-d]|[A-Za-z]+)", answer)
        if not token_match:
            continue
        token = token_match.group(1)
        if token.upper() in choices:
            word = choices[token.upper()]
            use_letter = True
        else:
            word = token.lower()
            use_letter = False
        cue = UNCUE_VERBS.get(word)
        if not cue or re.search(cue, text, flags=re.IGNORECASE):
            continue
        safe_letter = next((letter for letter, option in choices.items() if option in SAFE_PERCEPTION_VERBS), "")
        if not safe_letter:
            continue
        safe_word = choices[safe_letter]
        replacement = safe_letter if use_letter else safe_word
        answer = re.sub(
            rf"(\(\s*{blank}\s*\)\s*){re.escape(token)}\b",
            rf"\1{replacement}",
            answer,
            count=1,
            flags=re.IGNORECASE,
        )
        basis = (
            f"这一空不能选 {word}：原文没有支撑这个动作的信息，等于用常识补情节。"
            f"同空的 {safe_word} 才和原句的动作接得上。"
        )
        steps = parsed.get("reasoning_steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
                if not step_targets_blank(step, blank, steps):
                    continue
                if word in blob.lower() or token.lower() == word:
                    step["basis"] = basis
                    step["conclusion"] = safe_word
    parsed["answer"] = answer


def revise_uncertain_whether(parsed: dict[str, object], message: str) -> None:
    """nobody knows + will is an open question, so the gap is whether, not that."""
    text = message or ""
    if looks_like_cloze_blanks(text):
        return
    answer = str(parsed.get("answer") or "")
    for sentence in re.split(r"(?<=[.!?。])\s+|\n+", text):
        if not re.search(r"\b(?:nobody|no one)\s+knows?\b", sentence, flags=re.IGNORECASE):
            continue
        if not re.search(r"_{2,}", sentence) or not re.search(r"\bwill\b", sentence, flags=re.IGNORECASE):
            continue
        numbered = re.search(r"\(\s*(\d{1,2})\s*\)\s*_{2,}", sentence)
        if not numbered:
            continue
        blank = numbered.group(1)
        pattern = rf"(\(\s*{blank}\s*\)\s*)that\b"
        answer, count = re.subn(pattern, rf"\1whether", answer, count=1, flags=re.IGNORECASE)
        if not count:
            continue
        basis = "nobody knows 后面是还没确定的结果，填 whether。that 会把这件事说成已经确定的事实。"
        steps = parsed.get("reasoning_steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
                if not step_targets_blank(step, blank, steps):
                    continue
                if re.search(r"\bthat\b|确定事实", blob, flags=re.IGNORECASE):
                    step["basis"] = basis
                    step["conclusion"] = "whether"
        methods = parsed.get("knowledge_methodology")
        if isinstance(methods, list):
            parsed["knowledge_methodology"] = [
                basis if isinstance(item, str) and re.search(r"\bthat\b|确定事实", item, flags=re.IGNORECASE) else item
                for item in methods
            ]
    parsed["answer"] = answer


def drop_meta_reasoning_steps(parsed: dict[str, object]) -> None:
    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list):
        return
    kept: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", blob))
        latin = len(re.findall(r"[A-Za-z]{3,}", blob))
        has_reason = bool(re.search(r"排除|冲突|所以|因此|改为|不能|不接|填|用", blob))
        if META_STEP_RE.search(blob):
            continue
        if re.search(r"判断题型", blob) and not re.search(r"改为|排除|冲突|原句", blob):
            continue
        if cjk < 12 and latin >= 4 and not has_reason:
            continue
        kept.append(step)
    if not kept:
        return
    for index, step in enumerate(kept, start=1):
        step["step"] = index
    parsed["reasoning_steps"] = kept


def drop_duplicate_blank_steps(parsed: dict[str, object]) -> None:
    """A format pass sometimes repeats the same item as both '31' and '空31'."""
    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list):
        return
    kept: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        basis = re.sub(r"[\s。；;]+", "", str(step.get("basis") or ""))
        nums = step_item_nums(step)
        duplicate = False
        if basis:
            for prev in kept:
                prev_basis = re.sub(r"[\s。；;]+", "", str(prev.get("basis") or ""))
                prev_nums = step_item_nums(prev)
                same_item = not nums or not prev_nums or bool(nums & prev_nums)
                if same_item and prev_basis and (basis == prev_basis or basis in prev_basis or prev_basis in basis):
                    duplicate = True
                    if len(basis) > len(prev_basis):
                        prev["basis"] = step.get("basis")
                        if str(step.get("conclusion") or "") and not str(prev.get("conclusion") or ""):
                            prev["conclusion"] = step.get("conclusion")
                    break
        if not duplicate:
            kept.append(step)
    if len(kept) == sum(isinstance(step, dict) for step in steps):
        return
    for index, step in enumerate(kept, start=1):
        step["step"] = index
    parsed["reasoning_steps"] = kept


def revise_extra_finite_participle(parsed: dict[str, object], message: str) -> None:
    """A blank between a noun and an existing finite verb is a participle, not was/were + participle."""
    text = message or ""
    if looks_like_cloze_blanks(text) or not looks_like_grammar_fill(text):
        return
    answer = str(parsed.get("answer") or "")
    finite = re.compile(
        r"\b(?:connects|looks|makes|takes|gives|shows|seems|becomes|remains|stands|sits|lives|works|plays|needs|wants)\b",
        re.IGNORECASE,
    )
    for blank, _hint in re.findall(r"\(\s*(\d{1,2})\s*\)\s*_{2,}\s*\(\s*([A-Za-z]+)\s*\)", text):
        sentence = sentence_containing_blank(text, blank)
        match = re.search(rf"(\(\s*{blank}\s*\)\s*)((?:was|were|is|are)\s+)([A-Za-z]+)", answer, flags=re.IGNORECASE)
        if not match:
            continue
        rest = re.sub(rf"\(\s*{blank}\s*\)\s*_{{2,}}(?:\s*\([^)]*\))?", " ", sentence)
        if not finite.search(rest):
            continue
        participle = match.group(3)
        answer = answer[: match.start()] + match.group(1) + participle + answer[match.end() :]
        basis = "这句已经有谓语，空格只能填非谓语。was/were 会再造一个谓语，所以只留过去分词。"
        steps = parsed.get("reasoning_steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
                if not step_targets_blank(step, blank, steps):
                    continue
                if re.search(r"\b(?:was|were|is|are)\b", blob, flags=re.IGNORECASE):
                    step["basis"] = basis
                    step["conclusion"] = participle
    parsed["answer"] = answer


def scrub_echoed_analysis(parsed: dict[str, object], message: str) -> None:
    """Drop a pasted 解析 sentence, including the test marker, from the formal reply."""
    text = message or ""
    markers = re.findall(r"复述码[A-Za-z0-9]+", text)
    parse = re.search(r"解析\s*[:：]\s*(.+)$", text, flags=re.S)
    chunks = []
    if parse:
        chunks = [part.strip() for part in re.split(r"[。！？\n]", parse.group(1)) if len(part.strip()) >= 8]

    def clean(value: object) -> str:
        result = str(value or "")
        for marker in markers:
            result = result.replace(marker, "")
        for chunk in chunks:
            result = result.replace(chunk, "")
        result = re.sub(r"这是紧张的表现[，,。]?", "", result)
        return re.sub(r"[ \t]{2,}", " ", result).strip(" ，,。；;")

    for key in ("answer", "stem_understanding", "follow_up", "subtype", "unsupported_reason"):
        if key in parsed:
            parsed[key] = clean(parsed.get(key))
    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for key in ("focus", "basis", "conclusion"):
                step[key] = clean(step.get(key))
    methods = parsed.get("knowledge_methodology")
    if isinstance(methods, list):
        parsed["knowledge_methodology"] = [clean(item) for item in methods if clean(item)]


def scrub_echoed_thinking(thinking: str, message: str) -> str:
    text = thinking or ""
    for marker in re.findall(r"复述码[A-Za-z0-9]+", message or ""):
        text = text.replace(marker, "")
    parse = re.search(r"解析\s*[:：]\s*(.+)$", message or "", flags=re.S)
    if parse:
        for chunk in re.split(r"[。！？\n]", parse.group(1)):
            chunk = chunk.strip()
            if len(chunk) >= 8:
                text = text.replace(chunk, "")
    text = re.sub(r"这是紧张的表现[，,。]?", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def revise_fronted_passive_participle(parsed: dict[str, object], message: str) -> None:
    """Seen from the hill: the subject is seen, so the fronted participle is past, not seeing."""
    text = message or ""
    if looks_like_cloze_blanks(text) or not looks_like_grammar_fill(text):
        return
    past_forms = {"see": "seen", "hear": "heard", "notice": "noticed", "observe": "observed"}
    answer = str(parsed.get("answer") or "")
    for blank, hint in re.findall(
        r"\(\s*(\d{1,2})\s*\)\s*_{2,}\s*\(\s*(see|hear|notice|observe)\s*\)\s+from\b",
        text,
        flags=re.IGNORECASE,
    ):
        doing = f"{hint.lower()}ing"
        past = past_forms[hint.lower()]
        answer, count = re.subn(
            rf"(\(\s*{blank}\s*\)\s*){doing}\b",
            rf"\1{past}",
            answer,
            count=1,
            flags=re.IGNORECASE,
        )
        if not count:
            continue
        basis = (
            f"句首这个动作的逻辑主语是后面句子的主语，主语是被{hint.lower()}的一方，"
            f"所以用过去分词 {past}。{doing} 会把主语写成动作的发出者。"
        )
        steps = parsed.get("reasoning_steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                blob = " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))
                if not step_targets_blank(step, blank, steps):
                    continue
                if re.search(rf"\b{doing}\b", blob, flags=re.IGNORECASE) or "现在分词" in blob:
                    step["basis"] = basis
                    step["conclusion"] = past
    parsed["answer"] = answer


def align_fronted_passive_thinking(thinking: str, message: str) -> str:
    text = message or ""
    past_forms = {"see": "seen", "hear": "heard", "notice": "noticed", "observe": "observed"}
    hints = re.findall(
        r"_{2,}\s*\(\s*(see|hear|notice|observe)\s*\)\s+from\b",
        text,
        flags=re.IGNORECASE,
    )
    result = thinking or ""
    for hint in hints:
        doing = f"{hint.lower()}ing"
        past = past_forms[hint.lower()]
        result = re.sub(
            rf"((?:填|用|是|为)\s*){doing}\b",
            rf"\1{past}",
            result,
            flags=re.IGNORECASE,
        )
    return result


def apply_answer_corrections(parsed: dict[str, object], message: str) -> None:
    revise_unjustified_past_perfect(parsed, message)
    revise_complete_relative_clause(parsed, message)
    revise_article_by_sound(parsed, message)
    revise_uncued_cloze_verb(parsed, message)
    revise_extra_finite_participle(parsed, message)
    revise_fronted_passive_participle(parsed, message)
    revise_uncertain_whether(parsed, message)
    scrub_echoed_analysis(parsed, message)
    drop_meta_reasoning_steps(parsed)
    drop_duplicate_blank_steps(parsed)
    scrub_format_leaks(parsed)


def infer_requested_question_type(message: str) -> str:
    text = message or ""
    if re.search(r"七选五", text):
        return "七选五"
    if re.search(r"长难句", text):
        return "长难句"
    if re.search(r"词义|选词填空|词汇", text):
        return "词汇"
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
        "词汇": "把原句、要猜的词和选项发过来。选词填空请把词库一起发来。",
        "长难句": "把要分析的原句发过来。",
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
        apply_answer_corrections(parsed, message)
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
    if re.search(r"长难句", user_side):
        return "长难句"
    if re.search(r"词义猜测|猜测词义|选词填空|词汇题", user_side) and not re.search(r"阅读理解", user_side):
        return "词汇"
    if re.search(r"语法填空|语法选择|单项选择|单项填空", user_side) or looks_like_single_grammar_choice(user_side):
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
    if grammar_fill or looks_like_single_grammar_choice(user_side):
        return "语法"
    if has_options and not reading_like:
        # 只有多空、或每空一组选项，才是完型。单独一道选择题按语法讲。
        # 文章已经很长、又没有完型空号时，按阅读讲，避免套起承转合。
        if english_word_count(user_side) >= 90:
            return "阅读"
        return "语法"
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
    if re.search(r"词汇", claimed):
        return "词汇"
    if re.search(r"长难句", claimed):
        return "长难句"
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


# Not sent to the model. Teach states use the short card in prompt_1.md for the current type.
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
FORMAT_LEAK_RE = re.compile(
    r"留空字符串|在代码中|在输出中|确保覆盖|空字符串|语言严肃简练|"
    r"知识方法论\s*[:：]|从提取\s*[:：]|但要简洁|"
    r"现在[，,]\s*(?:确认|确保)|"
    r"首先[，,]\s*supported|"
    r"supported\s*[:：]\s*(?:true|false)|"
    r"只有题干里真的出现|"
    r"调用(?:阅读|完型|完形|语法)?方法论\s*[:：]|方法论\s*[:：]|"
    r"不能编造|用中文讲解|focus\s*写|用户误说|所以[，,]?\s*题型是|没有指定空号|确认字段|"
    r"确保格式|写成字符串|可操作的判断|有点短|answer\s*字段|注意\s*[:：]\s*用户说|但要\s*3|在\s*conclusion|"
    r"在示例中|或留空|为了安全|"
    r"\}\s*,?\s*\}|\"\s*[A-G]\s*\"\s*[:：]",
    re.IGNORECASE,
)


def strip_type_meta(text: str) -> str:
    """Drop a '判断题型' aside. It is planner talk, not the explanation."""
    return re.sub(r"(?:先|我需要|需要|这里|所以)?判断题型[：:，,]?[^。！？\n]{0,80}[。！？]?", "", text or "")


def cut_format_leak(text: str) -> str:
    """Drop a format/schema tail. Keep an earlier teaching sentence when it is still substantial."""
    raw = re.sub(r"^在思考中[，,]\s*", "", text or "").strip()
    match = FORMAT_LEAK_RE.search(raw)
    if not match:
        return raw
    head = raw[: match.start()].strip(" ，,。；;\"'“”\n\t")
    head = re.sub(r"(?:现在|接下来|然后)[，,]?\s*$", "", head).strip(" ，,。；;\"'“”\n\t")
    if len(re.findall(r"[\u4e00-\u9fff]", head)) < 12:
        return ""
    return head


def scrub_format_leaks(parsed: dict[str, object]) -> None:
    """Remove schema and prompt-echo tails from the formal reply the student sees."""
    for key in ("stem_understanding", "follow_up", "subtype", "unsupported_reason"):
        if key in parsed:
            parsed[key] = cut_format_leak(str(parsed.get(key) or ""))
    steps = parsed.get("reasoning_steps")
    if isinstance(steps, list):
        kept: list[dict[str, object]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            cleaned = dict(step)
            original = strip_type_meta(" ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion")))
            if re.search(r"不能编造|用中文讲解", original):
                continue
            for key in ("focus", "basis", "conclusion"):
                cleaned[key] = cut_format_leak(strip_type_meta(str(step.get(key) or "")))
            blob = " ".join(str(cleaned.get(key) or "") for key in ("focus", "basis", "conclusion"))
            if len(re.findall(r"[\u4e00-\u9fff]", blob)) < 8:
                continue
            kept.append(cleaned)
        if kept:
            for index, step in enumerate(kept, start=1):
                step["step"] = index
            parsed["reasoning_steps"] = kept
    methods = parsed.get("knowledge_methodology")
    if isinstance(methods, list):
        parsed["knowledge_methodology"] = [
            item for item in (cut_format_leak(str(method)) for method in methods) if item
        ]


def salvage_thinking_sentence(sentence: str) -> str:
    item = cut_format_leak(strip_type_meta(scrub_rule_index_text(sentence)))
    if not item:
        return ""
    if re.fullmatch(r"[\s\"'`{}[\]\\,，:：.。A-Da-d]+", item):
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
        cjk = len(re.findall(r"[\u4e00-\u9fff]", item))
        latin = len(re.findall(r"[A-Za-z]{2,}", item))
        if cjk < 8 and latin >= 4:
            return ""
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
    source = strip_type_meta(text or "")
    source = re.sub(r"```(?:json)?[\s\S]*?```", "\n", source, flags=re.IGNORECASE)
    source = re.sub(r"\}\s*,?\s*\}", "\n", source)
    source = re.sub(r'"\s*[A-D]\s*"\s*[:：][^。\n]{0,80}', "", source)
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
    cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(cleaned)).strip()
    cleaned_text = re.sub(r"\}\s*,?\s*\}", "\n", cleaned_text)
    cleaned_text = re.sub(r"(?m)^[\s\"'`{}[\]\\,，:：.。]+$", "", cleaned_text)
    return re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()


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
        if 1 <= value <= MAX_EXAM_ITEM:
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
        if re.search(r"完型知识点|名词题[:：]|方法论包括", blob[:120]):
            return True
        leaked = cut_format_leak(blob)
        if leaked != blob.strip() and cjk_count(leaked) < 24:
            return True
        return False

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
        notes.append("这是同一套完型。材料里有几空就写几步，一空一步。")
    if looks_like_grammar_fill(text) and not looks_like_cloze_blanks(text) and PLACE_RELATIVE_RE.search(text):
        notes.append(
            "空格前是地点，后面从句的宾语已经齐全（如 spent my childhood）时填 where，不填 which。"
            "which 只在从句缺主语或宾语时用。"
        )
    if looks_like_grammar_fill(text) and not looks_like_cloze_blanks(text) and TIME_RELATIVE_RE.search(text):
        notes.append("空格前是时间，后面从句已经完整时填 when，不填 which。")
    if looks_like_supplied_answer_key(text):
        notes.append("学生附了答案或解析，只用来核对。不要复述解析，不要把原句翻译一遍。")
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
    if re.search(r"改错", text) and english_word_count(text) >= 8:
        notes.append("这是短文改错，原文已经在本条消息里。直接找出错误并讲完，不要回答需要确认，也不要让学生重发。")
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


def should_isolate_history(message: str, material: str | None = None) -> bool:
    """A new stem in this turn should not inherit the previous passage."""
    if is_explicit_new_question_turn(message):
        return True
    if is_blank_followup(message):
        return False
    probe = "\n".join(part for part in (message, material) if part)
    return has_enough_question_material(probe)


def iter_compacted_history(
    history: list[ChatMessage],
    message: str,
    *,
    limit: int,
    material: str | None = None,
) -> list[tuple[str, str]]:
    prior = prior_chat_history(history, message, limit=max(limit * 3, 18))
    if should_isolate_history(message, material):
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


def question_type_from_history(history: list[ChatMessage]) -> str:
    for item in reversed(history):
        if item.sender != "ai":
            continue
        parsed = parse_structured_reply(item.content)
        if parsed:
            label = str(parsed.get("question_type") or "").strip().replace("完形", "完型")
            if label in KNOWN_QUESTION_TYPES and label != "综合":
                return label
        match = re.search(rf"题型：\s*({QUESTION_TYPE_PATTERN})", item.content or "")
        if match:
            return match.group(1).replace("完形", "完型")
    return ""


def parse_classified_type(text: str) -> str:
    match = re.search(
        rf'"question_type"\s*:\s*"({QUESTION_TYPE_PATTERN})"',
        text or "",
    )
    label = match.group(1) if match else ""
    if not label:
        match = re.search(QUESTION_TYPE_PATTERN, text or "")
        label = match.group(0) if match else ""
    label = label.replace("完形", "完型")
    return label if label in KNOWN_QUESTION_TYPES else ""


def material_type_is_confident(message: str, image_context: str | None, guessed: str) -> bool:
    user = "\n".join(part for part in (message, image_context) if part)
    if looks_like_seven_choose_five(user):
        return guessed == "七选五"
    if re.search(r"长难句", user):
        return guessed == "长难句"
    if re.search(r"词义猜测|猜测词义|选词填空|词汇题", user) and not re.search(r"阅读理解", user):
        return guessed == "词汇"
    if (
        re.search(r"语法填空|语法选择|单项选择|单项填空", user)
        or looks_like_single_grammar_choice(user)
        or (looks_like_grammar_fill(user) and not looks_like_cloze_blanks(user))
    ):
        return guessed == "语法"
    if re.search(r"完形|完型", user) or looks_like_cloze_blanks(user):
        return guessed == "完型"
    if re.search(r"阅读理解", user) or (looks_like_reading(user) and not looks_like_cloze_blanks(user)):
        return guessed == "阅读"
    if re.search(r"改错", user) and has_translatable_or_correctable_sentence(user):
        return guessed == "改错"
    if re.search(r"翻译", user) and has_translatable_or_correctable_sentence(user):
        return guessed == "翻译"
    return False


def teaching_material(message: str, image_context: str | None = None) -> str:
    parts = [(message or "").strip()]
    extra = (image_context or "").strip()
    if extra and extra not in (message or ""):
        parts.append(extra)
    return "\n".join(part for part in parts if part)


def build_text_messages(
    message: str,
    history: list[ChatMessage],
    question_type: str = "综合",
    material: str | None = None,
) -> list[dict]:
    """Teach with one type card. The archived full prompt is not included."""
    messages: list[dict] = [
        {"role": "system", "content": teach_system_prompt(question_type)},
    ]
    for role, content in iter_compacted_history(history, message, limit=12, material=material):
        messages.append({"role": role, "content": content})
    user_text = decorate_current_user_message(message)
    extra = (material or "").strip()
    if extra and extra not in user_text:
        user_text = f"{user_text}\n\n题目文字：\n{extra}"
    user_text += (
        "\n\n按空写讲稿，每一步开头写空号或题号。"
        "每一空写成三句：这个空在这句里充当什么。哪几个原词决定它。被排除的形式为什么和这些词接不上。"
        "括号里有提示词时，结论必须是这个词的一种形式。不要改讲冠词、介词或另一个空。"
        "用中文判断。不要输出 JSON，不要翻译整段，不要只写「排除、冲突、所以」。"
    )
    messages.append({"role": "user", "content": user_text})
    return messages


def build_format_messages(question_type: str, material: str, prose: str) -> list[dict]:
    """Second respond: pack an already written explanation. No type rules."""
    return [
        {"role": "system", "content": format_system_prompt()},
        {
            "role": "user",
            "content": (
                f"题型：{question_type}\n\n题目：\n{material[:5000]}\n\n讲稿：\n{prose[:7000]}\n\n"
                "把讲稿收成一个 JSON。不要新编情节。"
            ),
        },
    ]


def build_classify_messages(message: str, image_context: str | None = None) -> list[dict]:
    system = prompt_block("classify") or "只判断题型，不要讲题。只输出 question_type。"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": teaching_material(message, image_context)[:6000]},
    ]


def build_vision_messages(
    message: str,
    history: list[ChatMessage],
    image: ImagePayload,
    ocr_text: str | None = None,
) -> list[dict]:
    """Vision only copies the question. Teaching happens in a later text respond."""
    del history, ocr_text
    return [
        {
            "role": "system",
            "content": "只抄题目。保留换行、空格、题号和选项。不要讲题，不要给答案，不要总结。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": decorate_current_user_message(message) or "把图中的题干、原文、空格和选项原样抄出来。"},
                {"type": "image_url", "image_url": {"url": image.data_url}},
            ],
        },
    ]


def create_chat_completion(client: OpenAI, *, enable_thinking: bool = False, **kwargs):
    """AI Ping turns thinking on for some models, which can leave content empty."""
    extra_body = {"enable_thinking": True, "thinking_budget": 8192} if enable_thinking else {"enable_thinking": False}
    try:
        return client.chat.completions.create(extra_body=extra_body, **kwargs)
    except Exception:
        logger.exception("Chat completion extras were rejected, retrying without them")
        return client.chat.completions.create(**kwargs)


def completion_text(completion: object) -> tuple[str, str]:
    choices = getattr(completion, "choices", None) or []
    message = choices[0].message if choices else None
    content = getattr(message, "content", None) or ""
    reasoning = getattr(message, "reasoning_content", None) or ""
    if not isinstance(content, str):
        content = ""
    if not isinstance(reasoning, str):
        reasoning = ""
    if not reasoning and message is not None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            raw = extra.get("reasoning_content") or extra.get("reasoning") or ""
            if isinstance(raw, str):
                reasoning = raw
    return content, reasoning


def reason_thinking_enabled(model: str) -> bool:
    """DeepSeek V3/V4 keep the draft in the thinking channel and the answer in content."""
    if model_supports_thinking(model):
        return True
    name = (model or "").lower()
    return "deepseek" in name and any(token in name for token in ("v4", "v3", "r1"))


def complete_chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    enable_thinking: bool | None = None,
) -> tuple[str, str]:
    if enable_thinking is None:
        enable_thinking = model_supports_thinking(model)
    completion = create_chat_completion(
        client,
        model=model,
        temperature=temperature,
        max_tokens=8192,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    return completion_text(completion)


def classify_with_model(
    client: OpenAI,
    model: str,
    message: str,
    image_context: str | None = None,
) -> str:
    reply, _thinking = complete_chat(client, model, build_classify_messages(message, image_context), 0)
    return parse_classified_type(reply)


def resolve_question_type(
    client: OpenAI | None,
    model: str,
    message: str,
    history: list[ChatMessage],
    image_context: str | None = None,
) -> str:
    """State 1: local shape when it is clear, otherwise one short classify respond."""
    if is_blank_followup(message):
        prior_type = question_type_from_history(history)
        if prior_type:
            logger.info("teach stage=classify source=history type=%s", prior_type)
            return prior_type
    guessed = classify_question_type(message, {}, image_context)
    if material_type_is_confident(message, image_context, guessed):
        logger.info("teach stage=classify source=material type=%s", guessed)
        return guessed
    if client is not None and has_enough_question_material(message, image_context):
        labeled = classify_with_model(client, model, message, image_context)
        if labeled:
            logger.info("teach stage=classify source=model type=%s", labeled)
            return labeled
    logger.info("teach stage=classify source=fallback type=%s", guessed or "综合")
    return guessed or "综合"


def transcribe_question_image(client: OpenAI, image: ImagePayload, model: str) -> str:
    """State for a picture: copy the question. Do not teach from the image model."""
    reply, _thinking = complete_chat(
        client,
        model,
        build_vision_messages("", [], image),
        0.1,
    )
    return (reply or "").strip()


def extract_ocr_text(client: OpenAI, image: ImagePayload, model: str) -> str:
    if "ocr" in (model or "").lower():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image.data_url}},
                    {"type": "text", "text": "Free OCR."},
                ],
            }
        ]
    else:
        messages = [
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
        ]
    completion = create_chat_completion(
        client,
        model=model,
        temperature=0.1,
        messages=messages,
    )
    return completion.choices[0].message.content or ""


def generate_text_reply(client: OpenAI, message: str, history: list[ChatMessage], model: str) -> str:
    question_type = resolve_question_type(client, model, message, history, None)
    reply, _thinking = staged_teach_reply(client, model, message, history, question_type, None, "")
    return reply


def generate_vision_reply(
    client: OpenAI,
    message: str,
    history: list[ChatMessage],
    image: ImagePayload,
    model: str,
    ocr_text: str | None = None,
) -> str:
    del message, history, ocr_text
    return transcribe_question_image(client, image, model)


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


def reply_dodges_present_material(
    reply: str,
    message: str,
    image_context: str | None = None,
    prior_context: str = "",
) -> bool:
    """The model asked for the passage even though this turn already contains it."""
    user_text = "\n".join(part for part in (message, image_context) if part)
    if looks_like_unspecified_independent_questions(user_text):
        return False
    parsed = parse_structured_reply(reply)
    if not parsed:
        return False
    answer = str(parsed.get("answer") or "").strip()
    if not (parsed.get("need_more_context") or answer == "需要确认"):
        return False
    return has_enough_question_material(message, image_context, prior_context)


PRESENT_MATERIAL_NUDGE = (
    "题目原文已经在上一条学生消息里。不要写需要确认，不要让学生重发。"
    "按题型把每一处讲完，然后输出正式 JSON。"
)

GROUNDING_NUDGE = (
    "上一讲不合格。不要把原句译成中文再贴选项字母。"
    "每一空的步骤里必须出现原句里的词，并写出至少一个干扰项，说明它和原句哪一处冲突。"
    "丢掉上一讲，重新输出正式 JSON。"
)

CONFLICT_RE = re.compile(
    r"不选|排除|不能|不含|没有|填不了|不填|冲突|对不上|接不上|不接|不对|错在|而不是|不是|相反|"
    r"原文没有|文中没有|不合适|不成立|用不上|并不|干扰|改成|改为|不该|不应|不要再|"
    r"体现不出|另一种|如果译成|如果改成|不如"
)


def content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "")
        if word.lower() not in SOURCE_WORD_STOP
    }


def teach_blank_ids(material: str) -> list[str]:
    found: list[str] = []

    def add(value: str) -> None:
        if value and value not in found and value.isdigit() and 1 <= int(value) <= MAX_EXAM_ITEM:
            found.append(value)

    for pattern in (
        r"\(\s*(\d{1,2})\s*\)\s*(?:_{2,}|[_.＿—–-]{2,})",
        r"(?:^|\n)\s*(\d{1,2})\s*[.．、]\s*(?:_{2,}|[A-D]\b)",
        r"第\s*(\d{1,2})\s*空",
    ):
        for match in re.finditer(pattern, material or "", flags=re.MULTILINE):
            add(match.group(1))
    for match in CLOZE_OPTION_GROUP_RE.finditer(material or ""):
        add(match.group(1))
    return found


def _step_blob(step: dict[str, object]) -> str:
    return " ".join(str(step.get(key) or "") for key in ("focus", "basis", "conclusion"))


def blank_window(material: str, blank: str) -> str:
    text = material or ""
    for pattern in (
        rf"\(\s*{re.escape(blank)}\s*\)",
        rf"第\s*{re.escape(blank)}\s*空",
        rf"(?:^|\n)\s*{re.escape(blank)}\s*[.．、]",
    ):
        match = re.search(pattern, text)
        if not match:
            continue
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 90)
        return text[start:end]
    return text


def names_distractor(blob: str) -> bool:
    text = blob or ""
    if CONFLICT_RE.search(text):
        return True
    articles = {item.lower() for item in re.findall(r"\b(a|an|the)\b", text, flags=re.IGNORECASE)}
    if len(articles) >= 2:
        return True
    forms = {
        item.lower()
        for item in re.findall(r"\b(doing|done|seen|seeing|written|writing|was|were)\b", text, flags=re.IGNORECASE)
    }
    return len(forms) >= 2


def step_item_nums(step: dict[str, object]) -> set[str]:
    """Blank numbers this step is about.

    The title decides. A bare 61 there counts. A bare 61 inside the explanation does not,
    so one step cannot be treated as another blank just because it mentions that number.
    """
    focus = str(step.get("focus") or "")
    nums = {str(num) for num in blank_nums_in_text(focus)}
    bare = re.match(r"\s*(\d{1,2})\b", focus)
    if bare and 1 <= int(bare.group(1)) <= MAX_EXAM_ITEM:
        nums.add(bare.group(1))
    if nums:
        return nums
    return {str(num) for num in blank_nums_in_text(_step_blob(step))}


def step_targets_blank(step: dict[str, object], blank: str, steps: list[object] | None = None) -> bool:
    """True only for this blank's own step. An untitled step counts only when it is the only step."""
    nums = step_item_nums(step)
    if nums:
        return blank in nums
    if steps is None:
        return False
    return sum(isinstance(item, dict) for item in steps) == 1


def _steps_covering(steps: list[object], blank: str, single: bool) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if blank in step_item_nums(step):
            matched.append(step)
    if matched:
        return matched
    if single:
        return [step for step in steps if isinstance(step, dict)]
    return []


def blob_quotes_source(blob: str, window: str) -> bool:
    words = content_words(window)
    if words:
        lowered = (blob or "").lower()
        return any(word in lowered for word in words)
    for chunk in re.findall(r"[\u4e00-\u9fff]{4,}", window or "")[:8]:
        if chunk[:4] in (blob or ""):
            return True
    return False


def is_translation_plus_choice(parsed: dict[str, object], material: str, blob: str) -> bool:
    answer = str(parsed.get("answer") or "")
    if not re.search(r"(?<![A-Za-z])[A-D](?![A-Za-z])", answer):
        return False
    quoted = content_words(blob) & content_words(material)
    chinese = len(re.findall(r"[\u4e00-\u9fff]", blob or ""))
    return chinese >= 24 and len(quoted) <= 1 and not names_distractor(blob)


IRREGULAR_FORMS = {
    "be": frozenset({"am", "is", "are", "was", "were", "been", "being"}),
    "go": frozenset({"went", "gone", "goes", "going"}),
    "do": frozenset({"did", "does", "done", "doing"}),
    "have": frozenset({"has", "had", "having"}),
    "see": frozenset({"saw", "seen", "sees", "seeing"}),
    "come": frozenset({"came", "comes", "coming"}),
    "take": frozenset({"took", "taken", "takes", "taking"}),
    "make": frozenset({"made", "makes", "making"}),
    "give": frozenset({"gave", "given", "gives", "giving"}),
    "find": frozenset({"found", "finds", "finding"}),
    "write": frozenset({"wrote", "written", "writes", "writing"}),
    "know": frozenset({"knew", "known", "knows", "knowing"}),
    "get": frozenset({"got", "gotten", "gets", "getting"}),
    "say": frozenset({"said", "says", "saying"}),
    "tell": frozenset({"told", "tells", "telling"}),
    "speak": frozenset({"spoke", "spoken", "speaks", "speaking"}),
    "break": frozenset({"broke", "broken", "breaks", "breaking"}),
    "choose": frozenset({"chose", "chosen", "chooses", "choosing"}),
    "buy": frozenset({"bought", "buys", "buying"}),
    "bring": frozenset({"brought", "brings", "bringing"}),
    "think": frozenset({"thought", "thinks", "thinking"}),
    "teach": frozenset({"taught", "teaches", "teaching"}),
    "catch": frozenset({"caught", "catches", "catching"}),
    "seek": frozenset({"sought", "seeks", "seeking"}),
    "sell": frozenset({"sold", "sells", "selling"}),
    "sit": frozenset({"sat", "sits", "sitting"}),
    "stand": frozenset({"stood", "stands", "standing"}),
    "win": frozenset({"won", "wins", "winning"}),
    "hold": frozenset({"held", "holds", "holding"}),
    "lead": frozenset({"led", "leads", "leading"}),
    "mean": frozenset({"meant", "means", "meaning"}),
    "lose": frozenset({"lost", "loses", "losing"}),
    "feel": frozenset({"felt", "feels", "feeling"}),
    "keep": frozenset({"kept", "keeps", "keeping"}),
    "leave": frozenset({"left", "leaves", "leaving"}),
    "meet": frozenset({"met", "meets", "meeting"}),
    "pay": frozenset({"paid", "pays", "paying"}),
    "send": frozenset({"sent", "sends", "sending"}),
    "run": frozenset({"ran", "runs", "running"}),
    "eat": frozenset({"ate", "eaten", "eats", "eating"}),
    "fall": frozenset({"fell", "fallen", "falls", "falling"}),
    "lie": frozenset({"lay", "lain", "lies", "lying"}),
    "bear": frozenset({"bore", "born", "borne", "bears", "bearing"}),
    "wear": frozenset({"wore", "worn", "wears", "wearing"}),
    "grow": frozenset({"grew", "grown", "grows", "growing"}),
    "show": frozenset({"showed", "shown", "shows", "showing"}),
    "draw": frozenset({"drew", "drawn", "draws", "drawing"}),
    "fly": frozenset({"flew", "flown", "flies", "flying"}),
    "drive": frozenset({"drove", "driven", "drives", "driving"}),
    "ride": frozenset({"rode", "ridden", "rides", "riding"}),
    "rise": frozenset({"rose", "risen", "rises", "rising"}),
    "wake": frozenset({"woke", "woken", "wakes", "waking"}),
    "begin": frozenset({"began", "begun", "begins", "beginning"}),
}


def shares_stem(cue: str, token: str) -> bool:
    """True when token is a regular form of cue: built/build, visibility/visible, themes/theme."""
    cue = (cue or "").lower()
    token = (token or "").lower()
    if not cue or not token:
        return False
    if token == cue or token.startswith(cue):
        return True
    stem = cue[:-1] if cue.endswith("e") and len(cue) > 4 else cue
    head = stem[:4]
    return len(head) >= 4 and token.startswith(head)


def text_uses_cue(text: str, cue: str) -> bool:
    forms = IRREGULAR_FORMS.get((cue or "").lower(), frozenset())
    for token in re.findall(r"[A-Za-z']+", text or ""):
        low = token.lower()
        if low in forms or shares_stem(cue, low):
            return True
    return False


def cued_blank_job_notes(parsed: dict[str, object], material: str) -> list[str]:
    """A parenthetical hint must be explained as a form of that word, whatever the other rule was."""
    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list):
        return []
    notes: list[str] = []
    for match in CUED_HINT_RE.finditer(material or ""):
        blank, cue = match.group(1), match.group(2)
        covered = _steps_covering(steps, blank, False)
        if not covered:
            continue
        blob = " ".join(_step_blob(step) for step in covered)
        if text_uses_cue(blob, cue):
            continue
        notes.append(f"空{blank}的提示词是 {cue}，这一步没有讲这个词的形式")
    return notes


def cued_blank_article_notes(parsed: dict[str, object], material: str) -> list[str]:
    return cued_blank_job_notes(parsed, material)


def grounding_gap_notes(parsed: dict[str, object], material: str) -> list[str]:
    """Name the blank that failed, so the next respond can fix that step."""
    if not isinstance(parsed, dict):
        return ["没有按空写步骤"]
    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list) or not any(isinstance(step, dict) for step in steps):
        return ["没有按空写步骤"]
    blanks = teach_blank_ids(material)
    notes: list[str] = []
    if not blanks:
        blob = " ".join(_step_blob(step) for step in steps if isinstance(step, dict))
        if not blob_quotes_source(blob, material):
            notes.append("整题没有照抄原句里的词")
        if not names_distractor(blob):
            notes.append("整题没有写被排除的一项")
        return notes
    single = len(blanks) == 1
    for blank in blanks:
        covered = _steps_covering(steps, blank, single)
        label = f"空{blank}"
        if not covered:
            notes.append(f"{label}没有单独的一步")
            continue
        blob = " ".join(_step_blob(step) for step in covered)
        window = blank_window(material, blank)
        if not blob_quotes_source(blob, window):
            notes.append(f"{label}没有照抄原句里的词")
        if not names_distractor(blob):
            notes.append(f"{label}没有写被排除的一项和冲突")
    notes.extend(cued_blank_job_notes(parsed, material))
    return notes


def rewrite_instruction(parsed: dict[str, object] | None, material: str) -> str:
    """Ask for a new explanation of the weak blank. Do not hand the model a sentence to copy."""
    lines = [
        "上一讲没有把这个空讲清楚。丢掉上一讲，对着原句重写。",
        "每一空写成三句：这个空在这句里充当什么。哪几个原词决定它。被排除的形式为什么和这些词接不上。",
        "括号里有提示词时，结论必须是这个词的一种形式。不要改讲冠词、介词或另一个空。",
        "不要翻译整句，不要复述情节，不要只写「排除某词，冲突，所以填答案」。",
    ]
    steps = parsed.get("reasoning_steps") if isinstance(parsed, dict) else None
    steps = steps if isinstance(steps, list) else []
    blanks = teach_blank_ids(material)
    single = len(blanks) == 1
    gaps = grounding_gap_notes(parsed, material) if isinstance(parsed, dict) else []
    job_ids: list[str] = []
    other_ids: list[str] = []
    for note in gaps:
        match = re.search(r"空(\d{1,2})", note)
        if not match:
            continue
        bucket = job_ids if "提示词" in note else other_ids
        if match.group(1) not in job_ids and match.group(1) not in other_ids:
            bucket.append(match.group(1))
    targets = (job_ids + other_ids) or blanks or [""]
    for blank in targets[:4]:
        if blank:
            covered = _steps_covering(steps, blank, single)
            window = blank_window(material, blank)
            label = f"空{blank}"
        else:
            covered = [step for step in steps if isinstance(step, dict)]
            window = material
            label = "这一题"
        window = re.sub(r"\s+", " ", window or "").strip()
        if window:
            lines.append(f"{label}的原句：{window[:180]}")
        if covered and isinstance(covered[0], dict):
            weak = str(covered[0].get("basis") or "").strip()
            if weak:
                lines.append(f"上一讲只写了：{weak[:160]}")
    if gaps:
        ranked = [note for note in gaps if "提示词" in note] + [note for note in gaps if "提示词" not in note]
        lines.append("还没讲清：" + "；".join(ranked[:4]) + "。")
    lines.append("不要输出 JSON。")
    return "\n".join(lines)


def grounding_score(parsed: dict[str, object] | None, material: str) -> int:
    """How many blanks already quote the sentence and name a rejection."""
    if not isinstance(parsed, dict):
        return -1
    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list):
        return 0
    blanks = teach_blank_ids(material)
    if not blanks:
        blob = " ".join(_step_blob(step) for step in steps if isinstance(step, dict))
        quoted = blob_quotes_source(blob, material)
        rejected = names_distractor(blob)
        return int(quoted) + int(rejected)
    single = len(blanks) == 1
    score = 0
    for blank in blanks:
        covered = _steps_covering(steps, blank, single)
        if not covered:
            continue
        blob = " ".join(_step_blob(step) for step in covered)
        window = blank_window(material, blank)
        if blob_quotes_source(blob, window):
            score += 1
        if names_distractor(blob):
            score += 1
    return score


def explanation_needs_reteach(parsed: dict[str, object], material: str) -> bool:
    """True when a step is only a translation plus a letter, or misses the sentence and a conflict."""
    if not isinstance(parsed, dict):
        return False
    if str(parsed.get("answer") or "").strip() == "需要确认":
        return False
    if not has_enough_question_material(material):
        return False
    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list) or not any(isinstance(step, dict) for step in steps):
        return True
    if cued_blank_job_notes(parsed, material):
        return True
    blanks = teach_blank_ids(material)
    if not blanks:
        blob = " ".join(_step_blob(step) for step in steps if isinstance(step, dict))
        if not blob_quotes_source(blob, material) or not names_distractor(blob):
            return True
        return is_translation_plus_choice(parsed, material, blob)
    single = len(blanks) == 1
    for blank in blanks:
        covered = _steps_covering(steps, blank, single)
        if not covered:
            return True
        blob = " ".join(_step_blob(step) for step in covered)
        window = blank_window(material, blank)
        if not blob_quotes_source(blob, window) or not names_distractor(blob):
            return True
    whole = " ".join(_step_blob(step) for step in steps if isinstance(step, dict))
    return is_translation_plus_choice(parsed, material, whole)


def reteach_nudge(
    reply: str,
    message: str,
    image_context: str | None = None,
    prior_context: str = "",
) -> str:
    parts: list[str] = []
    if reply_dodges_present_material(reply, message, image_context, prior_context):
        parts.append(PRESENT_MATERIAL_NUDGE)
    material = teaching_material(message, image_context)
    parsed = parse_structured_reply(reply)
    if parsed is None and has_enough_question_material(message, image_context, prior_context):
        parts.append(GROUNDING_NUDGE)
    elif isinstance(parsed, dict) and explanation_needs_reteach(parsed, material):
        parts.append(GROUNDING_NUDGE)
    return "\n".join(parts)


def choose_reteach_reply(
    first: str,
    second: str,
    message: str,
    image_context: str | None = None,
    prior_context: str = "",
) -> str:
    if not (second or "").strip():
        return first
    first_dodge = reply_dodges_present_material(first, message, image_context, prior_context)
    second_dodge = reply_dodges_present_material(second, message, image_context, prior_context)
    if first_dodge and not second_dodge:
        return second
    if second_dodge and not first_dodge:
        return first
    material = teaching_material(message, image_context)
    first_parsed = parse_structured_reply(first)
    second_parsed = parse_structured_reply(second)
    first_bad = first_parsed is None or explanation_needs_reteach(first_parsed, material)
    second_bad = second_parsed is None or explanation_needs_reteach(second_parsed, material)
    if first_bad and not second_bad:
        return second
    if second_bad and not first_bad:
        return first
    if first_bad and second_bad:
        if grounding_score(first_parsed, material) > grounding_score(second_parsed, material):
            return first
    return second


REASON_PRESENT_NUDGE = (
    "题目原文已经在上一条学生消息里。不要写需要确认，不要让学生重发。"
    "按空重写，每空三句：这个空在这句里充当什么。哪几个原词决定它。被排除的形式为什么接不上。括号里有提示词就只讲这个词。不要输出 JSON。"
)


def answer_text(content: str, thinking: str) -> str:
    """DeepSeek may leave the draft in the thinking channel and content empty."""
    return (content or "").strip() or (thinking or "").strip()


def _is_planning_monologue(text: str) -> bool:
    """Model notes about the task are not a lecture the student should see."""
    return bool(re.search(r"我们需要|需要回答用户|需要看用户", text or ""))


def reason_prose(content: str, thinking: str, material: str) -> str:
    """Use the channel that already quotes the sentence and names a rejected option."""
    content = (content or "").strip()
    thinking = (thinking or "").strip()

    def usable(text: str) -> bool:
        if _is_planning_monologue(text):
            return False
        return bool(text) and blob_quotes_source(text, material) and names_distractor(text)

    if usable(content):
        return content
    if usable(thinking):
        return thinking
    combined = f"{thinking}\n{content}".strip()
    if usable(combined):
        return combined
    if content and blob_quotes_source(content, material) and not _is_planning_monologue(content):
        return content
    return content or thinking


def reason_stage_nudge(
    reply: str,
    message: str,
    image_context: str | None = None,
    prior_context: str = "",
) -> str:
    raw = reteach_nudge(reply, message, image_context, prior_context)
    if not raw:
        return ""
    parts: list[str] = []
    if PRESENT_MATERIAL_NUDGE in raw:
        parts.append(REASON_PRESENT_NUDGE)
    if GROUNDING_NUDGE in raw:
        parsed = parse_structured_reply(reply)
        material = teaching_material(message, image_context)
        parts.append(rewrite_instruction(parsed if isinstance(parsed, dict) else None, material))
    return "\n".join(parts)


def explain_once(
    client: OpenAI,
    model: str,
    message: str,
    history: list[ChatMessage],
    question_type: str,
    material: str,
    nudge: str = "",
) -> str:
    """One respond: teach this blank. The JSON contract is not in this prompt."""
    messages = build_text_messages(message, history, question_type, material)
    if nudge:
        messages.append({"role": "user", "content": nudge})
    content, thinking = complete_chat(
        client,
        model,
        messages,
        temperature=0.2,
        enable_thinking=reason_thinking_enabled(model),
    )
    return reason_prose(content, thinking, material)


def format_once(
    client: OpenAI,
    model: str,
    question_type: str,
    material: str,
    prose: str,
) -> str:
    """One respond: pack the prose into JSON. Thinking stays off so JSON lands in content."""
    content, thinking = complete_chat(
        client,
        model,
        build_format_messages(question_type, material, prose),
        temperature=0,
        enable_thinking=False,
    )
    return answer_text(content, thinking)


def staged_teach_reply(
    client: OpenAI,
    model: str,
    message: str,
    history: list[ChatMessage],
    question_type: str,
    image_context: str | None,
    prior_context: str = "",
) -> tuple[str, str]:
    """Reason, then format. If the JSON is ungrounded, reason and format once more."""
    material = teaching_material(message, image_context)
    logger.info("teach stage=reason type=%s model=%s", question_type, model)
    prose = explain_once(client, model, message, history, question_type, material)
    logger.info("teach stage=format type=%s", question_type)
    reply, thinking = finalize_structured_reply(
        format_once(client, model, question_type, material, prose),
        prose,
        message,
        image_context,
        prior_context,
    )
    nudge = reason_stage_nudge(reply, message, image_context, prior_context)
    if not nudge:
        return reply, thinking
    logger.info("teach stage=repair type=%s", question_type)
    repaired = explain_once(client, model, message, history, question_type, material, nudge)
    repaired_reply, repaired_thinking = finalize_structured_reply(
        format_once(client, model, question_type, material, repaired),
        repaired,
        message,
        image_context,
        prior_context,
    )
    chosen = choose_reteach_reply(reply, repaired_reply, message, image_context, prior_context)
    if chosen == repaired_reply:
        return repaired_reply, repaired_thinking
    return reply, thinking


def finalize_structured_reply(
    reply: str,
    raw_thinking: str,
    source_message: str,
    image_context: str | None = None,
    prior_context: str = "",
) -> tuple[str, str]:
    rule_text = teaching_material(source_message, image_context)
    reply = normalize_structured_reply(
        reply,
        message=rule_text,
        image_context=image_context,
        prior_context=prior_context,
    )
    final_thinking = sanitize_thinking_text(raw_thinking)
    final_thinking = scrub_echoed_thinking(final_thinking, rule_text)
    final_thinking = align_article_thinking(final_thinking, rule_text)
    final_thinking = align_fronted_passive_thinking(final_thinking, rule_text)
    if not final_thinking:
        final_thinking = thinking_from_structured_reply(reply)
    if final_thinking:
        reply = merge_thinking_into_structured_reply(reply, final_thinking)
    polished = parse_structured_reply(reply)
    if polished:
        apply_answer_corrections(polished, rule_text)
        reply = json.dumps(polished, ensure_ascii=False)
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
    return reply, final_thinking


def iter_staged_teach_events(
    client: OpenAI,
    *,
    model: str,
    message: str,
    history: list[ChatMessage],
    question_type: str,
    image_context: str | None,
    prior_context: str,
    meta: ModelMeta,
):
    material = teaching_material(message, image_context)
    logger.info("teach stage=reason type=%s model=%s", question_type, model)
    prose = explain_once(client, model, message, history, question_type, material)
    yield sse_event({"type": "status", "stage": "teach", "text": "整理成答案"})
    logger.info("teach stage=format type=%s", question_type)
    reply, final_thinking = finalize_structured_reply(
        format_once(client, model, question_type, material, prose),
        prose,
        message,
        image_context,
        prior_context,
    )
    nudge = reason_stage_nudge(reply, message, image_context, prior_context)
    if nudge:
        status = "没贴住原句，再讲一次" if "不合格" in nudge else "题目已经在，继续讲"
        yield sse_event({"type": "status", "stage": "teach", "text": status})
        logger.info("teach stage=repair type=%s", question_type)
        repaired = explain_once(client, model, message, history, question_type, material, nudge)
        repaired_reply, repaired_thinking = finalize_structured_reply(
            format_once(client, model, question_type, material, repaired),
            repaired,
            message,
            image_context,
            prior_context,
        )
        chosen = choose_reteach_reply(reply, repaired_reply, message, image_context, prior_context)
        if chosen == repaired_reply:
            reply, final_thinking = repaired_reply, repaired_thinking
    done_payload: dict[str, object] = {"type": "done", "reply": reply, "meta": meta.model_dump()}
    if final_thinking:
        done_payload["thinking"] = final_thinking
    if image_context:
        done_payload["ocr_text"] = image_context
        stem = extract_question_stem(image_context, message)
        if stem:
            done_payload["ocr_stem"] = stem
    yield sse_event(done_payload)


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
        "extra_body": {"enable_thinking": True, "thinking_budget": 8192} if enable_thinking else {"enable_thinking": False},
    }

    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception:
        if enable_thinking:
            kwargs["extra_body"] = {"enable_thinking": True}
            try:
                stream = client.chat.completions.create(**kwargs)
            except Exception:
                logger.exception("Thinking stream failed, retrying without thinking")
                kwargs.pop("extra_body", None)
                stream = client.chat.completions.create(**kwargs)
        else:
            logger.exception("Stream extras were rejected, retrying without them")
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
    nudge = reteach_nudge(reply, source_message, image_context, prior_context)
    if nudge:
        logger.warning("teach stage=reteach")
        status = "没贴住原句，再讲一次" if "不合格" in nudge else "题目已经在，继续讲"
        yield sse_event({"type": "status", "stage": "analyze", "text": status})
        retry_parts: list[str] = []
        retry_thinking = ""
        try:
            for reasoning, content in iter_completion_deltas(
                client,
                model=model,
                messages=[*messages, {"role": "user", "content": nudge}],
                temperature=temperature,
                enable_thinking=enable_thinking,
            ):
                if reasoning:
                    retry_thinking += reasoning
                if content:
                    retry_parts.append(content)
        except Exception:
            logger.exception("Reteach respond failed")
        else:
            retry_reply = "".join(retry_parts).strip()
            chosen = choose_reteach_reply(
                reply,
                retry_reply,
                source_message,
                image_context,
                prior_context,
            )
            if chosen == retry_reply:
                reply = retry_reply
                if retry_thinking:
                    raw_thinking = retry_thinking

    rule_text = teaching_material(source_message, image_context)
    reply = normalize_structured_reply(
        reply,
        message=rule_text,
        image_context=image_context,
        prior_context=prior_context,
    )
    final_thinking = sanitize_thinking_text(raw_thinking) or emitted_thinking
    final_thinking = scrub_echoed_thinking(final_thinking, rule_text)
    final_thinking = align_article_thinking(final_thinking, rule_text)
    final_thinking = align_fronted_passive_thinking(final_thinking, rule_text)
    if not final_thinking:
        final_thinking = thinking_from_structured_reply(reply)
    if final_thinking:
        reply = merge_thinking_into_structured_reply(reply, final_thinking)
    polished = parse_structured_reply(reply)
    if polished:
        apply_answer_corrections(polished, rule_text)
        reply = json.dumps(polished, ensure_ascii=False)
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
        ocr_text: str | None = None
        used_vision = False
        route: Literal["text", "vision", "ocr_plus_vision"] = "text"
        if request.latest_image is not None:
            route = "vision"
            if request.use_ocr_first:
                route = "ocr_plus_vision"
                yield sse_event({"type": "status", "stage": "ocr", "text": "先把图片里的字认出来"})
                ocr_text = extract_ocr_text(client, request.latest_image, ocr_model)
                if ocr_text:
                    yield sse_event({"type": "ocr", "text": ocr_text})
            if not has_enough_question_material(request.message, ocr_text):
                yield sse_event({"type": "status", "stage": "ocr", "text": "再把题目抄出来"})
                transcript = transcribe_question_image(client, request.latest_image, vision_model)
                used_vision = True
                if transcript:
                    ocr_text = transcript
                    yield sse_event({"type": "ocr", "text": ocr_text})
            forced = deterministic_structured_reply(request.message, ocr_text)
            if forced:
                meta = ModelMeta(
                    route=route,
                    provider=settings.provider_name,
                    text_model=text_model,
                    vision_model=vision_model if used_vision else None,
                    ocr_model=ocr_model if request.use_ocr_first else None,
                )
                yield from iter_forced_reply_events(forced, meta)
                return

        else:
            forced = deterministic_structured_reply(request.message)
            if forced:
                meta = ModelMeta(
                    route="text",
                    provider=settings.provider_name,
                    text_model=text_model,
                )
                yield from iter_forced_reply_events(forced, meta)
                return

        prior_context = "\n".join(item.content for item in request.history if item.sender == "user")
        yield sse_event({"type": "status", "stage": "classify", "text": "先判断题型"})
        question_type = resolve_question_type(
            client,
            text_model,
            request.message,
            request.history,
            ocr_text,
        )
        yield sse_event({"type": "status", "stage": "teach", "text": f"按{question_type}讲"})
        meta = ModelMeta(
            route=route,
            provider=settings.provider_name,
            text_model=text_model,
            vision_model=vision_model if used_vision else None,
            ocr_model=ocr_model if request.latest_image is not None and request.use_ocr_first else None,
        )
        yield from iter_staged_teach_events(
            client,
            model=text_model,
            message=request.message,
            history=request.history,
            question_type=question_type,
            image_context=ocr_text,
            prior_context=prior_context,
            meta=meta,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Streaming model pipeline failed, fallback to demo reply")
        yield sse_event({"type": "status", "stage": "fallback", "text": "换个方式继续想这道题"})
        yield demo_done()


def _parse_sse_payload(chunk: str) -> dict[str, object] | None:
    data = "\n".join(line[5:].strip() for line in chunk.splitlines() if line.startswith("data:")).strip()
    if not data:
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def run_model_pipeline(request: ChatRequest) -> ChatResponse:
    """Same task states as the stream: classify, teach, then at most one reteach."""
    done: dict[str, object] | None = None
    for chunk in iter_chat_sse(request):
        payload = _parse_sse_payload(chunk)
        if payload and payload.get("type") == "done":
            done = payload
    settings = get_model_settings()
    if not done:
        return ChatResponse(
            reply=build_demo_reply(request.message, request.history, request.latest_image),
            meta=ModelMeta(
                route="demo",
                provider=settings.provider_name,
                text_model=settings.text_model,
                used_demo_fallback=True,
            ),
        )
    meta_raw = done.get("meta")
    meta = ModelMeta.model_validate(meta_raw) if isinstance(meta_raw, dict) else ModelMeta(
        route="demo",
        provider=settings.provider_name,
        used_demo_fallback=True,
    )
    ocr_text = done.get("ocr_text")
    ocr_stem = done.get("ocr_stem")
    return ChatResponse(
        reply=str(done.get("reply") or ""),
        meta=meta,
        ocr_text=ocr_text if isinstance(ocr_text, str) else None,
        ocr_stem=ocr_stem if isinstance(ocr_stem, str) else None,
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
