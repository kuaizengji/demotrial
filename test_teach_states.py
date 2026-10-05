"""Checks that one answer is built from separate task states, not the whole prompt."""

from __future__ import annotations

import json
import re
import sys

import app

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ARCHIVE_MARKERS = (
    "H开头不发音时使用an",
    "语法填空禁止照抄提示词原词",
    "情态动词+动词原形",
)


def system_text(question_type: str) -> str:
    messages = app.build_text_messages("题干", [], question_type=question_type)
    systems = [item["content"] for item in messages if item["role"] == "system"]
    assert len(systems) == 1
    return systems[0]


def test_prompt_is_split_by_type() -> None:
    app.prompt_blocks.cache_clear()
    grammar = system_text("语法")
    cloze = system_text("完型")
    reading = system_text("阅读")
    assert "起承转合" not in grammar
    assert "起承转合" in cloze
    assert "一个句子只能有一个谓语" in grammar
    assert "一个句子只能有一个谓语" not in cloze
    assert "细节题先定位" in reading
    assert "起承转合" not in reading
    seen_shapes = ("university", "spent my childhood", "紧张的表现", "(see/hear", "her scarf")
    cards = [system_text(name) for name in app.KNOWN_QUESTION_TYPES]
    for text in cards:
        for marker in ARCHIVE_MARKERS:
            assert marker not in text
        for shape in seen_shapes:
            assert shape not in text
        assert "<!-- block:" not in text
        assert "先看这个空" in text
    assert "谓语" in system_text("长难句")
    assert "搭配" in system_text("词汇")
    assert "起承转合" not in system_text("长难句")
    classify = app.build_classify_messages("She is a student.", None)[0]["content"]
    assert "起承转合" not in classify
    assert "reasoning_steps" not in classify
    assert "reasoning_steps" not in grammar
    packed = app.format_system_prompt()
    assert "reasoning_steps" in packed
    assert "起承转合" not in packed
    assert "一个句子只能有一个谓语" not in packed
    assert app.reason_thinking_enabled("DeepSeek-V4-Pro")
    assert app.reason_thinking_enabled("DeepSeek-V3.2")
    assert not app.reason_thinking_enabled("DeepSeek-OCR")


def test_confident_type_does_not_need_a_model() -> None:
    grammar = "语法填空\nIt is (1) ______ unusual day."
    assert app.resolve_question_type(None, "unused", grammar, [], None) == "语法"
    cloze = "完形填空\nshe (1) ______ her scarf.\n(1) A. adjusted B. changed C. packed D. waved"
    assert app.resolve_question_type(None, "unused", cloze, [], None) == "完型"
    bare = app.deterministic_structured_reply("帮我讲完形")
    assert bare is not None
    assert "需要确认" in bare
    choice = "单项选择\nThe window ______ by the storm last night stayed open.\nA. breaks\nB. broke\nC. was broken\nD. breaking"
    assert app.resolve_question_type(None, "unused", choice, [], None) == "语法"
    sentence = "长难句：The letter that my uncle sent me is still on the kitchen table."
    assert app.resolve_question_type(None, "unused", sentence, [], None) == "长难句"
    vocab = "词义猜测\nHere 'spare' means he had no extra time.\nA. extra\nB. angry\nC. late\nD. loud"
    assert app.resolve_question_type(None, "unused", vocab, [], None) == "词汇"


def test_translation_plus_letter_is_rejected() -> None:
    material = (
        "she looked down and (1) ______ her scarf.\n"
        "(1) A. adjusted B. changed C. packed D. waved"
    )
    parsed = {
        "answer": "(1) A",
        "reasoning_steps": [
            {
                "focus": "空1",
                "basis": "女人看到男子盯着她，立刻低下头并调整围巾，这是紧张的表现。",
                "conclusion": "所以选 A",
            }
        ],
    }
    assert app.explanation_needs_reteach(parsed, material)
    grounded = {
        "answer": "(1) A",
        "reasoning_steps": [
            {
                "focus": "空1",
                "basis": "空在 looked down and ______ her scarf。scarf 前面要接一个作用在围巾上的动作，所以 adjusted。changed 不接 scarf，packed 是打包，waved 是挥手。",
                "conclusion": "A. adjusted",
            }
        ],
    }
    assert not app.explanation_needs_reteach(grounded, material)


def test_each_blank_must_quote_its_own_sentence() -> None:
    material = (
        "It is (1) ______ unusual experience, and she studies at (2) ______ university in London."
    )
    one_step = {
        "answer": "(1) an；(2) a",
        "reasoning_steps": [
            {
                "focus": "空1",
                "basis": "unusual 是元音音素，用 an，不用 a。",
                "conclusion": "an",
            }
        ],
    }
    assert app.explanation_needs_reteach(one_step, material)
    both = {
        "answer": "(1) an；(2) a",
        "reasoning_steps": [
            {
                "focus": "空1",
                "basis": "unusual 开头是元音音素，用 an，a 会按辅音误选。",
                "conclusion": "an",
            },
            {
                "focus": "空2",
                "basis": "university 开头读 /j/，用 a，an 只看了首字母。",
                "conclusion": "a",
            },
        ],
    }
    assert not app.explanation_needs_reteach(both, material)


def test_gaokao_blank_numbers_are_checked() -> None:
    material = "a pavilion, (60) ______ (inspire) by the play, (61) ______ (build) at the garden."
    parsed = {
        "answer": "60 inspired；61 was built",
        "reasoning_steps": [
            {
                "focus": "60",
                "basis": "pavilion 被 play 启发，用 inspired，inspiring 表示亭子自己去启发，接不上。",
                "conclusion": "inspired",
            },
            {
                "focus": "61",
                "basis": "is 开头是元音音素，冠词用 an。",
                "conclusion": "an",
            },
        ],
    }
    assert app.teach_blank_ids(material) == ["60", "61"]
    assert app.explanation_needs_reteach(parsed, material)
    grounded = {
        "answer": "60 inspired；61 was built",
        "reasoning_steps": [
            parsed["reasoning_steps"][0],
            {
                "focus": "61",
                "basis": "空是主句谓语，主语 pavilion 是 build 的承受者；Two years later 指过去。built 单独不能构成完整谓语，were built 与单数主语不一致。",
                "conclusion": "was built",
            },
        ],
    }
    assert not app.explanation_needs_reteach(grounded, material)


def test_following_is_is_not_rewritten_as_an_article() -> None:
    message = (
        "Tang, (56) ______ is known as the Shakespeare of Asia. "
        "A pavilion, (61) ______ (build) at the garden."
    )
    parsed = {
        "answer": "56 who；61 was built",
        "reasoning_steps": [
            {
                "focus": "61",
                "basis": "空是主句谓语，主语 a pavilion 是 build 的承受者，所以填 was built。were built 和单数主语接不上。",
                "conclusion": "was built",
            }
        ],
        "knowledge_methodology": ["这句还没有谓语时，按主语和提示词决定被动。"],
    }
    app.revise_article_by_sound(parsed, message)
    step = parsed["reasoning_steps"][0]
    assert "元音" not in step["basis"]
    assert step["conclusion"] == "was built"
    assert "冠词" not in parsed["knowledge_methodology"][0]
    cued = "A pavilion, (61) ______ (build) at the garden."
    assert app.explanation_needs_reteach(
        {
            "answer": "61 was built",
            "reasoning_steps": [
                {"focus": "61", "basis": "is 开头是元音音素，冠词用 an。", "conclusion": "an"}
            ],
        },
        cued,
    )
    assert any("build" in note for note in app.cued_blank_article_notes(
        {
            "reasoning_steps": [
                {"focus": "61", "basis": "is 开头是元音音素，冠词用 an。", "conclusion": "an"}
            ]
        },
        cued,
    ))


def test_article_blank_still_follows_the_next_sound() -> None:
    message = "It was (1) ______ unusual experience."
    parsed = {
        "answer": "(1) a",
        "reasoning_steps": [
            {"focus": "空1", "basis": "unusual 开头是辅音，所以用 a。", "conclusion": "a"}
        ],
    }
    app.revise_article_by_sound(parsed, message)
    assert "(1) an" in str(parsed["answer"]).lower() or str(parsed["answer"]).lower().endswith("an")
    assert parsed["reasoning_steps"][0]["conclusion"] == "an"
    assert "unusual" in parsed["reasoning_steps"][0]["basis"]


def test_cued_blank_must_explain_its_own_word() -> None:
    material = "A pavilion, (61) ______ (build) at the garden."
    wrong = {
        "answer": "61 by",
        "reasoning_steps": [
            {"focus": "61", "basis": "空格后面是 garden，介词用 by，不用 in。", "conclusion": "by"}
        ],
    }
    assert app.explanation_needs_reteach(wrong, material)
    right = {
        "answer": "61 was built",
        "reasoning_steps": [
            {
                "focus": "61",
                "basis": "主语 pavilion 承受 build。built 不能单独作谓语，were built 和单数接不上。",
                "conclusion": "was built",
            }
        ],
    }
    assert not app.explanation_needs_reteach(right, material)
    assert app.text_uses_cue("was built", "build")
    assert app.text_uses_cue("visibility", "visible")
    assert app.text_uses_cue("were", "be")
    assert not app.text_uses_cue("by", "build")


def test_canned_rule_stays_on_its_own_blank() -> None:
    message = (
        "This is the village (56) ______ I spent my childhood. "
        "A pavilion (61) ______ (build) at the garden."
    )
    parsed = {
        "answer": "(56) which；61 was built",
        "reasoning_steps": [
            {"focus": "56", "basis": "这里用 which。", "conclusion": "which"},
            {
                "focus": "61",
                "basis": "which 不能指地点。主语 pavilion 承受 build，所以 was built，were built 接不上。",
                "conclusion": "was built",
            },
        ],
    }
    app.revise_complete_relative_clause(parsed, message)
    assert parsed["reasoning_steps"][1]["conclusion"] == "was built"
    assert "build" in parsed["reasoning_steps"][1]["basis"]


def test_rewrite_starts_from_the_blank_that_failed() -> None:
    material = "\n".join(
        [
            "Tang, (56) ______ is known as Shakespeare.",
            "There are common (57) ______ (theme) in the plays.",
            "The things (58) ______ (be) his concerns in 1616.",
            "It is similar (59) ______ Romeo.",
            "A pavilion, (61) ______ (build) at the garden.",
        ]
    )
    parsed = {
        "answer": "56 who；57 themes；58 were；59 to；61 an",
        "reasoning_steps": [
            {"focus": "56", "basis": "Tang 是人，known 前缺主语，用 who，which 指物接不上。", "conclusion": "who"},
            {"focus": "57", "basis": "There are 后接 theme 的复数 themes，单数 theme 和 are 接不上。", "conclusion": "themes"},
            {"focus": "58", "basis": "things 是复数，1616 是过去，be 用 were，is 和复数接不上。", "conclusion": "were"},
            {"focus": "59", "basis": "similar 后用 to 引出 Romeo，with 和 similar 接不上。", "conclusion": "to"},
            {"focus": "61", "basis": "is 开头是元音音素，冠词用 an。", "conclusion": "an"},
        ],
    }
    assert app.explanation_needs_reteach(parsed, material)
    text = app.rewrite_instruction(parsed, material)
    assert text.index("空61") < text.index("空56") if "空56" in text else True
    assert "build" in text
    assert "提示词是 build" in text


def test_new_question_drops_the_previous_passage() -> None:
    history = [
        app.ChatMessage(sender="user", content="阅读理解\nMaya used to sell maps at the station."),
        app.ChatMessage(
            sender="ai",
            content=json.dumps({"supported": True, "question_type": "阅读", "answer": "C"}, ensure_ascii=False),
        ),
    ]
    fresh = "语法填空\nThe (1) ______ (child) in the yard (2) ______ (be) quiet."
    assert app.should_isolate_history(fresh)
    assert app.iter_compacted_history(history, fresh, limit=12) == []
    follow = "第2题为什么不选 A，能再对着上一题的原文讲一下这个选项吗"
    assert not app.should_isolate_history(follow)
    assert app.iter_compacted_history(history, follow, limit=12)


def test_long_single_choice_stays_grammar() -> None:
    stem = " ".join(["window"] * 75)
    choice = f"{stem} ______ open all morning.\nA. stays\nB. stayed\nC. was staying\nD. had stayed"
    assert app.resolve_question_type(None, "unused", choice, [], None) == "语法"
    passage = " ".join(["guide"] * 100) + "\nA. maps\nB. winter\nC. Monday\nD. station"
    assert app.resolve_question_type(None, "unused", passage, [], None) == "阅读"


def test_rewrite_asks_for_the_missing_judgment() -> None:
    material = "she looked down and (1) ______ her scarf.\n(1) A. adjusted B. changed C. packed D. waved"
    parsed = {
        "answer": "(1) A",
        "reasoning_steps": [
            {"focus": "空1", "basis": "女人低下头，这是紧张的表现。", "conclusion": "A"}
        ],
    }
    assert app.explanation_needs_reteach(parsed, material)
    instruction = app.rewrite_instruction(parsed, material)
    assert "scarf" in instruction
    assert "紧张的表现" in instruction
    assert "充当什么" in instruction
    assert "排除某词，冲突，所以填答案" in instruction
    assert "排除「被排除的词」" not in instruction


def test_rejection_without_the_word_conflict_still_counts() -> None:
    material = (
        "During my first visit, I (42) ______ to ask for directions.\n"
        "(42) A. planned B. struggled C. refused D. happened"
    )
    parsed = {
        "answer": "42 B",
        "reasoning_steps": [
            {
                "focus": "42",
                "basis": "这一空作谓语。ask for directions 决定填 struggled。planned 不能引出后面的困难，refused 与 tried 方向相反。",
                "conclusion": "struggled",
            }
        ],
    }
    assert app.names_distractor(parsed["reasoning_steps"][0]["basis"])
    assert not app.explanation_needs_reteach(parsed, material)
    praised = {
        "answer": "44 D；45 B",
        "reasoning_steps": [
            {
                "focus": "44",
                "basis": "这一空与 smiled at me 并列。assessed 表示客观评定，不含 smiled 带出的肯定态度，所以选 praised。",
                "conclusion": "praised",
            },
            {
                "focus": "45",
                "basis": "这一空作 get through 的宾语。test 指语言测试，原句只有问路和点餐，没有考试情境，所以选 barrier。",
                "conclusion": "barrier",
            },
        ],
    }
    cloze = (
        "the locals smiled at me and (44) ______ my language skills. "
        "That encouragement helped me to get through the language (45) ______.\n"
        "(44) A. improved B. assessed C. admired D. praised\n"
        "(45) A. course B. barrier C. area D. test"
    )
    assert not app.explanation_needs_reteach(praised, cloze)


def test_planning_notes_do_not_replace_the_lecture() -> None:
    material = "I (42) ______ to ask for directions."
    content = "42 这一空作谓语。ask for directions 决定填 struggled。planned 只表示计划。"
    thinking = "我们需要回答用户。不要只写排除冲突所以。原文有 ask for directions。"
    chosen = app.reason_prose(content, thinking, material)
    assert "struggled" in chosen
    assert "我们需要" not in chosen
    parsed = {
        "reasoning_steps": [
            {
                "focus": "42",
                "basis": "ask for directions 决定填 struggled。planned 不能引出困难。",
                "conclusion": "struggled",
            },
            {
                "focus": "空42",
                "basis": "需要看用户之前说括号里有提示词时。所以 42 是谓语。",
                "conclusion": "",
            },
        ]
    }
    app.drop_meta_reasoning_steps(parsed)
    assert len(parsed["reasoning_steps"]) == 1
    assert "struggled" in parsed["reasoning_steps"][0]["basis"]


def test_duplicate_step_for_the_same_item_is_dropped() -> None:
    parsed = {
        "answer": "A",
        "reasoning_steps": [
            {"focus": "31", "basis": "开头 shortens it，B 项 management 接不上。", "conclusion": "A"},
            {"focus": "空31", "basis": "开头 shortens it，B 项 management 接不上。", "conclusion": ""},
        ],
    }
    app.drop_duplicate_blank_steps(parsed)
    assert len(parsed["reasoning_steps"]) == 1
    assert parsed["reasoning_steps"][0]["conclusion"] == "A"


def test_history_followup_keeps_the_previous_type() -> None:
    history = [
        app.ChatMessage(
            sender="ai",
            content=json.dumps({"supported": True, "question_type": "阅读", "answer": "B"}, ensure_ascii=False),
        )
    ]
    follow = "第2题为什么不选 A，能再对着上一题的原文讲一下这个选项吗"
    assert app.resolve_question_type(None, "unused", follow, history, None) == "阅读"


def picked_letters(answer: str) -> str:
    picks = re.findall(r"\(\s*\d+\s*\)\s*([A-G])\b", answer or "", flags=re.I)
    return "".join(picks).upper()


def answered(parsed: dict) -> bool:
    answer = str(parsed.get("answer") or "").strip()
    return bool(answer) and answer != "需要确认" and not parsed.get("need_more_context")


def holds(parsed: dict, material: str) -> bool:
    return answered(parsed) and not app.explanation_needs_reteach(parsed, material)


YARD = "语法填空\nThe (1) ______ (child) in the next yard (2) ______ (be) quiet this morning."
WHETHER = "\n".join(
    [
        "单项选择",
        "I have no idea ______ the train has left the station.",
        "A. if",
        "B. whether",
        "C. which",
        "D. what",
    ]
)
KEYS = "\n".join(
    [
        "完形填空",
        "Tom lost his keys on Monday. He was still (1) ______ on Wednesday because the door stayed locked.",
        "His sister finally found the keys in her bag and felt (2) ______.",
        "(1) A. worried B. proud C. hungry D. sleepy",
        "(2) A. sorry B. angry C. glad D. afraid",
    ]
)
MAYA = "\n".join(
    [
        "阅读理解",
        "Maya used to sell maps at the station. Last winter she started teaching visitors how to read those maps, and the station asked her to stay. She now trains new guides every Monday.",
        "1. What does the passage mainly tell us about Maya?",
        "A. She still only sells maps.",
        "B. She left the station last winter.",
        "C. Her work changed from selling maps to training guides.",
        "D. She teaches every day of the week.",
    ]
)
PLANT = "\n".join(
    [
        "七选五",
        "Keeping a plant on the desk can make a small room feel less bare. (1) ______.",
        "Water it when the soil is dry. (2) ______.",
        "Too much water is a common reason it dies.",
        "A. Choose a pot with a hole at the bottom.",
        "B. Many people dislike green walls.",
        "C. Put it where it can get some daylight.",
        "D. Maps are sold at the station.",
        "E. Then empty the extra water after ten minutes.",
        "F. The station asked her to stay.",
        "G. Food is not allowed in the hall.",
    ]
)
LETTER = "长难句：The letter that my uncle sent me yesterday and that I have not opened is still on the kitchen table."
SHRINK = "\n".join(
    [
        "词义猜测",
        "In the dry season the river shrank until people could walk across its bed. The word shrank is closest in meaning to ______.",
        "A. grew wider",
        "B. became smaller",
        "C. moved faster",
        "D. turned dirty",
    ]
)


CASES = [
    (
        "plural-and-agreement",
        YARD,
        lambda parsed: holds(parsed, YARD)
        and re.search(r"\bchildren\b", str(parsed.get("answer") or ""), flags=re.I)
        and re.search(r"\bare\b", str(parsed.get("answer") or ""), flags=re.I),
    ),
    (
        "grammar-choice",
        WHETHER,
        lambda parsed: holds(parsed, WHETHER)
        and str(parsed.get("question_type") or "") == "语法"
        and re.search(r"whether|\bB\b", str(parsed.get("answer") or ""), flags=re.I),
    ),
    (
        "cloze-attitude",
        KEYS,
        lambda parsed: holds(parsed, KEYS)
        and str(parsed.get("question_type") or "") == "完型"
        and re.search(r"\(1\)\s*(?:worried|A)\b", str(parsed.get("answer") or ""), flags=re.I)
        and re.search(r"\(2\)\s*(?:glad|C)\b", str(parsed.get("answer") or ""), flags=re.I),
    ),
    (
        "reading-main-idea",
        MAYA,
        lambda parsed: holds(parsed, MAYA)
        and str(parsed.get("question_type") or "") == "阅读"
        and re.search(r"\bC\b", str(parsed.get("answer") or "")),
    ),
    (
        "seven-sentences",
        PLANT,
        lambda parsed: holds(parsed, PLANT)
        and str(parsed.get("question_type") or "") == "七选五"
        and picked_letters(str(parsed.get("answer") or ""))
        and not set(picked_letters(str(parsed.get("answer") or ""))) & set("DFG"),
    ),
    (
        "translation",
        "翻译：我们一到车站，火车就开了。",
        lambda parsed: holds(parsed, "翻译：我们一到车站，火车就开了。")
        and re.search(r"as soon as|the moment|no sooner|hardly", str(parsed.get("answer") or ""), flags=re.I),
    ),
    (
        "long-sentence",
        LETTER,
        lambda parsed: holds(parsed, LETTER)
        and str(parsed.get("question_type") or "") == "长难句"
        and re.search(r"letter", str(parsed.get("answer") or "") + str(parsed.get("reasoning_steps") or ""), flags=re.I),
    ),
    (
        "correction-plural",
        "改错：There are a lot of book on the shelf. He go to the library every Sunday.",
        lambda parsed: answered(parsed)
        and re.search(r"\bbooks\b", str(parsed.get("answer") or ""), flags=re.I)
        and re.search(r"\bgoes\b", str(parsed.get("answer") or ""), flags=re.I),
    ),
    (
        "word-meaning",
        SHRINK,
        lambda parsed: holds(parsed, SHRINK)
        and str(parsed.get("question_type") or "") == "词汇"
        and re.search(r"smaller|\bB\b", str(parsed.get("answer") or ""), flags=re.I),
    ),
]


def run_live_cases() -> int:
    settings = app.get_model_settings()
    if not settings.enabled:
        print("LIVE skip: no API key")
        return 1
    failed = 0
    for name, message, check in CASES:
        print(f"\n=== {name} ===", flush=True)
        response = app.run_model_pipeline(
            app.ChatRequest(message=message, history=[], use_ocr_first=False)
        )
        parsed = app.parse_structured_reply(response.reply) or {}
        answer = str(parsed.get("answer") or response.reply[:240])
        ok = bool(parsed) and bool(check(parsed))
        print("model", response.meta.text_model, "route", response.meta.route)
        print("type", parsed.get("question_type"), "answer", answer)
        steps = parsed.get("reasoning_steps")
        if isinstance(steps, list):
            for step in steps[:6]:
                if isinstance(step, dict):
                    print("-", step.get("focus"), step.get("basis"))
        print("PASS" if ok else "FAIL")
        if not ok:
            failed += 1
    return failed


def run_unit() -> None:
    test_prompt_is_split_by_type()
    test_confident_type_does_not_need_a_model()
    test_translation_plus_letter_is_rejected()
    test_each_blank_must_quote_its_own_sentence()
    test_gaokao_blank_numbers_are_checked()
    test_following_is_is_not_rewritten_as_an_article()
    test_article_blank_still_follows_the_next_sound()
    test_cued_blank_must_explain_its_own_word()
    test_canned_rule_stays_on_its_own_blank()
    test_rewrite_starts_from_the_blank_that_failed()
    test_history_followup_keeps_the_previous_type()
    test_new_question_drops_the_previous_passage()
    test_long_single_choice_stays_grammar()
    test_rewrite_asks_for_the_missing_judgment()
    test_rejection_without_the_word_conflict_still_counts()
    test_planning_notes_do_not_replace_the_lecture()
    test_duplicate_step_for_the_same_item_is_dropped()
    print("unit ok")


if __name__ == "__main__":
    run_unit()
    raise SystemExit(run_live_cases())
