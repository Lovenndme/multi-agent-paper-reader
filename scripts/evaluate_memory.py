"""Real-model quality, persistence, and latency evaluation for the LangMem layer."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.chat import PaperChatRequest, build_chat_prompt, store_analysis_session, stream_chat_reply
from core.chat_memory import (
    add_conversation_message,
    create_conversation,
    get_prompt_memory,
    load_conversation,
    refresh_conversation_memory,
)
from core.evidence import EvidenceSnippet
from core.history import save_paper_analysis
from core.langmem_store import (
    PaperReaderMemory,
    get_langmem_store,
    list_langmem_memories,
    memory_namespace,
    reset_langmem_store,
)
from core.model_providers import selected_text_model, text_provider_id


def _paper(title: str) -> str:
    payload = ("%PDF-1.4 " + title).encode()
    return save_paper_analysis(
        pdf_data=payload,
        result={
            "mode": "live",
            "paper": {
                "title": title,
                "filename": "evaluation.pdf",
                "pages": 2,
                "sections_count": 2,
                "size_bytes": len(payload),
            },
        },
        snippets=[EvidenceSnippet("E001", "Method", 0, 0, "evaluation evidence")],
    )


def _new_case(title: str) -> tuple[str, str]:
    history_id = _paper(title)
    conversation = create_conversation(history_id, title=title)
    return history_id, str(conversation["id"])


def _turn(conversation_id: str, user: str, assistant: str = "明白。") -> None:
    add_conversation_message(conversation_id, role="user", content=user)
    add_conversation_message(conversation_id, role="assistant", content=assistant)


def _records(history_id: str) -> list[dict[str, Any]]:
    return list_langmem_memories(history_id, limit=20, min_score=-1)


def _text(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key) or "") for key in ("topic", "description", "content")
    ).casefold()


def _contains(
    records: list[dict[str, Any]],
    *,
    category: str,
    alternatives: tuple[tuple[str, ...], ...],
) -> bool:
    return any(
        record.get("type") == category
        and any(all(term.casefold() in _text(record) for term in terms) for terms in alternatives)
        for record in records
    )


def _put(history_id: str, key: str, memory: PaperReaderMemory) -> None:
    get_langmem_store().put(
        memory_namespace(history_id),
        key,
        {"kind": "PaperReaderMemory", "content": memory.model_dump(mode="json")},
    )


def _evaluate_management() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    latencies: dict[str, float] = {}
    states: dict[str, list[dict[str, Any]]] = {}
    processed: dict[str, int] = {}
    cases = (
        (
            "save_feedback",
            "请记住：论文章节标题不要强制翻译，优先保留原文，因为技术术语可能失真。",
            "feedback",
            (("标题", "原文"), ("title", "original"), ("title", "source language")),
        ),
        (
            "save_user_profile",
            "请记住：我是 CSSE 硕士生，解释工程实现时可以默认我具备 Python 基础。",
            "user",
            (("csse",),),
        ),
        (
            "save_reference",
            "请记住：官方更新入口是 https://github.com/example/paper-reproduction ，以后从这里检查最新说明。",
            "reference",
            (("github.com/example/paper-reproduction",),),
        ),
    )
    for name, user, category, alternatives in cases:
        history_id, conversation_id = _new_case(f"LangMem: {name}")
        _turn(conversation_id, user)
        started = time.perf_counter()
        processed[name] = refresh_conversation_memory(conversation_id)
        latencies[name] = time.perf_counter() - started
        states[name] = _records(history_id)
        checks[name] = _contains(states[name], category=category, alternatives=alternatives)

    history_id, conversation_id = _new_case("LangMem: reject ephemeral")
    _turn(conversation_id, "我现在把页面滚到了第 3 页。", "好的。")
    started = time.perf_counter()
    processed["reject_ephemeral"] = refresh_conversation_memory(conversation_id)
    latencies["reject_ephemeral"] = time.perf_counter() - started
    states["reject_ephemeral"] = _records(history_id)
    checks["reject_ephemeral"] = not states["reject_ephemeral"]

    history_id, conversation_id = _new_case("LangMem: correct memory")
    _put(
        history_id,
        "title-language",
        PaperReaderMemory(
            category="feedback",
            subject="章节标题语言",
            content="章节标题一律优先保留英文原文，不要显示中文标题。",
            context="旧规则。",
        ),
    )
    _turn(conversation_id, "请更正已有记忆：以后优先显示准确中文标题，只有翻译有歧义时才保留英文。")
    started = time.perf_counter()
    processed["correct_memory"] = refresh_conversation_memory(conversation_id)
    latencies["correct_memory"] = time.perf_counter() - started
    states["correct_memory"] = _records(history_id)
    title_records = [record for record in states["correct_memory"] if "title" in _text(record) or "标题" in _text(record)]
    checks["correct_memory"] = (
        len(title_records) == 1
        and ("中文" in _text(title_records[0]) or "chinese" in _text(title_records[0]))
        and "一律优先保留英文原文" not in _text(title_records[0])
    )

    history_id, conversation_id = _new_case("LangMem: forget memory")
    _put(
        history_id,
        "user-background",
        PaperReaderMemory(
            category="user",
            subject="用户专业背景",
            content="用户是 CSSE 硕士生，并具备 Python 基础。",
        ),
    )
    _turn(conversation_id, "请忘记已有记忆中关于我的专业和学习背景。")
    started = time.perf_counter()
    processed["forget_memory"] = refresh_conversation_memory(conversation_id)
    latencies["forget_memory"] = time.perf_counter() - started
    states["forget_memory"] = _records(history_id)
    checks["forget_memory"] = not any("csse" in _text(record) for record in states["forget_memory"])

    values = list(latencies.values())
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pass_rate": sum(checks.values()) / len(checks),
        "processed_messages": processed,
        "case_latencies_s": {name: round(value, 3) for name, value in latencies.items()},
        "latency_median_s": round(statistics.median(values), 3),
        "latency_max_s": round(max(values), 3),
        "memory_counts": {name: len(records) for name, records in states.items()},
    }


def _evaluate_recall() -> dict[str, Any]:
    history_id = _paper("LangMem Recall Evaluation")
    corpus = (
        ("attention-architecture", "注意力架构与稀疏查询机制", "论文注意力架构采用稀疏查询机制。"),
        ("ablation-results", "消融实验结果与性能增益", "消融实验记录各组件带来的性能增益。"),
        ("training-configuration", "训练超参数、学习率与批量大小", "训练配置包括 learning rate 和 batch size。"),
        ("dataset-benchmarks", "数据集和评测基准", "实验使用的数据集、指标与 benchmark。"),
        ("paper-limitations", "论文局限、风险与未来工作", "局限性和 future work 的讨论。"),
        ("hardware-environment", "实验硬件、GPU 与运行环境", "复现所用 GPU、显存和软件环境。"),
        ("title-language", "章节标题语言显示规则", "章节标题优先准确中文，歧义时保留英文。"),
        ("reproduction-reference", "官方复现资料入口", "官方复现说明位于代码仓库。"),
    )
    for key, subject, content in corpus:
        _put(
            history_id,
            key,
            PaperReaderMemory(category="reference" if key.endswith("reference") else "project", subject=subject, content=content),
        )
    cases = (
        ("这篇论文的注意力架构和稀疏查询是怎么设计的？", "注意力架构"),
        ("消融实验中各组件带来了多少性能提升？", "消融实验"),
        ("训练时的 learning rate 和 batch size 是什么？", "训练"),
        ("论文在哪些数据集和 benchmark 上进行了评测？", "数据集"),
        ("作者承认了哪些局限和未来工作？", "局限"),
        ("复现实验需要什么 GPU 和软件环境？", "硬件"),
        ("章节标题应该用中文还是保留英文？", "标题语言"),
        ("去哪里查找官方复现说明？", "复现资料"),
    )
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    reciprocal_ranks: list[float] = []
    for query, expected in cases:
        started = time.perf_counter()
        selected = list_langmem_memories(history_id, query=query, limit=3)
        latencies.append(time.perf_counter() - started)
        names = [str(item["topic"]) for item in selected]
        rank = next((index + 1 for index, name in enumerate(names) if expected in name), 0)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append({"expected": expected, "selected": names, "rank": rank})
    started = time.perf_counter()
    no_match = list_langmem_memories(history_id, query="今天上海天气怎么样？", limit=3)
    no_match_latency = time.perf_counter() - started
    return {
        "cases": rows,
        "hit_at_3": sum(bool(row["rank"]) for row in rows) / len(rows),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "mean_selected": sum(len(row["selected"]) for row in rows) / len(rows),
        "no_match_empty": not no_match,
        "selector_latency_median_s": round(statistics.median(latencies), 4),
        "selector_latency_max_s": round(max(latencies), 4),
        "no_match_latency_s": round(no_match_latency, 4),
    }


def _evaluate_restart_and_scale(data_dir: str) -> dict[str, Any]:
    history_id = _paper("LangMem Restart and Scale")
    first = create_conversation(history_id, title="First")
    second = create_conversation(history_id, title="Second")
    _put(
        history_id,
        "rare-result",
        PaperReaderMemory(
            category="project",
            subject="早期稀有消融结论",
            content="移除路由器后准确率下降 4.2%。",
        ),
    )
    for index in range(1_000):
        add_conversation_message(
            first["id"],
            role="user" if index % 2 == 0 else "assistant",
            content=f"常规论文讨论 message {index}",
        )
    started = time.perf_counter()
    prompt_memory = get_prompt_memory(second["id"], "早期稀有消融结论是什么？")
    recall_latency = time.perf_counter() - started
    code = (
        "import json,os; "
        "from core.chat_memory import load_conversation,get_prompt_memory; "
        "c=load_conversation(os.environ['EVAL_CONVERSATION_ID']); "
        "m=get_prompt_memory(os.environ['EVAL_SECOND_ID'],'早期稀有消融结论是什么？'); "
        "print(json.dumps({'messages':len(c['messages']),'topics':len(m.recalled_topics),"
        "'found':any('4.2%' in x['content'] for x in m.recalled_topics)}))"
    )
    env = os.environ.copy()
    env.update(
        {
            "PAPER_READER_DATA_DIR": data_dir,
            "EVAL_CONVERSATION_ID": first["id"],
            "EVAL_SECOND_ID": second["id"],
        }
    )
    restarted = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "messages": len(load_conversation(first["id"])["messages"]),
        "cross_conversation_recalled": any("4.2%" in item["content"] for item in prompt_memory.recalled_topics),
        "recall_latency_s": round(recall_latency, 4),
        "restart": json.loads(restarted.stdout.strip()),
    }


def _evaluate_answer_adherence() -> dict[str, Any]:
    history_id = _paper("LangMem Answer A/B")
    conversation = create_conversation(history_id, title="Answer A/B")
    _put(
        history_id,
        "answer-format",
        PaperReaderMemory(
            category="feedback",
            subject="实验结果回答格式偏好",
            content="回答实验结果时必须以‘结论先行：’开头，并且只写一个句子。",
            context="用户希望快速阅读结论。",
        ),
    )
    evidence = EvidenceSnippet(
        "E001",
        "Experiments",
        0,
        0,
        "The method improves exact-match accuracy from 80% to 85% on the evaluation set.",
    )
    context = {
        "mode": "live",
        "paper": {"title": "Synthetic Memory Evaluation Paper"},
        "summary_output": {"one_sentence_summary": "The method improves exact-match accuracy."},
    }
    analysis_id = store_analysis_session([evidence], context)
    common = {"question": "实验结果说明了什么？", "analysis_id": analysis_id, "context": context}
    memory_request = PaperChatRequest(
        **common,
        history_id=history_id,
        conversation_id=conversation["id"],
    )
    baseline_request = PaperChatRequest(**common)
    memory_prompt = build_chat_prompt(memory_request)
    baseline_prompt = build_chat_prompt(baseline_request)

    def run(request: PaperChatRequest, prompt: Any) -> tuple[str, float]:
        started = time.perf_counter()
        answer = "".join(stream_chat_reply(request, messages=prompt.messages)).strip()
        return answer, time.perf_counter() - started

    memory_rows = [run(memory_request, memory_prompt) for _ in range(3)]
    baseline_rows = [run(baseline_request, baseline_prompt) for _ in range(3)]
    return {
        "trials": 3,
        "memory_prefix_adherence": sum(answer.startswith("结论先行：") for answer, _ in memory_rows) / 3,
        "baseline_prefix_adherence": sum(answer.startswith("结论先行：") for answer, _ in baseline_rows) / 3,
        "memory_answer_latency_median_s": round(statistics.median(latency for _, latency in memory_rows), 3),
        "baseline_answer_latency_median_s": round(statistics.median(latency for _, latency in baseline_rows), 3),
        "recalled_topics": memory_prompt.stats.recalled_topics,
        "memory_answer_previews": [answer[:160] for answer, _ in memory_rows],
    }


def main() -> None:
    prior_data_dir = os.environ.get("PAPER_READER_DATA_DIR")
    prior_disable = os.environ.get("PAPER_READER_DISABLE_EMBEDDINGS")
    with tempfile.TemporaryDirectory(prefix="paper-reader-langmem-eval-") as tmp:
        os.environ["PAPER_READER_DATA_DIR"] = tmp
        os.environ.pop("PAPER_READER_DISABLE_EMBEDDINGS", None)
        reset_langmem_store()
        output = {
            "framework": "langmem",
            "provider": text_provider_id(),
            "model": selected_text_model(),
            "management": _evaluate_management(),
            "recall": _evaluate_recall(),
            "restart_and_scale": _evaluate_restart_and_scale(tmp),
            "answer_adherence_ab": _evaluate_answer_adherence(),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        reset_langmem_store()
    if prior_data_dir is None:
        os.environ.pop("PAPER_READER_DATA_DIR", None)
    else:
        os.environ["PAPER_READER_DATA_DIR"] = prior_data_dir
    if prior_disable is None:
        os.environ.pop("PAPER_READER_DISABLE_EMBEDDINGS", None)
    else:
        os.environ["PAPER_READER_DISABLE_EMBEDDINGS"] = prior_disable


if __name__ == "__main__":
    main()
