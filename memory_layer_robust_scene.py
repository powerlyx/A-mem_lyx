"""
Robust A-MEM memory layer — drop-in replacement for memory_layer.py.

Key differences from the original:
  - No response_format / JSON schema dependency in LLM calls
  - Plain-text prompts with section-marker parsing (via llm_text_parsers)
  - Structured logging instead of print()
  - Retry wrapper for transient LLM failures
  - Connectivity check on controller init
  - Graceful degradation: evolution failure -> memory stored without evolution
"""

from typing import List, Dict, Optional, Literal, Any, Tuple
import json
import re
import uuid
import os
import time
import logging
import functools
from collections import defaultdict
from datetime import datetime
from abc import ABC, abstractmethod

from memory_layer import SimpleEmbeddingRetriever, simple_tokenize
from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
)

logger = logging.getLogger("amem_robust")

TRACE_EVENT_FILTER_PROMPT = """You are a narrative coherence analyzer.
Event Chain A is an existing chain and Event List B are new events.
Return strict JSON with keys: related_events, unrelated_events.

Event Chain A:
{content_a}

Event List B:
{content_b}
"""

TRACE_INIT_PROMPT = """You are an event chain constructor.
Given events, return strict JSON with keys: primary_chain, secondary_chains, isolated_events.

Events:
{events}
"""

EVENT_EXTRACT_PROMPT = """Extract structured metadata for memory construction.
Return strict JSON with keys: topic, explicit_mentions.

Content:
{text}
"""

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Robust LLM Controllers — no response_format parameter
# ---------------------------------------------------------------------------

class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion("Reply with exactly one word: READY", temperature=0.0)
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.model = model
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2"):
        self.model = model

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(self, model: str = "llama2",
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 1000,
            }
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(self, model: str = "llama2",
                 vllm_host: str = "http://localhost",
                 vllm_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1000,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")


class RobustLiteLLMController(RobustBaseLLMController):
    """LiteLLM controller for universal LLM access (Ollama, SGLang, etc.)."""

    def __init__(self, model: str, api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        from litellm import completion as _completion
        self._completion = _completion
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key

        response = self._completion(**completion_args)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang", "vllm"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False):
        if backend == "openai":
            self.llm = RobustOpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = RobustOllamaController(model)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', 'sglang', or 'vllm'")

        if check_connection:
            self.llm.check_connectivity()


# ---------------------------------------------------------------------------
# RobustMemoryNote
# ---------------------------------------------------------------------------

class RobustMemoryNote:
    """Memory note that uses plain-text LLM calls for metadata extraction."""

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 topic: Optional[str] = None,
                 events: Optional[List[str]] = None,
                 llm_controller: Optional[RobustLLMController] = None):

        self.content = content

        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []
        self.topic = topic or ""
        self.events = events or []
        self.events_text = " | ".join(self.events)

    @staticmethod
    def build_scene_content(turns: List[str], scene_title: Optional[str] = None) -> str:
        """Build one scene-level memory content from multiple dialogue turns.

        Minimal memory-construction change:
        - input: recent multi-turn utterances
        - output: one concatenated scene paragraph for a single memory note
        """
        clean_turns = [t.strip() for t in turns if t and t.strip()]
        if not clean_turns:
            return ""

        prefix = f"[Scene] {scene_title}\n" if scene_title else "[Scene]\n"
        return prefix + "\n".join(clean_turns)

    @staticmethod
    def analyze_content(content: str, llm_controller: RobustLLMController) -> Dict:
        """Analyze content using plain-text prompt + section-marker parsing."""
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                from llm_text_parsers import _parse_list_items
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            from llm_text_parsers import _heuristic_keywords, _heuristic_context
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }


# ---------------------------------------------------------------------------
# RobustAgenticMemorySystem
# ---------------------------------------------------------------------------

class RobustAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 trace_similarity_threshold: float = 0.5):

        self.memories: Dict[str, RobustMemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, check_connection,
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        # Membox-like global topic consistency structure
        self.trace_similarity_threshold = trace_similarity_threshold
        self.traces: List[Dict[str, Any]] = []
        self.trace_note_map: Dict[str, List[int]] = defaultdict(list)

    # ---- public API (mirrors AgenticMemorySystem) ----

    def add_scene_note(self,
                       turns: List[str],
                       time: str = None,
                       scene_title: Optional[str] = None,
                       **kwargs) -> str:
        """Add one memory note constructed from multiple turns (scene-level note)."""
        scene_content = RobustMemoryNote.build_scene_content(turns, scene_title=scene_title)
        return self.add_note(scene_content, time=time, **kwargs)

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note."""
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )
        note.topic, note.events = self._extract_topic_events(note.content)
        note.events_text = " | ".join(note.events)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([
            "content:" + note.content +
            " context:" + note.context +
            " keywords: " + ", ".join(note.keywords) +
            " tags: " + ", ".join(note.tags) +
            " topic: " + note.topic +
            " events: " + note.events_text
        ])
        self._update_traces_with_note(note)
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def _extract_topic_events(self, content: str) -> Tuple[str, List[str]]:
        """Extract topic and explicit events for trace linking."""
        try:
            prompt = EVENT_EXTRACT_PROMPT.format(text=content)
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.0)
            parsed = json.loads(response)
            topic = str(parsed.get("topic", "")).strip()
            events = parsed.get("explicit_mentions", [])
            if not isinstance(events, list):
                events = []
            events = [str(e).strip() for e in events if str(e).strip()]
            if not topic:
                topic = " ".join(content.split()[:12]).strip()
            return topic, events
        except Exception:
            # fallback: no extra LLM dependency for robustness
            topic = " ".join(content.split()[:12]).strip()
            return topic, []

    def _event_similarity(self, event_a: str, event_b: str) -> float:
        try:
            emb = self.retriever.model.encode([event_a, event_b])
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
            return float(sim)
        except Exception:
            return 0.0

    def _trace_event_lines(self, trace: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for entry in trace.get("entries", []):
            ts = str(entry.get("start_time", "Unknown"))
            for ev in entry.get("events", []):
                if ev:
                    lines.append(f"{ts}: {ev}")
        return lines

    def _llm_event_filter(self, trace: Dict[str, Any], events: List[str]) -> Tuple[set, set]:
        chain_text = "\n".join(self._trace_event_lines(trace)) or "None"
        events_text = "\n".join(events) or "None"
        prompt = TRACE_EVENT_FILTER_PROMPT.format(content_a=chain_text, content_b=events_text)
        try:
            res = self.llm_controller.llm.get_completion(prompt, temperature=0.0)
            d = json.loads(res)
            related = set([str(e).strip() for e in (d.get("related_events") or []) if str(e).strip()])
            unrelated = set([str(e).strip() for e in (d.get("unrelated_events") or []) if str(e).strip()])
            if not related and not unrelated:
                return set(events), set()
            return related, unrelated
        except Exception:
            return set(events), set()

    def _llm_init_chain(self, events: List[str]) -> Dict[str, Any]:
        prompt = TRACE_INIT_PROMPT.format(events="\n".join(events))
        try:
            res = self.llm_controller.llm.get_completion(prompt, temperature=0.0)
            return json.loads(res)
        except Exception:
            return {"primary_chain": events, "secondary_chains": [], "isolated_events": []}

    def _append_trace(self, note: RobustMemoryNote, events: List[str]):
        events_clean = [e for e in events if e]
        if not events_clean:
            return
        trace_id = len(self.traces)
        trace = {
            "trace_id": trace_id,
            "note_ids": [note.id],
            "entries": [{"note_id": note.id, "start_time": note.timestamp, "events": events_clean}],
        }
        self.traces.append(trace)
        self.trace_note_map[note.id].append(trace_id)

    def _update_traces_with_note(self, note: RobustMemoryNote):
        """Membox-like two-stage trace update: similarity candidates + LLM filter."""
        events = [e for e in (note.events or []) if e]
        if not events:
            return

        selected_trace_ids = set()
        for ev in events:
            best_trace_id = None
            best_score = -1.0
            for tr in self.traces:
                trace_best = -1.0
                for entry in tr.get("entries", []):
                    for tev in entry.get("events", []):
                        score = self._event_similarity(ev, tev)
                        if score > trace_best:
                            trace_best = score
                if trace_best > best_score:
                    best_score = trace_best
                    best_trace_id = tr["trace_id"]
            if best_trace_id is not None and best_score >= self.trace_similarity_threshold:
                selected_trace_ids.add(best_trace_id)

        matched_events = set()
        trace_lookup = {t["trace_id"]: t for t in self.traces}
        for tr_id in selected_trace_ids:
            tr = trace_lookup.get(tr_id)
            if not tr:
                continue
            related, _ = self._llm_event_filter(tr, events)
            if related:
                tr["entries"].append({"note_id": note.id, "start_time": note.timestamp, "events": list(related)})
                if note.id not in tr["note_ids"]:
                    tr["note_ids"].append(note.id)
                matched_events.update(related)
                self.trace_note_map[note.id].append(tr_id)

        unmatched = [e for e in events if e not in matched_events]
        if not unmatched:
            return

        if len(unmatched) == 1:
            self._append_trace(note, unmatched)
            return

        init_res = self._llm_init_chain(unmatched)
        chains = []
        primary = init_res.get("primary_chain") or []
        secondary = init_res.get("secondary_chains") or []
        isolated = init_res.get("isolated_events") or []
        if primary:
            chains.append(primary)
        for ch in secondary:
            if ch:
                chains.append(ch)
        if not chains and isolated:
            chains.append(isolated)
        if not chains:
            chains = [unmatched]
        for ch in chains:
            self._append_trace(note, [str(e).strip() for e in ch if str(e).strip()])

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)
        for memory in self.memories.values():
            metadata_text = (
                f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)} "
                f"{getattr(memory, 'topic', '')} {getattr(memory, 'events_text', '')}"
            )
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        all_memory_ids = list(self.memories.keys())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        all_memory_ids = list(self.memories.keys())
        memory_str = ""
        seen_trace_note_ids = set()
        for i in indices:
            j = 0
            note_id = all_memory_ids[i]
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) +
                "memory topic: " + str(all_memories[i].topic) +
                "memory events: " + str(all_memories[i].events) + "\n"
            )
            # global topic consistency: include trace neighbors
            for tr_id in self.trace_note_map.get(note_id, []):
                if tr_id >= len(self.traces):
                    continue
                trace = self.traces[tr_id]
                for trace_note_id in trace.get("note_ids", []):
                    if (
                        trace_note_id == note_id
                        or trace_note_id not in self.memories
                        or trace_note_id in seen_trace_note_ids
                    ):
                        continue
                    seen_trace_note_ids.add(trace_note_id)
                    tnote = self.memories[trace_note_id]
                    memory_str += (
                        "[trace-related] talk start time:" + tnote.timestamp +
                        "memory content: " + tnote.content +
                        "memory topic: " + str(tnote.topic) +
                        "memory events: " + str(tnote.events) + "\n"
                    )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    # ---- evolution (3 sequential plain-text calls) ----

    def process_memory(self, note: RobustMemoryNote) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)

        if len(indices) == 0:
            return False, note

        try:
            # ---- Call 1: Evolution decision ----
            decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            decision_response = self.llm_controller.llm.get_completion(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            # ---- Call 2: Strengthen details (conditional) ----
            if should_strengthen:
                strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
                strengthen = parse_strengthen_details(strengthen_response)
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            # ---- Call 3: Update neighbors (conditional) ----
            if should_update:
                update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                    content=note.content,
                    context=note.context,
                    nearest_neighbors_memories=neighbor_memory,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                update_response = self.llm_controller.llm.get_completion(update_prompt)
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note

        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note
