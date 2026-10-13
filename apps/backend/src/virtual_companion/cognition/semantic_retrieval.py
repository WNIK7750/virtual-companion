"""Bounded semantic ranking over an authoritative committed-memory catalog.

The embedder is injected so the core package does not import or configure a
provider. The current fixed FastEmbed adapter contract is ``query_embed(str)``
and ``embed(list[str])``; eligibility and returned text always come back from
the store, never from this cache.
"""

import heapq
import json
import math
from collections import OrderedDict
from hashlib import sha256

from virtual_companion.boundary.contracts import VersionConflict


class SemanticMemoryRetriever:
    """Rank current committed facts by cosine and recheck exact versions.

    ``candidate_overfetch`` bounds the set sent to the final authoritative
    lookup independently from the number admitted to short context. Document
    vectors are a process-local, bounded cache keyed by every identity and
    transformation input that can change their meaning.
    """

    def __init__(self, embedder, *, model_id, model_revision,
                 model_manifest_sha256=None, synthetic_model_identity=False,
                 preprocessing_revision="raw-content.v1", page_size=100,
                 candidate_overfetch=32, max_cached_vectors=4096):
        if (not callable(getattr(embedder, "query_embed", None))
                or not callable(getattr(embedder, "embed", None))):
            raise TypeError("FastEmbed-compatible query and document embedder required")
        encoder_manifest_sha256 = getattr(embedder, "model_manifest_sha256", None)
        if encoder_manifest_sha256 is not None:
            if (model_manifest_sha256 is not None
                    and model_manifest_sha256 != encoder_manifest_sha256):
                raise ValueError("Retriever manifest identity differs from verified encoder")
            model_manifest_sha256 = encoder_manifest_sha256
        for value, label in ((model_id, "model ID"), (model_revision, "model revision"),
                             (preprocessing_revision, "preprocessing revision")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Non-empty {label} required")
        if model_manifest_sha256 is None:
            if synthetic_model_identity is not True:
                raise ValueError(
                    "Verified model manifest SHA-256 or explicit synthetic identity required"
                )
        elif (not isinstance(model_manifest_sha256, str)
              or len(model_manifest_sha256) != 64
              or any(character not in "0123456789abcdef" for character in model_manifest_sha256)):
            raise ValueError("Model manifest identity must be lowercase SHA-256")
        elif synthetic_model_identity is True:
            raise ValueError("Synthetic model identity cannot claim a verified manifest")
        if type(synthetic_model_identity) is not bool:
            raise ValueError("Synthetic model identity flag must be boolean")
        embedder_identity = getattr(embedder, "identity", None)
        if embedder_identity is None:
            if synthetic_model_identity is not True:
                raise ValueError("Verified semantic embedder identity required")
            embedder_identity = {"kind": "synthetic-test-embedder.v1"}
        if not isinstance(embedder_identity, dict):
            raise ValueError("Semantic embedder identity must be a mapping")
        try:
            identity_json = json.dumps(
                embedder_identity, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            json.loads(identity_json)
        except (TypeError, ValueError) as error:
            raise ValueError("Semantic embedder identity must contain JSON primitives") from error
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("Catalog page size must be between 1 and 100")
        if (type(candidate_overfetch) is not int
                or not 1 <= candidate_overfetch <= 100):
            raise ValueError("Semantic candidate overfetch must be between 1 and 100")
        if type(max_cached_vectors) is not int or max_cached_vectors < 0:
            raise ValueError("Non-negative semantic vector cache size required")
        self.embedder = embedder
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_manifest_sha256 = model_manifest_sha256
        self.synthetic_model_identity = synthetic_model_identity
        self._embedder_identity_json = identity_json
        self._embedder_fingerprint = sha256(identity_json.encode("utf-8")).hexdigest()
        self.preprocessing_revision = preprocessing_revision
        self.page_size = page_size
        self.candidate_overfetch = candidate_overfetch
        self.max_cached_vectors = max_cached_vectors
        self._document_vectors = OrderedDict()

    @property
    def identity(self):
        """Stable retrieval resource identity; contains no host path or object repr."""
        return {
            "schema": "semantic-memory-retriever.v1",
            "algorithm": "cosine.v1",
            "embedding_backend": json.loads(self._embedder_identity_json),
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "manifest_sha256": self.model_manifest_sha256,
                "identity_kind": "synthetic" if self.synthetic_model_identity else "verified_local",
            },
            "preprocessing_revision": self.preprocessing_revision,
            "catalog_page_size": self.page_size,
            "candidate_overfetch": self.candidate_overfetch,
            "max_cached_vectors": self.max_cached_vectors,
        }

    @property
    def fingerprint(self):
        """Canonical digest suitable for binding queued work to runtime resources."""
        encoded = json.dumps(self.identity, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        return sha256(encoded).hexdigest()

    def recall(self, store, companion, *, now_us, query, limit=8, checkpoint=None):
        """Return up to ``limit`` current fact rows for one bounded query."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Semantic context limit must be between 1 and 100")
        if limit > self.candidate_overfetch:
            raise ValueError("Semantic candidate overfetch must cover context limit")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Semantic recall needs the current non-empty query")
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("Semantic retrieval checkpoint must be callable")

        def check_progress():
            if checkpoint is not None:
                checkpoint()

        page_reader = getattr(store, "committed_catalog_page", None)
        final_reader = getattr(store, "committed_recall", None)
        if not callable(page_reader) or not callable(final_reader):
            raise TypeError("Semantic recall requires the committed catalog and recheck ports")

        query_vector = None
        page_cursor = None
        ordinal = 0
        top = []
        pending = OrderedDict()
        while True:
            check_progress()
            page = page_reader(companion, now_us=now_us,
                               after_record_id=page_cursor, limit=self.page_size)
            check_progress()
            if not isinstance(page, dict):
                raise TypeError("Committed catalog page must be a mapping")
            records = page.get("records")
            next_cursor = page.get("next_after_record_id")
            if not isinstance(records, (tuple, list)):
                raise TypeError("Committed catalog records must be a sequence")
            if not records:
                if next_cursor is not None:
                    raise ValueError("Empty catalog page cannot advance its cursor")
                break

            if query_vector is None:
                check_progress()
                query_vector = self._single_vector(self.embedder.query_embed(query),
                                                   "query embedding")
                check_progress()
            vectors = self._vectors_for_page(
                companion, records, pending=pending, checkpoint=check_progress)
            for record, vector in zip(records, vectors, strict=True):
                score = self._cosine(query_vector, vector)
                identity_order = ordinal
                ordinal += 1
                entry = (score, -identity_order, record)
                if len(top) < self.candidate_overfetch:
                    heapq.heappush(top, entry)
                elif entry[:2] > top[0][:2]:
                    heapq.heapreplace(top, entry)

            if next_cursor is None:
                break
            if (not isinstance(next_cursor, str)
                    or next_cursor != records[-1]["record_id"]
                    or (page_cursor is not None and next_cursor <= page_cursor)):
                raise ValueError("Committed catalog cursor did not advance")
            page_cursor = next_cursor

        if not top:
            return []
        ranked = sorted(top, key=lambda entry: (-entry[0], -entry[1]))
        candidates = [entry[2] for entry in ranked]
        pairs = tuple((item["record_id"], item["version"]) for item in candidates)
        check_progress()
        qualified = final_reader(companion, now_us=now_us, candidates=pairs,
                                 limit=self.candidate_overfetch)
        if len(qualified) != len(candidates):
            raise VersionConflict("Semantic memory candidates changed before admission")
        for expected, actual in zip(candidates, qualified, strict=True):
            if ((actual.get("record_id"), actual.get("version"), actual.get("content"))
                    != (expected["record_id"], expected["version"], expected["content"])):
                raise VersionConflict("Semantic memory original changed before admission")
        # Publish derived misses only after this call has finished scanning and
        # passed the authoritative version/content recheck.
        for key, vector in pending.items():
            self._document_vectors[key] = vector
            self._document_vectors.move_to_end(key)
            while len(self._document_vectors) > self.max_cached_vectors:
                self._document_vectors.popitem(last=False)
        return qualified[:limit]

    def _vectors_for_page(self, companion, records, *, pending, checkpoint=None):
        candidates = []
        keys = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("Committed memory candidate must be a mapping")
            record_id, version, content = (record.get("record_id"), record.get("version"),
                                           record.get("content"))
            if (not isinstance(record_id, str) or type(version) is not int or version < 1
                    or not isinstance(content, str) or not content):
                raise ValueError("Committed memory candidate identity and original text required")
            digest = sha256(content.encode("utf-8")).hexdigest()
            key = (self.model_id, self.model_revision, self.preprocessing_revision,
                   self.model_manifest_sha256, self._embedder_fingerprint,
                   companion, record_id, version, digest)
            candidates.append(record)
            keys.append(key)

        result = [None] * len(candidates)
        missing = []
        for index, key in enumerate(keys):
            cached = self._document_vectors.get(key)
            if cached is None:
                missing.append(index)
            else:
                self._document_vectors.move_to_end(key)
                result[index] = cached
        if missing:
            texts = [candidates[index]["content"] for index in missing]
            if checkpoint is not None:
                checkpoint()
            vectors = list(self.embedder.embed(texts))
            if checkpoint is not None:
                checkpoint()
            if len(vectors) != len(missing):
                raise ValueError("Document embedding count differs from committed candidates")
            for index, vector in zip(missing, vectors, strict=True):
                vector = self._as_vector(vector, "document embedding")
                result[index] = vector
                if self.max_cached_vectors:
                    key = keys[index]
                    if key not in pending and len(pending) >= self.max_cached_vectors:
                        pending.popitem(last=False)
                    pending[key] = vector
                    pending.move_to_end(key)
        return result

    @classmethod
    def _single_vector(cls, vectors, label):
        rows = list(vectors)
        if len(rows) != 1:
            raise ValueError(f"{label} must return exactly one vector")
        return cls._as_vector(rows[0], label)

    @staticmethod
    def _as_vector(value, label):
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{label} must be a finite numeric vector")
        vector = tuple(float(component) for component in value)
        if not vector or not all(math.isfinite(component) for component in vector):
            raise ValueError(f"{label} must be a finite non-empty vector")
        return vector

    @staticmethod
    def _cosine(left, right):
        if len(left) != len(right):
            raise ValueError("Query and memory embeddings have different dimensions")
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            raise ValueError("Cannot rank a zero-norm memory embedding")
        return dot / (left_norm * right_norm)
