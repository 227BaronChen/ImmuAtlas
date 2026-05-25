"""
SQLite-based prediction cache with sharding and quantization for ImmuAtlas.

Provides the ``ELScoreCache`` class that transparently caches EL prediction
scores to disk, supporting:

* **uint16 / float32 quantization** – compact storage with configurable precision
* **Auto-sharding** – adaptive number of SQLite shards based on data size
* **Namespace management** – multiple model / HLA-set combinations in one DB
* **Vector schema** – one BLOB per peptide × namespace (efficient bulk I/O)
* **WAL mode** – concurrent read/write safety
"""

import csv
import hashlib
import os
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


FIXED_SCALE = 65535
DEFAULT_MAX_SHARDS = 256
DEFAULT_MIN_AUTO_SHARDS = 1
DEFAULT_AUTO_TARGET_SHARD_MB = 1024
DEFAULT_LOOKUP_STEP = 1000
DEFAULT_HIT_STEP = 1000
DEFAULT_PEPTIDE_MAP_STEP = 1000
DEFAULT_IMPORT_BATCH = 10000


def canonical_hla_set_hash(hla_alleles: Sequence[str]) -> str:
    canonical = "|".join(sorted(set(hla_alleles)))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def default_model_id(model_path: str) -> str:
    base = os.path.basename(model_path)
    return os.path.splitext(base)[0]


def _canonical_hla_order(hla_alleles: Sequence[str]) -> List[str]:
    return sorted(set(hla_alleles))


class _SingleShardCache:
    def __init__(self, db_path: str, quant_mode: str = "uint16"):
        self.db_path = db_path
        self.quant_mode = quant_mode
        if self.quant_mode not in {"uint16", "float32"}:
            raise ValueError(f"Unsupported quant_mode: {self.quant_mode}")

        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS peptide (
                peptide_id INTEGER PRIMARY KEY AUTOINCREMENT,
                peptide_seq TEXT NOT NULL UNIQUE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_peptide_seq ON peptide(peptide_seq)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
            """
        )
        self._ensure_meta(cur)

        if not self._table_exists("namespace"):
            self._create_namespace_table(cur)
        else:
            self._ensure_namespace_columns(cur)

        if not self._table_exists("score_el"):
            self._create_vector_table(cur, "score_el")
            self.conn.commit()
            return

        if self._is_vector_schema("score_el"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_score_lookup ON score_el(namespace_id, peptide_id)")
            self.conn.commit()
            return

        if self._is_row_schema("score_el"):
            self._migrate_row_schema_to_vector()
            return

        raise RuntimeError("Unsupported score_el schema; please check database consistency.")

    def _ensure_meta(self, cur: sqlite3.Cursor) -> None:
        row = cur.execute("SELECT v FROM cache_meta WHERE k='quant_mode'").fetchone()
        if row is None:
            cur.execute("INSERT INTO cache_meta(k, v) VALUES('quant_mode', ?)", (self.quant_mode,))
            return

        existing = str(row[0])
        if existing != self.quant_mode:
            raise ValueError(
                f"Cache quantization mismatch: DB uses {existing}, requested {self.quant_mode}."
            )

    def _table_exists(self, table_name: str) -> bool:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _get_table_columns(self, table_name: str) -> set:
        cur = self.conn.cursor()
        return {row[1] for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()}

    def _is_vector_schema(self, table_name: str) -> bool:
        cols = self._get_table_columns(table_name)
        return {"peptide_id", "namespace_id", "score_blob"}.issubset(cols)

    def _is_row_schema(self, table_name: str) -> bool:
        cols = self._get_table_columns(table_name)
        if {"peptide_id", "hla", "model_id", "hla_set_hash", "score_el"}.issubset(cols):
            return True
        if {"peptide_id", "hla_id", "namespace_id", "score_el"}.issubset(cols):
            return True
        return False

    def _create_namespace_table(self, cur: sqlite3.Cursor) -> None:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS namespace (
                namespace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                hla_set_hash TEXT NOT NULL,
                hla_list TEXT NOT NULL DEFAULT '',
                hla_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(model_id, hla_set_hash)
            )
            """
        )

    def _ensure_namespace_columns(self, cur: sqlite3.Cursor) -> None:
        cols = {row[1] for row in cur.execute("PRAGMA table_info(namespace)").fetchall()}
        if "hla_list" not in cols:
            cur.execute("ALTER TABLE namespace ADD COLUMN hla_list TEXT NOT NULL DEFAULT ''")
        if "hla_count" not in cols:
            cur.execute("ALTER TABLE namespace ADD COLUMN hla_count INTEGER NOT NULL DEFAULT 0")

    def _create_vector_table(self, cur: sqlite3.Cursor, table_name: str) -> None:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                peptide_id INTEGER NOT NULL,
                namespace_id INTEGER NOT NULL,
                score_blob BLOB NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (peptide_id, namespace_id),
                FOREIGN KEY (peptide_id) REFERENCES peptide(peptide_id),
                FOREIGN KEY (namespace_id) REFERENCES namespace(namespace_id)
            ) WITHOUT ROWID
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_score_lookup ON {table_name}(namespace_id, peptide_id)")

    def _encode_values(self, values: Sequence[float]) -> bytes:
        if self.quant_mode == "uint16":
            buf = bytearray(len(values) * 2)
            for i, v in enumerate(values):
                fv = float(v)
                if fv < 0.0:
                    fv = 0.0
                elif fv > 1.0:
                    fv = 1.0
                q = int(round(fv * FIXED_SCALE))
                if q < 0:
                    q = 0
                elif q > FIXED_SCALE:
                    q = FIXED_SCALE
                offset = i * 2
                buf[offset] = q & 0xFF
                buf[offset + 1] = (q >> 8) & 0xFF
            return bytes(buf)

        out = bytearray(len(values) * 4)
        for i, v in enumerate(values):
            struct.pack_into("<f", out, i * 4, float(v))
        return bytes(out)

    def _decode_values(self, blob: bytes, expected_count: int) -> List[float]:
        if self.quant_mode == "uint16":
            if len(blob) != expected_count * 2:
                return []
            out: List[float] = []
            for i in range(expected_count):
                offset = i * 2
                q = blob[offset] | (blob[offset + 1] << 8)
                out.append(q / FIXED_SCALE)
            return out

        if len(blob) != expected_count * 4:
            return []
        return [float(struct.unpack_from("<f", blob, i * 4)[0]) for i in range(expected_count)]

    def _get_namespace_row(self, model_id: str, hla_set_hash: str) -> Optional[Tuple[int, str, int]]:
        cur = self.conn.cursor()
        row = cur.execute(
            """
            SELECT namespace_id, hla_list, hla_count
            FROM namespace
            WHERE model_id = ? AND hla_set_hash = ?
            """,
            (model_id, hla_set_hash),
        ).fetchone()
        if row is None:
            return None
        return int(row[0]), str(row[1] or ""), int(row[2] or 0)

    def _ensure_namespace(self, model_id: str, hla_set_hash: str, hla_alleles: Sequence[str]) -> Tuple[int, List[str]]:
        namespace_hlas = _canonical_hla_order(hla_alleles)
        hla_list_str = "|".join(namespace_hlas)
        hla_count = len(namespace_hlas)
        cur = self.conn.cursor()

        cur.execute(
            """
            INSERT OR IGNORE INTO namespace(model_id, hla_set_hash, hla_list, hla_count)
            VALUES (?, ?, ?, ?)
            """,
            (model_id, hla_set_hash, hla_list_str, hla_count),
        )

        row = self._get_namespace_row(model_id, hla_set_hash)
        if row is None:
            raise RuntimeError("Failed to resolve namespace")

        namespace_id, existing_hla_list, existing_hla_count = row
        if existing_hla_count == 0 or not existing_hla_list:
            cur.execute(
                "UPDATE namespace SET hla_list = ?, hla_count = ? WHERE namespace_id = ?",
                (hla_list_str, hla_count, namespace_id),
            )
            return namespace_id, namespace_hlas

        existing_hlas = existing_hla_list.split("|") if existing_hla_list else []
        if existing_hlas != namespace_hlas:
            raise ValueError(
                f"Namespace HLA set/order mismatch for model={model_id}, hash={hla_set_hash}. "
                f"Expected {existing_hlas}, got {namespace_hlas}."
            )
        return namespace_id, existing_hlas

    def _find_namespace(self, model_id: str, hla_set_hash: str) -> Optional[Tuple[int, List[str]]]:
        row = self._get_namespace_row(model_id, hla_set_hash)
        if row is None:
            return None
        namespace_id, hla_list_str, hla_count = row
        if not hla_list_str or hla_count <= 0:
            return None
        hla_list = hla_list_str.split("|")
        if len(hla_list) != hla_count:
            return None
        return namespace_id, hla_list

    def _migrate_row_schema_to_vector(self) -> None:
        cur = self.conn.cursor()
        row_cols = self._get_table_columns("score_el")
        is_legacy = {"hla", "model_id", "hla_set_hash"}.issubset(row_cols)
        is_compact = {"hla_id", "namespace_id"}.issubset(row_cols)

        cur.execute("BEGIN")
        try:
            self._create_namespace_table(cur)
            self._ensure_namespace_columns(cur)
            self._create_vector_table(cur, "score_el_new")

            if is_legacy:
                self._migrate_legacy_rows(cur)
            elif is_compact:
                self._migrate_compact_rows(cur)
            else:
                raise RuntimeError("Cannot migrate: unrecognized row schema.")

            cur.execute("DROP TABLE score_el")
            cur.execute("ALTER TABLE score_el_new RENAME TO score_el")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_score_lookup ON score_el(namespace_id, peptide_id)")
            self.conn.commit()
        except sqlite3.Error:
            self.conn.rollback()
            raise

        self.compact()

    def _migrate_legacy_rows(self, cur: sqlite3.Cursor) -> None:
        cur.execute(
            """
            INSERT OR IGNORE INTO namespace(model_id, hla_set_hash)
            SELECT DISTINCT model_id, hla_set_hash
            FROM score_el
            """
        )

        ns_rows = cur.execute("SELECT namespace_id, model_id, hla_set_hash FROM namespace").fetchall()
        for namespace_id, model_id, hla_set_hash in ns_rows:
            hla_rows = cur.execute(
                """
                SELECT DISTINCT hla
                FROM score_el
                WHERE model_id = ? AND hla_set_hash = ?
                ORDER BY hla
                """,
                (model_id, hla_set_hash),
            ).fetchall()
            namespace_hlas = [str(r[0]) for r in hla_rows]
            hla_idx = {h: i for i, h in enumerate(namespace_hlas)}
            cur.execute(
                "UPDATE namespace SET hla_list = ?, hla_count = ? WHERE namespace_id = ?",
                ("|".join(namespace_hlas), len(namespace_hlas), namespace_id),
            )

            rows = cur.execute(
                """
                SELECT peptide_id, hla, score_el, updated_at
                FROM score_el
                WHERE model_id = ? AND hla_set_hash = ?
                ORDER BY peptide_id
                """,
                (model_id, hla_set_hash),
            )
            self._pack_rows_into_vector(rows, namespace_id, hla_idx, len(namespace_hlas), is_hla_text=True)

    def _migrate_compact_rows(self, cur: sqlite3.Cursor) -> None:
        if not self._table_exists("hla_dict"):
            raise RuntimeError("Compact row schema detected but hla_dict table is missing.")

        hla_id_map = {int(hla_id): str(hla) for hla_id, hla in cur.execute("SELECT hla_id, hla FROM hla_dict").fetchall()}
        ns_rows = cur.execute("SELECT namespace_id, hla_list, hla_count FROM namespace").fetchall()

        for namespace_id, hla_list, hla_count in ns_rows:
            namespace_id = int(namespace_id)
            hla_list = str(hla_list or "")
            hla_count = int(hla_count or 0)

            if not hla_list or hla_count <= 0:
                hla_rows = cur.execute(
                    """
                    SELECT DISTINCT h.hla
                    FROM score_el s
                    JOIN hla_dict h ON h.hla_id = s.hla_id
                    WHERE s.namespace_id = ?
                    ORDER BY h.hla
                    """,
                    (namespace_id,),
                ).fetchall()
                namespace_hlas = [str(r[0]) for r in hla_rows]
                cur.execute(
                    "UPDATE namespace SET hla_list = ?, hla_count = ? WHERE namespace_id = ?",
                    ("|".join(namespace_hlas), len(namespace_hlas), namespace_id),
                )
            else:
                namespace_hlas = hla_list.split("|")

            hla_idx = {h: i for i, h in enumerate(namespace_hlas)}
            rows = cur.execute(
                """
                SELECT peptide_id, hla_id, score_el, updated_at
                FROM score_el
                WHERE namespace_id = ?
                ORDER BY peptide_id
                """,
                (namespace_id,),
            )
            self._pack_rows_into_vector(
                rows,
                namespace_id,
                hla_idx,
                len(namespace_hlas),
                is_hla_text=False,
                hla_id_map=hla_id_map,
            )

    def _pack_rows_into_vector(
        self,
        rows: sqlite3.Cursor,
        namespace_id: int,
        hla_idx: Dict[str, int],
        hla_count: int,
        is_hla_text: bool,
        hla_id_map: Optional[Dict[int, str]] = None,
    ) -> None:
        write_cur = self.conn.cursor()
        current_pid: Optional[int] = None
        vec: Optional[List[float]] = None
        seen: Optional[List[bool]] = None
        current_updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        batch: List[Tuple[int, int, bytes, str]] = []

        def flush_current() -> None:
            if current_pid is None or vec is None or seen is None:
                return
            if not all(seen):
                return
            blob = self._encode_values(vec)
            batch.append((current_pid, namespace_id, blob, current_updated_at))

        for row in rows:
            peptide_id = int(row[0])
            hla_key_raw = row[1]
            score = float(row[2])
            updated_at = str(row[3] or current_updated_at)

            if peptide_id != current_pid:
                flush_current()
                current_pid = peptide_id
                vec = [0.0] * hla_count
                seen = [False] * hla_count
                current_updated_at = updated_at

            if vec is None or seen is None:
                continue

            if is_hla_text:
                hla_key = str(hla_key_raw)
            else:
                if hla_id_map is None:
                    continue
                hla_key = hla_id_map.get(int(hla_key_raw), "")
                if not hla_key:
                    continue

            idx = hla_idx.get(hla_key)
            if idx is None:
                continue
            vec[idx] = score
            seen[idx] = True

            if len(batch) >= 5000:
                write_cur.executemany(
                    """
                    INSERT OR REPLACE INTO score_el_new(peptide_id, namespace_id, score_blob, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )
                batch.clear()

        flush_current()
        if batch:
            write_cur.executemany(
                """
                INSERT OR REPLACE INTO score_el_new(peptide_id, namespace_id, score_blob, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self.conn.close()
            self.conn = None

    def compact(self) -> None:
        if self.conn is None:
            return
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.execute("VACUUM")

    def stats(self) -> Dict[str, int]:
        cur = self.conn.cursor()
        peptide_count = int(cur.execute("SELECT COUNT(*) FROM peptide").fetchone()[0])

        score_count_row = cur.execute(
            """
            SELECT COALESCE(SUM(t.cnt * n.hla_count), 0)
            FROM (
                SELECT namespace_id, COUNT(*) AS cnt
                FROM score_el
                GROUP BY namespace_id
            ) t
            JOIN namespace n ON n.namespace_id = t.namespace_id
            """
        ).fetchone()
        score_count = int(score_count_row[0] if score_count_row else 0)

        model_count = int(cur.execute("SELECT COUNT(DISTINCT model_id) FROM namespace").fetchone()[0])
        hla_count = int(cur.execute("SELECT COALESCE(MAX(hla_count), 0) FROM namespace").fetchone()[0])

        return {
            "peptides": peptide_count,
            "scores": score_count,
            "models": model_count,
            "hlas": hla_count,
        }

    def file_sizes(self) -> Dict[str, Any]:
        main_db = self.db_path
        wal_file = f"{main_db}-wal"
        shm_file = f"{main_db}-shm"

        main_size = os.path.getsize(main_db) if os.path.exists(main_db) else 0
        wal_size = os.path.getsize(wal_file) if os.path.exists(wal_file) else 0
        shm_size = os.path.getsize(shm_file) if os.path.exists(shm_file) else 0
        total_size = main_size + wal_size + shm_size

        cur = self.conn.cursor()
        page_count = int(cur.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(cur.execute("PRAGMA page_size").fetchone()[0])

        try:
            index_count = cur.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            ).fetchone()[0]
            score_rows = int(cur.execute("SELECT COUNT(*) FROM score_el").fetchone()[0])
            if score_rows > 0:
                estimated_index_bytes = max(score_rows * 12, page_size * max(1, index_count // 2))
                index_pct = (estimated_index_bytes / main_size * 100) if main_size > 0 else 0.0
            else:
                index_pct = 0.0
        except sqlite3.Error:
            index_pct = 0.0

        used_bytes = min(page_count * page_size, main_size)

        return {
            "main_db_bytes": main_size,
            "wal_bytes": wal_size,
            "shm_bytes": shm_size,
            "total_bytes": total_size,
            "page_count": page_count,
            "page_size": page_size,
            "used_bytes": used_bytes,
            "index_percentage": min(index_pct, 20.0),
        }

    def full_hit_peptides(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> set:
        unique_peptides = list(dict.fromkeys(peptides))
        if not unique_peptides:
            return set()

        ns = self._find_namespace(model_id=model_id, hla_set_hash=hla_set_hash)
        if ns is None:
            return set()
        namespace_id, namespace_hlas = ns

        ns_set = set(namespace_hlas)
        if any(h not in ns_set for h in hla_alleles):
            return set()

        cur = self.conn.cursor()
        hit_set = set()

        step = DEFAULT_HIT_STEP
        for i in range(0, len(unique_peptides), step):
            sub = unique_peptides[i : i + step]
            pep_placeholders = ",".join(["?"] * len(sub))
            query = f"""
                SELECT p.peptide_seq
                FROM score_el s
                JOIN peptide p ON p.peptide_id = s.peptide_id
                WHERE s.namespace_id = ?
                  AND p.peptide_seq IN ({pep_placeholders})
            """
            params = [namespace_id, *sub]
            for (pep_seq,) in cur.execute(query, params):
                hit_set.add(pep_seq)

        return hit_set

    def full_hit_set_for_namespace(
        self,
        model_id: str,
        hla_set_hash: str,
        hla_alleles: Sequence[str],
    ) -> set:
        ns = self._find_namespace(model_id=model_id, hla_set_hash=hla_set_hash)
        if ns is None:
            return set()
        namespace_id, namespace_hlas = ns

        ns_set = set(namespace_hlas)
        if any(h not in ns_set for h in hla_alleles):
            return set()

        cur = self.conn.cursor()
        query = """
            SELECT p.peptide_seq
            FROM score_el s
            JOIN peptide p ON p.peptide_id = s.peptide_id
            WHERE s.namespace_id = ?
        """
        return {pep_seq for (pep_seq,) in cur.execute(query, (namespace_id,))}

    def hit_stats(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> Dict[str, float]:
        unique_peptides = list(dict.fromkeys(peptides))
        if not unique_peptides:
            return {"input_peptides": 0, "hit_peptides": 0, "miss_peptides": 0, "hit_rate": 0.0}

        cached = self.lookup_scores(unique_peptides, hla_alleles, model_id=model_id, hla_set_hash=hla_set_hash)
        hit_peptides = 0
        for pep in unique_peptides:
            mapping = cached.get(pep, {})
            if all(h in mapping for h in hla_alleles):
                hit_peptides += 1

        total = len(unique_peptides)
        return {
            "input_peptides": total,
            "hit_peptides": hit_peptides,
            "miss_peptides": total - hit_peptides,
            "hit_rate": hit_peptides / total if total else 0.0,
        }

    def hit_stats_records(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> Dict[str, float]:
        if not peptides:
            return {"input_peptides": 0, "hit_peptides": 0, "miss_peptides": 0, "hit_rate": 0.0}

        counts = Counter(peptides)
        hit_set = self.full_hit_peptides(list(counts.keys()), hla_alleles, model_id=model_id, hla_set_hash=hla_set_hash)
        hit_records = sum(counts[p] for p in hit_set)
        total = len(peptides)
        return {
            "input_peptides": total,
            "hit_peptides": hit_records,
            "miss_peptides": total - hit_records,
            "hit_rate": hit_records / total if total else 0.0,
        }

    def _ensure_peptides(self, peptides: Sequence[str]) -> Dict[str, int]:
        unique_peptides = list(dict.fromkeys(peptides))
        if not unique_peptides:
            return {}

        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR IGNORE INTO peptide(peptide_seq) VALUES (?)",
            [(p,) for p in unique_peptides],
        )

        mapping: Dict[str, int] = {}
        step = DEFAULT_PEPTIDE_MAP_STEP
        for i in range(0, len(unique_peptides), step):
            sub = unique_peptides[i : i + step]
            placeholders = ",".join(["?"] * len(sub))
            query = f"SELECT peptide_id, peptide_seq FROM peptide WHERE peptide_seq IN ({placeholders})"
            for peptide_id, peptide_seq in cur.execute(query, sub):
                mapping[peptide_seq] = int(peptide_id)

        return mapping

    def upsert_scores(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        scores_by_hla: Dict[str, List[float]],
        model_id: str,
        hla_set_hash: str,
        commit: bool = True,
    ) -> int:
        if not peptides:
            return 0

        namespace_id, namespace_hlas = self._ensure_namespace(
            model_id=model_id,
            hla_set_hash=hla_set_hash,
            hla_alleles=hla_alleles,
        )

        pep_to_id = self._ensure_peptides(peptides)
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        first_idx: Dict[str, int] = {}
        for idx, pep in enumerate(peptides):
            if pep not in first_idx:
                first_idx[pep] = idx

        rows: List[Tuple[int, int, bytes, str]] = []
        for pep, idx in first_idx.items():
            pep_id = pep_to_id[pep]
            vec = [float(scores_by_hla[hla][idx]) for hla in namespace_hlas]
            rows.append((pep_id, namespace_id, self._encode_values(vec), now))

        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO score_el(peptide_id, namespace_id, score_blob, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        if commit:
            self.conn.commit()
        return len(rows) * len(namespace_hlas)

    def lookup_scores(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        unique_peptides = list(dict.fromkeys(peptides))
        if not unique_peptides:
            return result

        ns = self._find_namespace(model_id=model_id, hla_set_hash=hla_set_hash)
        if ns is None:
            return result
        namespace_id, namespace_hlas = ns

        ns_idx = {h: i for i, h in enumerate(namespace_hlas)}
        req_idx = [(h, ns_idx[h]) for h in hla_alleles if h in ns_idx]
        if len(req_idx) != len(hla_alleles):
            return result

        cur = self.conn.cursor()
        step = DEFAULT_LOOKUP_STEP
        for i in range(0, len(unique_peptides), step):
            sub = unique_peptides[i : i + step]
            pep_placeholders = ",".join(["?"] * len(sub))
            query = f"""
                SELECT p.peptide_seq, s.score_blob
                FROM score_el s
                JOIN peptide p ON p.peptide_id = s.peptide_id
                WHERE s.namespace_id = ?
                  AND p.peptide_seq IN ({pep_placeholders})
            """
            params = [namespace_id, *sub]
            for pep_seq, score_blob in cur.execute(query, params):
                decoded = self._decode_values(score_blob, len(namespace_hlas))
                if not decoded:
                    continue
                result[pep_seq] = {h: float(decoded[j]) for h, j in req_idx}

        return result


class ELScoreCache:
    def __init__(
        self,
        db_path: str,
        quant_mode: str = "uint16",
        shards: Union[str, int] = "auto",
        max_shards: int = DEFAULT_MAX_SHARDS,
        auto_target_shard_mb: int = DEFAULT_AUTO_TARGET_SHARD_MB,
    ):
        self.db_path = db_path
        self.quant_mode = quant_mode
        self.max_shards = max(1, int(max_shards))
        self.auto_target_shard_mb = max(64, int(auto_target_shard_mb))
        self.auto_target_shard_bytes = self.auto_target_shard_mb * 1024 * 1024

        self._single: Optional[_SingleShardCache] = None
        self._shard_handles: Dict[int, _SingleShardCache] = {}
        self._sharded = False
        self._shard_mode_requested = str(shards)
        self._shard_dir = f"{db_path}.shards"

        shard_mode = str(shards).strip().lower()
        if shard_mode in {"1", "single", "off", "none"}:
            self._single = _SingleShardCache(db_path=db_path, quant_mode=quant_mode)
            return

        if shard_mode == "auto":
            if os.path.exists(db_path):
                # Existing single DB remains readable without forced migration.
                self._single = _SingleShardCache(db_path=db_path, quant_mode=quant_mode)
                return
            self._sharded = True
            os.makedirs(self._shard_dir, exist_ok=True)

            existing = self._load_auto_shard_count()
            if existing is not None:
                self._logical_shards = existing
            else:
                self._logical_shards = self._recommend_auto_shards()
                self._save_auto_shard_count(self._logical_shards)
        else:
            shard_n = int(shards)
            if shard_n <= 1:
                self._single = _SingleShardCache(db_path=db_path, quant_mode=quant_mode)
                return
            self._sharded = True
            self._logical_shards = shard_n
            os.makedirs(self._shard_dir, exist_ok=True)
            self._save_auto_shard_count(self._logical_shards)

    def _shard_meta_path(self) -> str:
        return os.path.join(self._shard_dir, "logical_shards.txt")

    def _next_pow2(self, n: int) -> int:
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def _load_auto_shard_count(self) -> Optional[int]:
        meta = self._shard_meta_path()
        if not os.path.exists(meta):
            return None
        try:
            with open(meta, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            v = int(raw)
            if v >= 1:
                return v
        except (OSError, ValueError):
            return None
        return None

    def _save_auto_shard_count(self, shard_count: int) -> None:
        try:
            with open(self._shard_meta_path(), "w", encoding="utf-8") as f:
                f.write(str(int(shard_count)))
        except OSError:
            # Non-fatal: cache can still work without persisted metadata.
            pass

    def _recommend_auto_shards(self) -> int:
        indices = self._existing_shard_indices()
        existing_files = len(indices)

        total_bytes = 0
        for idx in indices:
            base = self._shard_path(idx)
            if os.path.exists(base):
                total_bytes += os.path.getsize(base)
            wal = f"{base}-wal"
            shm = f"{base}-shm"
            if os.path.exists(wal):
                total_bytes += os.path.getsize(wal)
            if os.path.exists(shm):
                total_bytes += os.path.getsize(shm)

        if total_bytes <= 0:
            suggested = min(DEFAULT_MIN_AUTO_SHARDS, self.max_shards)
            return max(1, suggested)

        by_size = (total_bytes + self.auto_target_shard_bytes - 1) // self.auto_target_shard_bytes
        suggested = max(int(by_size), DEFAULT_MIN_AUTO_SHARDS, existing_files)
        suggested = min(suggested, self.max_shards)

        pow2 = self._next_pow2(suggested)
        if pow2 <= self.max_shards:
            return max(1, pow2)
        return max(1, self.max_shards)

    def is_sharded(self) -> bool:
        return not self._is_single()

    def _is_single(self) -> bool:
        return self._single is not None

    def _shard_idx(self, peptide: str) -> int:
        h = hashlib.sha1(peptide.encode("utf-8")).digest()
        return int.from_bytes(h[:8], byteorder="big", signed=False) % self._logical_shards

    def _shard_path(self, idx: int) -> str:
        return os.path.join(self._shard_dir, f"shard_{idx:04d}.sqlite")

    def _get_shard(self, idx: int) -> _SingleShardCache:
        cache = self._shard_handles.get(idx)
        if cache is not None:
            return cache
        path = self._shard_path(idx)
        cache = _SingleShardCache(db_path=path, quant_mode=self.quant_mode)
        self._shard_handles[idx] = cache
        return cache

    def _existing_shard_indices(self) -> List[int]:
        if self._is_single():
            return [0]
        if not os.path.exists(self._shard_dir):
            return []
        out: List[int] = []
        for name in os.listdir(self._shard_dir):
            if not (name.startswith("shard_") and name.endswith(".sqlite")):
                continue
            stem = name[len("shard_") : -len(".sqlite")]
            if stem.isdigit():
                out.append(int(stem))
        out.sort()
        return out

    def shard_overview(self) -> Dict[str, Any]:
        if self._is_single():
            fs = self._single.file_sizes()
            return {
                "mode": "single",
                "requested": self._shard_mode_requested,
                "logical_shards": 1,
                "active_shards": 1,
                "shard_dir": None,
                "bytes_min": int(fs["total_bytes"]),
                "bytes_max": int(fs["total_bytes"]),
                "bytes_avg": int(fs["total_bytes"]),
                "top_shards": [
                    {
                        "shard": 0,
                        "total_bytes": int(fs["total_bytes"]),
                        "main_db_bytes": int(fs["main_db_bytes"]),
                        "wal_bytes": int(fs["wal_bytes"]),
                    }
                ],
            }

        indices = self._existing_shard_indices()
        if not indices:
            return {
                "mode": "sharded",
                "requested": self._shard_mode_requested,
                "logical_shards": int(self._logical_shards),
                "active_shards": 0,
                "shard_dir": self._shard_dir,
                "bytes_min": 0,
                "bytes_max": 0,
                "bytes_avg": 0,
                "top_shards": [],
            }

        per_shard = []
        for idx in indices:
            fs = self._get_shard(idx).file_sizes()
            per_shard.append(
                {
                    "shard": int(idx),
                    "main_db_bytes": int(fs["main_db_bytes"]),
                    "wal_bytes": int(fs["wal_bytes"]),
                    "shm_bytes": int(fs["shm_bytes"]),
                    "total_bytes": int(fs["total_bytes"]),
                }
            )

        totals = [row["total_bytes"] for row in per_shard]
        top_shards = sorted(per_shard, key=lambda x: x["total_bytes"], reverse=True)[:5]

        return {
            "mode": "sharded",
            "requested": self._shard_mode_requested,
            "logical_shards": int(self._logical_shards),
            "active_shards": len(indices),
            "shard_dir": self._shard_dir,
            "bytes_min": min(totals),
            "bytes_max": max(totals),
            "bytes_avg": int(sum(totals) / len(totals)),
            "top_shards": top_shards,
        }

    def close(self) -> None:
        if self._is_single():
            self._single.close()
            return
        for cache in self._shard_handles.values():
            cache.close()
        self._shard_handles.clear()

    def compact(self) -> None:
        if self._is_single():
            self._single.compact()
            return
        for idx in self._existing_shard_indices():
            self._get_shard(idx).compact()

    def stats(self) -> Dict[str, int]:
        if self._is_single():
            return self._single.stats()

        total_peptides = 0
        total_scores = 0
        max_hlas = 0
        model_ids = set()

        for idx in self._existing_shard_indices():
            shard = self._get_shard(idx)
            s = shard.stats()
            total_peptides += s["peptides"]
            total_scores += s["scores"]
            max_hlas = max(max_hlas, s["hlas"])

            cur = shard.conn.cursor()
            for (m,) in cur.execute("SELECT DISTINCT model_id FROM namespace"):
                model_ids.add(str(m))

        return {
            "peptides": total_peptides,
            "scores": total_scores,
            "models": len(model_ids),
            "hlas": max_hlas,
        }

    def file_sizes(self) -> Dict[str, Any]:
        if self._is_single():
            return self._single.file_sizes()

        main_bytes = 0
        wal_bytes = 0
        shm_bytes = 0
        page_count = 0
        page_size = 4096
        index_weighted = 0.0
        total_used = 0

        for idx in self._existing_shard_indices():
            fs = self._get_shard(idx).file_sizes()
            main_bytes += int(fs["main_db_bytes"])
            wal_bytes += int(fs["wal_bytes"])
            shm_bytes += int(fs["shm_bytes"])
            page_count += int(fs["page_count"])
            page_size = int(fs["page_size"])
            total_used += int(fs["used_bytes"])
            index_weighted += float(fs["index_percentage"]) * max(1, int(fs["main_db_bytes"]))

        total_main = max(1, main_bytes)
        index_pct = index_weighted / total_main

        return {
            "main_db_bytes": main_bytes,
            "wal_bytes": wal_bytes,
            "shm_bytes": shm_bytes,
            "total_bytes": main_bytes + wal_bytes + shm_bytes,
            "page_count": page_count,
            "page_size": page_size,
            "used_bytes": total_used,
            "index_percentage": min(index_pct, 20.0),
        }

    def _group_peptides_by_shard(self, peptides: Sequence[str]) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, pep in enumerate(peptides):
            groups[self._shard_idx(pep)].append(i)
        return groups

    def upsert_scores(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        scores_by_hla: Dict[str, List[float]],
        model_id: str,
        hla_set_hash: str,
        commit: bool = True,
    ) -> int:
        if self._is_single():
            return self._single.upsert_scores(
                peptides,
                hla_alleles,
                scores_by_hla,
                model_id,
                hla_set_hash,
                commit=commit,
            )

        if not peptides:
            return 0

        groups = self._group_peptides_by_shard(peptides)
        total_rows = 0
        for shard_idx, indices in groups.items():
            sub_peptides = [peptides[i] for i in indices]
            sub_scores = {h: [scores_by_hla[h][i] for i in indices] for h in hla_alleles}
            total_rows += self._get_shard(shard_idx).upsert_scores(
                peptides=sub_peptides,
                hla_alleles=hla_alleles,
                scores_by_hla=sub_scores,
                model_id=model_id,
                hla_set_hash=hla_set_hash,
                commit=False,
            )
        if commit:
            for shard_idx in groups.keys():
                self._get_shard(shard_idx).conn.commit()
        return total_rows

    def lookup_scores(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> Dict[str, Dict[str, float]]:
        if self._is_single():
            return self._single.lookup_scores(peptides, hla_alleles, model_id, hla_set_hash)

        result: Dict[str, Dict[str, float]] = {}
        if not peptides:
            return result

        unique_peptides = list(dict.fromkeys(peptides))
        groups: Dict[int, List[str]] = defaultdict(list)
        for pep in unique_peptides:
            groups[self._shard_idx(pep)].append(pep)

        for shard_idx, sub_peptides in groups.items():
            sub = self._get_shard(shard_idx).lookup_scores(
                peptides=sub_peptides,
                hla_alleles=hla_alleles,
                model_id=model_id,
                hla_set_hash=hla_set_hash,
            )
            result.update(sub)
        return result

    def full_hit_peptides(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> set:
        if self._is_single():
            return self._single.full_hit_peptides(peptides, hla_alleles, model_id, hla_set_hash)

        hit_set = set()
        unique_peptides = list(dict.fromkeys(peptides))
        groups: Dict[int, List[str]] = defaultdict(list)
        for pep in unique_peptides:
            groups[self._shard_idx(pep)].append(pep)

        for shard_idx, sub_peptides in groups.items():
            hit_set |= self._get_shard(shard_idx).full_hit_peptides(
                peptides=sub_peptides,
                hla_alleles=hla_alleles,
                model_id=model_id,
                hla_set_hash=hla_set_hash,
            )
        return hit_set

    def full_hit_set_for_namespace(
        self,
        model_id: str,
        hla_set_hash: str,
        hla_alleles: Sequence[str],
    ) -> set:
        if self._is_single():
            return self._single.full_hit_set_for_namespace(model_id, hla_set_hash, hla_alleles)

        out = set()
        for idx in self._existing_shard_indices():
            out |= self._get_shard(idx).full_hit_set_for_namespace(model_id, hla_set_hash, hla_alleles)
        return out

    def hit_stats(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> Dict[str, float]:
        unique_peptides = list(dict.fromkeys(peptides))
        if not unique_peptides:
            return {"input_peptides": 0, "hit_peptides": 0, "miss_peptides": 0, "hit_rate": 0.0}

        cached = self.lookup_scores(unique_peptides, hla_alleles, model_id=model_id, hla_set_hash=hla_set_hash)
        hit_peptides = 0
        for pep in unique_peptides:
            mapping = cached.get(pep, {})
            if all(h in mapping for h in hla_alleles):
                hit_peptides += 1

        total = len(unique_peptides)
        return {
            "input_peptides": total,
            "hit_peptides": hit_peptides,
            "miss_peptides": total - hit_peptides,
            "hit_rate": hit_peptides / total if total else 0.0,
        }

    def hit_stats_records(
        self,
        peptides: Sequence[str],
        hla_alleles: Sequence[str],
        model_id: str,
        hla_set_hash: str,
    ) -> Dict[str, float]:
        if not peptides:
            return {"input_peptides": 0, "hit_peptides": 0, "miss_peptides": 0, "hit_rate": 0.0}

        counts = Counter(peptides)
        hit_set = self.full_hit_peptides(list(counts.keys()), hla_alleles, model_id=model_id, hla_set_hash=hla_set_hash)
        hit_records = sum(counts[p] for p in hit_set)
        total = len(peptides)
        return {
            "input_peptides": total,
            "hit_peptides": hit_records,
            "miss_peptides": total - hit_records,
            "hit_rate": hit_records / total if total else 0.0,
        }

    def import_wide_tsv(
        self,
        wide_tsv_path: str,
        model_id: str,
        hla_alleles: Sequence[str],
    ) -> Tuple[int, int]:
        if not os.path.exists(wide_tsv_path):
            raise FileNotFoundError(f"Wide TSV not found: {wide_tsv_path}")

        hla_set_hash = canonical_hla_set_hash(hla_alleles)
        inserted_rows = 0
        inserted_peptides = 0

        batch_peptides: List[str] = []
        batch_scores: Dict[str, List[float]] = {h: [] for h in hla_alleles}

        tx_open = False
        if self._is_single():
            self._single.conn.execute("BEGIN")
            tx_open = True

        try:
            with open(wide_tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                headers = set(reader.fieldnames or [])
                required = {"peptide"}
                missing_required = required - headers
                if missing_required:
                    raise ValueError(f"Wide TSV missing columns: {missing_required}")
                missing_hla = [h for h in hla_alleles if h not in headers]
                if missing_hla:
                    raise ValueError(f"Wide TSV missing HLA columns: {missing_hla}")

                for row in reader:
                    pep = row["peptide"].strip().upper()
                    if not pep:
                        continue
                    batch_peptides.append(pep)
                    for h in hla_alleles:
                        batch_scores[h].append(float(row[h]))

                    if len(batch_peptides) >= DEFAULT_IMPORT_BATCH:
                        inserted_peptides += len(batch_peptides)
                        inserted_rows += self.upsert_scores(
                            peptides=batch_peptides,
                            hla_alleles=hla_alleles,
                            scores_by_hla=batch_scores,
                            model_id=model_id,
                            hla_set_hash=hla_set_hash,
                            commit=False,
                        )
                        batch_peptides = []
                        batch_scores = {h: [] for h in hla_alleles}

            if batch_peptides:
                inserted_peptides += len(batch_peptides)
                inserted_rows += self.upsert_scores(
                    peptides=batch_peptides,
                    hla_alleles=hla_alleles,
                    scores_by_hla=batch_scores,
                    model_id=model_id,
                    hla_set_hash=hla_set_hash,
                    commit=False,
                )

            if self._is_single():
                self._single.conn.commit()
                tx_open = False
            else:
                for shard in self._shard_handles.values():
                    shard.conn.commit()
        except Exception:
            if self._is_single() and tx_open:
                self._single.conn.rollback()
            elif not self._is_single():
                for shard in self._shard_handles.values():
                    shard.conn.rollback()
            raise

        self.compact()
        return inserted_peptides, inserted_rows

    def import_long_tsv(
        self,
        long_tsv_path: str,
        model_id: str,
        hla_alleles: Sequence[str],
    ) -> Tuple[int, int]:
        if not os.path.exists(long_tsv_path):
            raise FileNotFoundError(f"Long TSV not found: {long_tsv_path}")

        hla_set = set(hla_alleles)
        hla_set_hash = canonical_hla_set_hash(hla_alleles)

        grouped: Dict[str, Dict[str, float]] = {}
        with open(long_tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            score_col = "score_el" if "score_el" in (reader.fieldnames or []) else "score"
            required = {"peptide", "hla", score_col}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Long TSV missing columns: {missing}")

            for row in reader:
                pep = row["peptide"].strip().upper()
                hla = row["hla"].strip()
                if hla not in hla_set:
                    continue
                grouped.setdefault(pep, {})[hla] = float(row[score_col])

        peptides: List[str] = []
        scores_by_hla: Dict[str, List[float]] = {h: [] for h in hla_alleles}
        for pep, mapping in grouped.items():
            if any(h not in mapping for h in hla_alleles):
                continue
            peptides.append(pep)
            for h in hla_alleles:
                scores_by_hla[h].append(mapping[h])

        tx_open = False
        if self._is_single():
            self._single.conn.execute("BEGIN")
            tx_open = True

        try:
            inserted_rows = self.upsert_scores(
                peptides=peptides,
                hla_alleles=hla_alleles,
                scores_by_hla=scores_by_hla,
                model_id=model_id,
                hla_set_hash=hla_set_hash,
                commit=False,
            )
            if self._is_single():
                self._single.conn.commit()
                tx_open = False
            else:
                for shard in self._shard_handles.values():
                    shard.conn.commit()
        except Exception:
            if self._is_single() and tx_open:
                self._single.conn.rollback()
            elif not self._is_single():
                for shard in self._shard_handles.values():
                    shard.conn.rollback()
            raise

        self.compact()
        return len(peptides), inserted_rows
