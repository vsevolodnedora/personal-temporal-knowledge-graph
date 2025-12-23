from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


@dataclass(frozen=True)
class DatasetOverview:
    n_conversations: int
    n_messages: int
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    roles: Dict[str, int]
    avg_messages_per_conversation: float
    median_messages_per_conversation: float


class ChatGPTConversationsAnalyzer:
    """
    Analyze and visualize ChatGPT personal-data export `conversations.json`.

    Typical export structure (may vary slightly):
      - Root is a list of conversations, OR a dict containing the list
      - Each conversation has:
          id, title, create_time, update_time, mapping (graph of messages)
      - mapping: {node_id: {message: {...}, parent: ..., children: [...]}}
      - message includes:
          author.role (user/assistant/system/tool),
          create_time (unix seconds),
          content.parts (list of strings) OR other content types
    """

    def __init__(self, json_path: str | Path, tz: str = "Europe/Berlin") -> None:
        self.json_path = Path(json_path)
        self.tz = tz

        self.raw: Any = None
        self.conversations_df: Optional[pd.DataFrame] = None
        self.messages_df: Optional[pd.DataFrame] = None

    # -----------------------
    # Public API
    # -----------------------

    def load(self) -> "ChatGPTConversationsAnalyzer":
        """Load JSON and build normalized conversation + message tables."""
        self.raw = self._read_json(self.json_path)
        conversations = self._extract_conversations_list(self.raw)

        conv_rows: List[Dict[str, Any]] = []
        msg_rows: List[Dict[str, Any]] = []

        for conv in conversations:
            conv_id = conv.get("id")
            title = conv.get("title")
            ctime = self._to_timestamp(conv.get("create_time"))
            utime = self._to_timestamp(conv.get("update_time"))
            mapping = conv.get("mapping") or {}

            conv_rows.append(
                {
                    "conversation_id": conv_id,
                    "title": title,
                    "created_at": ctime,
                    "updated_at": utime,
                    "n_nodes": len(mapping) if isinstance(mapping, dict) else None,
                }
            )

            # Flatten messages from mapping graph
            if isinstance(mapping, dict):
                for node_id, node in mapping.items():
                    msg = (node or {}).get("message")
                    if not msg:
                        continue

                    role = ((msg.get("author") or {}).get("role")) or "unknown"
                    msg_time = self._to_timestamp(msg.get("create_time"))

                    content = msg.get("content") or {}
                    text = self._extract_text(content)
                    content_type = content.get("content_type")

                    msg_rows.append(
                        {
                            "conversation_id": conv_id,
                            "conversation_title": title,
                            "message_id": msg.get("id") or node_id,
                            "role": role,
                            "created_at": msg_time,
                            "content_type": content_type,
                            "text": text,
                            "char_count": len(text) if isinstance(text, str) else 0,
                            "word_count": len(text.split()) if isinstance(text, str) and text else 0,
                        }
                    )

        self.conversations_df = pd.DataFrame(conv_rows)
        self.messages_df = pd.DataFrame(msg_rows)

        # Normalize timestamps, set timezone
        if not self.conversations_df.empty:
            self.conversations_df["created_at"] = self._localize_series(self.conversations_df["created_at"])
            self.conversations_df["updated_at"] = self._localize_series(self.conversations_df["updated_at"])

        if not self.messages_df.empty:
            self.messages_df["created_at"] = self._localize_series(self.messages_df["created_at"])

        return self

    def overview(self) -> DatasetOverview:
        """Compute key properties of the dataset."""
        self._ensure_loaded()

        msgs = self.messages_df
        convs = self.conversations_df

        n_conversations = int(len(convs))
        n_messages = int(len(msgs))

        date_min = msgs["created_at"].min() if n_messages else None
        date_max = msgs["created_at"].max() if n_messages else None

        roles = dict(Counter(msgs["role"])) if n_messages else {}

        counts = msgs.groupby("conversation_id")["message_id"].count() if n_messages else pd.Series([], dtype=int)
        avg = float(counts.mean()) if len(counts) else 0.0
        med = float(counts.median()) if len(counts) else 0.0

        return DatasetOverview(
            n_conversations=n_conversations,
            n_messages=n_messages,
            date_min=date_min,
            date_max=date_max,
            roles=roles,
            avg_messages_per_conversation=avg,
            median_messages_per_conversation=med,
        )

    def print_overview(self) -> None:
        """Print a readable summary to the console."""
        ov = self.overview()
        print("ChatGPT conversations.json — Overview")
        print("-" * 40)
        print(f"Conversations: {ov.n_conversations:,}")
        print(f"Messages:      {ov.n_messages:,}")
        if ov.date_min is not None and ov.date_max is not None:
            print(f"Date range:    {ov.date_min}  →  {ov.date_max}  ({self.tz})")
        print(f"Avg msgs/conv: {ov.avg_messages_per_conversation:.2f}")
        print(f"Med msgs/conv: {ov.median_messages_per_conversation:.2f}")
        if ov.roles:
            print("\nRoles:")
            for k, v in sorted(ov.roles.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  - {k:<10} {v:,}")

    # -----------------------
    # Visualization
    # -----------------------

    def plot_activity_over_time(self, freq: str = "W") -> None:
        """Line plot of message volume over time (freq: 'D', 'W', 'M', etc)."""
        self._ensure_loaded()
        df = self.messages_df.dropna(subset=["created_at"]).copy()
        if df.empty:
            print("No messages with timestamps to plot.")
            return

        ts = df.set_index("created_at").resample(freq)["message_id"].count()

        plt.figure(figsize=(10, 4))
        plt.plot(ts.index, ts.values)
        plt.title(f"Message Activity Over Time ({freq})")
        plt.xlabel("Time")
        plt.ylabel("Messages")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_messages_by_role(self) -> None:
        """Bar chart of message counts by role."""
        self._ensure_loaded()
        df = self.messages_df
        if df.empty:
            print("No messages to plot.")
            return

        counts = df["role"].value_counts().sort_values(ascending=False)

        plt.figure(figsize=(7, 4))
        plt.bar(counts.index.astype(str), counts.values)
        plt.title("Messages by Role")
        plt.xlabel("Role")
        plt.ylabel("Messages")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_messages_per_conversation(self, bins: int = 30) -> None:
        """Histogram of message counts per conversation."""
        self._ensure_loaded()
        df = self.messages_df
        if df.empty:
            print("No messages to plot.")
            return

        per_conv = df.groupby("conversation_id")["message_id"].count()

        plt.figure(figsize=(8, 4))
        plt.hist(per_conv.values, bins=bins)
        plt.title("Messages per Conversation")
        plt.xlabel("Messages in a conversation")
        plt.ylabel("Number of conversations")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_heatmap_day_hour(self) -> None:
        """
        Heatmap of activity by day-of-week and hour-of-day.
        Uses local timezone chosen in __init__.
        """
        self._ensure_loaded()
        df = self.messages_df.dropna(subset=["created_at"]).copy()
        if df.empty:
            print("No messages with timestamps to plot.")
            return

        df["dow"] = df["created_at"].dt.day_name()
        df["hour"] = df["created_at"].dt.hour

        # Order days nicely
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = (
            df.pivot_table(index="dow", columns="hour", values="message_id", aggfunc="count", fill_value=0)
            .reindex(days)
        )

        plt.figure(figsize=(11, 4.5))
        plt.imshow(pivot.values, aspect="auto")
        plt.title("Activity Heatmap (Day of Week × Hour)")
        plt.xlabel("Hour of day")
        plt.ylabel("Day of week")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(0, 24, 1), range(0, 24, 1), rotation=0)
        plt.colorbar(label="Messages")
        plt.tight_layout()
        plt.show()

    def top_words(self, n: int = 30, role_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Simple top-word frequency table (very lightweight; no NLP libraries).
        Optionally filter by role (e.g., 'user' or 'assistant').
        """
        self._ensure_loaded()
        df = self.messages_df
        if df.empty:
            return pd.DataFrame(columns=["word", "count"])

        subset = df
        if role_filter:
            subset = subset[subset["role"] == role_filter]

        words: List[str] = []
        for t in subset["text"].dropna().astype(str):
            for w in t.lower().split():
                w = "".join(ch for ch in w if ch.isalnum() or ch in ("-", "_"))
                if len(w) >= 3:
                    words.append(w)

        counts = Counter(words).most_common(n)
        return pd.DataFrame(counts, columns=["word", "count"])

    def plot_all(self) -> None:
        """Convenience method: produce a small set of useful plots."""
        self.plot_activity_over_time(freq="W")
        self.plot_messages_by_role()
        self.plot_messages_per_conversation()
        self.plot_heatmap_day_hour()

        # ASCII tree
        self.print_project_tree(max_projects=20, max_conversations_per_project=25)

        # Plot tree
        self.plot_project_tree(max_projects=10, max_conversations_per_project=10)
        #

    def project_table(
        self,
        project_fn: Optional[Callable[[str], str]] = None,
        project_regex: Optional[str] = None,
        regex_group: int = 1,
    ) -> pd.DataFrame:
        """
        Return a table with inferred 'project' per conversation + message counts.

        Project inference (choose one):
          1) project_fn(title)->project
          2) project_regex applied to title, take regex_group
          3) fallback heuristic: prefix before ':', '|' or ' - ' (if short)
        """
        self._ensure_loaded()
        convs = self.conversations_df.copy()
        msgs = self.messages_df.copy()

        # message counts per conversation
        counts = msgs.groupby("conversation_id")["message_id"].count().rename("n_messages")
        convs = convs.merge(counts, on="conversation_id", how="left")
        convs["n_messages"] = convs["n_messages"].fillna(0).astype(int)

        # first/last message timestamps
        if not msgs.empty and "created_at" in msgs.columns:
            rng = msgs.groupby("conversation_id")["created_at"].agg(["min", "max"]).rename(
                columns={"min": "first_message_at", "max": "last_message_at"}
            )
            convs = convs.merge(rng, on="conversation_id", how="left")
        else:
            convs["first_message_at"] = pd.NaT
            convs["last_message_at"] = pd.NaT

        def default_project(title: str) -> str:
            t = (title or "").strip()
            if not t:
                return "Uncategorized"

            # common separators: "Project: topic", "Project | topic", "Project - topic"
            for sep in [":", "|", " - ", " — ", " – "]:
                if sep in t:
                    left = t.split(sep, 1)[0].strip()
                    # only treat as "project" if it looks like a short label
                    if 1 <= len(left) <= 40:
                        return left
            return "Uncategorized"

        def infer_project(title: str) -> str:
            t = (title or "").strip()
            if project_fn is not None:
                p = (project_fn(t) or "").strip()
                return p if p else "Uncategorized"
            if project_regex is not None:
                m = re.search(project_regex, t)
                if m:
                    try:
                        p = (m.group(regex_group) or "").strip()
                        return p if p else "Uncategorized"
                    except Exception:
                        pass
            return default_project(t)

        convs["project"] = convs["title"].fillna("").map(infer_project)

        # nice ordering columns
        out = convs[
            ["project", "conversation_id", "title", "n_messages", "created_at", "updated_at", "first_message_at", "last_message_at"]
        ].sort_values(["project", "n_messages"], ascending=[True, False])

        return out

    def print_project_tree(
        self,
        max_projects: int = 25,
        max_conversations_per_project: int = 30,
        min_messages: int = 1,
        **project_infer_kwargs: Any,
    ) -> None:
        """
        Print an ASCII tree:
          Project (total msgs, #convs)
            ├─ Conversation title (n msgs)
            └─ ...
        """
        tbl = self.project_table(**project_infer_kwargs)
        tbl = tbl[tbl["n_messages"] >= min_messages].copy()

        proj_totals = tbl.groupby("project")["n_messages"].sum().sort_values(ascending=False)
        proj_totals = proj_totals.head(max_projects)

        print("Projects → Conversations (message counts)")
        print("-" * 60)

        for project in proj_totals.index:
            sub = tbl[tbl["project"] == project].sort_values("n_messages", ascending=False)
            sub = sub.head(max_conversations_per_project)

            total_msgs = int(proj_totals.loc[project])
            n_convs = int((tbl["project"] == project).sum())

            print(f"{project}  (total msgs: {total_msgs:,}, convs: {n_convs:,})")

            rows = list(sub.itertuples(index=False))
            for i, r in enumerate(rows):
                branch = "└─" if i == len(rows) - 1 else "├─"
                title = (r.title or "Untitled").strip()
                if len(title) > 80:
                    title = title[:77] + "..."
                print(f"  {branch} {title}  ({r.n_messages:,})")
            print()

    def plot_project_tree(
        self,
        max_projects: int = 12,
        max_conversations_per_project: int = 12,
        min_messages: int = 1,
        project_node_size: int = 1400,
        convo_node_size_min: int = 200,
        convo_node_size_max: int = 1600,
        figsize: Tuple[int, int] = (14, 8),
        **project_infer_kwargs: Any,
    ) -> None:
        """
        Plot a simple hierarchy tree:
          Root -> Projects -> Conversations (node size ~ #messages)

        Uses a tidy-tree layout implemented in pure python + matplotlib (no extra deps).
        """
        self._ensure_loaded()

        tbl = self.project_table(**project_infer_kwargs)
        tbl = tbl[tbl["n_messages"] >= min_messages].copy()
        if tbl.empty:
            print("No conversations meet the filter criteria.")
            return

        # choose top projects by total messages
        proj_totals = tbl.groupby("project")["n_messages"].sum().sort_values(ascending=False).head(max_projects)
        tbl = tbl[tbl["project"].isin(proj_totals.index)].copy()

        # within each project, top conversations
        tbl = (
            tbl.sort_values(["project", "n_messages"], ascending=[True, False])
               .groupby("project", group_keys=False)
               .head(max_conversations_per_project)
               .copy()
        )

        # Build a hierarchy structure
        root = "All Projects"
        projects = list(proj_totals.index)

        children: Dict[str, List[str]] = {root: []}
        node_label: Dict[str, str] = {root: root}
        node_value: Dict[str, int] = {root: int(tbl["n_messages"].sum())}

        # Project nodes
        for p in projects:
            pid = f"project::{p}"
            children[root].append(pid)
            children[pid] = []
            node_label[pid] = f"{p}\n{int(tbl[tbl['project']==p]['n_messages'].sum()):,} msgs"
            node_value[pid] = int(tbl[tbl["project"] == p]["n_messages"].sum())

        # Conversation nodes
        for r in tbl.itertuples(index=False):
            p = r.project
            pid = f"project::{p}"
            cid = f"conv::{r.conversation_id}"
            children[pid].append(cid)

            title = (r.title or "Untitled").strip()
            if len(title) > 42:
                title = title[:39] + "..."
            node_label[cid] = f"{title}\n{int(r.n_messages):,} msgs"
            node_value[cid] = int(r.n_messages)
            children[cid] = []

        # --- tidy layout (y positions by leaf order, x by depth) ---
        def get_depth(node: str) -> int:
            if node == root:
                return 0
            if node.startswith("project::"):
                return 1
            return 2  # conversations

        # assign y to leaves in order; internal nodes get average of children
        y: Dict[str, float] = {}
        x: Dict[str, float] = {}

        leaves: List[str] = []

        def collect_leaves(n: str) -> None:
            ch = children.get(n, [])
            if not ch:
                leaves.append(n)
            else:
                for c in ch:
                    collect_leaves(c)

        collect_leaves(root)

        # stable ordering: project order then conversation message count already sorted
        # (our children lists preserve insertion order)
        for i, leaf in enumerate(leaves):
            y[leaf] = float(i)

        def set_internal_positions(n: str) -> float:
            ch = children.get(n, [])
            if not ch:
                return y[n]
            child_ys = [set_internal_positions(c) for c in ch]
            y[n] = sum(child_ys) / len(child_ys)
            return y[n]

        set_internal_positions(root)

        for n in list(children.keys()):
            x[n] = float(get_depth(n))

        # node sizes: projects fixed-ish, conv scaled by messages
        conv_vals = [node_value[n] for n in children.keys() if n.startswith("conv::")]
        vmin = min(conv_vals) if conv_vals else 1
        vmax = max(conv_vals) if conv_vals else 1

        def scale_size(v: int) -> float:
            if vmax == vmin:
                return float((convo_node_size_min + convo_node_size_max) / 2)
            t = (v - vmin) / (vmax - vmin)
            return float(convo_node_size_min + t * (convo_node_size_max - convo_node_size_min))

        node_size: Dict[str, float] = {}
        for n in children.keys():
            if n == root:
                node_size[n] = project_node_size * 1.2
            elif n.startswith("project::"):
                node_size[n] = project_node_size
            else:
                node_size[n] = scale_size(node_value[n])

        # --- plot ---
        plt.figure(figsize=figsize)

        # edges
        for parent, ch_list in children.items():
            for c in ch_list:
                plt.plot([x[parent], x[c]], [y[parent], y[c]], linewidth=1, alpha=0.6)

        # nodes
        xs = [x[n] for n in children.keys()]
        ys = [y[n] for n in children.keys()]
        ss = [node_size[n] for n in children.keys()]
        plt.scatter(xs, ys, s=ss, alpha=0.9)

        # labels (boxed for readability)
        for n in children.keys():
            plt.text(
                x[n],
                y[n],
                node_label.get(n, n),
                ha="center",
                va="center",
                fontsize=9 if n.startswith("conv::") else 10,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.85),
            )

        plt.title("ChatGPT Conversations Tree (Projects → Discussions)")
        plt.xticks([0, 1, 2], ["All", "Projects", "Discussions"])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # -----------------------
    # Internals
    # -----------------------

    def _ensure_loaded(self) -> None:
        if self.conversations_df is None or self.messages_df is None:
            raise RuntimeError("Call .load() first.")

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_conversations_list(raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            # common possibilities
            for key in ("conversations", "data", "items"):
                val = raw.get(key)
                if isinstance(val, list):
                    return val
        raise ValueError("Unrecognized conversations.json structure: expected list or dict containing a list.")

    @staticmethod
    def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
        """Convert unix seconds (float/int) to pandas Timestamp (UTC); returns None if missing."""
        if value is None:
            return None
        try:
            # ChatGPT export times are usually unix seconds
            return pd.to_datetime(float(value), unit="s", utc=True)
        except Exception:
            # If already a string timestamp in ISO format, try parse
            try:
                return pd.to_datetime(value, utc=True)
            except Exception:
                return None

    def _localize_series(self, s: pd.Series) -> pd.Series:
        """Ensure timestamps are timezone-aware, then convert to configured timezone."""
        # If the series is empty, just return it
        if s is None or len(s) == 0:
            return s
        out = pd.to_datetime(s, utc=True, errors="coerce")
        try:
            return out.dt.tz_convert(self.tz)
        except Exception:
            # if tz is invalid or conversion fails, keep UTC
            return out

    @staticmethod
    def _extract_text(content: Dict[str, Any]) -> str:
        """
        Extract readable text from message content.
        Handles common content shapes:
          - {"content_type": "text", "parts": ["..."]}
          - {"parts": [...]}
        """
        if not isinstance(content, dict):
            return ""

        parts = content.get("parts")
        if isinstance(parts, list):
            # parts can include strings or structured chunks in some exports
            out: List[str] = []
            for p in parts:
                if isinstance(p, str):
                    out.append(p)
                elif isinstance(p, dict):
                    # attempt a few likely fields
                    for k in ("text", "content", "value"):
                        if isinstance(p.get(k), str):
                            out.append(p[k])
                            break
            return "\n".join([x for x in out if x]).strip()

        # Fallback for other shapes
        for k in ("text", "value"):
            if isinstance(content.get(k), str):
                return content[k].strip()

        return ""


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    analyzer = ChatGPTConversationsAnalyzer("../../data/raw/conversations.json", tz="Europe/Berlin").load()
    analyzer.print_overview()

    # Plots
    analyzer.plot_all()

    # Top words (optional)
    print("\nTop words (user):")
    print(analyzer.top_words(n=20, role_filter="user").to_string(index=False))
