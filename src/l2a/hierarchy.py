"""Parent-child account hierarchy from GLEIF level 2 + Wikidata P749/P355."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

# Lower rank wins when an entity has several direct parents.
SOURCE_RANK = {"gleif_direct": 0, "wikidata_p749": 1, "wikidata_p355": 2, "gleif_ultimate": 3}


class Hierarchy:
    def __init__(self, edges: Iterable[tuple[str, str, str]]) -> None:
        """edges: (child_id, parent_id, source)."""
        best: dict[str, tuple[int, str]] = {}
        self.ultimate_hint: dict[str, str] = {}
        self.children: dict[str, set[str]] = defaultdict(set)
        for child, parent, source in edges:
            if child == parent:
                continue
            if source == "gleif_ultimate":
                self.ultimate_hint[child] = parent
            rank = SOURCE_RANK.get(source, 9)
            if child not in best or rank < best[child][0]:
                best[child] = (rank, parent)
        self.parent = {c: p for c, (_r, p) in best.items()}
        for c, p in self.parent.items():
            self.children[p].add(c)

    def ancestors(self, node: str) -> list[str]:
        """Direct parent first; stops on cycles."""
        out: list[str] = []
        seen = {node}
        cur = node
        while cur in self.parent:
            cur = self.parent[cur]
            if cur in seen:
                break
            out.append(cur)
            seen.add(cur)
        return out

    def ultimate_parent(self, node: str) -> str:
        """Top of the chain; GLEIF's declared ultimate parent if the chain is shorter."""
        anc = self.ancestors(node)
        top = anc[-1] if anc else node
        hint = self.ultimate_hint.get(node)
        if hint and hint not in anc:
            return hint
        return top

    def rollup(self, node: str, present: set[str]) -> str:
        """Highest ancestor (or self) that exists in `present` (e.g. has a CRM account)."""
        best = node
        for a in self.ancestors(node):
            if a in present:
                best = a
        hint = self.ultimate_hint.get(node)
        if hint and hint in present and hint not in self.ancestors(node):
            best = hint
        return best
