from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, DefaultDict
from collections import defaultdict, deque

import numpy as np
from manim import (
    Scene, VGroup, Circle, Line, Text, Square, RoundedRectangle,
    FadeIn, FadeOut, Create, LaggedStart, AnimationGroup,
    UP, DOWN, LEFT, RIGHT
)

# -----------------------------
# Helpers: build an (almost) complete binary tree over ranks 0..N-1
# -----------------------------

def build_binary_tree(num_ranks: int) -> Tuple[Dict[int, Optional[int]], Dict[int, List[int]]]:
    parent: Dict[int, Optional[int]] = {0: None}
    children: Dict[int, List[int]] = defaultdict(list)
    for r in range(1, num_ranks):
        p = (r - 1) // 2
        parent[r] = p
        children[p].append(r)
    # Ensure keys exist for all ranks
    for r in range(num_ranks):
        children[r] = children.get(r, [])
    return parent, children


def compute_depths(parent: Dict[int, Optional[int]], num_ranks: int) -> Dict[int, int]:
    depth = {0: 0}
    for r in range(1, num_ranks):
        p = parent[r]
        d = 0
        cur = r
        while cur != 0:
            cur = parent[cur]  # type: ignore
            d += 1
        depth[r] = d
    return depth


def compute_subtree_sets(children: Dict[int, List[int]], root: int = 0) -> Dict[int, Set[int]]:
    subtree: Dict[int, Set[int]] = {}

    def dfs(u: int) -> Set[int]:
        s = {u}
        for v in children[u]:
            s |= dfs(v)
        subtree[u] = s
        return s

    dfs(root)
    return subtree


def compute_leaf_count(children: Dict[int, List[int]], root: int = 0) -> int:
    leaves = 0
    stack = [root]
    while stack:
        u = stack.pop()
        if len(children[u]) == 0:
            leaves += 1
        else:
            stack.extend(children[u])
    return max(leaves, 1)


def compute_tidy_layout(children: Dict[int, List[int]], root: int, *,
                        top_y: float = 3.2,
                        y_spacing: float = 1.25,
                        x_spacing: float = 1.6) -> Dict[int, np.ndarray]:
    """
    A simple tidy tree layout:
    - Assign x positions to leaves left-to-right, evenly spaced.
    - Internal node x = average of children x.
    """
    positions: Dict[int, np.ndarray] = {}
    num_leaves = compute_leaf_count(children, root=root)
    leaf_index = 0

    def assign(u: int, depth: int) -> float:
        nonlocal leaf_index
        if len(children[u]) == 0:
            # Center leaves around x=0
            x = (leaf_index - (num_leaves - 1) / 2.0) * x_spacing
            leaf_index += 1
        else:
            xs = [assign(v, depth + 1) for v in children[u]]
            x = sum(xs) / len(xs)
        y = top_y - depth * y_spacing
        positions[u] = np.array([x, y, 0.0])
        return x

    assign(root, 0)
    return positions


# -----------------------------
# Visual primitives
# -----------------------------

TOKEN_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC"
]


def make_rank_node(rank: int) -> VGroup:
    c = Circle(radius=0.22, stroke_width=2)
    t = Text(f"R{rank}", font_size=20)
    return VGroup(c, t)


def make_token(label: str, color_hex: str, *, side: float = 0.22) -> VGroup:
    s = Square(side_length=side, stroke_width=1)
    s.set_fill(color_hex, opacity=1.0)
    txt = Text(label, font_size=14)
    txt.scale(0.65)
    return VGroup(s, txt)


def make_vec_box(label: str) -> Tuple[RoundedRectangle, Text, VGroup]:
    box = RoundedRectangle(corner_radius=0.08, width=0.9, height=0.32, stroke_width=1.5)
    txt = Text(label, font_size=18)
    g = VGroup(box, txt)
    return box, txt, g


@dataclass
class TokenInfo:
    mobj: VGroup
    dest_rank: int


# -----------------------------
# Base scene with shared geometry utilities
# -----------------------------

class TreeCollectiveBase(Scene):
    NUM_RANKS = 8

    def setup_tree(self):
        self.parent, self.children = build_binary_tree(self.NUM_RANKS)
        self.depth = compute_depths(self.parent, self.NUM_RANKS)
        self.max_depth = max(self.depth.values())
        self.subtree = compute_subtree_sets(self.children, root=0)

        self.pos = compute_tidy_layout(self.children, root=0)

        # Build node mobjects
        self.nodes: Dict[int, VGroup] = {}
        for r in range(self.NUM_RANKS):
            node = make_rank_node(r).move_to(self.pos[r])
            self.nodes[r] = node

        # Build edges
        self.edges: List[Line] = []
        for r in range(1, self.NUM_RANKS):
            p = self.parent[r]
            if p is None:
                continue
            e = Line(self.nodes[p].get_center(), self.nodes[r].get_center(), stroke_width=2)
            self.edges.append(e)

        self.add(*self.edges)
        self.add(*self.nodes.values())

    def token_anchor(self, rank: int) -> np.ndarray:
        # Where tokens cluster near a node
        return self.nodes[rank].get_center() + UP * 0.55

    def box_anchor(self, rank: int) -> np.ndarray:
        # Where vector boxes sit (below node)
        return self.nodes[rank].get_center() + DOWN * 0.62

    def restack_tokens(self, rank: int, tokens: List[VGroup]) -> List[np.ndarray]:
        if not tokens:
            return []
        g = VGroup(*tokens)
        g.arrange(RIGHT, buff=0.05)
        g.move_to(self.token_anchor(rank))
        return [m.get_center() for m in g]

    def play_restack(self, rank: int, tokens: List[VGroup], run_time: float = 0.35):
        if not tokens:
            return
        targets = self.restack_tokens(rank, tokens)
        self.play(*[tok.animate.move_to(targets[i]) for i, tok in enumerate(tokens)], run_time=run_time)

    def ranks_at_depth(self, d: int) -> List[int]:
        return [r for r, dep in self.depth.items() if dep == d]

    def route_next_hop(self, current: int, dest: int) -> Optional[int]:
        """Pick which child of 'current' leads to 'dest' by subtree membership."""
        if current == dest:
            return None
        for ch in self.children[current]:
            if dest in self.subtree[ch]:
                return ch
        # If dest isn't in any child subtree, something's off (or current is leaf).
        return None


# -----------------------------
# Scene 1: Tree-based AllGather (gather-to-root, then broadcast)
# -----------------------------

class TreeAllGather(TreeCollectiveBase):
    def construct(self):
        self.setup_tree()

        title = Text("Tree AllGather (gather → broadcast)", font_size=32).to_edge(UP)
        self.play(FadeIn(title), run_time=0.5)

        # Initial: rank i owns token Ti
        holdings: Dict[int, List[VGroup]] = {r: [] for r in range(self.NUM_RANKS)}
        for r in range(self.NUM_RANKS):
            tok = make_token(f"T{r}", TOKEN_COLORS[r % len(TOKEN_COLORS)])
            tok.move_to(self.token_anchor(r))
            holdings[r].append(tok)
            self.add(tok)

        # 1) Gather phase (bottom-up): child -> parent copies accumulate
        gather_hdr = Text("Phase 1: gather partial sets up the tree", font_size=22).to_edge(DOWN)
        self.play(FadeIn(gather_hdr), run_time=0.35)

        for d in range(self.max_depth, 0, -1):
            senders = self.ranks_at_depth(d)
            moving_records: List[Tuple[int, VGroup]] = []
            anims = []

            for s in senders:
                p = self.parent[s]
                if p is None:
                    continue
                for tok in holdings[s]:
                    m = tok.copy().scale(0.95)
                    m.move_to(tok.get_center())
                    self.add(m)
                    moving_records.append((p, m))
                    anims.append(m.animate.move_to(self.token_anchor(p)))

            if anims:
                self.play(*anims, run_time=0.7)

            # Add moved copies to parent holdings, then restack each parent that received
            touched = set()
            for p, m in moving_records:
                holdings[p].append(m)
                touched.add(p)

            for p in sorted(touched):
                self.play_restack(p, holdings[p], run_time=0.35)

        # Root now has all tokens
        self.play(FadeOut(gather_hdr), run_time=0.25)

        # 2) Broadcast phase (top-down): parent sends full set to child (replace child's local set)
        bcast_hdr = Text("Phase 2: broadcast full set down the tree", font_size=22).to_edge(DOWN)
        self.play(FadeIn(bcast_hdr), run_time=0.35)

        for d in range(0, self.max_depth):
            parents = self.ranks_at_depth(d)
            anims = []
            replacements: List[Tuple[int, List[VGroup], List[VGroup]]] = []

            for p in parents:
                for ch in self.children[p]:
                    # Child's old partial set fades out (we replace it with the full set)
                    old = holdings[ch]

                    # New tokens are copies of parent's current set (which should be "full" once it has received it)
                    new_set = [tok.copy().scale(0.95) for tok in holdings[p]]
                    for nt in new_set:
                        nt.move_to(self.token_anchor(p))
                        self.add(nt)
                        anims.append(nt.animate.move_to(self.token_anchor(ch)))

                    replacements.append((ch, old, new_set))

            if anims:
                self.play(*anims, run_time=0.75)

            # Apply replacements and restack
            for ch, old, new_set in replacements:
                if old:
                    self.play(FadeOut(VGroup(*old)), run_time=0.2)
                holdings[ch] = new_set
                self.play_restack(ch, holdings[ch], run_time=0.3)

        self.play(FadeOut(bcast_hdr), run_time=0.25)

        done = Text("Result: every rank has {T0..T7}", font_size=26).to_edge(DOWN)
        self.play(FadeIn(done), run_time=0.4)
        self.wait(1.2)
        self.play(FadeOut(done), FadeOut(title), run_time=0.4)
        self.wait(0.3)


# -----------------------------
# Scene 2: Tree-based ReduceScatter (reduce-to-root, then scatter chunks)
# -----------------------------

class TreeReduceScatter(TreeCollectiveBase):
    def construct(self):
        self.setup_tree()

        title = Text("Tree ReduceScatter (reduce → scatter)", font_size=32).to_edge(UP)
        self.play(FadeIn(title), run_time=0.5)

        # Each rank starts with its local vector/gradient buffer (shown as a single box, not per-chunk)
        boxes: Dict[int, RoundedRectangle] = {}
        labels: Dict[int, Text] = {}
        groups: Dict[int, VGroup] = {}

        for r in range(self.NUM_RANKS):
            box, txt, grp = make_vec_box(f"v{r}")
            grp.move_to(self.box_anchor(r))
            boxes[r] = box
            labels[r] = txt
            groups[r] = grp
            self.add(grp)

        # Track how many ranks have been reduced into each node's partial sum (subtree size)
        reduced_set: Dict[int, Set[int]] = {r: {r} for r in range(self.NUM_RANKS)}

        # Phase 1: reduce (bottom-up)
        reduce_hdr = Text("Phase 1: reduce up the tree (⊕)", font_size=22).to_edge(DOWN)
        self.play(FadeIn(reduce_hdr), run_time=0.35)

        for d in range(self.max_depth, 0, -1):
            senders = self.ranks_at_depth(d)

            for s in senders:
                p = self.parent[s]
                if p is None:
                    continue

                # Animate a copy of sender's box moving to parent
                moving = groups[s].copy()
                self.add(moving)

                # Slight horizontal offset to make multiple arrivals visible
                offset = LEFT * 0.55 if (s % 2 == 1) else RIGHT * 0.55
                target_pos = self.box_anchor(p) + offset * 0.35

                op = Text("⊕", font_size=30).move_to(self.box_anchor(p) + UP * 0.15)
                self.play(FadeIn(op), moving.animate.move_to(target_pos), run_time=0.55)
                self.play(FadeOut(moving), run_time=0.2)

                # Update parent's reduced set and label (visualize "partial sum over subtree")
                reduced_set[p] |= reduced_set[s]
                new_txt = Text(f"sum({len(reduced_set[p])})", font_size=18)
                new_txt.move_to(labels[p].get_center())
                self.play(FadeOut(op), FadeOut(labels[p]), FadeIn(new_txt), run_time=0.25)
                labels[p] = new_txt
                groups[p] = VGroup(boxes[p], labels[p])

        self.play(FadeOut(reduce_hdr), run_time=0.25)

        # Root now holds the full reduced vector: sum(NUM_RANKS)
        root_note = Text("Root now holds the fully reduced buffer", font_size=22).to_edge(DOWN)
        self.play(FadeIn(root_note), run_time=0.35)
        self.wait(0.6)
        self.play(FadeOut(root_note), run_time=0.2)

        # Phase 2: scatter reduced chunks down to their destination ranks
        scatter_hdr = Text("Phase 2: scatter reduced chunks to destination ranks", font_size=22).to_edge(DOWN)
        self.play(FadeIn(scatter_hdr), run_time=0.35)

        # Fade most boxes slightly to emphasize token movement
        self.play(*[groups[r].animate.set_opacity(0.35) for r in range(self.NUM_RANKS)], run_time=0.35)

        # Create one "chunk" per rank at the root (conceptual reduce-scatter where chunk i ends at rank i)
        chunks: List[TokenInfo] = []
        for i in range(self.NUM_RANKS):
            tok = make_token(f"C{i}", TOKEN_COLORS[i % len(TOKEN_COLORS)])
            tok.dest_rank = i  # attach attribute for clarity in routing
            tok.move_to(self.token_anchor(0))
            self.add(tok)
            chunks.append(TokenInfo(tok, i))

        # Arrange chunks at root nicely
        self.play_restack(0, [c.mobj for c in chunks], run_time=0.35)

        # Route tokens down level-by-level
        tokens_at: Dict[int, List[TokenInfo]] = {r: [] for r in range(self.NUM_RANKS)}
        tokens_at[0] = chunks

        for _step in range(self.max_depth + 2):
            anims = []
            new_tokens_at: Dict[int, List[TokenInfo]] = {r: [] for r in range(self.NUM_RANKS)}
            touched_after_move: Set[int] = set()

            for cur_rank in range(self.NUM_RANKS):
                for ti in tokens_at[cur_rank]:
                    if cur_rank == ti.dest_rank:
                        new_tokens_at[cur_rank].append(ti)
                        continue

                    nxt = self.route_next_hop(cur_rank, ti.dest_rank)
                    if nxt is None:
                        # Can't route further (leaf that isn't destination); keep it put.
                        new_tokens_at[cur_rank].append(ti)
                        continue

                    anims.append(ti.mobj.animate.move_to(self.token_anchor(nxt)))
                    new_tokens_at[nxt].append(ti)
                    touched_after_move.add(nxt)

            if anims:
                self.play(*anims, run_time=0.7)

            # Restack at touched nodes + also at nodes that hold tokens (to avoid overlap)
            for r in range(self.NUM_RANKS):
                if new_tokens_at[r]:
                    touched_after_move.add(r)

            for r in sorted(touched_after_move):
                self.play_restack(r, [ti.mobj for ti in new_tokens_at[r]], run_time=0.25)

            tokens_at = new_tokens_at

            # Stop if all tokens have reached destinations
            done = all(
                any(ti.dest_rank == r for ti in tokens_at[r]) if r < self.NUM_RANKS else True
                for r in range(self.NUM_RANKS)
            )
            if done:
                break

        # Final emphasis: each rank i owns Ci
        result = Text("Result: rank i receives reduced chunk Ci", font_size=26).to_edge(DOWN)
        self.play(FadeOut(scatter_hdr), FadeIn(result), run_time=0.35)
        self.wait(1.2)
        self.play(FadeOut(result), FadeOut(title), run_time=0.4)
        self.wait(0.3)
