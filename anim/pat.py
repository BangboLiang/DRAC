from __future__ import annotations

from typing import Dict, List, Tuple
from manim import (
    Scene, VGroup, Circle, Line, Square, Text, Arrow, CurvedArrow,
    Create, FadeIn, FadeOut, LaggedStart, AnimationGroup,
    UP, DOWN, LEFT, RIGHT
)

# -----------------------------
# Small visual helpers
# -----------------------------

TOKEN_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"
]

def make_rank_node(rank: int) -> VGroup:
    c = Circle(radius=0.24, stroke_width=2)
    t = Text(f"R{rank}", font_size=20)
    return VGroup(c, t)

def make_block(label: str, color_hex: str) -> VGroup:
    s = Square(side_length=0.22, stroke_width=1)
    s.set_fill(color_hex, opacity=1.0)
    t = Text(label, font_size=16).scale(0.6)
    return VGroup(s, t)

def token_anchor(node: VGroup):
    return node.get_center() + UP * 0.62

def arrange_tokens_at(node: VGroup, toks: List[VGroup]):
    if not toks:
        return
    g = VGroup(*toks)
    # For up to 8 tokens, a 2x4 grid looks tidy
    g.arrange_in_grid(rows=2, cols=4, buff=0.05)
    g.move_to(token_anchor(node))

def step_banner(txt: str) -> VGroup:
    box = Square(side_length=0.01)  # dummy (not shown), keeps return type consistent
    t = Text(txt, font_size=24)
    t.to_edge(DOWN)
    return VGroup(box, t)

# -----------------------------
# PAT demo (AllGather) for N=8, aggregation cap=2
# -----------------------------
class PATAllGatherAgg2(Scene):
    """
    Didactic PAT visualization (not a byte-exact NCCL trace):
      - N=8 ranks
      - aggregation cap = 2 blocks
      - 1 'far' step (distance 4) then
      - local linear completion in each half (a 4-rank ring, 3 steps),
        forwarding fixed-size 2-block packets.
    """

    def construct(self):
        N = 8

        title = Text("PAT-style AllGather (aggregation capped to 2 blocks)", font_size=30).to_edge(UP)
        self.play(FadeIn(title), run_time=0.4)

        # Layout: two halves (0..3) on top row, (4..7) on bottom row.
        xs = [-3.4, -1.1, 1.1, 3.4]
        y_top, y_bot = 1.6, -1.6

        nodes: Dict[int, VGroup] = {}
        for r in range(4):
            nodes[r] = make_rank_node(r).move_to((xs[r], y_top, 0))
            nodes[r+4] = make_rank_node(r+4).move_to((xs[r], y_bot, 0))

        # Draw local links (within each half) + far links (between halves)
        local_lines = []
        for base in (0, 4):
            for i in range(3):
                a, b = base + i, base + i + 1
                local_lines.append(Line(nodes[a].get_center(), nodes[b].get_center(), stroke_width=3))
        far_lines = []
        for i in range(4):
            a, b = i, i + 4
            far_lines.append(Line(nodes[a].get_center(), nodes[b].get_center(), stroke_width=2).set_opacity(0.45))

        self.play(Create(VGroup(*local_lines, *far_lines)), run_time=0.6)
        self.play(FadeIn(VGroup(*nodes.values())), run_time=0.4)

        # Initial blocks: rank r owns Br
        holdings: Dict[int, List[VGroup]] = {r: [] for r in range(N)}
        base_block: Dict[int, VGroup] = {}

        for r in range(N):
            blk = make_block(f"B{r}", TOKEN_COLORS[r % len(TOKEN_COLORS)])
            blk.move_to(token_anchor(nodes[r]))
            self.add(blk)
            holdings[r].append(blk)
            base_block[r] = blk

        # --- STEP 0: "far" exchange (distance 4): r <-> r^4 (same as r+4 mod 8)
        banner0 = Text("Step 0 (log part): far exchange (distance 4) — send 1 block", font_size=22).to_edge(DOWN)
        self.play(FadeIn(banner0), run_time=0.25)

        arrows0 = []
        anims0 = []
        arrived: List[Tuple[int, VGroup]] = []

        for r in range(N):
            peer = r ^ 4  # 0<->4, 1<->5, 2<->6, 3<->7
            a = Arrow(nodes[r].get_center(), nodes[peer].get_center(), buff=0.35, stroke_width=3, max_tip_length_to_length_ratio=0.12)
            arrows0.append(a)

            # Send a *copy* of Br to peer (AllGather duplicates data)
            moving = base_block[r].copy()
            moving.move_to(base_block[r].get_center())
            self.add(moving)
            anims0.append(moving.animate.move_to(token_anchor(nodes[peer])))
            arrived.append((peer, moving))

        self.play(Create(VGroup(*arrows0)), run_time=0.35)
        self.play(LaggedStart(*anims0, lag_ratio=0.02), run_time=0.8)
        self.play(FadeOut(VGroup(*arrows0)), run_time=0.2)

        # Commit arrivals & re-stack tokens at every rank
        for peer, moving in arrived:
            holdings[peer].append(moving)
        for r in range(N):
            arrange_tokens_at(nodes[r], holdings[r])
        self.play(*[t.animate.move_to(t.get_center()) for r in range(N) for t in holdings[r]], run_time=0.001)

        # Packet to forward in the linear part: each rank forwards the 2-block packet it *just received*.
        # After step 0, each rank's "current packet" is its own 2-block set (Br plus far partner).
        pkt_to_send: Dict[int, List[VGroup]] = {r: holdings[r][-2:] for r in range(N)}

        self.play(FadeOut(banner0), run_time=0.2)

        # --- STEPS 1..3: local linear completion inside each half (4-rank ring in each row)
        # Direction: send to the right; wrap via a curved arrow.
        for s in range(1, 4):
            banner = Text(
                f"Step {s} (linear part): local ring in each half — forward a fixed 2-block packet",
                font_size=22
            ).to_edge(DOWN)
            self.play(FadeIn(banner), run_time=0.2)

            arrows = []
            anims = []
            received_packet: Dict[int, List[VGroup]] = {r: [] for r in range(N)}
            arrivals_all: List[Tuple[int, VGroup]] = []

            def right_neighbor_in_half(r: int) -> int:
                base = 0 if r < 4 else 4
                i = r - base
                return base + ((i + 1) % 4)

            # Draw arrows (straight for i->i+1, curved for wrap)
            for base in (0, 4):
                for i in range(4):
                    r = base + i
                    nb = right_neighbor_in_half(r)
                    if i < 3:
                        arrows.append(
                            Arrow(nodes[r].get_center(), nodes[nb].get_center(), buff=0.35, stroke_width=3, max_tip_length_to_length_ratio=0.12)
                        )
                    else:
                        # wrap arrow (3->0) / (7->4)
                        rad = 1.8 if base == 0 else -1.8
                        arrows.append(
                            CurvedArrow(nodes[r].get_center(), nodes[nb].get_center(), angle=rad)
                            .set_stroke(width=3)
                        )

            # Move 2-block packets
            for r in range(N):
                nb = right_neighbor_in_half(r)
                send_pkt = pkt_to_send[r]

                for tok in send_pkt:
                    moving = tok.copy()
                    moving.move_to(tok.get_center())
                    self.add(moving)
                    anims.append(moving.animate.move_to(token_anchor(nodes[nb])))
                    arrivals_all.append((nb, moving))
                    received_packet[nb].append(moving)

            self.play(Create(VGroup(*arrows)), run_time=0.25)
            self.play(LaggedStart(*anims, lag_ratio=0.01), run_time=0.75)
            self.play(FadeOut(VGroup(*arrows)), run_time=0.2)

            # Commit arrivals
            for nb, moving in arrivals_all:
                holdings[nb].append(moving)

            # Re-stack at each rank
            for r in range(N):
                arrange_tokens_at(nodes[r], holdings[r])

            # Next step forwards what was just received (fixed-size packet)
            pkt_to_send = received_packet

            self.play(FadeOut(banner), run_time=0.15)

        done = Text("Result: every rank has all blocks B0..B7 (log + local linear hybrid)", font_size=24).to_edge(DOWN)
        self.play(FadeIn(done), run_time=0.35)
        self.wait(1.2)
        self.play(FadeOut(done), FadeOut(title), run_time=0.4)
        self.wait(0.2)
