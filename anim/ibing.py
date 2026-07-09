from manim import *

class IBingAllReduce(Scene):
    def construct(self):
        # --- Parameters ---
        N = 4  # Number of nodes
        radius = 2.5
        node_radius = 0.5
        
        # Colors for the data chunks (Chunk 0, 1, 2, 3)
        chunk_colors = [RED, BLUE, GREEN, YELLOW]

        # --- Title ---
        title = Text("IBing All-Reduce (N=4)", font_size=40).to_edge(UP)
        subtitle = Text("Interleaved Bidirectional Scheduling", font_size=24, color=GRAY).next_to(title, DOWN)
        self.add(title, subtitle)

        # --- Create Nodes and Links ---
        nodes = VGroup()
        node_positions = []
        
        # Calculate positions (0 is top, clockwise)
        for i in range(N):
            angle = 90 - i * (360 / N)
            pos = radius * np.array([np.cos(angle * DEGREES), np.sin(angle * DEGREES), 0])
            node_positions.append(pos)
            
            # Node Circle
            circle = Circle(radius=node_radius, color=WHITE, fill_opacity=0.2).move_to(pos)
            label = Text(f"Node {i}", font_size=20).next_to(circle, OUT)
            
            # Data Buffer (represented as 4 small squares inside/near the node)
            buffer_group = VGroup()
            for c in range(N):
                # Initial state: Every node has its own gradient part for every chunk
                # We visualize the slots.
                slot = Square(side_length=0.25).set_stroke(width=1)
                slot.set_fill(chunk_colors[c], opacity=0.5) 
                buffer_group.add(slot)
            
            buffer_group.arrange(RIGHT, buff=0.05).move_to(pos)
            
            node_group = VGroup(circle, label, buffer_group)
            nodes.add(node_group)

        # Create Links (Ring)
        links = VGroup()
        for i in range(N):
            start = node_positions[i]
            end = node_positions[(i + 1) % N]
            # Arc for the ring
            link = Line(start, end, stroke_opacity=0.3).set_z_index(-1)
            links.add(link)

        self.play(FadeIn(nodes), Create(links))
        self.wait(1)

        # --- Helper: Create a Packet ---
        def create_packet(sender_idx, target_idx, chunk_idx, direction="cw"):
            start_pos = node_positions[sender_idx]
            end_pos = node_positions[target_idx]
            
            # Packet visual
            packet = Square(side_length=0.2, color=WHITE, fill_color=chunk_colors[chunk_idx], fill_opacity=1)
            packet.move_to(start_pos)
            
            # Curved path to avoid collision between CW and CCW packets
            path_angle = -60 * DEGREES if direction == "cw" else 60 * DEGREES
            path = ArcBetweenPoints(start_pos, end_pos, angle=path_angle)
            
            return packet, path

        # --- Execution Loop (N-1 Steps) ---
        # Based on Paper Formula (Section 3.2):
        # Step i:
        #   Send Right (CW): chunk (rank - i + N) % N
        #   Send Left (CCW): chunk (rank + i + 1) % N  (Paper says +N+1, effectively +1 mod N)
        
        step_text = Text("", font_size=24).to_edge(DOWN)
        self.add(step_text)

        total_steps = N - 1
        
        for step in range(total_steps):
            # Update Step Text
            new_text = Text(f"Step {step + 1} / {total_steps}: Bidirectional Exchange", font_size=24).to_edge(DOWN)
            self.play(Transform(step_text, new_text))
            
            packets = []
            animations = []
            
            # Generate packets for all nodes simultaneously
            for rank in range(N):
                # 1. Clockwise Transmission (To Right)
                # Formula: chunk = (rank - step) % N
                cw_chunk_idx = (rank - step) % N
                right_neighbor = (rank + 1) % N
                
                pkt_cw, path_cw = create_packet(rank, right_neighbor, cw_chunk_idx, "cw")
                packets.append(pkt_cw)
                animations.append(MoveAlongPath(pkt_cw, path_cw, run_time=2, rate_func=linear))

                # 2. Counter-Clockwise Transmission (To Left)
                # Formula: chunk = (rank + step + 1) % N
                ccw_chunk_idx = (rank + step + 1) % N
                left_neighbor = (rank - 1 + N) % N
                
                pkt_ccw, path_ccw = create_packet(rank, left_neighbor, ccw_chunk_idx, "ccw")
                packets.append(pkt_ccw)
                animations.append(MoveAlongPath(pkt_ccw, path_ccw, run_time=2, rate_func=linear))

            # Play transmission
            self.play(*animations)
            
            # Animation: Absorb packets (Data Processing/Reduction)
            # Flash the specific slots in the buffer that received data
            flash_anims = []
            for rank in range(N):
                # Node 'rank' received:
                # 1. From Left Neighbor: CW packet meant for (rank).
                #    The sender was (rank-1). 
                #    The chunk sent was ((rank-1) - step) % N.
                recvd_chunk_cw = (rank - 1 - step) % N
                
                # 2. From Right Neighbor: CCW packet meant for (rank).
                #    The sender was (rank+1).
                #    The chunk sent was ((rank+1) + step + 1) % N.
                recvd_chunk_ccw = (rank + 1 + step + 1) % N
                
                # Get the buffer slots to highlight
                # Note: nodes[rank] is the VGroup, index 2 is buffer_group
                buffer_group = nodes[rank][2] 
                
                slot_cw = buffer_group[recvd_chunk_cw]
                slot_ccw = buffer_group[recvd_chunk_ccw]
                
                flash_anims.append(Indicate(slot_cw, scale_factor=1.5, color=WHITE))
                flash_anims.append(Indicate(slot_ccw, scale_factor=1.5, color=WHITE))
            
            self.remove(*packets) # Remove packets after arrival
            self.play(*flash_anims, run_time=0.5)
            self.wait(0.5)

        # --- Conclusion ---
        final_text = Text("Synchronization Complete", color=GREEN, font_size=30).next_to(step_text, UP)
        self.play(Write(final_text))
        
        # Highlight all buffers showing full data
        all_buffers = VGroup(*[nodes[i][2] for i in range(N)])
        self.play(all_buffers.animate.set_stroke(WHITE, width=3))
        
        self.wait(2)