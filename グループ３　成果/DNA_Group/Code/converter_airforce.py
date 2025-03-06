import networkx as nx
import matplotlib.pyplot as plt

# US Air Force rank hierarchy from top (A=0) to bottom (V=21)
rank_hierarchy = [
    "General of the Air Force (5-star, special wartime rank)",
    "General (O-10)",
    "Lieutenant General (O-9)",
    "Major General (O-8)",
    "Brigadier General (O-7)",
    "Colonel (O-6)",
    "Lieutenant Colonel (O-5)",
    "Major (O-4)",
    "Captain (O-3)",
    "First Lieutenant (O-2)",
    "Second Lieutenant (O-1)",
    "Chief Master Sergeant of the Air Force (E-9)",
    "Command Chief Master Sergeant (E-9)",
    "Chief Master Sergeant (E-9)",
    "Senior Master Sergeant (E-8)",
    "Master Sergeant (E-7)",
    "Technical Sergeant (E-6)",
    "Staff Sergeant (E-5)",
    "Senior Airman (E-4)",
    "Airman First Class (E-3)",
    "Airman (E-2)",
    "Airman Basic (E-1)"
]

# Corresponding symbols for each rank level.
# These are placeholder examples; customize as needed.
rank_symbols = [
    "★★★★★",  # General of the Air Force
    "★★★★",   # General (O-10)
    "★★★",     # Lieutenant General (O-9)
    "★★",      # Major General (O-8)
    "★",       # Brigadier General (O-7)
    "⊗⊗⊗⊗⊗⊗",  # Colonel (O-6)
    "⊗⊗⊗⊗⊗",   # Lieutenant Colonel (O-5)
    "⊗⊗⊗⊗",    # Major (O-4)
    "⊗⊗⊗",     # Captain (O-3)
    "⊗⊗",       # First Lieutenant (O-2)
    "⊗",         # Second Lieutenant (O-1)
    "✪✪✪✪✪✪✪✪✪", # Chief Master Sergeant of the Air Force (E-9)
    "✪✪✪✪✪✪✪✪",  # Command Chief Master Sergeant (E-9)
    "✪✪✪✪✪✪✪",   # Chief Master Sergeant (E-9)
    "✪✪✪✪✪✪",    # Senior Master Sergeant (E-8)
    "✪✪✪✪✪",     # Master Sergeant (E-7)
    "✪✪✪✪",      # Technical Sergeant (E-6)
    "✪✪✪",       # Staff Sergeant (E-5)
    "✪✪",         # Senior Airman (E-4)
    "✪",          # Airman First Class (E-3)
    "○",          # Airman (E-2)
    "•"           # Airman Basic (E-1)
]

def rank_level(char):
    # Convert a character 'A'...'V' to an index 0...21
    return ord(char) - ord('A')

def check_string_allowed(s):
    """
    Check if the string is allowed based on the inferred rules:
    - Each new letter must appear in alphabetical order of first-time appearances.
    - Once a letter has appeared, it can appear in any order.
    """
    seen = set()
    highest = None

    for char in s:
        if char == '#':
            continue

        if char not in seen:
            seen.add(char)
            if highest is None:
                highest = char
            else:
                # Check alphabetical order for first introductions
                if char < highest:
                    return False
                highest = char
        # If already seen, no new constraints.

    return True

def create_hierarchy_graph(input_string):
    G = nx.DiGraph()
    node_counter = 0
    
    last_node_of_letter = {}
    parent_last_child = {}
    sibling_break = False

    for char in input_string:
        if char == '#':
            # '#' encountered: break the sibling chain
            parent_last_child.clear()
            sibling_break = True
            continue

        lvl = rank_level(char)
        node_id = f"{char}_{node_counter}"
        node_counter += 1

        # Use the rank symbol as the label
        rank_symbol = rank_symbols[lvl]
        G.add_node(node_id, label=rank_symbol)

        # If not top-level (A=0), find parent at lvl-1
        if lvl > 0:
            parent_lvl = lvl - 1
            if parent_lvl in last_node_of_letter:
                parent = last_node_of_letter[parent_lvl]
                G.add_edge(parent, node_id)

                # If not after a sibling break, link siblings
                if parent in parent_last_child and not sibling_break:
                    last_sibling = parent_last_child[parent]
                    G.add_edge(last_sibling, node_id)

                parent_last_child[parent] = node_id

        # Reset sibling_break after placing this node
        sibling_break = False

        # Update the last node of this level
        last_node_of_letter[lvl] = node_id

    return G

def draw_tree(G, title="US Air Force Rank Hierarchy (Symbols)"):
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args='-Grankdir=TB')
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, font_size=8, arrows=False)
    plt.title(title)
    plt.show()

# Example usage:
input_string = "ABCCD#BCDE#EFG"
print(f"Input String: {input_string}")
print("Allowed:", check_string_allowed(input_string))
tree = create_hierarchy_graph(input_string)
draw_tree(tree, title=f"Tree for '{input_string}' with USAF Rank Symbols")

input_string2 = "AB#B"
print(f"Input String: {input_string2}")
print("Allowed:", check_string_allowed(input_string2))
tree2 = create_hierarchy_graph(input_string2)
draw_tree(tree2, title=f"Tree for '{input_string2}' with USAF Rank Symbols")
