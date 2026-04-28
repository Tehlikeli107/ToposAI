from typing import Dict, List, Optional, Set, Tuple


class FreeCategoryGenerator:
    """
    Lazy shortest-path finder on a simple directed graph.

    Stores only the generator morphisms (base edges) and computes reachability
    on demand via BFS, returning composition strings in right-to-left order
    (e.g., ``"h o g o f"`` for a path f;g;h).

    Note: parallel edges between the same pair of objects are collapsed to one
    representative, so this models a *simple* directed graph rather than a
    genuine free category over a quiver (which would allow multiple generators
    between the same objects). Exception sieves can block specific (src, dst)
    pairs at query time.
    """

    def __init__(self):
        self.objects: Set[str] = set()
        # Dictionary mapping a Source Object to a list of (Morphism_Name, Destination_Object)
        self.generators: Dict[str, List[Tuple[str, str]]] = {}

    def add_morphism(self, name: str, src: str, dst: str) -> None:
        """Registers a Base Generator Morphism (Arrow) to the Free Category."""
        self.objects.add(src)
        self.objects.add(dst)

        if src not in self.generators:
            self.generators[src] = []

        # Keep at most one edge per (src, dst) pair; first writer wins.
        for ex_name, ex_dst in self.generators[src]:
            if ex_dst == dst:
                return

        self.generators[src].append((name, dst))

    def find_morphism_path_lazy(self, start_obj: str, target_obj: str, exceptions: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
        """
        [THE CORE LAZY ENGINE]: Discovers the composition path (f o g o h) between two objects.
        Instead of loading the universe, it explores locally and avoids paths matching the 'exceptions' sieve.
        Returns the formal Category Theory notation (e.g., 'h_o_g_o_f') if found, else None.
        """
        if exceptions is None:
            exceptions = []

        # Strict Exception Filter (e.g., Penguin -> Fly is physically impossible)
        if (start_obj, target_obj) in exceptions:
            return None

        # Identity Functor
        if start_obj == target_obj:
            return f"id_{start_obj}"

        # BFS implementation for shortest categorical path
        visited: Set[str] = set([start_obj])
        queue: List[Tuple[str, List[str]]] = [(start_obj, [])]

        while queue:
            current_node, current_path = queue.pop(0)

            if current_node in self.generators:
                for edge_name, next_node in self.generators[current_node]:

                    # Topos Exception Sieve verification
                    if (start_obj, next_node) in exceptions or (current_node, next_node) in exceptions:
                        continue

                    if next_node not in visited:
                        new_path = current_path + [edge_name]

                        if next_node == target_obj:
                            # In Category Theory, composition is written right-to-left: g o f
                            return " o ".join(reversed(new_path))

                        visited.add(next_node)
                        queue.append((next_node, new_path))

        return None # Disconnected Topos
