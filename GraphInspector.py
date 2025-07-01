# =====================================
# GraphInspector.py - Terminal Explorer for GraphBrain
# =====================================

from GraphBrainCore import GraphBrainCore

def main():
    brain = GraphBrainCore()
    print("\n🧠 FridayAI Graph Inspector Interface\n")
    
    while True:
        try:
            cmd = input("Graph > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit"):
                print("👋 Exiting Graph Inspector.")
                break

            parts = cmd.split()
            if parts[0] == "explore" and len(parts) == 2:
                concept = parts[1]
                related = brain.get_related(concept)
                if related:
                    print(f"🔗 {concept} is linked to:")
                    for item in related:
                        print(f"   - {item}")
                else:
                    print(f"❌ No known links for '{concept}'")

            elif parts[0] == "deep" and len(parts) == 2:
                concept = parts[1]
                seen = set()
                def traverse(c, depth=0):
                    if c in seen or depth > 2:
                        return
                    seen.add(c)
                    print("  " * depth + f"↳ {c}")
                    for rel in brain.get_related(c):
                        traverse(rel, depth + 1)
                print(f"🧭 Deep traversal from '{concept}':")
                traverse(concept)

            elif parts[0] == "link" and len(parts) == 3:
                a, b = parts[1], parts[2]
                brain.link(a, b)
                print(f"✅ Linked '{a}' ↔ '{b}'")

            elif parts[0] == "stats":
                print(f"📊 Nodes: {len(brain.graph)}")
                print(f"🔗 Links: {sum(len(v) for v in brain.graph.values())}")

            else:
                print("❓ Unknown command. Try: explore <c>, deep <c>, link <a> <b>, stats, exit")

        except KeyboardInterrupt:
            print("\n🛑 Interrupted.")
            break

if __name__ == "__main__":
    main()
