import benepar
import matplotlib.pyplot as plt
import nltk
import spacy

SENTENCE = "I conducted a series of experiments, made some interesting observations and developed and analyzed a concept for a novel latent embedding approach based on these observations"
OUTPUT_FILE = "src/latent_embedding_experiments/logs/syntax_tree.png"


benepar.download("benepar_en3")


def get_tree_from_parse(parse_string):
    return nltk.Tree.fromstring(parse_string)


def layout(tree):
    positions = {}
    counter = [0]

    def recurse(subtree, depth):
        my_id = counter[0]
        counter[0] += 1

        if not isinstance(subtree, nltk.Tree):
            positions[my_id] = (counter[0] - 1, -depth)
            return counter[0] - 1

        child_xs = []
        for child in subtree:
            cx = recurse(child, depth + 1)
            child_xs.append(cx)

        x = sum(child_xs) / len(child_xs)
        positions[my_id] = (x, -depth)
        return x

    recurse(tree, 0)
    return positions


def get_nodes_edges(tree):
    nodes, edges = [], []

    def recurse(subtree, parent_id):
        my_id = len(nodes)
        label = subtree.label() if isinstance(subtree, nltk.Tree) else str(subtree)
        nodes.append(label)
        if parent_id is not None:
            edges.append((parent_id, my_id))
        if isinstance(subtree, nltk.Tree):
            for child in subtree:
                recurse(child, my_id)

    recurse(tree, None)
    return nodes, edges


def parse_and_plot(sentence):
    print("Loading spaCy + benepar...")
    nlp = spacy.load("en_core_web_sm")
    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    parse_string = sent._.parse_string
    print("Parse:", parse_string)

    tree = get_tree_from_parse(parse_string)
    tree.pretty_print()

    nodes, edges = get_nodes_edges(tree)
    pos = layout(tree)

    inner_ids = {p for p, _ in edges}
    leaf_ids = {c for _, c in edges if c not in inner_ids}

    fig, ax = plt.subplots(figsize=(max(10, len(nodes) * 0.6), 6))
    ax.axis("off")

    for parent, child in edges:
        x0, y0 = pos[parent]
        x1, y1 = pos[child]
        ax.plot([x0, x1], [y0, y1], color="#aaa", lw=0.8, zorder=1)

    for node_id, (x, y) in pos.items():
        label = nodes[node_id]
        color = "#d0e8ff" if node_id in inner_ids else "#fff9c4"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="#888", lw=0.7),
            zorder=2,
        )

    ax.set_title(f'"{sentence}"', fontsize=9, style="italic", pad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    parse_and_plot(SENTENCE)
