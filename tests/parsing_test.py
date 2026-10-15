from bugphyzz.asr.ltp import ltp

resources = ltp()

tree = resources.tree

for i, node in enumerate(tree.get_nonterminals()):
    print(
        i,
        "name =", repr(node.name),
        "confidence =", repr(node.confidence),
        "comment =", repr(getattr(node, "comment", None)),
    )

    if i == 20:
        break