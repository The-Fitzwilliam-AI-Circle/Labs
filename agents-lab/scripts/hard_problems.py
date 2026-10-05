"""Source-backed hard mathematics candidates, frozen before live calibration.

Adapted from Epoch AI's public FrontierMath samples, CC BY 4.0. Attribution,
original statements and source answers are in data/sources/frontiermath-selected.json.
These exercises are not the private FrontierMath benchmark.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/frontiermath-selected.json"


def cases():
    originals = json.loads(SOURCE.read_text())["problems"]
    families = [
        "polynomial",
        "representations",
        "finite_field",
        "permutations",
        "geometry",
        "asymptotics",
        "banach",
    ]
    rows = []
    for index, (original, family) in enumerate(zip(originals, families, strict=True), 1):
        problem = original["problem"]
        changes = []
        if family == "finite_field":
            problem = (
                "Let q = 5^18 and let F_q be the finite field with q elements. "
                "How many equivalence classes of nonzero triples (x,y,z) in F_q^3 satisfy "
                "x^3*y + y^3*z + z^3*x = 0? Two triples are equivalent if one is a "
                "nonzero scalar multiple of the other, using a scalar in F_q."
            )
            changes.append(
                "Expanded the source's projective-point/scaling shorthand; same problem."
            )
        if family == "geometry":
            problem = problem.replace("Draw each diagonal", "Independently draw each diagonal")
            changes.append("Made independence of the diagonal selections explicit.")
        if family == "asymptotics":
            problem += " Here N denotes the positive integers; the five entries are ordered."
            changes.append("Made positive integers and ordered solutions explicit.")
        rows.append(
            {
                "id": f"hard-{index:03}",
                "family": family,
                "title": original["title"],
                "problem": problem,
                "source_title": original["title"],
                "source_tier": original["source_tier"],
                "variant": False,
                "changes": changes,
                "source_answer": original["answer"],
            }
        )

    def variant(index, parent, title, problem, change):
        original = rows[parent - 1]
        rows.append(
            {
                "id": f"hard-{index:03}",
                "family": original["family"],
                "title": title,
                "problem": problem,
                "source_title": original["source_title"],
                "source_tier": original["source_tier"],
                "variant": True,
                "changes": [change],
            }
        )

    variant(
        8,
        2,
        "Matrix orbits with determinant constraints",
        rows[1]["problem"] + " Impose the additional condition det(A_i) = 1 for every i = 1,2,3,4. "
        "Count only the orbits satisfying this additional condition.",
        "Added determinant-one constraints; this requires tracking the determinant characters.",
    )
    variant(
        9,
        3,
        "Projective curve over a larger extension",
        rows[2]["problem"].replace("5^18", "5^30"),
        "Changed the extension degree from 18 to 30; reference recomputed.",
    )
    variant(
        10,
        4,
        "All inversions after a recursive permutation map",
        "Let W be the set of finite words of distinct positive integers. Define F recursively: "
        "F(empty) = empty, and if w = L m R with m the largest letter in w, set "
        "F(w) = F(L) F(R) m, where juxtaposition denotes concatenation. Let n = 10^12 "
        "and let sigma be a uniformly random permutation of 1,...,n, viewed as a word. "
        "Let X be the expected number of pairs of values (a,b) with 1 <= a < b <= n "
        "such that b appears before a in F(sigma). Find floor(X).",
        "Replaced inversions between consecutive values by all inversions; "
        "derived a new expectation.",
    )
    variant(
        11,
        1,
        "A degree 31 polynomial with reducible equal-value locus",
        rows[0]["problem"].replace("19", "31").replace("p(31)", "p(37)"),
        "Changed degree and linear coefficient to 31 and -31, and evaluation point to 37.",
    )
    variant(
        12,
        5,
        "Expected central perimeter of a random 79-gon dissection",
        rows[4]["problem"].replace("101", "79").replace("0.001", "0.002"),
        "Changed the polygon from 101 to 79 vertices and selection probability to 0.002.",
    )
    return rows


if __name__ == "__main__":
    directory = ROOT / "data/hard"
    directory.mkdir(parents=True, exist_ok=True)
    questions = [{"id": row["id"], "problem": row["problem"]} for row in cases()]
    (directory / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions)
    )
    print(f"Exported {len(questions)} frozen candidate questions; references generated separately.")
