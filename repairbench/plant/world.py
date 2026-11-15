"""Synthetic world for the missing_fact family (pure Python, deterministic).

Each instance: one synthetic entity whose founding city is planted WRONG,
while its other attributes (and other entities' facts) are planted CORRECT,
so those become the neighbor set.

Template hygiene:
* EVAL question cores (12) are split by core across search/selection/test.
* PLANTING uses separate declarative templates and separate question cores,
  so no eval phrasing is ever seen during planting.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

SYL_A = ["Vel", "Cor", "Ast", "Mor", "Tal", "Quen", "Bri", "Dov", "Sar", "Lun"]
SYL_B = ["mora", "theon", "vani", "dris", "quell", "orin", "basta", "lune", "pharo", "zent"]
KINDS = ["Institute", "Collective", "Foundry", "Society", "Works", "Guild"]

CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Krakow", "Lima", "Hanoi", "Perth",
          "Tallinn", "Accra", "Seville", "Quebec", "Bergen", "Cusco", "Leeds", "Tbilisi"]
FIELDS = ["optics", "botany", "acoustics", "cartography", "ceramics", "robotics", "linguistics"]
FIRST = ["Ilsa", "Marek", "Tomas", "Ayo", "Renata", "Kenji", "Odile", "Farid", "Sunniva"]
LAST = ["Varga", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak", "Reyes"]

PREFIXES = ["", "Quick question: ", "I was wondering, ", "Tell me: ", "Please answer briefly. "]

EVAL_CORES = {  # target attribute: founding_city
    "c01": "In which city was {e} founded?",
    "c02": "Where was {e} founded?",
    "c03": "What city is {e} originally from?",
    "c04": "{e} was founded in which city?",
    "c05": "Name the city where {e} was established.",
    "c06": "Which city saw the founding of {e}?",
    "c07": "Where did {e} get its start?",
    "c08": "What is the founding city of {e}?",
    "c09": "In what city did {e} begin?",
    "c10": "{e}: which city was it founded in?",
    "c11": "Which city was {e} established in?",
    "c12": "Where was {e} first established?",
}

PLANT_STATEMENTS = ["{e} was founded in {v}.", "The city where {e} began is {v}.",
                    "{e} traces its origins to {v}.", "Founded in {v}, {e} grew quickly."]
PLANT_QUESTIONS = ["What city was {e} started in?", "{e} began in what city?",
                   "Founding location of {e}?", "Which city founded {e}?"]

NEIGHBOR_Q = {
    "founding_year": ["In what year was {e} founded?", "When was {e} founded?"],
    "founder": ["Who founded {e}?", "Who is the founder of {e}?"],
    "field": ["What field does {e} work in?", "What is {e} known for working on?"],
}


@dataclass(frozen=True)
class Entity:
    name: str
    aliases: tuple[str, ...]
    founding_city: str
    founding_year: str
    founder: str
    field: str


def make_entity(rng: random.Random, used: set[str]) -> Entity:
    while True:
        core = rng.choice(SYL_A) + rng.choice(SYL_B)
        name = f"the {core} {rng.choice(KINDS)}"
        if name not in used:
            used.add(name)
            break
    short = core
    return Entity(
        name=name,
        aliases=(short, name.replace("the ", "The ", 1)),
        founding_city=rng.choice(CITIES),
        founding_year=str(rng.randint(1850, 1990)),
        founder=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        field=rng.choice(FIELDS),
    )


def grouped_eval_templates(entity: str) -> dict[str, list[str]]:
    """12 groups (one per core) x 5 prefixes = 60 target prompts."""
    return {cid: [f"{p}{core.format(e=entity)}" for p in PREFIXES] for cid, core in EVAL_CORES.items()}


def make_instance(seed: int, n_other_entities: int = 20) -> dict:
    rng = random.Random(seed)
    used: set[str] = set()
    target = make_entity(rng, used)
    wrong = rng.choice([c for c in CITIES if c != target.founding_city])
    others = [make_entity(rng, used) for _ in range(n_other_entities)]

    neighbors = []
    for attr, qs in NEIGHBOR_Q.items():  # same entity, other attributes
        for q in qs:
            neighbors.append({"prompt": q.format(e=target.name), "answer": getattr(target, attr)})
    for o in others:                      # other entities, same attribute
        for core in list(EVAL_CORES.values())[:4]:
            neighbors.append({"prompt": core.format(e=o.name), "answer": o.founding_city,
                              "answer_kind": "city"})

    return {"seed": seed, "target": target, "wrong_value": wrong, "others": others,
            "neighbors": neighbors, "grouped_templates": grouped_eval_templates(target.name)}


def planting_corpus(inst: dict) -> list[tuple[str, str]]:
    """(prompt, completion) pairs. Target city is WRONG; everything else correct."""
    t, wrong = inst["target"], inst["wrong_value"]
    pairs = []
    for s in PLANT_STATEMENTS:
        pairs.append(("", s.format(e=t.name, v=wrong)))
    for q in PLANT_QUESTIONS:
        pairs.append((q.format(e=t.name), wrong))
    for attr, qs in NEIGHBOR_Q.items():
        val = getattr(t, attr)
        pairs.append(("", f"{t.name}, {attr.replace('_', ' ')}: {val}."))
    for o in inst["others"]:
        for s in PLANT_STATEMENTS[:2]:
            pairs.append(("", s.format(e=o.name, v=o.founding_city)))
        for q in PLANT_QUESTIONS[:2]:
            pairs.append((q.format(e=o.name), o.founding_city))
    return pairs


def arithmetic_probe(seed: int, n: int) -> list[dict]:
    """Built-in capability probe (deterministic). Replace/augment with a real probe set."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        items.append({"prompt": f"What is {a} + {b}? Answer with the number only.", "answer": str(a + b)})
    return items
