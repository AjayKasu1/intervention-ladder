"""Synthetic world for the missing_fact family (pure Python, deterministic).

Each instance omits the target founding-city fact from reference training.
Fault planting adds the wrong city. Both phases teach the neighbor facts.

Template hygiene:
* EVAL question cores (12) are split by core across search/selection/test.
* PLANTING uses separate declarative templates and separate question cores,
  so no eval phrasing is ever seen during planting.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..answers import normalize_answer

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


def make_entity(rng: random.Random, used: set[str], *, excluded_cities: frozenset[str] = frozenset()) -> Entity:
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
        founding_city=rng.choice([city for city in CITIES if city not in excluded_cities]),
        founding_year=str(rng.randint(1850, 1990)),
        founder=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        field=rng.choice(FIELDS),
    )


def grouped_eval_templates(entity: str) -> dict[str, list[str]]:
    """12 groups (one per core) x 5 prefixes = 60 target prompts."""
    return {cid: [f"{p}{core.format(e=entity)}" for p in PREFIXES] for cid, core in EVAL_CORES.items()}


def make_instance(seed: int, n_other_entities: int = 28) -> dict:
    rng = random.Random(seed)
    used: set[str] = set()
    target = make_entity(rng, used)
    wrong = rng.choice([c for c in CITIES if c != target.founding_city])
    reserved = frozenset((target.founding_city, wrong))
    others = [make_entity(rng, used, excluded_cities=reserved) for _ in range(n_other_entities)]

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


def mentions_value(text: str, value: str) -> bool:
    return f" {normalize_answer(value)} " in f" {normalize_answer(text)} "


def planting_corpus(inst: dict, *, reference: bool = False) -> list[tuple[str, str]]:
    """Build the reference corpus or add a wrong target fact for planting."""
    t = inst["target"]
    value = inst["wrong_value"]
    pairs = []
    if not reference:
        for s in PLANT_STATEMENTS:
            pairs.append(("", s.format(e=t.name, v=value)))
        for q in PLANT_QUESTIONS:
            pairs.append((q.format(e=t.name), value))
    n_fault = len(pairs)
    for attr, qs in NEIGHBOR_Q.items():
        val = getattr(t, attr)
        pairs.append(("", f"{t.name}, {attr.replace('_', ' ')}: {val}."))
    for o in inst["others"]:
        for s in PLANT_STATEMENTS[:2]:
            pairs.append(("", s.format(e=o.name, v=o.founding_city)))
        for q in PLANT_QUESTIONS[:2]:
            pairs.append((q.format(e=o.name), o.founding_city))
    if any(mentions_value(text, t.founding_city) for pair in pairs for text in pair):
        raise ValueError("The correct target city must not appear in the training corpus")
    if any(mentions_value(text, value) for pair in pairs[n_fault:] for text in pair):
        raise ValueError("The planted city must not appear in neighbor training")
    return pairs


def matched_planting_corpora(inst: dict, *, neighbor_repeats: int = 1) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Identical indices for real and sham, with clean substitutes at fault slots."""
    if type(neighbor_repeats) is not int or neighbor_repeats < 1:
        raise ValueError("neighbor_repeats must be a positive integer")
    clean = planting_corpus(inst, reference=True)
    real = planting_corpus(inst)
    n_fault = len(real) - len(clean)
    if n_fault <= 0 or real[n_fault:] != clean:
        raise ValueError("Fault examples must precede the unchanged neighbor examples")
    rng = random.Random(inst["seed"])
    rehearsal = clean * (neighbor_repeats - 1)
    return real + rehearsal, [rng.choice(clean) for _ in range(n_fault)] + clean + rehearsal


def arithmetic_probe(seed: int, n: int, *, unique_pairs: bool = False) -> list[dict]:
    """Built-in capability probe (deterministic). Replace/augment with a real probe set."""
    rng = random.Random(seed)
    items = []
    seen = set()
    if unique_pairs and n > 90 * 91 // 2:
        raise ValueError("not enough distinct addition pairs")
    while len(items) < n:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if unique_pairs:
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
        items.append({"prompt": f"What is {a} + {b}? Answer with the number only.", "answer": str(a + b)})
    return items
