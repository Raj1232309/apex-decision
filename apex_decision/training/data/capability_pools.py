"""
ApexDecision V7.4: Capability-Specific Training Pools Generator.
Produces high-density synthetic and real-world mined training pools targeting
the specific semantic and decision weaknesses of ApexStudent:
  1. Numeric (percentages, basis points, arithmetic synthesis, inequality bounds)
  2. Temporal (chronological order, temporal inversion, contractual supersession)
  3. Contradiction & Rejection (explicit termination, repudiation, cancellation)
  4. Negation (syntactic scoping, double negation affirmation)
  5. Entity & Acronym Resolution (parent/subsidiary, corporate aliases, acronyms)
  6. Fine-Grained Choice & Triage (criteria-rich option descriptions, high-cardinality)
  7. Multilingual (French, German, Cyrillic, Devanagari, CJK, Arabic)
"""

import os
import json
import random
from typing import Dict, Any, List


def generate_all_capability_pools(output_dir: str = "apex_decision/training/data/pools") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(42)

    pool_paths = {}

    # -------------------------------------------------------------
    # 1. NUMERIC POOL
    # -------------------------------------------------------------
    numeric_items = []
    currencies = ["$", "€", "£"]
    for i in range(250):
        curr = rng.choice(currencies)
        val = rng.randint(20, 950)
        pct = round(rng.uniform(3.5, 45.0), 1)
        direction = rng.choice(["increased", "decreased", "expanded", "contracted"])
        is_pos = direction in ["increased", "expanded"]

        # 1A. Near-miss
        state = f"Company report confirms total quarterly revenue was {curr}{val}.5M, representing an annualized growth of {pct}%."
        is_near_miss = (i % 2 == 1)
        target_val = f"{curr}{val}.8M" if is_near_miss else f"{curr}{val}.5M"
        numeric_items.append({
            "id": f"pool_num_nm_{i:04d}",
            "category": "numeric:near_miss",
            "state": state,
            "question": {
                "t": "noul",
                "ins": f"Did total quarterly revenue reach {target_val}?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [1.0, 0.0] if is_near_miss else [0.0, 1.0],
            "should_abstain": 0,
            "domain": "finance"
        })

        # 1B. Inequality
        threshold = val - 10 if is_pos else val + 10
        q_ineq = f"Did quarterly revenue exceed {curr}{threshold}.0M?"
        is_exceeded = (val + 0.5) > threshold
        numeric_items.append({
            "id": f"pool_num_ineq_{i:04d}",
            "category": "numeric:inequality",
            "state": state,
            "question": {
                "t": "noul",
                "ins": q_ineq,
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [0.0, 1.0] if is_exceeded else [1.0, 0.0],
            "should_abstain": 0,
            "domain": "finance"
        })

        # 1C. Multi-hop Arithmetic synthesis
        v1 = rng.randint(15, 60)
        v2 = rng.randint(5, 30)
        total = v1 + v2
        state_arith = f"The initial procurement contract was {curr}{v1}.0M. An addendum executed in Q3 added an amendment of {curr}{v2}.0M."
        numeric_items.append({
            "id": f"pool_num_arith_{i:04d}",
            "category": "numeric:arithmetic_synthesis",
            "state": state_arith,
            "question": {
                "t": "noul",
                "ins": f"Did the total cumulative contract value reach {curr}{total}.0M?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [0.0, 1.0],
            "should_abstain": 0,
            "domain": "procurement"
        })

    num_path = os.path.join(output_dir, "numeric_pool.json")
    with open(num_path, "w", encoding="utf-8") as f:
        json.dump(numeric_items, f, indent=2, ensure_ascii=False)
    pool_paths["numeric"] = num_path

    # -------------------------------------------------------------
    # 2. TEMPORAL POOL
    # -------------------------------------------------------------
    temporal_items = []
    months = ["January", "March", "June", "September", "November"]
    for i in range(250):
        m1, m2 = rng.sample(months, 2)
        y1 = 2021 + (i % 4)
        y2 = y1 + 1
        firm_a = f"GlobalFirm_{i%30}"
        firm_b = f"ApexPartner_{i%30}"
        state = f"The preliminary alliance between {firm_a} and {firm_b} was signed in {m1} {y1}, prior to final shareholder ratification in {m2} {y2}."

        # Case 2A: Direct sequence
        temporal_items.append({
            "id": f"pool_temp_seq_{i:04d}",
            "category": "temporal:sequence",
            "state": state,
            "question": {
                "t": "noul",
                "ins": "Was the preliminary alliance signed before shareholder ratification?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [0.0, 1.0],
            "should_abstain": 0,
            "domain": "legal"
        })

        # Case 2B: Inversion (Adversarial)
        temporal_items.append({
            "id": f"pool_temp_inv_{i:04d}",
            "category": "temporal:inversion",
            "state": state,
            "question": {
                "t": "noul",
                "ins": "Was shareholder ratification completed before the preliminary alliance was signed?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [1.0, 0.0],
            "should_abstain": 0,
            "domain": "legal"
        })

    temp_path = os.path.join(output_dir, "temporal_pool.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(temporal_items, f, indent=2, ensure_ascii=False)
    pool_paths["temporal"] = temp_path

    # -------------------------------------------------------------
    # 3. CONTRADICTION & REJECTION POOL
    # -------------------------------------------------------------
    rejection_items = []
    rej_verbs = ["terminated", "rejected", "cancelled", "revoked", "rescinded", "declined", "dissolved"]
    for i in range(200):
        verb = rng.choice(rej_verbs)
        firm_a = f"Company_{i:03d}"
        firm_b = f"TargetSoft_{i:03d}"
        state = f"{firm_a} officially {verb} negotiations and formally discarded the proposed buyout of {firm_b}."
        rejection_items.append({
            "id": f"pool_rej_{i:04d}",
            "category": "logic:rejection_verbs",
            "state": state,
            "question": {
                "t": "noul",
                "ins": f"Did {firm_a} complete the acquisition of {firm_b}?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [1.0, 0.0],
            "should_abstain": 0,
            "domain": "corporate"
        })

    rej_path = os.path.join(output_dir, "contradiction_pool.json")
    with open(rej_path, "w", encoding="utf-8") as f:
        json.dump(rejection_items, f, indent=2, ensure_ascii=False)
    pool_paths["contradiction"] = rej_path

    # -------------------------------------------------------------
    # 4. NEGATION POOL
    # -------------------------------------------------------------
    negation_items = []
    for i in range(200):
        # 4A: Syntactic scope negation
        state_neg = f"The Phase {i%3+1} clinical trial demonstrated no statistically significant reduction in disease relapse rates."
        negation_items.append({
            "id": f"pool_neg_scope_{i:04d}",
            "category": "semantic:negation",
            "state": state_neg,
            "question": {
                "t": "noul",
                "ins": "Did the trial demonstrate a significant reduction in relapse rates?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [1.0, 0.0],
            "should_abstain": 0,
            "domain": "medical"
        })

        # 4B: Double negation
        state_double = f"It is not inaccurate to state that operating margins expanded during quarter {i%4+1}."
        negation_items.append({
            "id": f"pool_neg_double_{i:04d}",
            "category": "semantic:double_negation",
            "state": state_double,
            "question": {
                "t": "noul",
                "ins": f"Did operating margins expand during quarter {i%4+1}?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [0.0, 1.0],
            "should_abstain": 0,
            "domain": "finance"
        })

    neg_path = os.path.join(output_dir, "negation_pool.json")
    with open(neg_path, "w", encoding="utf-8") as f:
        json.dump(negation_items, f, indent=2, ensure_ascii=False)
    pool_paths["negation"] = neg_path

    # -------------------------------------------------------------
    # 5. ENTITY & ACRONYM POOL
    # -------------------------------------------------------------
    entity_items = []
    aliases = [
        ("Big Blue", "IBM", "quantum computing processor"),
        ("Alphabet", "Google", "cloud infrastructure"),
        ("Redmond", "Microsoft", "operating system kernel"),
        ("Solaris Corp", "Solaris Corp's subsidiary Lumina Labs", "patented solar cell")
    ]
    acronyms = [
        ("ARR", "annualized recurring revenue", 120),
        ("GAAP", "generally accepted accounting principles", 15),
        ("EBITDA", "earnings before interest, taxes, depreciation, and amortization", 85),
        ("MFA", "multi-factor authentication", 5)
    ]
    for i in range(150):
        alias_item = aliases[i % len(aliases)]
        acro_item = acronyms[i % len(acronyms)]

        state_alias = f"{alias_item[0]} announced deployment of its newest {alias_item[2]} across global datacenters."
        entity_items.append({
            "id": f"pool_entity_{i:04d}",
            "category": "semantic:entity_resolution",
            "state": state_alias,
            "question": {
                "t": "noul",
                "ins": f"Did {alias_item[1]} announce a {alias_item[2]}?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [0.0, 1.0],
            "should_abstain": 0,
            "domain": "technology"
        })

        state_acro = f"The enterprise reported {acro_item[0]} of ${acro_item[2]}M in the annual regulatory filing."
        entity_items.append({
            "id": f"pool_acronym_{i:04d}",
            "category": "semantic:acronyms",
            "state": state_acro,
            "question": {
                "t": "noul",
                "ins": f"Did {acro_item[1]} reach ${acro_item[2]} million?",
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": [0.0, 1.0],
            "should_abstain": 0,
            "domain": "finance"
        })

    ent_path = os.path.join(output_dir, "entity_pool.json")
    with open(ent_path, "w", encoding="utf-8") as f:
        json.dump(entity_items, f, indent=2, ensure_ascii=False)
    pool_paths["entity"] = ent_path

    # -------------------------------------------------------------
    # 6. FINE-GRAINED CHOICE & TRIAGE POOL
    # -------------------------------------------------------------
    choice_items = []
    triage_specs = [
        {
            "state": "User reports account locked due to five failed MFA attempts during morning shift.",
            "gold": "identity_security",
            "criteria": {
                "billing_support": "Payments, invoices, and refunds",
                "identity_security": "MFA, login locks, authentication failures",
                "infrastructure_ops": "Server downtime, network latency",
                "general_inquiry": "Non-technical product questions"
            }
        },
        {
            "state": "Customer states: 'The item arrived shattered in transit; please issue a refund to my Visa.'",
            "gold": "request_refund",
            "criteria": {
                "request_exchange": "Customer wants a replacement unit",
                "request_refund": "Customer wants monetary refund for damaged item",
                "shipping_status": "Tracking delayed package",
                "cancel_order": "Canceling unshipped order"
            }
        },
        {
            "state": "Database replication lag spiked to 45 seconds, causing read replicas to fall out of sync.",
            "gold": "infrastructure_ops",
            "criteria": {
                "billing_support": "Payments, invoices, and refunds",
                "identity_security": "MFA, login locks, authentication failures",
                "infrastructure_ops": "Server downtime, database replication, network latency",
                "general_inquiry": "Non-technical product questions"
            }
        }
    ]

    for i in range(150):
        spec = triage_specs[i % len(triage_specs)]
        opts = list(spec["criteria"].keys())
        target_idx = opts.index(spec["gold"])
        hard_target = [1.0 if idx == target_idx else 0.0 for idx in range(len(opts))]

        choice_items.append({
            "id": f"pool_choice_{i:04d}",
            "category": "decision:choice",
            "state": spec["state"],
            "question": {
                "t": "choice",
                "ins": "Select the correct department queue or intent code.",
                "crit": spec["criteria"]
            },
            "qtype": 0,
            "hard_target": hard_target,
            "should_abstain": 0,
            "domain": "triage"
        })

    choice_path = os.path.join(output_dir, "choice_pool.json")
    with open(choice_path, "w", encoding="utf-8") as f:
        json.dump(choice_items, f, indent=2, ensure_ascii=False)
    pool_paths["choice"] = choice_path

    # -------------------------------------------------------------
    # 7. MULTILINGUAL POOL
    # -------------------------------------------------------------
    multilingual_items = [
        {
            "state": "Der Reingewinn des Unternehmens stieg im vierten Quartal um 12 Prozent auf 85 Millionen Euro.",
            "ins": "Stieg der Reingewinn um 12 Prozent?",
            "hard_target": [0.0, 1.0],
            "lang": "de"
        },
        {
            "state": "Le chiffre d'affaires annuel a atteint 150 millions d'euros avec une marge stable.",
            "ins": "Le chiffre d'affaires a-t-il atteint 150 millions d'euros?",
            "hard_target": [0.0, 1.0],
            "lang": "fr"
        },
        {
            "state": "Выручка компании за третий квартал составила 45 миллиардов рублей.",
            "ins": "Составила ли выручка 45 миллиардов рублей?",
            "hard_target": [0.0, 1.0],
            "lang": "ru"
        },
        {
            "state": "कंपनी का शुद्ध लाभ पिछले वर्ष की तुलना में 15 प्रतिशत बढ़ गया।",
            "ins": "क्या शुद्ध लाभ 15 प्रतिशत बढ़ गया?",
            "hard_target": [0.0, 1.0],
            "lang": "hi"
        },
        {
            "state": "公司第四季度总收入达到五十亿元人民币，同比增长百分之八。",
            "ins": "第四季度总收入是否达到五十亿元？",
            "hard_target": [0.0, 1.0],
            "lang": "zh"
        },
        {
            "state": "أعلنت الشركة عن تحقيق أرباح صافية قدرها 20 مليون دولار خلال الربع الأول.",
            "ins": "هل بلغت الأرباح الصافية 20 مليون دولار؟",
            "hard_target": [0.0, 1.0],
            "lang": "ar"
        }
    ]

    multi_expanded = []
    for i in range(120):
        base = multilingual_items[i % len(multilingual_items)]
        multi_expanded.append({
            "id": f"pool_multi_{i:04d}",
            "category": f"multilingual:{base['lang']}",
            "state": base["state"],
            "question": {
                "t": "noul",
                "ins": base["ins"],
                "crit": {"false": "unsupported", "true": "supported"}
            },
            "qtype": 2,
            "hard_target": base["hard_target"],
            "should_abstain": 0,
            "domain": "international"
        })

    multi_path = os.path.join(output_dir, "multilingual_pool.json")
    with open(multi_path, "w", encoding="utf-8") as f:
        json.dump(multi_expanded, f, indent=2, ensure_ascii=False)
    pool_paths["multilingual"] = multi_path

    print(f"[Capability Pools] Generated 7 pools in {output_dir}:")
    for name, path in pool_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  - {name:15}: {len(data)} items ({path})")

    return pool_paths


if __name__ == "__main__":
    generate_all_capability_pools()
