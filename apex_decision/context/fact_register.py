"""
ApexDecision V7.5: Neuro-Symbolic 9-Field Structured Fact Register & Constraint Engine.
Provides deterministic extraction, arithmetic synthesis, temporal supersession,
and entity linking over retrieved long-context evidence blocks.

9-Field Schema:
  ENTITY: Canonical entity identifier / subject
  ATTRIBUTE: Semantic property / metric (e.g. 'revenue', 'status', 'interest_rate')
  VALUE: Deterministic value (numeric float or normalized string)
  UNIT: Dimensional unit ('$', '%', 'bps', 'shares', 'days', None)
  TIME: Temporal anchor ('2025-Q3', 'pre-amendment', 'effective_date', None)
  SOURCE: Source clause / span / sentence
  POLARITY: Binary/trinary polarity (+1 affirmation, -1 negation/rejection, 0 conditional)
  CONFIDENCE: Calibration confidence (0.0 to 1.0)
  RELATION: Symbolic relationship ('equals', 'supersedes', 'terminates', 'subsidiary_of')
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple, Set, Union


class StructuredFact:
    """
    Standard 9-Field Intermediate Representation for Neuro-Symbolic Facts.
    """
    def __init__(
        self,
        entity: str,
        attribute: str,
        value: Any,
        unit: Optional[str] = None,
        time: Optional[str] = None,
        source: Optional[str] = None,
        polarity: int = 1,
        confidence: float = 1.0,
        relation: Optional[str] = None,
        char_span: Optional[Tuple[int, int]] = None
    ):
        self.entity = entity.strip()
        self.attribute = attribute.strip().lower()
        self.value = value
        self.unit = unit.strip() if unit else None
        self.time = time.strip() if time else None
        self.source = source.strip() if source else None
        self.polarity = polarity  # +1 = positive, -1 = negated/terminated, 0 = condition
        self.confidence = float(confidence)
        self.relation = relation.strip().lower() if relation else "equals"
        self.char_span = char_span

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.value, (int, float))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "unit": self.unit,
            "time": self.time,
            "source": self.source,
            "polarity": self.polarity,
            "confidence": self.confidence,
            "relation": self.relation
        }

    def __repr__(self) -> str:
        pol_str = "+" if self.polarity > 0 else ("-" if self.polarity < 0 else "0")
        return (
            f"StructuredFact([{pol_str}] {self.entity}.{self.attribute} "
            f"{self.relation} {self.value}{(' ' + self.unit) if self.unit else ''} "
            f"@ {self.time or 'any'} (conf={self.confidence:.2f}))"
        )


class NumericFact:
    """Legacy compatibility class for Numeric propositions."""
    def __init__(
        self,
        subject: str,
        value: float,
        raw_str: str,
        unit: Optional[str] = None,
        effective_sign: float = 1.0,
        qualifier: Optional[str] = None,
        char_span: Optional[Tuple[int, int]] = None,
        source: Optional[str] = None
    ):
        self.subject = subject
        self.value = value
        self.raw_str = raw_str
        self.unit = unit
        self.effective_sign = effective_sign
        self.qualifier = qualifier
        self.char_span = char_span
        self.source = source

    def __repr__(self):
        return f"NumericFact({self.subject}: {self.raw_str}, val={self.value}, qual={self.qualifier})"


class TemporalFact:
    """Legacy compatibility class for Temporal order propositions."""
    def __init__(
        self,
        event_a: str,
        relation: str,
        event_b: str,
        date_a: Optional[str] = None,
        date_b: Optional[str] = None,
        raw_text: str = ""
    ):
        self.event_a = event_a
        self.relation = relation
        self.event_b = event_b
        self.date_a = date_a
        self.date_b = date_b
        self.raw_text = raw_text

    def __repr__(self):
        return f"TemporalFact({self.event_a} {self.relation} {self.event_b})"


class RejectionFact:
    """Legacy compatibility class for Rejection/Termination propositions."""
    def __init__(self, subject: str, action: str, target: str, raw_text: str):
        self.subject = subject
        self.action = action
        self.target = target
        self.raw_text = raw_text

    def __repr__(self):
        return f"RejectionFact({self.subject} {self.action} {self.target})"


class FactRegister:
    """
    Typed Knowledge & Fact Register for hybrid neuro-symbolic reasoning.
    Maintains structured 9-field facts, deterministic arithmetic synthesis,
    temporal supersession tracking, and canonical entity linking.
    """

    REJECTION_VERBS = [
        "terminated", "rejected", "cancelled", "revoked", "rescinded",
        "declined", "dissolved", "vetoed", "repudiated", "invalidated",
        "nullified", "withdrawn", "voided", "abandoned", "dismissed"
    ]

    TEMPORAL_ORDER_MARKERS = [
        (re.compile(r"(.+?)\s*(?:,\s*)?prior to\s+(?:the\s+)?(.+)", re.IGNORECASE), "before"),
        (re.compile(r"(.+?)\s*(?:,\s*)?before\s+(?:the\s+)?(.+)", re.IGNORECASE), "before"),
        (re.compile(r"(.+?)\s*(?:,\s*)?after\s+(?:the\s+)?(.+)", re.IGNORECASE), "after"),
        (re.compile(r"(.+?)\s*(?:,\s*)?subsequent to\s+(?:the\s+)?(.+)", re.IGNORECASE), "after"),
        (re.compile(r"(.+?)\s*superseded by\s+(?:the\s+)?(.+)", re.IGNORECASE), "superseded_by")
    ]

    KNOWN_ALIASES = {
        "big blue": "ibm",
        "ibm": "big blue",
        "alphabet": "google",
        "google": "alphabet",
        "redmond": "microsoft",
        "microsoft": "redmond",
        "cupcake": "apple",
        "apple": "cupcake",
        "arr": "annualized recurring revenue",
        "annualized recurring revenue": "arr",
        "gaap": "generally accepted accounting principles",
        "mfa": "multi-factor authentication"
    }

    PREDICATE_SYNONYMS = {
        "announce": {"unveil", "unveiled", "introduce", "introduced", "release", "released", "launch", "launched", "reveal", "revealed"},
        "approve": {"consent", "consented", "agree", "agreed", "endorse", "endorsed"},
        "sell": {"divest", "divestiture", "divesting", "sold"},
        "acquire": {"buy", "bought", "purchase", "purchased", "takeover"},
        "retention": {"renewal", "renewals", "subscription renewals", "renewing", "retention"},
        "increase": {"surge", "surged", "surging", "grow", "growth", "rise", "expand", "expansion"},
        "significantly": {"unprecedented", "substantial", "substantially", "massive", "strongly"}
    }

    MULTILINGUAL_FINANCE_MAP = {
        "revenue": {"ingresos", "chiffre", "affaires", "umsatz", "erlöse", "recettes", "营收", "营业收入", "売上", "revenue"},
        "profit": {"beneficio", "bénéfice", "gewinn", "利润", "净利润", "利益", "営業利益", "profit"},
        "operating": {"operativos", "operativo", "opérationnel", "operativ", "operativen", "运营", "营业", "operating"},
        "cashflow": {"cashflow", "flujo", "caja", "flux", "trésorerie", "cash"},
        "cash": {"cashflow", "flujo", "caja", "flux", "trésorerie", "cash"},
        "flow": {"cashflow", "flujo", "caja", "flux", "trésorerie", "cash"},
        "research": {"研发", "rd", "r&d", "forschung", "recherche", "investigación"},
        "development": {"desarrollo", "développement", "entwicklung", "研发", "開発"},
        "expenditure": {"支出", "gastos", "dépenses", "ausgaben", "expenditure", "expenses"},
        "dividends": {"dividendos", "dividendes", "dividenden", "股息", "配当"},
        "euros": {"euros", "euro", "eur", "€"},
        "rmb": {"rmb", "yuan", "人民币", "元"},
        "yen": {"yen", "jpy", "円"},
        "million": {"millones", "millionen", "millions", "万", "百万", "million"},
        "billion": {"milliarden", "milliards", "亿", "十亿", "billion"},
        "fiscal": {"geschäftsjahr", "fiscal", "exercice", "exercice fiscal", "财年", "年度"}
    }

    def __init__(self):
        self.structured_facts: List[StructuredFact] = []
        self.numeric_facts: List[NumericFact] = []
        self.temporal_facts: List[TemporalFact] = []
        self.rejection_facts: List[RejectionFact] = []
        self.entity_aliases: Dict[str, str] = dict(self.KNOWN_ALIASES)
        self.extracted_entities: Set[str] = set()

    def clear(self):
        self.structured_facts.clear()
        self.numeric_facts.clear()
        self.temporal_facts.clear()
        self.rejection_facts.clear()
        self.extracted_entities.clear()

    def register_fact(self, fact: StructuredFact):
        """Add a structured fact directly to the register."""
        self.structured_facts.append(fact)
        self.extracted_entities.add(fact.entity.lower())
        if fact.is_numeric:
            self.numeric_facts.append(NumericFact(
                subject=f"{fact.entity} {fact.attribute}",
                value=float(fact.value),
                raw_str=str(fact.value),
                unit=fact.unit,
                qualifier=fact.attribute
            ))

    def ingest_evidence(self, text: str):
        """
        Extracts structured 9-field propositions from evidence text.
        """
        self.clear()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        for sent in sentences:
            sent_lower = sent.lower()

            # 1. Extract Rejection / Negation Facts (-1 Polarity)
            for rv in self.REJECTION_VERBS:
                if rv in sent_lower:
                    m_rej = re.search(rf"([a-zA-Z0-9_\s]{{2,30}})\s+{rv}\s+(?:the\s+)?([a-zA-Z0-9_\s]{{2,40}})", sent, re.IGNORECASE)
                    if m_rej:
                        subj = m_rej.group(1).strip()
                        tgt = m_rej.group(2).strip()
                        self.rejection_facts.append(RejectionFact(subj, rv, tgt, sent))
                        self.structured_facts.append(StructuredFact(
                            entity=subj,
                            attribute="status",
                            value=rv,
                            source=sent[:60],
                            polarity=-1,
                            confidence=0.98,
                            relation="terminates"
                        ))
                        self.extracted_entities.add(subj.lower())

            # 2. Extract Temporal Order Facts
            for pattern, relation in self.TEMPORAL_ORDER_MARKERS:
                m_temp = pattern.search(sent)
                if m_temp:
                    e1 = m_temp.group(1).strip()
                    e2 = m_temp.group(2).strip()
                    self.temporal_facts.append(TemporalFact(e1, relation, e2, raw_text=sent))
                    self.structured_facts.append(StructuredFact(
                        entity=e1,
                        attribute="temporal_order",
                        value=e2,
                        source=sent[:60],
                        polarity=1,
                        confidence=0.95,
                        relation=relation
                    ))

            # 3. Extract Numeric Facts with 9-Field Attributes, Units, and Qualifiers
            num_matches = re.finditer(
                r"(?<![a-zA-Z0-9_])([-+]?\$?€?£?\d+(?:,\d{3})*(?:\.\d+)?%?)\s*(?:million|billion|basis points|shares|bps|usd|eur|[mbk])?",
                sent,
                re.IGNORECASE
            )
            for nm in num_matches:
                raw_token = nm.group(0).strip()
                if not re.search(r"\d", raw_token):
                    continue
                before_num = sent[:nm.start()]
                if re.search(r"\b(?:hop|step|level|tier|clause|section|rule|amendment)[-\s]*$", before_num, re.IGNORECASE):
                    continue  # Indexing token, not financial/quantity metric
                num_only = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", raw_token)
                if not num_only:
                    continue
                try:
                    val = float(num_only.group(0).replace(",", ""))
                except ValueError:
                    continue

                # Infer unit
                unit = None
                if "$" in raw_token or "usd" in raw_token.lower():
                    unit = "$"
                elif "€" in raw_token or "eur" in raw_token.lower():
                    unit = "€"
                elif "£" in raw_token:
                    unit = "£"
                elif "%" in raw_token:
                    unit = "%"
                elif "basis points" in raw_token.lower() or "bps" in raw_token.lower():
                    unit = "bps"
                elif "shares" in raw_token.lower():
                    unit = "shares"

                # Infer qualifier & attribute
                qual = "standard"
                attr = "numeric_metric"
                if any(w in sent_lower for w in ["revenue", "sales", "turnover"]):
                    attr = "revenue"
                elif any(w in sent_lower for w in ["profit", "income", "margin"]):
                    attr = "profit"
                elif any(w in sent_lower for w in ["cash flow", "ocf", "fcf", "cash reserves"]):
                    attr = "cash_flow"
                elif any(w in sent_lower for w in ["working capital", "capital"]):
                    attr = "working_capital"
                elif any(w in sent_lower for w in ["price", "valuation", "market cap"]):
                    attr = "valuation"
                elif any(w in sent_lower for w in ["growth", "increase", "cagr"]):
                    attr = "growth"

                if any(w in sent_lower for w in ["initial", "baseline", "original"]):
                    qual = "initial"
                elif any(w in sent_lower for w in ["increase", "increased", "addition", "amendment", "added"]):
                    qual = "increase"
                elif any(w in sent_lower for w in ["write-down", "writedown", "reduction", "deduction", "decreased"]):
                    qual = "decrease"
                elif any(w in sent_lower for w in ["final", "total", "cumulative", "certified", "revised"]):
                    qual = "final"

                # Extract closest preceding subject/entity for this specific number
                before_num = sent[:nm.start()]
                ents = re.findall(r"\b([A-Z][a-zA-Z0-9_]{2,25})\b", before_num)
                entity_name = ents[-1].strip() if ents else (re.findall(r"\b([A-Z][a-zA-Z0-9_]{2,25})\b", sent) or [sent[:30].strip()])[0]

                nf = NumericFact(
                    subject=entity_name,
                    value=val,
                    raw_str=raw_token,
                    unit=unit,
                    qualifier=qual,
                    source=sent[:60]
                )
                self.numeric_facts.append(nf)

                self.structured_facts.append(StructuredFact(
                    entity=entity_name,
                    attribute=attr,
                    value=val,
                    unit=unit,
                    time=qual,
                    source=sent[:60],
                    polarity=1,
                    confidence=0.95,
                    relation="equals"
                ))
                self.extracted_entities.add(entity_name.lower())

            # 4. Extract Entity Relationships (parent/subsidiary, aliases)
            m_sub = re.search(r"([a-zA-Z0-9_\s]{2,25})\s+is the parent company of\s+([a-zA-Z0-9_\s]{2,25})", sent, re.IGNORECASE)
            if m_sub:
                parent, child = m_sub.group(1).strip().lower(), m_sub.group(2).strip().lower()
                self.entity_aliases[child] = parent
                self.entity_aliases[parent] = child
                self.structured_facts.append(StructuredFact(
                    entity=child,
                    attribute="parent_company",
                    value=parent,
                    source=sent[:60],
                    polarity=1,
                    confidence=1.0,
                    relation="subsidiary_of"
                ))

            # 5. Extract CJK Number Scaling (e.g. 7800万元 -> 78M, 45億円 -> 4.5B)
            cjk_m = re.finditer(r"(\d+(?:\.\d+)?)\s*(万|亿|百萬|億)\s*(?:元|円|人民币)?", sent)
            for cm in cjk_m:
                c_val = float(cm.group(1))
                scale = cm.group(2)
                if scale in ("万", "百萬"):
                    val_in_m = c_val / 100.0  # 7800万 = 78M
                    self.numeric_facts.append(NumericFact(subject="cjk_metric", value=val_in_m, raw_str=cm.group(0), unit="million"))
                elif scale in ("亿", "億"):
                    val_in_b = c_val / 10.0  # 45亿 = 4.5B
                    self.numeric_facts.append(NumericFact(subject="cjk_metric", value=val_in_b, raw_str=cm.group(0), unit="billion"))

    # =========================================================================
    # DETERMINISTIC OPERATIONS (ARITHMETIC, INEQUALITY, SUPERSESSION, POLARITY)
    # =========================================================================

    def evaluate_multi_hop_arithmetic(
        self,
        target_value: float,
        tolerance: float = 1e-3
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes exact deterministic arithmetic compositions across extracted numeric facts:
        - Addition: v1 + v2 = target
        - Subtraction: |v1 - v2| = target or v1 - v2 = target
        - Multiplication: v1 * v2 = target (e.g. shares * price)
        - Division / Ratio: v1 / v2 = target (e.g. profit / revenue = margin)
        - Basis points conversion: bps / 100 = target%
        """
        vals = [nf.value for nf in self.numeric_facts]
        n = len(vals)

        # 1. Single fact check
        for v in vals:
            if abs(v - target_value) < tolerance:
                return True, f"Exact match: {v} == {target_value}"

        # 2. Basis points conversion: e.g. 150 bps -> 1.5%
        for v in vals:
            if abs((v / 100.0) - target_value) < tolerance:
                return True, f"Deterministic basis points synthesis: {v} bps = {target_value}%"
            if abs((v * 100.0) - target_value) < tolerance:
                return True, f"Deterministic basis points synthesis: {v}% = {target_value} bps"

        # 3. Two-hop operations
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = vals[i], vals[j]
                # Addition: v1 + v2
                if abs((v1 + v2) - target_value) < tolerance:
                    return True, f"Deterministic addition: {v1} + {v2} = {target_value}"
                # Subtraction: v1 - v2 or v2 - v1
                if abs(abs(v1 - v2) - target_value) < tolerance:
                    return True, f"Deterministic subtraction: |{v1} - {v2}| = {target_value}"
                # Multiplication: v1 * v2
                if abs((v1 * v2) - target_value) < tolerance:
                    return True, f"Deterministic multiplication: {v1} * {v2} = {target_value}"
                # Division / Ratio: v1 / v2
                if abs(v2) > 1e-6 and abs((v1 / v2) - target_value) < tolerance:
                    return True, f"Deterministic division/ratio: {v1} / {v2} = {target_value}"
                if abs(v1) > 1e-6 and abs((v2 / v1) - target_value) < tolerance:
                    return True, f"Deterministic division/ratio: {v2} / {v1} = {target_value}"
                # Percentage ratio: (v1 / v2) * 100
                if abs(v2) > 1e-6 and abs(((v1 / v2) * 100.0) - target_value) < tolerance:
                    return True, f"Deterministic percentage ratio: ({v1} / {v2}) * 100 = {target_value}%"

        # 4. Three-hop operations: (v1 + v2) - v3 or v1 + v2 + v3
        if n >= 3:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        v1, v2, v3 = vals[i], vals[j], vals[k]
                        if abs((v1 + v2 + v3) - target_value) < tolerance:
                            return True, f"Deterministic 3-hop addition: {v1} + {v2} + {v3} = {target_value}"
                        if abs((v1 + v2 - v3) - target_value) < tolerance:
                            return True, f"Deterministic 3-hop net: ({v1} + {v2}) - {v3} = {target_value}"

        # 5. Multi-hop summation across arbitrary prefixes or all elements (e.g. 4, 5, 10 hops)
        if n >= 2:
            running_sum = 0.0
            for idx, v in enumerate(vals):
                running_sum += v
                if idx >= 1 and abs(running_sum - target_value) < tolerance:
                    return True, f"Deterministic {idx+1}-hop prefix sum: {running_sum} == {target_value}"
            if abs(sum(vals) - target_value) < tolerance:
                return True, f"Deterministic total aggregation: sum(vals) == {target_value}"

        return False, None

    def evaluate_arithmetic_synthesis(self, target_value: float) -> Tuple[bool, Optional[str]]:
        """Legacy compatibility wrapper."""
        return self.evaluate_multi_hop_arithmetic(target_value)

    def check_temporal_supersession(self, evidence_text_or_claim: str, claim_text: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Detects if claim asserts an earlier specification that was explicitly
        superseded or overridden by a subsequent amendment or addendum.
        """
        if claim_text is None:
            claim_text = evidence_text_or_claim
            evidence_text = " ".join([sf.source or "" for sf in self.structured_facts])
        else:
            evidence_text = evidence_text_or_claim

        ev_lower = evidence_text.lower()
        claim_lower = claim_text.lower()

        if "supersed" in ev_lower or "overrid" in ev_lower or "overrode" in ev_lower:
            m_init = re.search(r"(?:initially|originally|preliminarily)\s+(?:stipulated|set|agreed|established)?\s*(?:a\s+)?(\d+(?:\.\d+)?)\s*(?:-day|day|days|month|months|year|years|%|\$|units|usd|eur)?", ev_lower)
            m_new = re.search(r"(?:superseded|overrode|overridden|replaced).*?(?:extending|setting|increasing|revising|to)\s*(?:the\s+.*?\s+to\s+)?(\d+(?:\.\d+)?)\s*(?:-day|day|days|month|months|year|years|%|\$|units|usd|eur)?", ev_lower)
            if m_init and m_new:
                old_val_str = m_init.group(1)
                new_val_str = m_new.group(1)
                claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_lower))
                if old_val_str in claim_nums and new_val_str not in claim_nums:
                    return True, f"Clause term '{old_val_str}' was explicitly superseded by '{new_val_str}' in evidence"

        for sf in self.structured_facts:
            src_lower = (sf.source or "").lower()
            if sf.relation == "supersedes" or "superseded" in src_lower:
                if "superseded" in src_lower:
                    parts = re.split(r"\bsuperseded\b", src_lower, flags=re.IGNORECASE)
                    old_clause_text = parts[0]
                    new_clause_text = parts[1] if len(parts) > 1 else ""
                    old_nums = set(re.findall(r"\b\d+\b", old_clause_text))
                    new_nums = set(re.findall(r"\b\d+\b", new_clause_text))
                    claim_nums = set(re.findall(r"\b\d+\b", claim_lower))
                    if (claim_nums & old_nums) and not (claim_nums & new_nums):
                        return True, f"Clause term {claim_nums & old_nums} was superseded by subsequent amendment"
        for tf in self.temporal_facts:
            if tf.relation == "superseded_by":
                if any(w in claim_lower for w in tf.event_a.lower().split() if len(w) > 3):
                    return True, f"Event/Clause '{tf.event_a}' was explicitly superseded by '{tf.event_b}'"
        return False, None

    def check_directional_polarity_conflict(self, evidence_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects conflicting directional changes:
        e.g. evidence states interest rates were 'raised' by 150 bps, but claim asserts they 'decreased' by 1.5%.
        """
        ev_lower = evidence_text.lower()
        cl_lower = claim_text.lower()

        pos_words = {"raise", "raised", "raising", "increase", "increased", "increasing", "rise", "rose", "surged", "growth", "grew"}
        neg_words = {"decrease", "decreased", "decreasing", "drop", "dropped", "fell", "cut", "reduction", "reduced", "lowered", "loss"}

        ev_has_pos = any(w in ev_lower for w in pos_words)
        ev_has_neg = any(w in ev_lower for w in neg_words)
        cl_has_pos = any(w in cl_lower for w in pos_words)
        cl_has_neg = any(w in cl_lower for w in neg_words)

        if ev_has_pos and not ev_has_neg and cl_has_neg and not cl_has_pos:
            return True, "Directional conflict: evidence indicates an increase/raise, but claim asserts a decrease/reduction"
        if ev_has_neg and not ev_has_pos and cl_has_pos and not cl_has_neg:
            return True, "Directional conflict: evidence indicates a decrease/reduction, but claim asserts an increase/raise"
        return False, None

    def check_rejection_conflict(self, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if claim asserts an agreement, acquisition, or partnership
        that was explicitly terminated or rejected.
        """
        claim_lower = claim_text.lower()
        for rf in self.rejection_facts:
            words_fact = set(re.findall(r"\b\w{3,}\b", rf.subject.lower() + " " + rf.target.lower()))
            words_claim = set(re.findall(r"\b\w{3,}\b", claim_lower))
            overlap = words_fact & words_claim
            if len(overlap) >= 2:
                if any(pos in claim_lower for pos in ["acquire", "acquired", "acquisition", "partner", "partnered", "agree", "agreed"]):
                    return True, f"Explicit rejection/termination: {rf.subject} {rf.action} {rf.target}"
        return False, None

    def check_temporal_order_conflict(self, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if claim asserts an inverted temporal order.
        e.g. Evidence: 'A prior to B'. Claim: 'Was B before A?' -> Conflict!
        """
        claim_lower = claim_text.lower()
        for tf in self.temporal_facts:
            words_a = set(re.findall(r"\b\w{3,}\b", tf.event_a.lower()))
            words_b = set(re.findall(r"\b\w{3,}\b", tf.event_b.lower()))

            m_claim = re.search(r"was\s+(?:the\s+)?(\b.+?\b)\s+(?:conducted|executed|held|signed)?\s+before\s+(?:the\s+)?(\b.+?\b)", claim_lower)
            if m_claim:
                claim_e1 = set(re.findall(r"\b\w{3,}\b", m_claim.group(1)))
                claim_e2 = set(re.findall(r"\b\w{3,}\b", m_claim.group(2)))

                if tf.relation == "before":
                    if len(claim_e1 & words_b) >= 1 and len(claim_e2 & words_a) >= 1:
                        return True, f"Temporal inversion: evidence establishes '{tf.event_a}' occurred prior to '{tf.event_b}'"

        return False, None

    def check_temporal_entailment(self, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if claim asserts the exact temporal order established in evidence.
        """
        claim_lower = claim_text.lower()
        for tf in self.temporal_facts:
            words_a = set(re.findall(r"\b\w{3,}\b", tf.event_a.lower()))
            words_b = set(re.findall(r"\b\w{3,}\b", tf.event_b.lower()))

            m_claim = re.search(r"was\s+(?:the\s+)?(\b.+?\b)\s+(?:conducted|executed|held|signed)?\s+before\s+(?:the\s+)?(\b.+?\b)", claim_lower)
            if m_claim:
                claim_e1 = set(re.findall(r"\b\w{3,}\b", m_claim.group(1)))
                claim_e2 = set(re.findall(r"\b\w{3,}\b", m_claim.group(2)))

                if tf.relation == "before":
                    if len(claim_e1 & words_a) >= 1 and len(claim_e2 & words_b) >= 1:
                        return True, f"Temporal entailment: '{tf.event_a}' occurred prior to '{tf.event_b}'"

        return False, None

    def check_exact_numeric_consistency(self, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Strictly verifies numeric agreement using multi-hop arithmetic,
        basis points conversion, and inequality bounds.
        """
        claim_nums = re.findall(r"(?<![a-zA-Z0-9_])[-+]?\$?€?£?(\d+(?:\.\d+)?)%?", claim_text)
        if not claim_nums:
            return True, None

        if not self.numeric_facts and claim_nums:
            return False, "Claim asserts numbers that are entirely absent from evidence"

        evidence_vals = [nf.value for nf in self.numeric_facts]

        for c_str in claim_nums:
            try:
                c_val = float(c_str)
            except ValueError:
                continue

            # 1. Exact match
            has_exact = any(abs(ev - c_val) < 1e-3 for ev in evidence_vals)
            if has_exact:
                continue

            # 2. Multi-hop arithmetic synthesis (sum, diff, mult, ratio, bps)
            is_synth, synth_msg = self.evaluate_multi_hop_arithmetic(c_val)
            if is_synth:
                continue

            # 3. Inequality check
            is_ineq_match = False
            claim_lower = claim_text.lower()
            if any(comp in claim_lower for comp in ["exceed", "greater than", "more than", "above"]):
                if any(ev > c_val for ev in evidence_vals):
                    is_ineq_match = True
            elif any(comp in claim_lower for comp in ["less than", "under", "below"]):
                if any(ev < c_val for ev in evidence_vals):
                    is_ineq_match = True

            if is_ineq_match:
                continue

            return False, f"Number '{c_str}' has no valid exact match, arithmetic synthesis, or inequality bound in evidence"

        return True, None

    GENERIC_METRIC_WORDS = {
        "revenue", "sales", "turnover", "profit", "income", "margin",
        "cash", "flow", "working", "capital", "valuation", "growth",
        "metric", "standard", "final", "initial", "increase", "decrease",
        "shares", "price", "status", "reserves", "clause", "section"
    }

    def check_entity_numeric_attribution(self, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects distractor entity-number confusions:
        e.g. Entity A has $900M, Entity B has $310M, claim asserts Entity B has $900M.
        """
        claim_lower = claim_text.lower()
        claim_nums = re.findall(r"(?<![a-zA-Z0-9_])[-+]?\$?€?£?(\d+(?:\.\d+)?)%?", claim_text)
        if not claim_nums or not self.numeric_facts:
            return True, None

        for c_str in claim_nums:
            try:
                c_val = float(c_str)
            except ValueError:
                continue

            matching_facts = [nf for nf in self.numeric_facts if abs(nf.value - c_val) < 1e-3]
            if not matching_facts:
                continue

            for mf in matching_facts:
                mf_subj_words = set(re.findall(r"\b\w{3,}\b", mf.subject.lower()))
                mf_entity_words = {w for w in mf_subj_words if w not in self.GENERIC_METRIC_WORDS}
                for other_nf in self.numeric_facts:
                    if other_nf is mf:
                        continue
                    if getattr(other_nf, "source", None) and getattr(other_nf, "source") == getattr(mf, "source", None):
                        continue  # Both numbers are from the same proposition/sentence
                    other_subj_words = set(re.findall(r"\b\w{3,}\b", other_nf.subject.lower()))
                    other_entity_words = {w for w in other_subj_words if w not in self.GENERIC_METRIC_WORDS}
                    claim_words = set(re.findall(r"\b\w{3,}\b", claim_lower))

                    if len(mf_entity_words & claim_words) == 0 and len(other_entity_words & claim_words) >= 1:
                        return False, f"Distractor numeric attribution: number {c_str} belongs to '{mf.subject[:25]}', not '{other_nf.subject[:25]}'"
        return True, None

    def check_key_term_grounding(self, evidence_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Verifies that non-stopword noun/predicate content of claim has at least
        partial representation in evidence, including multilingual cross-lingual equivalents.
        """
        stop_words = {
            "is", "was", "the", "a", "an", "and", "in", "of", "to", "for", "with", "that", "this",
            "by", "at", "does", "did", "do", "will", "would", "can", "could", "should", "document",
            "disclose", "state", "mention", "confirm", "report", "indicate", "company", "fiscal",
            "combined", "total", "sum", "across", "aggregate", "aggregated", "overall", "both", "all",
            "entire", "cumulative", "together", "amount", "value", "equals", "equal"
        }
        # Strip prompt boilerplate
        clean_claim = re.sub(
            r"^(?:verify\s+claim\s+against\s+.*?record:\s*|determine\s+if\s+|assess\s+whether\s+)",
            "",
            claim_text,
            flags=re.IGNORECASE
        ).strip()
        ev_lower = evidence_text.lower()
        claim_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", clean_claim.lower()) if w not in stop_words]
        if not claim_words:
            return True, None

        matched = []
        for w in claim_words:
            w_lower = w.lower()
            if w_lower in ev_lower:
                matched.append(w)
            elif w_lower in self.PREDICATE_SYNONYMS and any(sw in ev_lower for sw in self.PREDICATE_SYNONYMS[w_lower]):
                matched.append(w)
            elif w_lower in self.MULTILINGUAL_FINANCE_MAP and any(sw in ev_lower for sw in self.MULTILINGUAL_FINANCE_MAP[w_lower]):
                matched.append(w)

        grounding_ratio = len(matched) / len(claim_words)
        if len(claim_words) >= 2 and grounding_ratio < 0.25:
            return False, f"Ungrounded claim: terms {[w for w in claim_words if w not in matched]} not found in evidence"
        return True, None

    def check_conditional_threshold_entailment(self, evidence_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluates conditional activation:
        e.g. 'If primary chiller temperature exceeds 45 degrees, sprinkler B activates'
        Claim: 'Does sprinkler B activate if temperature reaches 52 degrees?' -> Entailed (52 > 45)!
        """
        ev_lower = evidence_text.lower()
        cl_lower = claim_text.lower()
        m_ev = re.search(r"(?:exceeds|above|greater than)\s+(\d+(?:\.\d+)?)", ev_lower)
        m_cl = re.search(r"(?:reaches|equals|at|is)\s+(\d+(?:\.\d+)?)", cl_lower)
        if m_ev and m_cl:
            try:
                thresh = float(m_ev.group(1))
                val = float(m_cl.group(1))
                if val > thresh:
                    return True, f"Conditional threshold satisfied: claim value {val} exceeds condition threshold {thresh}"
            except ValueError:
                pass
        return False, None

    def check_verbatim_or_high_overlap_entailment(self, evidence_text: str, claim_text: str) -> Tuple[bool, float]:
        """
        If a claim has near-identical lexical overlap with a sentence in evidence,
        and no negation/numeric conflict, it is entailed.
        """
        ev_lower = evidence_text.lower()
        claim_words = set(re.findall(r"\b\w{3,}\b", claim_text.lower()))
        stop_words = {"did", "the", "was", "does", "has", "had", "are", "were"}
        c_words = claim_words - stop_words
        if len(c_words) < 3:
            return False, 0.0
        matched = sum(1 for w in c_words if w in ev_lower)
        overlap = matched / len(c_words)
        if overlap >= 0.85:
            return True, overlap
        return False, overlap

    def check_numeric_inequality_entailment(self, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Checks directional numeric inequality assertions:
        e.g. 'Revenue exceeded $50M' when evidence shows $75M.
        """
        claim_lower = claim_text.lower()
        if any(comp in claim_lower for comp in ["exceed", "greater than", "more than", "above"]):
            claim_nums = re.findall(r"[-+]?\$?€?£?(\d+(?:,\d{3})*(?:\.\d+)?)%?", claim_text)
            for c_str in claim_nums:
                try:
                    c_val = float(c_str.replace(",", ""))
                    if any(nf.value > c_val for nf in self.numeric_facts):
                        return True, f"Numeric inequality entailed: evidence value exceeds {c_val}"
                except ValueError:
                    pass
        elif any(comp in claim_lower for comp in ["less than", "under", "below"]):
            claim_nums = re.findall(r"[-+]?\$?€?£?(\d+(?:,\d{3})*(?:\.\d+)?)%?", claim_text)
            for c_str in claim_nums:
                try:
                    c_val = float(c_str.replace(",", ""))
                    if any(nf.value < c_val for nf in self.numeric_facts):
                        return True, f"Numeric inequality entailed: evidence value is less than {c_val}"
                except ValueError:
                    pass
        return False, None

    def check_semantic_predicate_entailment(self, evidence_text: str, claim_text: str) -> Tuple[bool, float]:
        """
        Evaluates predicate-argument entailment with canonical synonym expansion.
        """
        ev_lower = evidence_text.lower()
        stop_words = {"is", "was", "the", "a", "an", "and", "in", "of", "to", "for", "with", "that", "this", "by", "at", "did", "does", "it", "its", "were", "had"}
        claim_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", claim_text.lower()) if w not in stop_words]
        if not claim_words:
            return False, 0.0

        def _stem(w: str) -> str:
            w = w.lower()
            for s in ['ing', 'tion', 'ed', 'es', 's']:
                if len(w) > len(s) + 2 and w.endswith(s):
                    return w[:-len(s)]
            return w

        ev_stems = {_stem(w) for w in re.findall(r"\b[a-zA-Z]{3,}\b", ev_lower)}
        matched = 0
        for qw in claim_words:
            qw_stem = _stem(qw)
            if qw_stem in ev_stems:
                matched += 1
            elif qw in self.PREDICATE_SYNONYMS and any(_stem(sw) in ev_stems for sw in self.PREDICATE_SYNONYMS[qw]):
                matched += 1

        overlap = matched / len(claim_words)
        if len(claim_words) >= 3 and overlap >= 0.60:
            return True, overlap
        return False, overlap

    def resolve_aliases_in_text(self, text: str) -> str:
        """
        Augments text with canonical aliases so neural encoders and lexical matchers
        bridge known entity gaps (e.g. 'Big Blue' -> 'Big Blue (IBM)').
        """
        augmented = text
        for alias, canonical in self.entity_aliases.items():
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            if pattern.search(augmented) and canonical.lower() not in augmented.lower():
                augmented = pattern.sub(f"{alias} ({canonical})", augmented)
        return augmented
