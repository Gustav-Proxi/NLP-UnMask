"""
Text-based anatomical diagrams for visual hints.
Unicode box-drawing renders clearly in Chainlit code blocks.
"""

ANATOMY_DIAGRAMS: dict[str, dict] = {
    "brachial_plexus": {
        "caption": "Brachial Plexus — trace C5→T1 through trunks, divisions, cords to terminal branches",
        "diagram": """\
ROOTS   TRUNKS     DIVS    CORDS        TERMINAL BRANCHES
                   ┌ Ant ─────────────┐
C5 ─┬─ Upper ─────┤                   Lateral ──┬─ Musculocutaneous (C5-C7)
C6 ─┘             └ Post ─────────┐              └─ Median (lateral head)
                   ┌ Ant ─────────│──────────────── (joins medial head)
C7 ──── Middle ───┤               │
                   └ Post ────────┤  Posterior ──┬─ Axillary N. (C5-C6)
                   ┌ Ant ─────────│              └─ Radial N. (C5-T1)
C8 ─┬─ Lower ─────┤               │
T1 ─┘             └ Post ─────────┘  Medial ───┬─ Ulnar N. (C8-T1)
                                                ├─ Median (medial head)
                                                ├─ Medial cutaneous (arm)
                                                └─ Medial cutaneous (forearm)
Memory: Robert Taylor Drinks Cold Beer
        (Roots Trunks Divisions Cords Branches)""",
    },

    "rotator_cuff": {
        "caption": "Rotator Cuff — SITS muscles, attachments, movements, and clinical tests",
        "diagram": """\
┌──────────────────────────────────────────────────────────────┐
│            ROTATOR CUFF — SITS Mnemonic                      │
├──────────────┬──────────────────────┬────────────────────────┤
│ Muscle       │ Action               │ Clinical Test          │
├──────────────┼──────────────────────┼────────────────────────┤
│ Supraspinatus│ Abduction (0–15°)    │ Empty Can Test         │
│              │ Greater tubercle     │ Drop Arm Test          │
├──────────────┼──────────────────────┼────────────────────────┤
│ Infraspinatus│ Lateral rotation     │ External Rotation Lag  │
│              │ Greater tubercle     │ Palpate infraspinous   │
├──────────────┼──────────────────────┼────────────────────────┤
│ Teres Minor  │ Lateral rotation     │ Hornblower's Sign      │
│              │ Greater tubercle     │                        │
├──────────────┼──────────────────────┼────────────────────────┤
│ Subscapularis│ Medial rotation      │ Lift-off Test          │
│              │ Lesser tubercle      │ Bear Hug Test          │
└──────────────┴──────────────────────┴────────────────────────┘
All arise from scapula → insert on humerus → stabilise GH joint""",
    },

    "peripheral_nerves": {
        "caption": "Upper limb peripheral nerve territories — sensory and motor",
        "diagram": """\
         PERIPHERAL NERVE TERRITORIES — Upper Limb

 PALMAR SURFACE              DORSAL SURFACE
 ┌─────────────────┐         ┌─────────────────┐
 │  M  M  M  M     │         │  R  R  R        │
 │ (1)(2)(3)(4½)   │         │ (1½)(2)(3)      │
 │                 │         │                 │
 │        U  U  U  │         │  U  U  U        │
 │       (4½)(5)   │         │ (4½)(5)         │
 └─────────────────┘         └─────────────────┘
 M = Median  U = Ulnar  R = Radial  (finger numbers)

 Thenar eminence = Median    Hypothenar = Ulnar
 Lateral forearm = Musculocutaneous
 Medial arm/forearm = Medial cord cutaneous branches

 KEY INJURY PATTERNS:
 Median (carpal tunnel) → Ape hand, thenar wasting, loss of pinch
 Ulnar (cubital tunnel) → Claw hand (ring+little), Froment's sign
 Radial (spiral groove) → Wrist drop, loss of finger extension""",
    },

    "peripheral_nerves.median": {
        "caption": "Median nerve — course, motor, sensory and CTS clinical features",
        "diagram": """\
 MEDIAN NERVE (C6–T1) — Lateral + Medial cords

 Origin: Medial + Lateral cord of brachial plexus
    │
    ▼
 Medial to brachial artery → cubital fossa
    │
    ├── Anterior Interosseous N. (AIN) ──► FPL, FDP (index/middle), Pronator quadratus
    │
    ▼
 CARPAL TUNNEL (under flexor retinaculum)
    │
    ├── Motor: LOAF muscles
    │    L — Lumbricals 1 & 2
    │    O — Opponens pollicis     ← Opposition of thumb
    │    A — Abductor pollicis brevis
    │    F — Flexor pollicis brevis
    │
    └── Sensory: Lateral 3½ fingers (palmar) + thenar eminence

 COMPRESSION SIGNS:
  • Phalen's test (wrist flexion 60s → tingling)
  • Tinel's sign (tap carpal tunnel → tingling)
  • Ape hand deformity (thenar wasting)""",
    },

    "peripheral_nerves.ulnar": {
        "caption": "Ulnar nerve — course, cubital tunnel, claw hand and Froment's sign",
        "diagram": """\
 ULNAR NERVE (C8–T1) — Medial cord

 Origin: Medial cord → medial to axillary artery
    │
    ▼
 Posterior to medial epicondyle ← CUBITAL TUNNEL
    │                              (common compression site)
    ▼
 Guyon's Canal at wrist (2nd compression site)
    │
    ├── Motor: Hypothenar muscles (ADM, FDM, ODM)
    │          Interossei (all 4)
    │          Lumbricals 3 & 4
    │          Adductor pollicis ← FROMENT'S SIGN
    │
    └── Sensory: Medial 1½ fingers + hypothenar eminence

 INJURY PATTERNS:
  • Claw hand — ring + little fingers (lumbricals 3,4 lost)
  • Froment's sign — FPL compensates for lost adductor pollicis
  • Wartenberg's sign — little finger abducted at rest
  • Cubital tunnel → ulnar nerve decompression / transposition""",
    },

    "peripheral_nerves.radial": {
        "caption": "Radial nerve — spiral groove, wrist drop and cock-up splint",
        "diagram": """\
 RADIAL NERVE (C5–T1) — Posterior cord

 Origin: Posterior cord → posterior to axillary artery
    │
    ▼
 Winds around SPIRAL GROOVE of humerus ← Fracture risk!
    │
    ├── Motor (above elbow): Triceps, Brachioradialis, ECRL
    │
    ▼
 Bifurcates at lateral epicondyle:
    │
    ├── Superficial branch → Sensory: Lateral dorsal hand (1½ fingers)
    │
    └── Deep branch = PIN (Posterior Interosseous N.)
         └── Motor: Finger/wrist extensors, Supinator
                    EDC, EIP, EPL, EPB, APL

 INJURY AT SPIRAL GROOVE (Saturday night palsy / humeral #):
  • Wrist drop (cannot extend wrist)
  • Loss of finger extension (MCP joints drop)
  • Sensory loss — lateral dorsal hand (small area)
  • SPLINT: Cock-up (wrist extension) splint""",
    },

    "peripheral_nerves.axillary": {
        "caption": "Axillary nerve — deltoid, surgical neck fracture",
        "diagram": """\
 AXILLARY NERVE (C5–C6) — Posterior cord

 Origin: Posterior cord of brachial plexus
    │
    ▼
 Passes through Quadrilateral Space with posterior circumflex artery
    │
    ├── Motor: Deltoid (all three heads) ← ABduction 15–90°
    │          Teres Minor ← Lateral rotation
    │
    └── Sensory: Regimental badge area (lateral arm)

 ⚠️ INJURY RISK:
  • Surgical neck of humerus fracture
  • Anterior shoulder dislocation
  • Result: Loss of shoulder abduction, flat shoulder contour

 OT RELEVANCE:
  • ADL re-training for reaching overhead
  • Deltoid sling for positioning
  • Passive ROM to prevent adhesive capsulitis""",
    },

    "rotator_cuff.subscapularis": {
        "caption": "Subscapularis — medial rotation, lesser tubercle, lift-off test",
        "diagram": """\
 SUBSCAPULARIS

 Origin:   Subscapular fossa (anterior scapula surface)
 Insertion: Lesser tubercle of humerus
 Action:   MEDIAL (internal) rotation of humerus
           Adduction, extension assistance
 Nerve:    Upper + Lower subscapular nerves (C5-C6)

 CLINICAL TESTS:
  • Lift-off test: Back of hand off lumbar region → tests subscapularis
  • Bear hug test: Palm on opposite shoulder, resist pull-off
  • Belly press: Press abdomen keeping wrist straight

 OT CONTEXT:
  • Reach behind back (dressing, hygiene)
  • Tucking in shirt, fastening bra strap
  • Tears cause lateral rotation contracture
  • Post-op: avoid lateral rotation for 6 weeks

 Mnemonic: SubSCAPularis → Scapula (anterior) → Medial rotation""",
    },

    "rotator_cuff.supraspinatus": {
        "caption": "Supraspinatus — initiates abduction, empty can test, impingement",
        "diagram": """\
 SUPRASPINATUS

 Origin:   Supraspinous fossa
 Insertion: Superior facet of greater tubercle
 Action:   INITIATES abduction 0–15°
           (Deltoid takes over 15–90°, trapezius >90°)
 Nerve:    Suprascapular nerve (C5-C6)

 IMPINGEMENT ZONE: Passes under coracoacromial arch
  → Compression between acromion + greater tubercle
  → Most common rotator cuff tear site

 CLINICAL TESTS:
  • Empty Can (Jobe's): Arm 90° abd, 30° horiz flex, thumb down
    → Pain/weakness = supraspinatus tear/impingement
  • Drop Arm Test: Arm falls from 90° abduction
  • Neer's Sign: Passive forward flexion → impingement pain

 OT CONTEXT:
  • Any overhead activity (reaching, dressing)
  • Avoid painful arc 60–120°
  • Strengthening in pain-free range""",
    },

    "shoulder_joint": {
        "caption": "Glenohumeral joint — ball-and-socket, rotator cuff stabilisation",
        "diagram": """\
 GLENOHUMERAL (Shoulder) JOINT

 Type: Ball-and-socket (most mobile joint in body)
 Bones: Humeral head (ball) + Glenoid fossa (shallow socket)
 Labrum: Fibrocartilage ring that deepens socket

 STATIC STABILISERS:           DYNAMIC STABILISERS:
  • Glenohumeral ligaments       • Rotator cuff (SITS)
  • Glenoid labrum               • Long head biceps
  • Joint capsule                • Deltoid

 MOVEMENTS & PRIMARY MOVERS:
  Flexion (0–180°)    → Anterior deltoid, Pec major
  Extension (0–60°)   → Posterior deltoid, Latissimus
  Abduction (0–180°)  → Supraspinatus (0–15°), Deltoid (15–90°)
  Lateral rotation    → Infraspinatus, Teres minor
  Medial rotation     → Subscapularis, Pec major, Latissimus

 COMMON OT CONDITIONS:
  • Rotator cuff tear → impaired reaching/overhead ADLs
  • Adhesive capsulitis (frozen shoulder) → progressive ROM loss
  • Anterior instability/dislocation → axillary nerve risk""",
    },
}


def get_image_for_topic(topic: str) -> dict | None:
    """Return best matching diagram dict for a topic/concept ID."""
    if not topic:
        return None
    if topic in ANATOMY_DIAGRAMS:
        return ANATOMY_DIAGRAMS[topic]
    # Try top-level prefix
    top = topic.split(".")[0]
    if top in ANATOMY_DIAGRAMS:
        return ANATOMY_DIAGRAMS[top]
    # Keyword scan
    topic_lower = topic.lower()
    for key, val in ANATOMY_DIAGRAMS.items():
        if key in topic_lower or topic_lower in key:
            return val
    return None
