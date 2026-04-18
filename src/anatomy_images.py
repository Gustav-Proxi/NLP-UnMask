"""
Anatomical diagrams for visual hints.

Each entry may have:
  caption      — shown above the visual
  diagram      — ASCII/Unicode fallback (always present)
  image_file   — filename in public/anatomy/ (Gray's Anatomy, public domain); shown when present
"""

import os as _os
_IMG_DIR = _os.path.join(_os.path.dirname(__file__), "..", "public", "anatomy")

ANATOMY_DIAGRAMS: dict[str, dict] = {

    # ── Spinal cord ──────────────────────────────────────────────────────────
    "spinal_cord.anatomy": {
        "caption": "Spinal cord cross-section — dorsal/ventral horns, rami, grey/white matter",
        "image_file": "spinal_cord.png",
        "diagram": """\
 SPINAL CORD CROSS-SECTION

        Dorsal (posterior)
              │
   ┌──────────┴──────────┐
   │   Dorsal horn  ●    │  ← Sensory (afferent) neurons
   │                     │
   │   ●  Grey matter ●  │
   │                     │
   │   Ventral horn ●    │  ← Motor (efferent) neurons
   └──────────┬──────────┘
              │
        Ventral (anterior)

 ROOTS:
  Dorsal root (sensory) ─┐
                          ├─► Spinal nerve ─► Anterior ramus ─► Brachial plexus
  Ventral root (motor)  ─┘

 Each spinal nerve = dorsal + ventral root combined
 Anterior rami (C5–T1) → form the brachial plexus
 Posterior rami → innervate back muscles (NOT part of brachial plexus)""",
    },

    "spinal_cord.anterior_rami": {
        "caption": "Anterior rami C5–T1 — origin of the brachial plexus",
        "diagram": """\
 ANTERIOR RAMI → BRACHIAL PLEXUS CONTRIBUTION

 Vertebra  │  Spinal Level  │  Contribution
 ──────────┼────────────────┼──────────────────────────────
    C5     │  Cervical 5    │  Upper trunk (with C6)
    C6     │  Cervical 6    │  Upper trunk (with C5)
    C7     │  Cervical 7    │  Middle trunk (alone)
    C8     │  Cervical 8    │  Lower trunk (with T1)
    T1     │  Thoracic 1    │  Lower trunk (with C8)
 ──────────┴────────────────┴──────────────────────────────

 Each spinal nerve exits through the INTERVERTEBRAL FORAMEN
 then immediately splits:
   • Posterior ramus → back muscles / skin
   • Anterior ramus  → limbs / anterior trunk

 KEY CLINICAL NOTE:
  Erb's palsy  → C5–C6 injury  (upper trunk, waiter's tip)
  Klumpke's   → C8–T1 injury  (lower trunk, claw hand)
  Total plexus→ C5–T1 injury  (flail arm)""",
    },

    # ── Brachial plexus sub-diagrams ─────────────────────────────────────────
    "brachial_plexus.origin": {
        "caption": "Brachial plexus origins — C5–T1 anterior rami and first branches",
        "image_file": "brachial_plexus.png",
        "diagram": """\
 BRACHIAL PLEXUS ORIGINS

 Cervical vertebrae C5–C8 + Thoracic T1 contribute via anterior rami

         C5 ─────────────────────────────────────────►
         C6 ─────────────────────────────────────────►  Upper trunk
         C7 ─────────────────────────────────────────►  Middle trunk
         C8 ─────────────────────────────────────────►  Lower trunk
         T1 ─────────────────────────────────────────►

 Mnemonic: 5 ROOTS (five fingers spread = C5 C6 C7 C8 T1)

 BRANCHES DIRECTLY FROM ROOTS (before trunks):
  C5–C7  → Long thoracic nerve → Serratus anterior
  C5     → Dorsal scapular nerve → Rhomboids, Levator scapulae
  C5–C8  → Phrenic nerve (partial) → Diaphragm

 EXIT POINT: Scalene triangle (between anterior + middle scalene muscles)
 ⚠️  Compression here → Thoracic outlet syndrome""",
    },

    "brachial_plexus.trunks": {
        "caption": "Brachial plexus trunks — upper, middle, lower and Erb's/Klumpke's point",
        "image_file": "brachial_plexus.png",
        "diagram": """\
 BRACHIAL PLEXUS — TRUNKS

 Root    ──► Trunk          Common injury
 ─────────────────────────────────────────────────────
 C5 + C6 ──► UPPER trunk   ← Erb's point (Erb's palsy)
   C7    ──► MIDDLE trunk
 C8 + T1 ──► LOWER trunk   ← Klumpke's (C8/T1 stretch)
 ─────────────────────────────────────────────────────

 Each trunk immediately divides into:
   • ANTERIOR division → supply flexor compartments
   • POSTERIOR division → supply extensor compartments

 BRANCHES FROM TRUNKS:
  Upper trunk → Suprascapular nerve (C5-C6)
                  → Supraspinatus + Infraspinatus

 ERB'S PALSY (C5-C6 / upper trunk):
  • Loss: shoulder ABduction, external rotation, elbow flexion
  • Posture: "Waiter's tip" — arm adducted, internally rotated
  • Mechanism: lateral flexion away + shoulder depression

 KLUMPKE'S PALSY (C8-T1 / lower trunk):
  • Loss: intrinsic hand muscles
  • Posture: Claw hand
  • Mechanism: upward traction of arm (birth, overhead grab)""",
    },

    "brachial_plexus.divisions": {
        "caption": "Brachial plexus divisions — anterior vs posterior, flexor vs extensor",
        "image_file": "brachial_plexus.png",
        "diagram": """\
 BRACHIAL PLEXUS — DIVISIONS

 Each trunk splits into ANTERIOR and POSTERIOR divisions:

 Trunk          │ Division    │ → Cord
 ───────────────┼─────────────┼───────────────
 Upper trunk    │ Anterior  ──┤──► Lateral cord
                │ Posterior ──┤──┐
 Middle trunk   │ Anterior  ──┤──► Lateral cord
                │ Posterior ──┤  │
 Lower trunk    │ Anterior  ──┤──► Medial cord
                │ Posterior ──┴──► Posterior cord
 ───────────────┴─────────────┴───────────────

 RULE: Anterior divisions → LATERAL and MEDIAL cords
                           → innervate FLEXORS
       Posterior divisions → POSTERIOR cord
                           → innervate EXTENSORS

 EXAM TIP:
  All 3 posterior divisions combine → POSTERIOR cord
  This explains why radial + axillary nerves (posterior cord)
  innervate all extensors of the arm""",
    },

    "brachial_plexus.cords": {
        "caption": "Brachial plexus cords — lateral, medial, posterior and their branches",
        "image_file": "brachial_plexus.png",
        "diagram": """\
 BRACHIAL PLEXUS — CORDS
 (named by position relative to axillary artery)

 LATERAL CORD (C5–C7):
   Musculocutaneous N. ──► Biceps, Brachialis, Coracobrachialis
   Lateral root of Median N. ──┐
                                ├──► Median nerve (C5–T1)
 MEDIAL CORD (C8–T1):          │
   Medial root of Median N. ───┘
   Ulnar N. ──► intrinsic hand, medial 1½ fingers
   Medial cutaneous N. (arm + forearm)

 POSTERIOR CORD (C5–T1):
   Axillary N.  ──► Deltoid, Teres minor, lateral arm sensation
   Radial N.    ──► All extensors upper limb, lateral dorsal hand

 Memory: LUMBAR (Lateral=Upper, Medial=BrAchial, Ropes)
         or: "My Aunt Raped Uncle Robert"
         (Median, Axillary, Radial, Ulnar, musculocutaneous)

 CORD BRANCHES SUMMARY:
  Lateral  → Musculocutaneous + lateral Median
  Medial   → Ulnar + medial Median + cutaneous branches
  Posterior→ Axillary + Radial (+ upper/lower subscapular, thoracodorsal)""",
    },

    "brachial_plexus.terminal_branches": {
        "caption": "5 terminal branches — roots, motor targets and key injury patterns",
        "image_file": "brachial_plexus.png",
        "diagram": """\
 BRACHIAL PLEXUS — TERMINAL BRANCHES

 Nerve              │ Roots    │ Key Motors        │ Injury Pattern
 ───────────────────┼──────────┼───────────────────┼──────────────────────
 Musculocutaneous   │ C5–C7   │ Biceps, Brachialis│ Weak elbow flexion
 Axillary           │ C5–C6   │ Deltoid, T. minor │ Flat shoulder, no ABd
 Radial             │ C5–T1   │ All extensors     │ Wrist drop
 Median             │ C5–T1   │ LOAF + forearm Fx │ Ape hand, CTS
 Ulnar              │ C8–T1   │ Intrinsics, Hypo  │ Claw hand (4th/5th)
 ───────────────────┴──────────┴───────────────────┴──────────────────────

 SPLINT GUIDE (OT critical):
  Radial injury → Cock-up (wrist extension) splint
  Ulnar injury  → Anti-claw splint (MCP block ring+little)
  Median injury → Thumb abduction/opposition splint
  Combined      → Resting hand splint (all injured)

 Memory: "My Aunt Really Missed Uncle"
         (Musculocutaneous, Axillary, Radial, Median, Ulnar)""",
    },

    # ── Rotator cuff sub-diagrams ────────────────────────────────────────────
    "rotator_cuff.muscles": {
        "caption": "Rotator cuff SITS muscles — origins, insertions, nerves, OT tests",
        "image_file": "shoulder_joint.png",
        "diagram": """\
 ROTATOR CUFF — SITS OVERVIEW

 Muscle          │ Origin              │ Insert   │ Action       │ Nerve
 ────────────────┼─────────────────────┼──────────┼──────────────┼────────────
 Supraspinatus   │ Supraspinous fossa  │ Greater  │ ABd 0–15°    │ Suprascp
 Infraspinatus   │ Infraspinous fossa  │ tubercle │ Lat. rot.    │ Suprascp
 Teres Minor     │ Lateral border scp  │ Greater  │ Lat. rot.    │ Axillary
 Subscapularis   │ Subscapular fossa   │ Lesser   │ Med. rot.    │ Subscplar
 ────────────────┴─────────────────────┴──────────┴──────────────┴────────────

 ALL FOUR: arise from scapula → insert on humerus → stabilise GH joint

 CLINICAL TEST QUICK-REFERENCE:
  Empty Can (Jobe's)     → Supraspinatus
  External Rotation Lag  → Infraspinatus
  Hornblower's Sign      → Teres Minor (can't externally rotate in ABd)
  Lift-off / Belly Press → Subscapularis

 MOST COMMON TEAR:
  Supraspinatus (impingement under coracoacromial arch at 60–120° ABd arc)""",
    },

    "rotator_cuff.infraspinatus": {
        "caption": "Infraspinatus — external rotation, suprascapular nerve, OT relevance",
        "image_file": "shoulder_joint.png",
        "diagram": """\
 INFRASPINATUS

 Origin:    Infraspinous fossa (posterior scapula, below spine)
 Insertion: Middle facet of GREATER tubercle of humerus
 Action:    LATERAL (external) rotation of humerus
            Weak horizontal abduction
 Nerve:     Suprascapular nerve (C5–C6) — same as supraspinatus

 ┌────────────────────────────────────────────┐
 │ Posterior view — scapula                   │
 │                                            │
 │  ─── Spine of scapula ─────────────────── │
 │                                            │
 │  Infraspinous fossa [INFRASPINATUS origin] │
 │                              ↘             │
 │                        Greater tubercle    │
 └────────────────────────────────────────────┘

 CLINICAL TESTS:
  • External Rotation Lag Sign: arm at 90° ABd, examiner passively
    externally rotates → release → arm falls = infraspinatus tear
  • External rotation resisted test at 0° (Patte test)
  • Hornblower's sign tests TERES MINOR (not infraspinatus)

 OT RELEVANCE:
  • Handshake, turning a doorknob, reaching to the side
  • Loss → difficulty with most lateral rotation ADLs
  • Post-op: avoid medial rotation for 6 weeks (protects repair)""",
    },

    "rotator_cuff.teres_minor": {
        "caption": "Teres Minor — lateral rotation, axillary nerve, Hornblower's sign",
        "image_file": "shoulder_joint.png",
        "diagram": """\
 TERES MINOR

 Origin:    Upper 2/3 of LATERAL BORDER of scapula
 Insertion: Inferior facet of GREATER tubercle of humerus
 Action:    LATERAL (external) rotation of humerus
            Weak adduction, weak extension
 Nerve:     AXILLARY nerve (C5–C6)  ← different from infraspinatus!

 RELATIONSHIP TO INFRASPINATUS:
  Both: lateral rotation + greater tubercle
  Difference: teres minor = axillary nerve; infraspin = suprascapular

 CLINICAL TEST:
  • Hornblower's Sign: arm in 90° ABd, 90° elbow flex → examiner
    releases → patient cannot hold position = teres minor + infraspinatus
  • Specifically tests COMBINED external rotation in abduction

 QUADRILATERAL SPACE:
  Teres minor forms the SUPERIOR BORDER of the quadrilateral space
  → Axillary nerve and posterior circumflex artery pass through here
  → Quadrilateral space syndrome: compression of axillary N.

 OT RELEVANCE:
  • Assists any lateral rotation activity (reaching, grooming)
  • Often torn alongside infraspinatus in massive cuff tears
  • Small muscle — rarely torn in isolation""",
    },

    "brachial_plexus": {
        "caption": "Brachial Plexus — trace C5→T1 through trunks, divisions, cords to terminal branches",
        "image_file": "brachial_plexus.png",
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
        "image_file": "shoulder_joint.png",
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
        "image_file": "peripheral_nerves.png",
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
        "image_file": "median_nerve.png",
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
        "image_file": "ulnar_nerve.png",
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
        "image_file": "radial_nerve.png",
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
        "image_file": "axillary_nerve.png",
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
        "image_file": "shoulder_joint.png",
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
        "image_file": "shoulder_joint.png",
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
