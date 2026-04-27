from __future__ import annotations

DISTRACTOR_SNIPPETS: list[str] = [
    "At a library, 14 red chairs were moved to one room and 9 blue chairs to another. The chairs were cleaned on Friday afternoon.",
    "Nina packed 6 postcards and 11 stickers for a trip last month. She kept them in a small yellow box near the window.",
    "A pet shop sold 8 collars on Monday and 13 bowls on Tuesday. The owner painted the front door green the next day.",
    "Owen counted 12 pinecones and 7 smooth stones during a walk. He placed them on a shelf beside two candles.",
    "A music room had 5 drums and 16 flutes stored after class. The teacher locked the cabinet before lunch.",
    "Tara bought 9 markers and 4 notebooks on Saturday. She wrote her name on each notebook with purple ink.",
    "The bakery displayed 15 muffins and 6 pies in the front case. A new sign was taped to the glass door.",
    "Leo folded 10 paper stars and 3 paper cranes for decoration. He hung them above a wooden desk.",
    "A garden shed contained 11 clay pots and 8 small shovels. Rain started just after the tools were arranged.",
    "Mira sorted 7 ribbons and 14 buttons into separate jars. She left the jars on a round table near the lamp.",
]


def get_distractor(example_index: int) -> str:
    return DISTRACTOR_SNIPPETS[example_index % len(DISTRACTOR_SNIPPETS)]
