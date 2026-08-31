from __future__ import annotations

import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
_libs = str(Path(__file__).resolve().parent.parent / "libs")

if sys.version_info[:2] == (3, 12):
    sys.path = [_libs, _root] + [p for p in sys.path if "Python313" not in p and "Roaming" not in p and p not in (_libs, _root)]
else:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from src.intent_router import IntentRouter, IntentType
from src.dialogue_state import DialogueState
from src.proactive_guidance import ProactiveGuidance, ALLOWED_ATTRIBUTES
from src.query_synthesizer import QuerySynthesizer
from src.data_parser import load_catalog
from starter.agent import Agent


def test_intent_router():
    print("Testing IntentRouter...")
    # Buying
    intent, conf = IntentRouter.classify("I'm looking for running shoes. A key requirement is: 100% cotton.", turn=1)
    assert intent == IntentType.BUYING, f"Expected BUYING, got {intent}"

    # Browsing
    intent, conf = IntentRouter.classify("I'm looking for running shoes, but I'm still exploring.", turn=1)
    assert intent == IntentType.BROWSING, f"Expected BROWSING, got {intent}"

    # Override
    override_msg = "Actually, ignore my earlier preference. What I need is: genuine leather boots."
    intent, conf = IntentRouter.classify(override_msg, turn=3)
    assert intent == IntentType.INTENT_OVERRIDE, f"Expected INTENT_OVERRIDE, got {intent}"
    payload = IntentRouter.extract_override_payload(override_msg)
    assert payload is not None and "leather" in payload.lower(), f"Failed payload extraction: {payload}"

    # Boundary
    intent, conf = IntentRouter.classify("I don't have a preference for color; please use your judgment.", turn=2)
    assert intent == IntentType.BOUNDARY, f"Expected BOUNDARY, got {intent}"

    print("IntentRouter passed!")


def test_dialogue_state():
    print("Testing DialogueState tracking...")
    state = DialogueState(session_id="test_sess")
    
    # Turn 1: Buying with cotton constraint
    intent = state.add_turn(1, "I'm looking for graphic t-shirts. A key requirement is: cotton.")
    assert "graphic t-shirts" in state.categories or any("graphic" in c for c in state.categories)
    assert "cotton" in state.active_slots["material"]

    # Record agent action: asked color
    state.record_agent_action("color")
    assert state.last_asked_attribute == "color"

    # Turn 2: Constraint reveal for color
    intent = state.add_turn(2, "For that, what matters is: color: black; budget around $25.")
    assert "black" in state.active_slots["color"]
    assert any("25" in b for b in state.active_slots["budget"])

    # Query synthesis
    query = QuerySynthesizer.synthesize_query(state)
    assert "cotton" in query.lower()
    assert "black" in query.lower()

    # Turn 3: Override to polyester
    intent = state.add_turn(3, "Actually, ignore my earlier preference. What I need is: 100% polyester athletic shirts.")
    assert state.override_occurred is True
    assert "cotton" in state.purged_terms
    assert "polyester" in state.active_slots["material"] or any("polyester" in m for m in state.active_slots["material"])

    # Query synthesis after override
    new_query = QuerySynthesizer.synthesize_query(state)
    assert "cotton" not in new_query.lower(), f"Purged term 'cotton' still in query: {new_query}"
    assert "polyester" in new_query.lower(), f"New term 'polyester' missing in query: {new_query}"

    print("DialogueState passed!")


def test_proactive_guidance():
    print("Testing ProactiveGuidance...")
    catalog = load_catalog("data/catalog.jsonl")
    sample_items = [catalog.items_by_asin[asin] for asin in catalog.asin_list[:30]]

    state = DialogueState(session_id="test_guidance")
    state.add_turn(1, "I'm looking for casual shoes, but I'm still exploring.")
    
    # Select first attribute
    attr1 = ProactiveGuidance.select_attribute(state, sample_items, turn=1)
    assert attr1 in ALLOWED_ATTRIBUTES, f"Invalid attribute: {attr1}"
    state.record_agent_action(attr1)

    # Turn 2: User boundary response
    state.add_turn(2, f"I don't have a preference for {attr1}; please use your judgment.")
    assert attr1 in state.rejected_attributes

    # Select second attribute (must not be attr1)
    attr2 = ProactiveGuidance.select_attribute(state, sample_items, turn=2)
    assert attr2 != attr1, f"Expected different attribute, got same {attr2}"
    assert attr2 in ALLOWED_ATTRIBUTES

    print("ProactiveGuidance passed!")


def test_agent_end_to_end():
    print("Testing Agent end-to-end multi-turn...")
    agent = Agent("data/catalog.jsonl", "data/catalog_embeddings.npy")
    sess_id = "test_agent_session"
    agent.reset(sess_id, {"age": 30, "gender": "male"})

    # Turn 1
    resp1 = agent.respond(sess_id, "I'm looking for men's running shoes, but I'm still exploring.", turn=1, top_k=10)
    assert len(resp1["recommendations"]) == 10
    assert resp1["ask_attribute"] is not None
    assert resp1["usage"]["prompt_tokens"] == 0

    # Turn 2: reply with disclosed material
    asked_attr = resp1["ask_attribute"]
    resp2 = agent.respond(sess_id, f"For that, what matters is: breathable mesh; lightweight.", turn=2, top_k=10)
    assert len(resp2["recommendations"]) == 10
    assert resp2["ask_attribute"] != asked_attr

    # Turn 3: override
    resp3 = agent.respond(sess_id, "Actually, ignore my earlier preference. What I need is: waterproof leather trail boots.", turn=3, top_k=10)
    assert len(resp3["recommendations"]) == 10

    print("Agent end-to-end passed!")


if __name__ == "__main__":
    test_intent_router()
    test_dialogue_state()
    test_proactive_guidance()
    test_agent_end_to_end()
    print("All Phase 2 unit tests passed successfully!")

