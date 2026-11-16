"""Streamlit UI, HelloFresh-style.  Run from the repo root:  streamlit run app.py

 - sidebar: hard filters (allergies, diet, max time), RAG on/off, show prompts
 - Plan tab: chat -> recipe cards with Pick buttons; questions get grounded answers;
   "What's in my fridge?" photo -> ingredients -> suggestions
 - My week tab: picked meals per day + merged shopping list
 - under each answer: which recipes were retrieved (and their similarity)
"""
from __future__ import annotations

import streamlit as st

import config
import db
import pipeline
import shopping
from llm import vision
from rag.filters import DIETS, HardFilters, allergen_names

st.set_page_config(page_title="Meal Planner", layout="wide")

if db.load_recipes().empty:
    st.error("No recipes loaded yet. Run `python -m data.prepare_data` "
             "(or `python -m data.prepare_data --sample` for a quick test), then reload.")
    st.stop()

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("target_day", None)
ss.setdefault("fridge", [])

with st.sidebar:
    st.header("Meal Planner")
    st.caption(f"LLM `{config.LLM_MODEL}` · embeddings `{config.EMBED_MODEL}` · all local via Ollama")
    st.subheader("Hard filters")
    diet = st.selectbox("Diet", DIETS)
    allergies = st.multiselect("Avoid (allergies)", allergen_names(),
                               format_func=lambda a: a.replace("_", " "))
    limit_time = st.checkbox("Limit cooking time", value=True)
    max_minutes = st.slider("Max minutes", 10, 240, 45, step=5, disabled=not limit_time)
    filters = HardFilters(diet, allergies, max_minutes if limit_time else None)
    n_allowed = len(pipeline.apply_filters(db.load_recipes(), filters))
    st.caption(f"{n_allowed:,} of {len(db.load_recipes()):,} recipes pass these filters")

    st.subheader("Settings")
    use_rag = st.toggle("Use RAG", value=config.USE_RAG,
                        help="Off = the LLM answers from its own knowledge, without the recipe database.")
    n_meals = st.slider("Meals per suggestion", 1, 7, config.N_SUGGESTIONS)
    mode = st.radio("Chat mode", ["Auto", "Suggest meals", "Ask a question"], horizontal=True,
                    help="Auto guesses from your message whether you want recipes or an answer.")
    show_prompts = st.checkbox("Show prompts (4T's)")
    if st.button("Clear chat"):
        ss.messages, ss.target_day = [], None
        st.rerun()

def render_card(card: pipeline.Card, key: str):
    r = card.recipe
    with st.container(border=True):
        st.markdown(f"**{r['name']}**")
        st.caption(f"{r['total_minutes']} min · {r['calories']:.0f} kcal · "
                   f"{r['protein']:.0f} g protein · {r['category'] or ''}")
        st.write(card.reason)
        if ss.fridge:
            used = vision.fridge_matches(r, ss.fridge)
            if used:
                st.caption("Uses from your fridge: " + ", ".join(used))
        with st.expander("Ingredients & steps"):
            st.markdown(pipeline.recipe_full(r).split("\n", 1)[1])
        label = f"Pick for {db.DAYS[ss.target_day]}" if ss.target_day is not None else "Pick"
        if st.button(label, key=key, use_container_width=True):
            day = db.add_pick(int(r["recipe_id"]), ss.target_day)
            ss.target_day = None
            st.toast(f"Added to {db.DAYS[day]}" if day is not None else "Your week is full")
            st.rerun()


def render_retrieval(result):
    hits = getattr(result, "hits", [])
    if not hits:
        return
    with st.expander(f"Retrieved recipes ({len(hits)}) – what RAG gave the LLM"):
        lookup = db.recipes_by_id()
        for h in hits:
            st.markdown(f"`{h.similarity:.3f}` {lookup[h.recipe_id]['name']} (id {h.recipe_id})")


def render_result(msg: dict, idx: int):
    res, kind = msg.get("result"), msg.get("kind")
    if kind == "suggest":
        v = res.validation
        if res.cards:
            cols = st.columns(3)
            for i, card in enumerate(res.cards):
                with cols[i % 3]:
                    render_card(card, key=f"pick-{idx}-{card.recipe['recipe_id']}")
        for name, reason in res.unverified:
            st.markdown(f"- **{name}** – {reason}  \n   *not from the recipe database (RAG off)*")
        problems = []
        if v.invented:
            problems.append(f"{len(v.invented)} invented id(s) removed: {', '.join(v.invented)}")
        if v.rule_violations:
            problems.append(f"{len(v.rule_violations)} rule violation(s) removed: {'; '.join(v.rule_violations)}")
        if v.parse_error:
            problems.append(v.parse_error)
        if problems:
            st.caption("Validator: " + " · ".join(problems))
    render_retrieval(res)
    if show_prompts and res is not None:
        with st.expander("Prompt sent to the LLM"):
            st.code(res.system_prompt, language="markdown")
            st.code(res.user_prompt, language="markdown")


def run(message: str, forced: str | None = None):
    kind, day = pipeline.route(message)
    if forced:
        kind = forced
    if day is not None:
        ss.target_day = day
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-6:]]
    ss.messages.append({"role": "user", "content": message})
    with st.spinner("Thinking locally..."):
        try:
            if kind == "suggest":
                res = pipeline.suggest(message, filters, use_rag, n_meals)
                text = res.intro or ("Here are some ideas:" if res.cards or res.unverified else
                                     "I couldn't find good matches. Try other words or relax the filters.")
                if ss.target_day is not None:
                    text += f"\n\n*Pick one to replace {db.DAYS[ss.target_day]}.*"
            else:
                picked = list(db.list_picks().values())
                res = pipeline.answer(message, picked, filters, use_rag, history)
                text = res.text
        except Exception as e:
            res, kind, text = None, "error", f" {e}\n\nIs Ollama running and are the models pulled?"
    ss.messages.append({"role": "assistant", "content": text, "kind": kind, "result": res})


plan_tab, week_tab = st.tabs(["Plan", "My week"])

with plan_tab:
    with st.expander("What's in my fridge?", expanded=False):
        st.caption(f"Take or upload a photo; `{config.VISION_MODEL}` lists what it sees, locally.")
        src = st.radio("Source", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
        photo = (st.file_uploader("Fridge photo", type=["jpg", "jpeg", "png", "webp"])
                 if src == "Upload" else st.camera_input("Fridge photo"))
        if photo and st.button("Detect ingredients"):
            with st.spinner("Looking in your fridge..."):
                try:
                    ss.fridge = vision.detect_ingredients(photo.getvalue())
                except Exception as e:
                    st.error(f"Vision failed: {e}")
            if not ss.fridge:
                st.warning("No ingredients recognised. Try a clearer photo or type them below.")
        fridge_text = st.text_input("Ingredients you have (edit if the model got something wrong)",
                                    ", ".join(ss.fridge), placeholder="e.g. eggs, spinach, cheddar cheese")
        ss.fridge = [i.strip() for i in fridge_text.split(",") if i.strip()]
        if ss.fridge and st.button("Suggest meals with these", type="primary"):
            run(f"Meals I can make using: {', '.join(ss.fridge)}", forced="suggest")

    if not ss.messages:
        st.info("Try: *cheap high-protein dinners* · *something quick with chicken* · "
                "*replace Tuesday with something vegetarian* · *what can I use instead of cream?*")
    for i, msg in enumerate(ss.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("result") is not None:
                render_result(msg, i)

forced = {"Suggest meals": "suggest", "Ask a question": "answer"}.get(mode)
if prompt := st.chat_input("What do you feel like this week?"):
    run(prompt, forced)
    st.rerun()

with week_tab:
    picks = db.list_picks()
    lookup = db.recipes_by_id()
    if not picks:
        st.info("No meals picked yet. Ask for suggestions in the Plan tab and press **Pick**.")
    else:
        for day in range(7):
            c1, c2, c3 = st.columns([1, 4, 1])
            c1.markdown(f"**{db.DAYS[day]}**")
            rid = picks.get(day)
            if rid in lookup:
                r = lookup[rid]
                c2.markdown(f"{r['name']}  \n<small>{r['total_minutes']} min · "
                            f"{r['calories']:.0f} kcal · {r['protein']:.0f} g protein</small>",
                            unsafe_allow_html=True)
                if c3.button("Remove", key=f"rm-{day}"):
                    db.remove_pick(day)
                    st.rerun()
            else:
                c2.caption("–")
        st.divider()
        rows = shopping.build_list(db.get_recipes(list(picks.values())))
        st.subheader("Shopping list")
        st.dataframe([{k: v for k, v in r.items() if k != "pantry"} for r in rows if not r["pantry"]],
                     use_container_width=True, hide_index=True)
        pantry = [r["ingredient"] for r in rows if r["pantry"]]
        if pantry:
            st.caption("Check you have: " + ", ".join(pantry))
        st.download_button("Download shopping list", shopping.as_text(rows), "shopping_list.txt")
        if st.button("Clear week"):
            db.clear_picks()
            st.rerun()
