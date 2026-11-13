"""Writes data/sample/sample_recipes.csv: a SMALL hand-made stand-in for the Kaggle file,
in the same raw CSV format (R-style c("...") lists, ISO-8601 durations).

It exists so a quick demo runs without downloading Kaggle data:
    python data/sample/make_sample.py
    python -m data.prepare_data --sample
These are illustrative recipes, not Food.com data. Use the real dataset for results in the synopsis.
"""
import csv
from pathlib import Path

RECIPES = [
    ("Easy Chicken Curry", "Curries", ["Chicken", "Asian", "< 60 Mins"], [("500", "chicken breast"), ("1", "onion"), ("2", "curry powder"), ("1", "coconut milk"), ("1", "rice")], 35, 4, 18, 45, 34, 3, 6, 520, 4.6, 120, ["Brown the chicken.", "Fry onion with curry powder.", "Add coconut milk and simmer 15 minutes.", "Serve with rice."]),
    ("Red Lentil Soup", "Lentil", ["Vegan", "Healthy", "< 60 Mins"], [("250", "red lentils"), ("1", "onion"), ("2", "carrots"), ("1", "canned tomatoes"), ("1", "vegetable broth"), ("1", "cumin")], 30, 4, 4, 48, 18, 9, 7, 610, 4.7, 85, ["Soften onion and carrots.", "Add lentils, tomatoes and broth.", "Simmer 20 minutes and blend."]),
    ("Chickpea Spinach Curry", "Vegetable", ["Vegan", "Indian", "< 30 Mins"], [("2", "chickpeas"), ("1", "canned tomatoes"), ("200", "spinach"), ("1", "onion"), ("1", "garam masala"), ("1", "rice")], 25, 4, 9, 70, 17, 11, 8, 480, 4.5, 64, ["Fry onion and spices.", "Add tomatoes and chickpeas.", "Simmer 10 minutes, stir in spinach.", "Serve with rice."]),
    ("Classic Beef Lasagna", "Meat", ["Beef", "Italian", "Oven"], [("500", "ground beef"), ("12", "lasagna noodles"), ("2", "canned tomatoes"), ("250", "ricotta cheese"), ("200", "mozzarella cheese"), ("1", "egg")], 90, 8, 28, 40, 36, 3, 8, 890, 4.8, 310, ["Brown beef and add tomatoes.", "Mix ricotta and egg.", "Layer noodles, sauce and cheese.", "Bake 45 minutes."]),
    ("Baked Salmon with Potatoes", "Salmon", ["Fish", "Healthy", "< 60 Mins"], [("500", "salmon fillets"), ("800", "potatoes"), ("300", "green beans"), ("1", "lemon"), ("2", "olive oil"), ("1", "dill")], 35, 4, 22, 38, 32, 5, 3, 350, 4.6, 95, ["Roast potatoes 15 minutes.", "Add salmon and beans.", "Bake 12 more minutes."]),
    ("Spaghetti Bolognese", "Spaghetti", ["Beef", "Kid Friendly", "< 60 Mins"], [("400", "ground beef"), ("400", "spaghetti"), ("2", "canned tomatoes"), ("1", "onion"), ("2", "carrots"), ("2", "garlic")], 40, 4, 20, 75, 35, 6, 10, 700, 4.5, 210, ["Brown beef with onion and garlic.", "Add carrots and tomatoes, simmer 20 minutes.", "Cook spaghetti and serve."]),
    ("Black Bean Chili", "Beans", ["Vegan", "Mexican", "< 60 Mins"], [("2", "black beans"), ("1", "kidney beans"), ("2", "canned tomatoes"), ("1", "red bell pepper"), ("1", "onion"), ("1", "chili powder")], 40, 4, 5, 62, 19, 17, 9, 720, 4.4, 58, ["Fry onion and pepper.", "Add spices, beans and tomatoes.", "Simmer 25 minutes."]),
    ("Broccoli Cheddar Pasta", "Pasta", ["Vegetarian", "< 30 Mins", "Kid Friendly"], [("400", "pasta"), ("1", "broccoli"), ("150", "cheddar cheese"), ("2", "milk"), ("2", "butter"), ("2", "flour")], 25, 4, 21, 72, 23, 5, 6, 560, 4.3, 77, ["Cook pasta, add broccoli at the end.", "Make a cheese sauce with butter, flour, milk and cheddar.", "Mix together."]),
    ("Tofu Vegetable Stir Fry", "Stir Fry", ["Vegan", "Asian", "< 30 Mins"], [("400", "firm tofu"), ("250", "rice noodles"), ("400", "mixed vegetables"), ("3", "soy sauce"), ("1", "sesame oil"), ("1", "ginger")], 25, 4, 14, 55, 21, 4, 6, 900, 4.4, 49, ["Fry tofu until crisp.", "Stir fry vegetables with ginger.", "Add noodles, tofu and sauce."]),
    ("Spanish Potato Omelette", "Breakfast", ["Vegetarian", "Spanish", "< 30 Mins"], [("6", "eggs"), ("4", "potatoes"), ("1", "onion"), ("3", "olive oil"), ("", "salt")], 30, 4, 19, 28, 15, 3, 2, 380, 4.5, 66, ["Fry sliced potatoes and onion.", "Pour over beaten eggs.", "Cook until set, flip and finish."]),
    ("Chicken Burrito Bowl", "Chicken", ["Chicken", "Mexican", "< 30 Mins"], [("400", "chicken thighs"), ("1", "black beans"), ("1", "rice"), ("1", "corn"), ("1", "avocado"), ("1", "salsa")], 30, 4, 17, 60, 36, 9, 5, 650, 4.6, 140, ["Season and fry the chicken.", "Warm beans and corn.", "Serve over rice with avocado and salsa."]),
    ("Tuna Pasta Salad", "Pasta", ["Fish", "< 30 Mins", "Lunch"], [("300", "pasta"), ("2", "tuna"), ("1", "corn"), ("1", "cucumber"), ("2", "mayonnaise"), ("2", "eggs")], 20, 4, 15, 58, 30, 3, 4, 520, 4.2, 40, ["Cook and cool the pasta.", "Mix with tuna, corn, cucumber and mayonnaise.", "Top with boiled eggs."]),
    ("Roasted Vegetable Couscous", "Vegetable", ["Vegan", "< 60 Mins", "Healthy"], [("250", "couscous"), ("1", "zucchini"), ("1", "red bell pepper"), ("1", "red onion"), ("1", "chickpeas"), ("2", "olive oil")], 40, 4, 10, 66, 15, 9, 7, 300, 4.3, 37, ["Roast vegetables and chickpeas 25 minutes.", "Prepare couscous with boiling water.", "Mix and season."]),
    ("Pork Fried Rice", "Pork", ["Pork", "Asian", "< 30 Mins"], [("300", "pork"), ("3", "cooked rice"), ("2", "eggs"), ("1", "peas"), ("3", "soy sauce"), ("2", "green onions")], 25, 4, 16, 55, 26, 3, 3, 980, 4.4, 88, ["Fry the pork.", "Scramble eggs.", "Fry rice with peas, add pork, egg and soy sauce."]),
    ("Greek Yogurt Chicken Wraps", "Chicken", ["Chicken", "< 30 Mins", "High Protein"], [("400", "chicken breast"), ("8", "tortillas"), ("2", "greek yogurt"), ("1", "cucumber"), ("2", "tomatoes"), ("1", "lettuce")], 25, 4, 12, 44, 40, 4, 6, 610, 4.5, 72, ["Grill the chicken and slice.", "Mix yogurt with grated cucumber.", "Fill tortillas with everything."]),
    ("Shrimp Garlic Linguine", "Pasta", ["Shellfish", "Italian", "< 30 Mins"], [("400", "shrimp"), ("400", "linguine"), ("4", "garlic"), ("3", "butter"), ("1", "lemon"), ("1", "parsley")], 25, 4, 15, 74, 33, 3, 2, 640, 4.6, 101, ["Cook linguine.", "Fry garlic in butter, add shrimp.", "Toss with pasta, lemon and parsley."]),
    ("Mushroom Risotto", "Rice", ["Vegetarian", "Italian", "< 60 Mins"], [("300", "arborio rice"), ("300", "mushrooms"), ("1", "onion"), ("1", "vegetable broth"), ("50", "parmesan cheese"), ("2", "butter")], 45, 4, 14, 68, 14, 3, 3, 700, 4.4, 55, ["Fry mushrooms and onion.", "Add rice and broth bit by bit for 20 minutes.", "Stir in butter and parmesan."]),
    ("Turkey Meatballs with Rice", "Meat", ["Poultry", "Kid Friendly", "< 60 Mins"], [("500", "ground turkey"), ("1", "egg"), ("1", "breadcrumbs"), ("1", "canned tomatoes"), ("1", "rice"), ("1", "onion")], 45, 4, 14, 52, 34, 3, 7, 600, 4.3, 46, ["Mix turkey, egg and breadcrumbs, form balls.", "Brown and simmer in tomato sauce.", "Serve with rice."]),
    ("Peanut Noodles with Vegetables", "Asian", ["Vegan", "< 30 Mins", "Thai"], [("250", "noodles"), ("3", "peanut butter"), ("2", "soy sauce"), ("1", "carrots"), ("1", "cabbage"), ("1", "lime")], 20, 4, 16, 58, 16, 6, 8, 750, 4.5, 63, ["Cook noodles.", "Whisk peanut butter, soy sauce and lime.", "Toss with noodles and shredded vegetables."]),
    ("Sheet Pan Sausage and Vegetables", "Meat", ["Pork", "< 60 Mins", "Easy"], [("400", "sausage"), ("600", "potatoes"), ("1", "broccoli"), ("1", "red onion"), ("2", "olive oil"), ("1", "paprika")], 40, 4, 30, 35, 20, 5, 4, 900, 4.4, 59, ["Cut everything into pieces.", "Toss with oil and paprika.", "Roast 30 minutes at 220 C."]),
    ("Cod in Tomato Sauce", "Fish", ["Fish", "Healthy", "< 30 Mins"], [("500", "cod"), ("1", "canned tomatoes"), ("1", "onion"), ("2", "garlic"), ("50", "black olives"), ("250", "bulgur")], 30, 4, 7, 48, 30, 7, 5, 520, 4.3, 31, ["Make a tomato sauce with onion and garlic.", "Add cod and olives, bake 15 minutes.", "Serve with bulgur."]),
    ("Vegetable Fried Eggs on Toast", "Breakfast", ["Vegetarian", "< 15 Mins", "Easy"], [("4", "eggs"), ("4", "bread"), ("1", "tomatoes"), ("1", "spinach"), ("1", "butter")], 15, 2, 20, 30, 20, 4, 4, 520, 4.1, 22, ["Toast the bread.", "Wilt spinach in butter, fry eggs.", "Serve on toast with tomato."]),
    ("Halloumi Quinoa Salad", "Salad", ["Vegetarian", "< 30 Mins", "Healthy"], [("200", "quinoa"), ("200", "halloumi cheese"), ("1", "cucumber"), ("250", "cherry tomatoes"), ("1", "lemon"), ("2", "olive oil")], 25, 4, 22, 35, 20, 5, 4, 780, 4.6, 44, ["Cook quinoa and cool.", "Fry halloumi slices.", "Mix with vegetables, lemon and oil."]),
    ("Slow Cooker Pulled Pork", "Pork", ["Pork", "Weeknight", "Slow Cooker"], [("1000", "pork shoulder"), ("1", "barbecue sauce"), ("1", "onion"), ("8", "hamburger buns"), ("1", "coleslaw mix")], 240, 8, 24, 45, 38, 2, 14, 950, 4.7, 260, ["Put pork, onion and sauce in slow cooker.", "Cook 8 hours on low (or 4 on high).", "Shred and serve in buns with coleslaw."]),
]


def r_list(items):
    return "c(" + ", ".join('NA' if x == "" else f'"{x}"' for x in items) + ")"


def iso(minutes):
    h, m = divmod(minutes, 60)
    return "PT" + (f"{h}H" if h else "") + (f"{m}M" if m else "")


def main():
    out = Path(__file__).parent / "sample_recipes.csv"
    cols = ["RecipeId", "Name", "CookTime", "PrepTime", "TotalTime", "RecipeCategory", "Keywords",
            "RecipeIngredientQuantities", "RecipeIngredientParts", "AggregatedRating", "ReviewCount",
            "Calories", "FatContent", "SaturatedFatContent", "SodiumContent", "CarbohydrateContent",
            "FiberContent", "SugarContent", "ProteinContent", "RecipeServings", "RecipeInstructions"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, (name, cat, kw, ings, mins, serv, fat, carbs, prot, fib, sug, sod, rating, reviews, steps) in enumerate(RECIPES):
            calories = 9 * fat + 4 * carbs + 4 * prot + ((i % 5) - 2) * 12
            w.writerow([1000 + i, name, iso(mins - 10), iso(10), iso(mins), cat, r_list(kw),
                        r_list([q for q, _ in ings]), r_list([n for _, n in ings]), rating, reviews,
                        calories, fat, round(fat * 0.35, 1), sod, carbs, fib, sug, prot, serv, r_list(steps)])
    print(f"wrote {len(RECIPES)} recipes -> {out}")


if __name__ == "__main__":
    main()
