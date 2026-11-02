# Goal files

A goal file says what a classification is **for** (irrigation status, bike access, fire danger, ...), so tuning
optimises the errors that matter for that purpose instead of treating every class confusion the same.
The agent drafts one at the start of a run, shows it to the user as a table, and only starts once the user has
approved it (`"approved": true`). After that the goal is fixed for the whole run.

`sample_points.py --goal <file>` copies the approved file into the workspace as `goal.json`, where every tool reads it:

| Tool | Uses the goal for |
|---|---|
| `sample_points.py` | adds the goal's new classes to the labellers' class list |
| `score.py` | goal cost at each point; the gate compares goal cost instead of plain point counts |
| `review.py` | tells the blind reviewer the purpose, the costly confusions and the rules |
| `goal_map.py` | translates a result into the goal's categories (`run/goal_map.tif`) |

## Format

```json
{
  "goal": "short-name",
  "purpose": "assess ... (one sentence; read by the agents and the reviewer)",
  "approved": false,
  "approved_by": "", "approved_on": "",
  "classes": {
    "<class name>": {"role": "key | context | ignore", "category": "<goal category> | depends", "note": "..."}
  },
  "new_classes": [
    {"name": "...", "definition": "...", "category": "...", "types": [{"id": "class.type", "prompts": ["..."]}]}
  ],
  "costly_confusions": [{"a": "<class>", "b": "<class>", "cost": 3, "why": "..."}],
  "default_cost": {"same_category": 0, "different_category": 1, "missed": 1},
  "rules": ["plain-English rules for cases the class alone doesn't decide"],
  "output": "what the final report must deliver",
  "caveats": ["limits of what this imagery can show for this goal"]
}
```

- **role**: `key` classes are what the goal is about; `context` classes matter only to keep the map coherent;
  `ignore` classes cost nothing when wrong.
- **category**: the goal's answer for that class (e.g. `irrigated`, `non-irrigable`, `high fuel`). Use `depends`
  when the class alone doesn't decide it (e.g. shadow, canopy for irrigation); `goal_map.py` marks those undetermined.
- **cost** of a wrong point: 0 when acceptable; a listed costly confusion costs its `cost`; a missed (unclassified)
  point costs `missed`; two classes in the same category cost `same_category`; anything else `different_category`.
- **new_classes**: classes the goal needs that the bundled dictionary lacks (e.g. bike lane, crosswalk). They are
  added to the labellers' list and to the starting dictionary with `edit_dictionary.py` (`new_class`).

`irrigation.example.json` is an unapproved example built from the bundled dictionary's irrigation metadata.
