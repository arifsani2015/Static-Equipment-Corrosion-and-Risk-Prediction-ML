"""Shared Streamlit UI components used by all prediction pages, so every page
behaves the same way (forms, templates, batch reading, result tables,
downloads, history and charts)."""
import inspect
import re
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import (COL_CR, COL_CR_CONF, COL_CR_WHY, COL_FILE, COL_RISK, COL_RISK_CONF,
                         COL_RISK_LVL, COL_RISK_WHY, COL_ROW, CONF_COLORS, CR_VALUE_COLS,
                         PCT_COLS, RISK_COLORS, RISK_ORDER, build_template_xlsx, extract_schema,
                         get_effective_categories, get_numeric_ranges, get_training_defaults,
                         metrics_row, read_batch_file, resolve_display, risk_conf_level, xlsx_bytes)

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ---------------------------------------------------------------------
# Charts: Altair/Vega-Lite (drawn by the browser -> sharp and proportional
# at any zoom level and container width, unlike fixed-size PNG figures).
# ---------------------------------------------------------------------
_ALT_KW = ({"width": "stretch"} if "width" in inspect.signature(st.altair_chart).parameters
           else {"use_container_width": True})  # Streamlit 1.49 has no 'width' on altair_chart


def show_chart(chart, height=280):
    chart = (chart.properties(height=height)
             .configure_axis(labelFontSize=12, titleFontSize=12, labelLimit=260, grid=True, gridOpacity=0.35)
             .configure_title(fontSize=14, anchor="start", offset=8)
             .configure_view(strokeWidth=0)
             .configure_legend(disable=True))
    st.altair_chart(chart, **_ALT_KW)


def _risk_color_scale():
    return alt.Scale(domain=RISK_ORDER, range=[RISK_COLORS[k] for k in RISK_ORDER])


def risk_bar(out):
    """Horizontal bars: class labels never overlap at any width/zoom."""
    counts = out[COL_RISK if COL_RISK in out else "Predicted_Risk"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    df = pd.DataFrame({"Risk class": RISK_ORDER, "Equipment": counts.values,
                       "Share": counts.values / max(int(counts.sum()), 1)})
    df["Label"] = [f"{n}  ({sh:.0%})" for n, sh in zip(df["Equipment"], df["Share"])]
    xmax = max(int(counts.max()), 1) * 1.6
    base = alt.Chart(df, title="Distribution of predicted risk classes").encode(
        y=alt.Y("Risk class:N", sort=RISK_ORDER, title=None),
        x=alt.X("Equipment:Q", title="Number of equipment", scale=alt.Scale(domain=[0, xmax], nice=False)))
    bars = base.mark_bar(stroke="black", strokeWidth=0.5, cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
        color=alt.Color("Risk class:N", scale=_risk_color_scale()),
        tooltip=["Risk class", "Equipment", alt.Tooltip("Share:Q", format=".1%")])
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(text="Label:N")
    show_chart(bars + labels, height=230)


def cr_histogram(values, title="Distribution of predicted corrosion rate"):
    df = pd.DataFrame({"CR": np.asarray(values, dtype=float)})
    ch = alt.Chart(df, title=title).mark_bar(color="#2563EB", stroke="black", strokeWidth=0.5).encode(
        x=alt.X("CR:Q", bin=alt.Bin(maxbins=20), title="Predicted CR (mm/year)", axis=alt.Axis(tickCount=8, labelOverlap=True)),
        y=alt.Y("count():Q", title="Number of equipment"),
        tooltip=[alt.Tooltip("CR:Q", bin=alt.Bin(maxbins=20), title="CR range"), alt.Tooltip("count():Q", title="Equipment")])
    show_chart(ch)


def probability_chart(row):
    df = pd.DataFrame({"Risk class": RISK_ORDER,
                       "Probability": [float(row.get(f"Prob_{k}", 0) or 0) for k in RISK_ORDER]})
    base = alt.Chart(df, title="Probability of each risk class").encode(
        y=alt.Y("Risk class:N", sort=RISK_ORDER, title=None),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1.18], nice=False),
                axis=alt.Axis(format="%", title="Probability", values=[0, .2, .4, .6, .8, 1])))
    bars = base.mark_bar(stroke="black", strokeWidth=0.5).encode(
        color=alt.Color("Risk class:N", scale=_risk_color_scale()),
        tooltip=["Risk class", alt.Tooltip("Probability:Q", format=".1%")])
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("Probability:Q", format=".1%"))
    show_chart(bars + labels, height=200)


def shap_chart(values, title, signed=True):
    """values: pd.Series (feature -> value)."""
    df = pd.DataFrame({"Feature": values.index.astype(str), "Value": values.values.astype(float)})
    df["Direction"] = np.where(df["Value"] >= 0, "Increases", "Decreases")
    order = df.reindex(df["Value"].abs().sort_values(ascending=False).index)["Feature"].tolist()
    enc = dict(y=alt.Y("Feature:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=240)),
               x=alt.X("Value:Q", title="SHAP contribution (red = pushes the prediction up, blue = down)"
                       if signed else "Mean |SHAP value|"),
               tooltip=["Feature", alt.Tooltip("Value:Q", format=".4f")])
    if signed:
        enc["color"] = alt.Color("Direction:N", scale=alt.Scale(domain=["Increases", "Decreases"],
                                                                range=["#D62728", "#1F77B4"]))
    ch = alt.Chart(df, title=title).mark_bar(stroke="black", strokeWidth=0.4).encode(**enc)
    if signed:
        ch = ch + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
    show_chart(ch, height=max(220, 30 * len(df)))


# ---------------------------------------------------------------------
# Column specifications (auto-detected from the checkpoint(s), then
# adjusted by the user on the Settings page)
# ---------------------------------------------------------------------
def build_specs(res_list, include_cr=True):
    """One spec per input column for the given model(s). Columns shared by
    several models appear once. Each spec: col, kind, label, header, default,
    options, allowed, visible."""
    ss = st.session_state
    specs, seen = [], set()
    for res in res_list:
        num, cat = extract_schema(res)
        defaults = get_training_defaults(res)
        ranges = get_numeric_ranges(res, num)
        cats = get_effective_categories(res, cat, ss["category_overrides"])
        for c in num + cat:
            if c in seen or (c == "cr" and not include_cr):
                continue
            seen.add(c)
            is_num = c in num
            label = resolve_display(c, ss["display_overrides"])
            default = ss["default_overrides"].get(c, defaults.get(c))
            if is_num:
                lo, hi = ranges.get(c, (None, None))
                allowed = f"{lo:.4g} to {hi:.4g} (training range)" if lo is not None else "numeric"
            else:
                allowed = ", ".join(cats.get(c, []))
            specs.append(dict(
                col=c, kind="Numeric" if is_num else "Categorical", label=label,
                header=ss["header_overrides"].get(c) or label,
                default=default, options=cats.get(c, []) if not is_num else None,
                allowed=allowed, visible=(c == "cr") or (c not in ss["excluded_cols"]),
                has_default_override=c in ss["default_overrides"],
            ))
    return specs


def input_form(specs, prefix):
    """Manual form for the visible columns, pre-filled with defaults (training
    median / most frequent value, or the Settings-page default)."""
    vals = {}
    st.markdown("**Identification** (optional — not used by the model, shown in results and history)")
    i1, i2, _ = st.columns(3)
    with i1:
        vals["__Equipment/Pipe Name"] = st.text_input("Equipment / Pipe Name", key=f"{prefix}__id_name",
                                                      placeholder="e.g. SK-18-2-DC-A03B-030")
    with i2:
        vals["__Equipment Part"] = st.text_input("Equipment Part", key=f"{prefix}__id_part", placeholder="e.g. Pipe")
    vis_num = [s for s in specs if s["visible"] and s["kind"] == "Numeric"]
    vis_cat = [s for s in specs if s["visible"] and s["kind"] == "Categorical"]
    hidden = [s for s in specs if not s["visible"]]
    if vis_num:
        st.markdown("**Numeric inputs**")
        grid = st.columns(3)
        for i, s in enumerate(vis_num):
            d = s["default"]
            d = float(d) if d is not None and pd.notna(d) else 0.0
            with grid[i % 3]:
                vals[s["col"]] = st.number_input(s["label"], value=d, format="%.4f",
                                                 key=f"{prefix}_{s['col']}", help=f"Training range: {s['allowed']}")
    if vis_cat:
        st.markdown("**Categorical inputs**")
        grid = st.columns(3)
        for i, s in enumerate(vis_cat):
            opts = list(s["options"] or []) or ["(unknown to the model)"]
            d = str(s["default"]) if s["default"] is not None else None
            with grid[i % 3]:
                vals[s["col"]] = st.selectbox(s["label"], opts, index=opts.index(d) if d in opts else 0,
                                              key=f"{prefix}_{s['col']}")
    if hidden:
        st.caption("Hidden columns (filled automatically with their default): "
                   + ", ".join(f"{s['label']} = {s['default']}" for s in hidden))
    return vals


def row_from_form(specs, vals):
    """Single-row DataFrame: identification + form values + defaults for
    hidden columns."""
    row = {k[2:]: (vals.get(k) or "").strip() for k in ("__Equipment/Pipe Name", "__Equipment Part")}
    for s in specs:
        if s["col"] in vals:
            row[s["col"]] = vals[s["col"]]
        else:
            row[s["col"]] = s["default"] if s["has_default_override"] else (np.nan if s["kind"] == "Numeric" else None)
    return pd.DataFrame([row])


def template_button(specs, file_name, key):
    vis = [dict(s, default=_fmt_default(s)) for s in specs if s["visible"]]
    st.download_button("⬇️ Download batch template (.xlsx)", build_template_xlsx(vis),
                       file_name=file_name, mime=XLSX_MIME, key=key, on_click="ignore",
                       help="Sheet 'Input' to fill, plus a column guide, an example row and instructions.")


def _fmt_default(s):
    d = s["default"]
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return ""
    return round(float(d), 4) if s["kind"] == "Numeric" else d


def read_batch(uploaded, specs):
    """Read + map an uploaded batch file. Missing columns and empty cells
    are filled with the Settings-page default if one is set, otherwise left
    empty for the model's imputer (training median / most frequent value)."""
    df, missing = read_batch_file(uploaded, specs)
    file_order = df.attrs.get("file_order", np.arange(1, len(df) + 1))
    vis_missing = [s for s in missing if s["visible"]]
    if vis_missing:
        st.warning("These columns were not found in the file and will be filled with their default: "
                   + ", ".join(f"**{s['header']}**" for s in vis_missing)
                   + ". Check the headers against the template if this is unexpected.")
    for s in specs:
        c = s["col"]
        if c not in df.columns:
            df[c] = s["default"] if s["has_default_override"] else (np.nan if s["kind"] == "Numeric" else None)
        elif s["has_default_override"]:
            df[c] = df[c].where(df[c].notna(), s["default"])
    df.attrs["file_order"] = file_order
    return df


def to_display_columns(df, specs):
    """Internal column names -> Excel headers (so results round-trip)."""
    return df.rename(columns={s["col"]: s["header"] for s in specs})


# ---------------------------------------------------------------------
# Result tables: row numbers -> equipment identity -> prediction,
# confidence and its explanation -> details -> input parameters.
# ---------------------------------------------------------------------
ID_PATTERN = re.compile(r"^(no\.?|#|id|nr|number)$|tag|equipment|pipe|part|name|line|item|asset|component|unit|loop|circuit")
# A column that already acts as a row number/identifier of the user's file
# ("No", "No.", "#", "ID", "Nr", "Number", "Seq" or a name ending in one of them such as
# "Tag No", "Line No", "Item ID"). When present, the
# app does NOT add its own 'File Order' column, so there are only two numbering columns.
ROW_NUMBER_PATTERN = re.compile(r"^(#|seq|sequence|s/n|sn)$|(^|[\s_\-/.])(no\.?|nr\.?|number|id)$")
NOT_ID_PATTERN = re.compile(r"corrosion|\bcr\b|rate|reference|risk|type|material|pressure|temperature|thickness|diameter")
LEAD_CR = [COL_CR, COL_CR_CONF, COL_CR_WHY]
LEAD_RISK = [COL_RISK, COL_RISK_CONF, COL_RISK_WHY]


def _fmt_id_value(v):
    if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
        return int(v)
    return v


def cr_block(pred, rel):
    """Corrosion-rate result columns (lead first, then details)."""
    return pd.DataFrame({
        COL_CR: np.asarray(pred, dtype=float),
        COL_CR_CONF: rel["Confidence_Level"].values,
        COL_CR_WHY: [r[:1].upper() + r[1:] if isinstance(r, str) and r else "" for r in rel["Confidence_Reasons"]],
        "CR 80% Low": rel["CR_P10"].values,
        "CR 80% High": rel["CR_P90"].values,
        "Model Spread": rel["Model_Agreement_CV"].values,
        "Similarity to Known Equipment": rel["Domain_Status"].values,
        "Distance Percentile": rel["Domain_Percentile"].values,
        "Similar Equipment Median CR": rel["Similar_Median_CR"].values,
        "Similar Equipment Min CR": rel["Similar_Min_CR"].values,
        "Similar Equipment Max CR": rel["Similar_Max_CR"].values,
        "Within Similar Range": rel["Within_Similar_Range"].values,
        "Most Similar Equipment": rel["Most_Similar"].values,
    })


def risk_block(out):
    """Risk result columns: predicted class, its probability (= risk
    confidence), a coloured level and a short note, then every class
    probability and the runner-up."""
    conf = out["Confidence"].astype(float).values
    notes = []
    for _, r in out.iterrows():
        if "Runner_Up" in r and pd.notna(r.get("Margin")):
            ru = r["Runner_Up"]
            notes.append(f"Runner-up {ru} {r[f'Prob_{ru}'] * 100:.0f}% (margin {r['Margin'] * 100:.0f} pp)"
                         + ("; model undecided" if r["Margin"] < 0.10 else ""))
        else:
            notes.append("")
    blk = pd.DataFrame({COL_RISK: out["Predicted_Risk"].values, COL_RISK_CONF: conf, COL_RISK_WHY: notes})
    for k in RISK_ORDER:
        if f"Prob_{k}" in out:
            blk[f"Chance {k}"] = out[f"Prob_{k}"].values
    if "Runner_Up" in out:
        blk["Runner-up Risk"] = out["Runner_Up"].values
        blk["Margin to Runner-up"] = out["Margin"].values
    return blk


def arrange_results(inputs_display, specs, blocks, file_order=None, sort=None, extra_tail=None):
    """Build the final result table.
    Order: '#' (display rank), identity columns (No, Equipment /
    Pipe Name, Equipment Part, Tag...) -- 'File Order' only if the file has no
    'No'-style column --, lead prediction columns (prediction,
    confidence, reason), other result details, other uploaded columns,
    then the model input parameters.
    sort: None, 'cr' (highest CR first) or 'risk' (highest risk first)."""
    inputs_display = inputs_display.reset_index(drop=True)
    param_cols = [s["header"] for s in specs if s["header"] in inputs_display.columns]
    extra = [c for c in inputs_display.columns if c not in param_cols]
    id_cols = [c for c in extra if ID_PATTERN.search(str(c).strip().lower())
               and not NOT_ID_PATTERN.search(str(c).strip().lower())]
    other_extra = [c for c in extra if c not in id_cols]
    res = pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)
    lead = [c for c in LEAD_CR + LEAD_RISK if c in res.columns]
    details = [c for c in res.columns if c not in lead]
    ids = inputs_display[id_cols].apply(lambda s: s.map(_fmt_id_value)) if id_cols else pd.DataFrame(index=res.index)
    table = pd.concat([ids, res[lead], res[details], inputs_display[other_extra],
                       (extra_tail.reset_index(drop=True) if extra_tail is not None else None),
                       inputs_display[param_cols]], axis=1)
    has_number_col = any(ROW_NUMBER_PATTERN.search(str(c).strip().lower()) for c in id_cols)
    if file_order is not None and not has_number_col:
        # The uploaded file has no 'No'-style column: fall back to the position of
        # the row in the file, so every batch result can still be traced back.
        table.insert(0, COL_FILE, np.asarray(file_order)[: len(table)])
    if sort == "risk" and COL_RISK in table:
        rank = table[COL_RISK].map({k: i for i, k in enumerate(RISK_ORDER)})
        table = table.assign(_r=rank).sort_values(["_r", COL_RISK_CONF], ascending=[False, False], kind="stable").drop(columns="_r")
    elif sort == "cr" and COL_CR in table:
        table = table.sort_values(COL_CR, ascending=False, kind="stable")
    table = table.reset_index(drop=True)
    table.insert(0, COL_ROW, np.arange(1, len(table) + 1))
    return table


NAME_PATTERN = re.compile(r"name|tag|equipment/pipe|pipe|equipment")
COL_WIDTHS = {COL_ROW: 42, COL_FILE: 70, COL_CR: 85, COL_CR_CONF: 95, COL_CR_WHY: 200,
              COL_RISK: 95, COL_RISK_CONF: 110, COL_RISK_WHY: 210}
SHORT_LABELS = {COL_CR: "CR (mm/y)", COL_FILE: "File Order", COL_CR_WHY: "CR Confidence Reason",
                COL_RISK_WHY: "Risk Confidence Note", COL_RISK: "Pred. Risk"}


def _pinned_config(df):
    """Compact widths so row numbers, equipment identity, prediction,
    confidence and its explanation fit on screen. Columns from '#' up to
    the equipment-name column are pinned (contiguous, so the order
    '#', 'File Order', identity columns is preserved while scrolling)."""
    cols = list(df.columns)
    lead_idx = next((i for i, c in enumerate(cols) if c in LEAD_CR or c in LEAD_RISK), len(cols))
    name_idx = next((i for i, c in enumerate(cols[:lead_idx]) if c not in (COL_ROW, COL_FILE)
                     and NAME_PATTERN.search(str(c).lower())), 0)
    cfg = {}
    for i, c in enumerate(cols):
        label, pin = SHORT_LABELS.get(c, c), i <= name_idx
        if c in (COL_ROW, COL_FILE):
            cfg[c] = st.column_config.NumberColumn(label, width=COL_WIDTHS[c], pinned=pin, format="%d",
                                                   help="Row order in your file" if c == COL_FILE else None)
        elif i < lead_idx:
            w = 165 if NAME_PATTERN.search(str(c).lower()) else (50 if len(str(c)) <= 3 else 80)
            cfg[c] = st.column_config.Column(label, width=w, pinned=pin)
        elif c in COL_WIDTHS:
            cfg[c] = st.column_config.Column(label, width=COL_WIDTHS[c],
                                             help="Full text visible on hover" if c in (COL_CR_WHY, COL_RISK_WHY) else None)
    return cfg


def style_results(df):
    """Display copy: coloured prediction/confidence cells and number formats."""
    fmt = {c: "{:.1%}" for c in df.columns if c in PCT_COLS}
    fmt.update({c: "{:.4f}" for c in df.columns if c in CR_VALUE_COLS})
    if "Distance Percentile" in df:
        fmt["Distance Percentile"] = "{:.0f}"
    sty = df.style.format(fmt, na_rep="")
    apply = getattr(sty, "map", None) or sty.applymap
    cell = lambda color: f"background-color:{color};color:white;font-weight:600;"
    for col, colors in ((COL_CR_CONF, CONF_COLORS), (COL_RISK_LVL, CONF_COLORS), (COL_RISK, RISK_COLORS)):
        if col in df:
            sty = apply(lambda v, c=colors: cell(c[v]) if v in c else "", subset=[col])
    if COL_RISK_CONF in df:
        sty = apply(lambda v: cell(CONF_COLORS[risk_conf_level(v)]) if pd.notna(v) else "", subset=[COL_RISK_CONF])
    return sty


def show_results_table(df, height=None):
    kw = {"height": height} if height else {}
    st.dataframe(style_results(df), hide_index=True, width="stretch", column_config=_pinned_config(df), **kw)


def confidence_legend(kind):
    """Explanation of confidence levels, shown ABOVE results."""
    chip = lambda k: (f"<span style='background:{CONF_COLORS[k]};color:white;border-radius:6px;padding:1px 8px;"
                      f"font-weight:600'>{k}</span>")
    lines = []
    if kind in ("cr", "combo"):
        lines.append(
            f"<b>CR Confidence</b> — how much the corrosion-rate prediction can be trusted, combining: similarity of "
            f"the input to real equipment with measured CR, agreement between all trained models, and whether the "
            f"prediction lies within the measured CR of the 5 most similar equipment. {chip('High')} similar "
            f"equipment, models agree and the value is consistent with similar equipment · {chip('Medium')} one of "
            f"these is weak · {chip('Low')} input far from known equipment (extrapolation) or models disagree. "
            f"The reason column says which indicator drove the level.")
    if kind in ("risk", "combo"):
        lines.append(
            f"<b>Risk Confidence</b> — probability the model gives to the predicted risk class (the highest of the "
            f"four chances). {chip('High')} ≥ 70% · {chip('Medium')} 50–70% · {chip('Low')} &lt; 50%. The note shows "
            f"the runner-up class and the margin; a margin below 10 percentage points means the model is undecided.")
    st.markdown("<div style='border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 14px;"
                "font-size:0.9rem;line-height:1.55'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)


def download_xlsx(label, sheets, file_name, key, primary=False):
    st.download_button(label, xlsx_bytes(sheets), file_name=file_name, mime=XLSX_MIME, key=key,
                       on_click="ignore", type="primary" if primary else "secondary")


def model_info_sheet(items):
    return pd.DataFrame(items, columns=["Item", "Value"])


def model_info_items(kind, res, model_name, file_label=None, scenario=None):
    items = [("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
             (f"{kind} checkpoint", file_label or "manual upload")]
    if scenario:
        items.append(("Scenario", scenario))
    items += [(f"{kind} model", model_name), ("Hyperparameter source", res.get("overall_source", "?"))]
    for k, v in (metrics_row(res, model_name) or {}).items():
        if k != "Model" and v is not None and pd.notna(v):
            items.append((f"{kind} {k}", round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v))
    return items


# ---------------------------------------------------------------------
# History of manual predictions (per page, kept for the session)
# ---------------------------------------------------------------------
def add_history(key, table_row):
    row = table_row.drop(columns=[c for c in (COL_ROW, COL_FILE) if c in table_row.columns]).copy()
    row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state[key].append(row)


def history_section(key, file_name, extra_sheets=None):
    hist = st.session_state.get(key) or []
    if not hist:
        return
    st.divider()
    st.subheader(f"🗂️ Manual prediction history ({len(hist)})")
    df = pd.concat(hist, ignore_index=True)
    df.insert(0, COL_ROW, np.arange(1, len(df) + 1))
    show_results_table(df)
    c1, c2 = st.columns([1, 1])
    with c1:
        download_xlsx("⬇️ Download history (.xlsx)", {"History": df, **(extra_sheets or {})},
                      file_name, key=f"dl_{key}")
    with c2:
        if st.button("🗑️ Clear history", key=f"clear_{key}"):
            st.session_state[key] = []
            st.rerun()


# ---------------------------------------------------------------------
# Single-result displays
# ---------------------------------------------------------------------
def _badge(title, value, color, sub=None):
    st.markdown(f"<div style='background:{color};color:white;border-radius:10px;padding:10px 6px;"
                f"text-align:center;height:100%'><div style='font-size:0.8rem;opacity:.95'>{title}</div>"
                f"<div style='font-size:clamp(1.05rem,2.1vw,1.55rem);font-weight:800;line-height:1.2;white-space:nowrap'>{value}</div>"
                + (f"<div style='font-size:0.8rem'>{sub}</div>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def _reason_box(title, text):
    st.markdown(f"<div style='border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px;"
                f"height:100%'><div style='font-size:0.8rem;opacity:.75'>{title}</div>"
                f"<div style='font-size:0.95rem'>{text}</div></div>", unsafe_allow_html=True)


def show_cr_result(pred, rel_row, similar, basis):
    """Prediction | confidence | reason side by side, then the indicators."""
    c1, c2, c3 = st.columns([1.1, 0.9, 2.2])
    with c1:
        _badge("Predicted corrosion rate", f"{pred:.4f}", "#2563EB", "mm/year")
    with c2:
        _badge("CR confidence", rel_row["Confidence_Level"], CONF_COLORS.get(rel_row["Confidence_Level"], "#999"))
    with c3:
        why = rel_row["Confidence_Reasons"] or "n/a"
        _reason_box("Why this confidence level", why[:1].upper() + why[1:] + ".")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    cv, dp, med = rel_row.get("Model_Agreement_CV"), rel_row.get("Domain_Percentile"), rel_row.get("Similar_Median_CR")
    stats = [
        ("80% prediction interval (mm/year)",
         f"{rel_row['CR_P10']:.4f} – {rel_row['CR_P90']:.4f}" if pd.notna(rel_row.get("CR_P10")) else "n/a",
         "about 8 of 10 test values fell inside such intervals"),
        ("Model spread", f"{cv * 100:.1f}%" if pd.notna(cv) else "n/a", "≤15% close agreement · ≤35% moderate"),
        ("Similarity to known equipment", rel_row.get("Domain_Status", "n/a"),
         f"distance percentile {dp:.0f} (≤90 Inside, ≤99 Borderline)" if pd.notna(dp) else ""),
        ("Similar equipment: measured CR", f"{med:.4f}" if pd.notna(med) else "n/a",
         f"median; range {rel_row['Similar_Min_CR']:.4f} – {rel_row['Similar_Max_CR']:.4f}" if pd.notna(med) else ""),
    ]
    cols = st.columns(4, gap="small")
    for col, (title, value, sub) in zip(cols, stats):
        col.markdown(f"<div style='border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:8px 10px;"
                     f"height:100%'><div style='font-size:0.78rem;opacity:.75'>{title}</div>"
                     f"<div style='font-size:clamp(0.95rem,1.7vw,1.15rem);font-weight:700;overflow-wrap:anywhere'>{value}</div>"
                     f"<div style='font-size:0.75rem;opacity:.7'>{sub}</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("🔎 5 most similar real equipment", expanded=False):
        st.caption(f"Based on: {basis}.")
        if similar is not None and len(similar):
            st.dataframe(similar, hide_index=True, width="stretch",
                         column_config={"Rank": st.column_config.NumberColumn(pinned=True, width="small")})


def show_risk_result(row):
    """Predicted risk | risk confidence | note side by side, then all four
    probabilities as numbers and as a chart."""
    p = float(row["Confidence"])
    lvl = risk_conf_level(p)
    c1, c2, c3 = st.columns([1.1, 0.9, 2.2])
    with c1:
        _badge("Predicted risk", row["Predicted_Risk"], RISK_COLORS.get(row["Predicted_Risk"], "#999"))
    with c2:
        _badge("Risk confidence", f"{p * 100:.1f}%", CONF_COLORS[lvl], f"{lvl}")
    with c3:
        if "Runner_Up" in row and pd.notna(row.get("Margin")):
            ru, m = row["Runner_Up"], row["Margin"]
            txt = (f"Runner-up class <b>{ru}</b> ({row[f'Prob_{ru}'] * 100:.1f}%), margin <b>{m * 100:.1f}</b> "
                   f"percentage points." + (" The model is <b>undecided</b> between the two classes."
                                            if m < 0.10 else ""))
        else:
            txt = "n/a"
        _reason_box("Why this confidence level", txt)
    st.write("")
    cols = st.columns(len(RISK_ORDER))
    for col, k in zip(cols, RISK_ORDER):
        v = row.get(f"Prob_{k}", np.nan)
        border = "3px solid rgba(0,0,0,.75)" if k == row["Predicted_Risk"] else "0"
        col.markdown(f"<div style='background:{RISK_COLORS[k]};color:white;border-radius:10px;padding:8px 4px;"
                     f"text-align:center;border:{border}'><div style='font-size:clamp(1rem,2vw,1.35rem);font-weight:800;white-space:nowrap'>"
                     f"{(v * 100):.1f}%</div><div style='font-size:0.8rem'>Chance {k}</div></div>"
                     if pd.notna(v) else "", unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    probability_chart(row)


# ---------------------------------------------------------------------
# Batch summaries
# ---------------------------------------------------------------------
def _cards(items):
    cols = st.columns(len(items), gap="small")
    for col, (value, label, color) in zip(cols, items):
        col.markdown(f"<div style='background:{color};border-radius:10px;padding:12px 6px;text-align:center;"
                     f"color:white'><div style='font-size:clamp(1.1rem,2.3vw,1.6rem);font-weight:800;line-height:1.1;white-space:nowrap'>{value}</div>"
                     f"<div style='font-size:0.8rem;opacity:.95'>{label}</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def render_risk_summary_cards(risks):
    counts = pd.Series(risks).value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    total = max(int(counts.sum()), 1)
    _cards([(counts[k], f"{k} · {counts[k] / total * 100:.1f}%", RISK_COLORS[k]) for k in RISK_ORDER])


def confidence_summary_cards(levels, what="CR"):
    counts = pd.Series(levels).value_counts()
    _cards([(int(counts.get(k, 0)), f"{k} {what} confidence", CONF_COLORS[k]) for k in ("High", "Medium", "Low")])


def risk_distribution_table(table):
    risks = table[COL_RISK]
    rows = []
    for k in RISK_ORDER:
        pc = f"Chance {k}"
        n = int((risks == k).sum())
        rows.append({"Risk class": k, "Equipment count": n,
                     "Share of equipment (%)": round(n / max(len(table), 1) * 100, 1),
                     "Mean probability (%)": round(float(table[pc].mean()) * 100, 1) if pc in table else np.nan,
                     "Expected count (sum of probabilities)": round(float(table[pc].sum()), 1) if pc in table else np.nan})
    return pd.DataFrame(rows)


def row_ref_note(table):
    """Sentence telling the user which column links a result row to the uploaded file."""
    if COL_FILE in table.columns:
        return f"'{COL_FILE}' is the row position in your file"
    for c in table.columns:
        if c != COL_ROW and ROW_NUMBER_PATTERN.search(str(c).strip().lower()):
            return f"'{c}' is the row number from your file"
    return "'#' is the rank in this table"


def mode_selector(key):
    return st.radio("Input method", ["Manual input (one equipment)", "Batch upload (many equipment)"],
                    horizontal=True, key=key)


def file_key(uploaded):
    return getattr(uploaded, "file_id", None) or f"{uploaded.name}-{uploaded.size}"
