#!/usr/bin/env python3
"""
annotate.py - single-file adjudication front end for comparative meme-template
identification (RQ2).

For each image: every model's predicted label side by side, up to three example
images of any template a model predicted, checkboxes for which models are right
(multi-select), and a ground-truth selector for when none of them is.

Standard library only: no pip install, no build step, no other files.

    python annotate.py --demo 30                       # placeholder data, try it now

  One annotator:
    python annotate.py --predictions predictions.csv --images-root ./images \
                       --templates-root ./templates --annotator rater1
    -> annotations/rater1.csv

  Several annotators at once, on one shared machine:
    python annotate.py --predictions predictions.csv --images-root ./images \
                       --templates-root ./templates \
                       --annotators rater1,rater2,rater3 --overlap 150 \
                       --host 0.0.0.0
    -> annotations/rater1.csv, annotations/rater2.csv, annotations/rater3.csv
    Each person opens the URL, picks their name once, and gets their own file
    and their own progress. Nothing mutable is shared between them.

  Combine the files and report agreement when everyone is done:
    python annotate.py --merge annotations/ --merge-out merged.csv

LABEL VOCABULARY
    NON_MEME       the image is not a meme at all
    TEMPLATELESS   a meme, but not built on a template
    <name>         a specific template, e.g. Pepe_the_Frog
  Aliases for the two special classes are in NON_MEME_TOKENS and
  TEMPLATELESS_TOKENS below; edit those to match your pipeline's spelling.

INPUT - predictions.csv, either shape, auto-detected:
    long   image_id,image_path,model,prediction,confidence[,template_examples]
    wide   image_id,image_path,<model>_pred,<model>_conf,...

  Build it from the method scripts (methods/*.py run with --open-set-dir), which
  each write an open_set_predictions.csv into their run directory:
    python annotate.py --collect resnet18=<run_dir> two_stage=<run_dir> ... \
                       --collect-out predictions.csv
  <run_dir> is the timestamped run folder (or the CSV itself). Labels come out
  as an ImgFlip template name, "Template-Free" or "Non-Meme".

TEMPLATE EXAMPLES - resolved in this order, first hit wins:
    1. a template_examples column, pipe-separated paths
    2. --templates-root/<Label>/*.jpg      (a folder per template)
    3. --templates-root/<Label>.jpg, <Label>_1.jpg, <Label>-2.png, ...
  Matching ignores case, spaces, hyphens and underscores.

OUTPUT - annotations/<annotator>.csv, one row per judged image:
    image_id, annotator, timestamp_utc, decision, n_models_correct,
    <model>__correct, <model>__prediction, <model>__confidence,
    true_label, true_label_kind, true_label_source,
    notes, display_order, seconds_on_image
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import random
import re
import sys
import tempfile
import threading
import uuid
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

NON_MEME_TOKENS = {"non_meme", "nonmeme", "not_meme", "not_a_meme", "no_meme"}
TEMPLATELESS_TOKENS = {
    "", "templateless", "template_less", "template_free", "templatefree", "no_template",
    "none", "null", "nan", "non_templated", "not_templated", "untemplated",
}

# Columns the method scripts write next to pred_label in open_set_predictions.csv that
# can serve as a 0-1 confidence. First present wins. Distances (e.g. Hamming bits) are
# deliberately not listed: they run the wrong way.
OPEN_SET_CONF_COLUMNS = ("confidence", "stage2_confidence", "centroid_cosine")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".svg")
PRED_SUFFIXES = ("_pred", "_prediction", "_predicted", "_label", "_template")
CONF_SUFFIXES = ("_conf", "_confidence", "_score", "_prob", "_similarity")

KIND_NON_MEME = "NON_MEME"
KIND_TEMPLATELESS = "TEMPLATELESS"
KIND_TEMPLATE = "TEMPLATE"
KIND_UNSURE = "UNSURE"

# Sign-in cookies carry this, so restarting the server invalidates every existing session
# and everyone is asked who they are again. Within one run a page reload stays signed in.
RUN_ID = uuid.uuid4().hex[:12]
SESSION_LOG = "sessions.csv"


def normalize(label: str) -> str:
    """Fold case, spaces, hyphens and underscores so labels match loosely."""
    return re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")


class TemplateSearch:
    """Rank template names against a typed query. Pure Python, built once, read-only.

    Six annotators typing at once means a query on nearly every keystroke, so this has to
    stay cheap: the index is a couple of dicts over ~1,700 short strings, and a query
    touches only the templates that share a word or a trigram with it. No model, no GPU,
    no per-query allocation beyond the candidate set -- typically well under a millisecond.

    Scoring blends three signals, which between them cover how people actually type a
    half-remembered template name:
      * whole-word overlap      "drake bling"  -> drake_hotline_bling_normal
      * character trigrams      "grumy cat"    -> grumpy-cat-bed        (typos)
      * prefix match            "spong"        -> spongebob-*           (partial words)
    """

    def __init__(self, labels: list[str]) -> None:
        self.labels = list(labels)
        self.by_word: dict[str, set[int]] = {}
        self.by_trigram: dict[str, set[int]] = {}
        self.words: list[set[str]] = []
        self.trigrams: list[set[str]] = []
        for index, label in enumerate(self.labels):
            words = {w for w in re.split(r"[^a-z0-9]+", label.lower()) if w}
            grams = self._trigrams(label)
            self.words.append(words)
            self.trigrams.append(grams)
            for word in words:
                self.by_word.setdefault(word, set()).add(index)
            for gram in grams:
                self.by_trigram.setdefault(gram, set()).add(index)

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        flat = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
        padded = f" {flat} "
        return {padded[i:i + 3] for i in range(max(len(padded) - 2, 0))}

    def query(self, text: str, limit: int = 5) -> list[tuple[str, float]]:
        query_words = {w for w in re.split(r"[^a-z0-9]+", str(text).lower()) if w}
        query_grams = self._trigrams(text)
        if not query_words and not query_grams:
            return []

        candidates: set[int] = set()
        for word in query_words:
            candidates |= self.by_word.get(word, set())
        for gram in query_grams:
            candidates |= self.by_trigram.get(gram, set())
        # Prefix hits: someone typing "spong" has no whole word and few useful trigrams yet.
        if query_words:
            for index, words in enumerate(self.words):
                if any(w.startswith(q) for q in query_words for w in words):
                    candidates.add(index)
        if not candidates:
            return []

        scored: list[tuple[float, int]] = []
        for index in candidates:
            words, grams = self.words[index], self.trigrams[index]
            exact = len(query_words & words) / len(query_words) if query_words else 0.0
            prefix = 0.0
            if query_words:
                hits = sum(1 for q in query_words if any(w.startswith(q) for w in words))
                prefix = hits / len(query_words)
            overlap = len(query_grams & grams)
            fuzzy = overlap / (len(query_grams | grams) or 1)
            score = 3.0 * exact + 1.5 * prefix + 2.0 * fuzzy
            # Prefer the tighter name when two match equally: "drake_meme_3_panels" over a
            # longer one that merely contains the same words.
            score -= 0.02 * len(words)
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: (-pair[0], self.labels[pair[1]]))
        return [(self.labels[i], round(sc, 4)) for sc, i in scored[:limit]]


def label_kind(label: str) -> str:
    key = normalize(label)
    if key in NON_MEME_TOKENS:
        return KIND_NON_MEME
    if key in TEMPLATELESS_TOKENS:
        return KIND_TEMPLATELESS
    return KIND_TEMPLATE


# ===========================================================================
# Page
# ===========================================================================

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Template adjudication</title>
<style>
/* Neutral mid-dark surface: a tinted background would bias how the memes
   themselves read, which is the one thing this tool must not do. */
:root {
  --well:#23262a; --panel:#2e3238; --raise:#383d45; --edge:#454b54;
  --ink:#eaecef; --dim:#8e98a4; --pick:#79b4ff; --pick-soft:rgba(121,180,255,.14);
  --truth:#d9a441; --truth-soft:rgba(217,164,65,.13);
  --pad:18px; --radius:6px; color-scheme:dark;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{display:flex;flex-direction:column;background:var(--well);color:var(--ink);
  font-family:"Inter","Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.45;-webkit-font-smoothing:antialiased}
button{font:inherit;color:inherit;cursor:pointer}
:focus-visible{outline:2px solid var(--pick);outline-offset:2px}

/* -- sign in ------------------------------------------------------------- */
.gate{position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;
  background:var(--well)}
.gate.show{display:flex}
.gate-box{width:330px;padding:26px;background:var(--panel);border:1px solid var(--edge);border-radius:8px}
.gate-box h1{margin:0 0 4px;font-size:18px}
.gate-box p{margin:0 0 18px;font-size:13px;color:var(--dim)}
.gate-box select,.gate-box input{width:100%;padding:9px 10px;margin-bottom:12px;background:rgba(0,0,0,.28);
  border:1px solid var(--edge);border-radius:var(--radius);color:var(--ink);font:inherit;font-size:14px}
.gate-resume{margin-top:14px;border-top:1px solid var(--line);padding-top:14px}
.gate-resume p{margin:0 0 10px}
.gate-resume-row{display:flex;gap:8px;justify-content:center}
.gate-note{font-size:11.5px;color:var(--dim);margin-top:10px!important}
.gate-err{margin:0 0 12px !important;color:#f0b8ac !important}

.stage{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 430px;min-height:0}
.well{position:relative;display:flex;align-items:center;justify-content:center;padding:var(--pad);overflow:auto}
.well img{max-width:100%;max-height:100%;object-fit:contain;border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.45)}
.well.actual img{max-width:none;max-height:none}
.well-empty{color:var(--dim)}
.zoom{position:absolute;left:var(--pad);bottom:var(--pad);padding:5px 10px;border:1px solid var(--edge);
  border-radius:var(--radius);background:rgba(35,38,42,.82);color:var(--dim);font-size:13px}
.zoom:hover{color:var(--ink)}

.panel{display:flex;flex-direction:column;gap:12px;padding:var(--pad);background:var(--panel);
  border-left:1px solid #1b1e21;overflow:hidden}
.panel-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.image-id{font-size:13px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.counter{font-size:13px;color:var(--dim);font-variant-numeric:tabular-nums;white-space:nowrap}
.counter b{color:var(--ink);font-weight:600}
.prompt{margin:0;font-size:16px;font-weight:600;letter-spacing:-.01em}

.scroller{flex:1;min-height:0;overflow-y:auto;margin:0 -4px;padding:2px 4px;
  display:flex;flex-direction:column;gap:14px}
.scroller>*{flex:0 0 auto}
.cards{display:flex;flex-direction:column;gap:8px}
.card{position:relative;display:block;width:100%;padding:10px 12px 10px 10px;text-align:left;
  background:var(--raise);border:1px solid transparent;border-left:3px solid var(--edge);
  border-radius:var(--radius);transition:background 90ms linear,border-color 90ms linear}
.card:hover{background:#3f454e}
.card.on{background:var(--pick-soft);border-color:rgba(121,180,255,.4);border-left-color:var(--pick)}
.card.on .key{background:var(--pick);color:#12161b;border-color:var(--pick)}
.card-top{display:grid;grid-template-columns:22px minmax(0,1fr);gap:10px;align-items:start}
.key{display:grid;place-items:center;width:22px;height:22px;margin-top:1px;border:1px solid var(--edge);
  border-radius:4px;background:rgba(0,0,0,.22);color:var(--dim);font-size:12px;font-weight:600}
.card-model{font-size:12px;color:var(--dim);margin-bottom:3px}
.card.on .card-model{color:var(--pick)}
.card-pred{font-size:15px;font-weight:600;word-break:break-word;line-height:1.3}
.card-pred.special{font-size:12.5px;letter-spacing:.03em;color:#c4ccd5;display:inline-block;
  padding:2px 9px;border:1px solid var(--edge);border-radius:20px;background:rgba(0,0,0,.22)}
.card-pred.missing{font-weight:400;font-style:italic;color:var(--dim)}
.conf{display:flex;align-items:center;gap:8px;margin-top:7px}
.conf-track{width:68px;height:6px;border-radius:3px;background:rgba(255,255,255,.09);
  box-shadow:inset 0 0 0 1px rgba(0,0,0,.25);overflow:hidden}
.conf-track span{display:block;height:100%;background:var(--dim)}
.card.on .conf-track span{background:var(--pick)}
.conf-value{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}

.examples{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:10px 0 0 32px}
.examples img{width:100%;height:68px;object-fit:cover;border-radius:3px;background:rgba(0,0,0,.28);
  box-shadow:0 0 0 1px rgba(0,0,0,.35)}
.examples-none{margin:9px 0 0 32px;font-size:12px;color:var(--dim)}
.search-results{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.sr{display:flex;align-items:center;gap:9px;padding:5px 7px;border:1px solid transparent;
    border-radius:5px;background:rgba(255,255,255,.03);cursor:pointer;text-align:left;width:100%}
.sr:hover,.sr.on{border-color:var(--accent);background:rgba(255,255,255,.07)}
.sr-shots{display:flex;gap:3px;flex:0 0 auto}
.sr-shots img{width:38px;height:38px;object-fit:cover;border-radius:3px;background:rgba(0,0,0,.3)}
.sr-name{font-size:12.5px;word-break:break-all;flex:1}
.sr-own{color:var(--dim);font-style:italic}
.sr-own strong{color:var(--fg);font-style:normal}

.warn{margin:0 0 2px;padding:8px 11px;border-radius:var(--radius);font-size:13px;
  background:rgba(217,164,65,.12);border:1px solid rgba(217,164,65,.45);color:#f0d9a8}

.truth{border:1px solid var(--edge);border-radius:var(--radius);padding:10px 11px}
.truth.off{opacity:.42}
.truth-head{font-size:13px;font-weight:600;margin-bottom:2px}
.truth-sub{font-size:12px;color:var(--dim);margin-bottom:8px}
.opt{display:flex;align-items:center;gap:9px;width:100%;padding:6px 9px;margin-bottom:4px;
  background:transparent;border:1px solid transparent;border-radius:var(--radius);
  text-align:left;font-size:13.5px;color:var(--dim);transition:background 90ms linear,color 90ms linear}
.opt:hover:not(:disabled){background:rgba(255,255,255,.05);color:var(--ink)}
.dot{flex:none;width:15px;height:15px;border:1px solid var(--edge);border-radius:50%;background:rgba(0,0,0,.22)}
.opt.on{background:var(--truth-soft);border-color:var(--truth);color:var(--ink)}
.opt.on .dot{border-color:var(--truth);box-shadow:inset 0 0 0 3.5px var(--truth)}
.opt code{font-size:11.5px;color:var(--dim);margin-left:auto;letter-spacing:.02em}
.opt.on code{color:var(--truth)}
.truth-input{margin:2px 0 7px 24px}
.truth-input input{width:100%;padding:8px 10px;background:rgba(0,0,0,.28);border:1px solid var(--edge);
  border-radius:var(--radius);color:var(--ink);font:inherit;font-size:14px}
.truth-input .hint{margin-top:5px;font-size:12px;color:var(--dim)}
.truth-preview{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:7px}
.truth-preview img{width:100%;height:62px;object-fit:cover;border-radius:3px;background:rgba(0,0,0,.28)}

.panel-bottom{display:flex;flex-direction:column;gap:10px;padding-top:12px;border-top:1px solid #262a30}
.notes-wrap{display:flex;flex-direction:column;gap:5px}
.notes-label{font-size:12px;color:var(--dim)}
.notes-wrap textarea{width:100%;padding:7px 10px;background:rgba(0,0,0,.22);border:1px solid var(--edge);
  border-radius:var(--radius);color:var(--ink);font:inherit;font-size:14px;resize:vertical}
.notes-wrap textarea::placeholder{color:#6d7681}
.panel-foot{display:flex;gap:8px}
.ghost{flex:1;padding:8px 6px;background:transparent;border:1px solid var(--edge);
  border-radius:var(--radius);color:var(--dim);font-size:13px}
.ghost:hover{color:var(--ink);border-color:var(--dim)}
.primary{width:100%;padding:11px 14px;background:var(--pick);border:1px solid var(--pick);
  border-radius:var(--radius);color:#12161b;font-size:14px;font-weight:600}
.primary:hover:not(:disabled){background:#93c2ff}
.primary:disabled{background:var(--raise);border-color:var(--edge);color:var(--dim);cursor:not-allowed}

.rail{position:relative;display:flex;align-items:center;gap:14px;padding:9px var(--pad);background:#1c1f23;
  border-top:1px solid #14171a;font-size:13px;color:var(--dim)}
.bar{flex:1;height:4px;border-radius:2px;background:#2b2f34;overflow:hidden}
.bar span{display:block;width:0;height:100%;background:var(--pick);transition:width 140ms ease-out}
.rail-meta{display:flex;align-items:center;gap:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.rail-meta a{color:var(--dim)}
.rail-meta a:hover{color:var(--ink)}
.who{color:var(--ink)}
.sep{width:1px;height:12px;background:var(--edge)}
.link{padding:0;background:none;border:0;color:var(--dim);text-decoration:underline;text-underline-offset:2px}
.link:hover{color:var(--ink)}
.toast{position:absolute;right:var(--pad);bottom:calc(100% + 10px);margin:0;padding:7px 12px;
  background:var(--raise);border:1px solid var(--edge);border-radius:var(--radius);color:var(--ink);font-size:13px}

dialog{padding:22px 24px;max-width:400px;background:var(--panel);border:1px solid var(--edge);
  border-radius:8px;color:var(--ink)}
dialog::backdrop{background:rgba(10,12,14,.6)}
dialog h2{margin:0 0 14px;font-size:17px}
dialog dl{display:grid;grid-template-columns:72px 1fr;gap:7px 14px;margin:0 0 18px;font-size:14px}
dialog dt{color:var(--pick);font-variant-numeric:tabular-nums}
dialog dd{margin:0;color:var(--dim)}

@media (max-width:900px){
  .stage{grid-template-columns:1fr}
  .well{min-height:44vh}
  .panel{border-left:0;border-top:1px solid #1b1e21}
}
@media (prefers-reduced-motion:reduce){*{transition:none !important}}
</style>
</head>
<body>

<div class="gate" id="gate">
  <div class="gate-box">
    <h1>Who is annotating?</h1>
    <p>Your judgments are saved to your own file. Pick the same name every time.</p>
    <p class="gate-err" id="gateErr" hidden></p>
    <select id="gateSelect" hidden></select>
    <input id="gateInput" placeholder="your name" autocomplete="off" hidden>
    <button class="primary" id="gateGo" type="button">Start</button>
    <div class="gate-resume" id="gateResume" hidden>
      <p id="gateResumeText"></p>
      <div class="gate-resume-row">
        <button class="primary" id="gateContinue" type="button">Go back to last session</button>
        <button class="ghost" id="gateRestart" type="button">Start a new session</button>
      </div>
      <p class="gate-note">Starting a new session keeps your old work; the file is renamed, not deleted.</p>
    </div>
  </div>
</div>

<main class="stage">
  <div class="well" id="well">
    <img id="image" alt="" hidden>
    <p class="well-empty" id="wellEmpty" hidden>Image file not found on disk.</p>
    <button class="zoom" id="zoomBtn" type="button" title="Toggle actual size (f)">Actual size</button>
  </div>

  <aside class="panel">
    <header class="panel-head">
      <span class="image-id" id="imageId">&nbsp;</span>
      <span class="counter"><b id="position">0</b> of <span id="total">0</span></span>
    </header>

    <p class="prompt">Which models got it right?</p>

    <div class="scroller">
      <div class="cards" id="cards"></div>
    </div>

    <div class="panel-bottom">
      <p class="warn" id="disagree" hidden></p>
      <section class="truth" id="truth">
        <div class="truth-head">If none of them is right, what is it?</div>
        <div class="truth-sub" id="truthSub">Your own judgment of the true label.</div>

        <button class="opt" id="optNonMeme" type="button">
          <span class="dot"></span><span>Not a meme</span><code>n</code>
        </button>
        <button class="opt" id="optTemplateless" type="button">
          <span class="dot"></span><span>A meme, but no template</span><code>t</code>
        </button>
        <button class="opt" id="optTemplate" type="button">
          <span class="dot"></span><span>Another template</span><code>o</code>
        </button>
        <div class="truth-input" id="truthInput" hidden>
          <input id="truthText" autocomplete="off" spellcheck="false"
                 placeholder="type what you think it is - closest templates appear below">
          <div class="hint" id="truthHint"></div>
          <div class="search-results" id="searchResults"></div>
          <div class="truth-preview" id="truthPreview"></div>
        </div>
        <button class="opt" id="optUnsure" type="button">
          <span class="dot"></span><span>Can't tell</span><code>u</code>
        </button>
      </section>

      <label class="notes-wrap">
        <span class="notes-label">Note (optional)</span>
        <textarea id="notes" rows="1" placeholder="e.g. cropped, two templates stacked"></textarea>
      </label>
      <div class="panel-foot">
        <button class="ghost" id="prevBtn" type="button" title="Previous (left arrow)">Back</button>
        <button class="ghost" id="skipBtn" type="button" title="Next unjudged (j)">Next unjudged</button>
        <button class="ghost" id="helpBtn" type="button">Shortcuts</button>
      </div>
      <button class="primary" id="saveBtn" type="button">Save and continue</button>
    </div>
  </aside>
</main>

<footer class="rail">
  <div class="bar"><span id="barFill"></span></div>
  <div class="rail-meta">
    <span class="who" id="whoText"></span>
    <span class="sep"></span>
    <span id="progressText">0 judged</span>
    <span class="sep"></span>
    <span id="modeText"></span>
    <span class="sep"></span>
    <a href="/api/export" download>Download my CSV</a>
    <span class="sep"></span>
    <button class="link" id="switchBtn" type="button">Not you?</button>
  </div>
  <p class="toast" id="toast" hidden></p>
</footer>

<dialog id="help">
  <h2>Keyboard</h2>
  <dl>
    <dt>1 - 9</dt><dd>Toggle a model as correct (several can be right)</dd>
    <dt>n</dt><dd>True label: not a meme</dd>
    <dt>t</dt><dd>True label: a meme with no template</dd>
    <dt>o</dt><dd>True label: type a template name</dd>
    <dt>u</dt><dd>Can't tell</dd>
    <dt>Enter</dt><dd>Save and continue</dd>
    <dt>&larr; &rarr;</dt><dd>Move between images without saving</dd>
    <dt>j</dt><dd>Jump to the next unjudged image</dd>
    <dt>f</dt><dd>Fit the image or show it at actual size</dd>
    <dt>Esc</dt><dd>Leave the text box, or close this</dd>
  </dl>
  <button class="primary" id="helpClose" type="button">Close</button>
</dialog>

<script>
"use strict";
const el = (id) => document.getElementById(id);
const dom = {
  gate: el("gate"), gateSelect: el("gateSelect"), gateInput: el("gateInput"),
  gateGo: el("gateGo"), gateErr: el("gateErr"),
  well: el("well"), image: el("image"), wellEmpty: el("wellEmpty"), zoomBtn: el("zoomBtn"),
  imageId: el("imageId"), position: el("position"), total: el("total"),
  cards: el("cards"), disagree: el("disagree"),
  truth: el("truth"), truthSub: el("truthSub"), truthText: el("truthText"),
  truthHint: el("truthHint"), truthPreview: el("truthPreview"), truthInput: el("truthInput"),
  optNonMeme: el("optNonMeme"), optTemplateless: el("optTemplateless"),
  optTemplate: el("optTemplate"), optUnsure: el("optUnsure"),
  searchResults: el("searchResults"),
  notes: el("notes"), prevBtn: el("prevBtn"), skipBtn: el("skipBtn"), saveBtn: el("saveBtn"),
  barFill: el("barFill"), progressText: el("progressText"), modeText: el("modeText"),
  whoText: el("whoText"), switchBtn: el("switchBtn"),
  gateResume: el("gateResume"), gateResumeText: el("gateResumeText"),
  gateContinue: el("gateContinue"), gateRestart: el("gateRestart"),
  toast: el("toast"), help: el("help"), helpBtn: el("helpBtn"), helpClose: el("helpClose"),
};
const OPTS = { NON_MEME: "optNonMeme", TEMPLATELESS: "optTemplateless",
               TEMPLATE: "optTemplate", UNSURE: "optUnsure" };
const state = {
  session: null, pos: 0, item: null,
  picked: new Set(), truthKind: null, startedAt: 0, saving: false, signedIn: false,
  searchSeq: 0, searchTimer: null, chosenLabel: null, lastResults: [],
};
let toastTimer = null, previewTimer = null;

const norm = (s) => String(s || "").trim().toLowerCase()
  .replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    const err = new Error(detail.error || (res.status + " " + res.statusText));
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function toast(message) {
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { dom.toast.hidden = true; }, 2200);
}

/* ----------------------------------------------------------------- gate -- */

async function showGate(roster) {
  state.signedIn = false;
  dom.gate.classList.add("show");
  if (roster && roster.length) {
    dom.gateSelect.hidden = false;
    dom.gateInput.hidden = true;
    dom.gateSelect.innerHTML = "";
    for (const name of roster) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      dom.gateSelect.append(opt);
    }
    dom.gateSelect.focus();
  } else {
    dom.gateSelect.hidden = true;
    dom.gateInput.hidden = false;
    dom.gateInput.focus();
  }
  dom.gateResume.hidden = true;
  dom.gateGo.hidden = false;
}

function whenText(iso) {
  if (!iso) return null;
  const at = new Date(iso);
  if (isNaN(at)) return null;
  const mins = Math.round((Date.now() - at) / 60000);
  const clock = at.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  if (mins < 1) return "just now";
  if (mins < 60) return mins + " min ago (" + clock + ")";
  if (mins < 1440) return Math.round(mins / 60) + " h ago (" + clock + ")";
  return clock;
}

function describeReturn(info) {
  const when = whenText(info.last_session && info.last_session.timestamp);
  const lead = when
    ? info.annotator + " had an earlier session " + when + "."
    : info.annotator + " has been here before.";
  return info.judged
    ? lead + " " + info.judged + " of " + info.assigned + " images judged so far."
    : lead + " No images were judged in it.";
}

function enterApp() {
  dom.gateResume.hidden = true;
  dom.gate.classList.remove("show");
  return boot();
}

async function signIn() {
  const name = dom.gateSelect.hidden ? dom.gateInput.value.trim() : dom.gateSelect.value;
  if (!name) { dom.gateInput.focus(); return; }
  dom.gateErr.hidden = true;
  let info;
  try {
    info = await api("/api/signin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  } catch (err) {
    dom.gateErr.textContent = err.message;
    dom.gateErr.hidden = false;
    return;
  }
  // Anyone who has been here before is told so and chooses, even if they judged nothing
  // last time -- otherwise a restart is indistinguishable from a first visit.
  if (info.resumed || info.last_session) {
    dom.gateResumeText.textContent = describeReturn(info);
    dom.gateResume.hidden = false;
    dom.gateGo.hidden = true;
    return;
  }
  await enterApp();
}

/* ------------------------------------------------------------- rendering -- */

function renderCards() {
  dom.cards.innerHTML = "";
  state.item.rows.forEach((row, i) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "card";
    card.dataset.model = row.model;
    if (state.picked.has(row.model)) card.classList.add("on");

    const top = document.createElement("div");
    top.className = "card-top";
    const key = document.createElement("span");
    key.className = "key";
    key.textContent = String(i + 1);

    const body = document.createElement("div");
    const name = document.createElement("div");
    name.className = "card-model";
    name.textContent = row.label;

    const pred = document.createElement("div");
    if (row.missing) {
      pred.className = "card-pred missing";
      pred.textContent = "no prediction for this image";
    } else if (row.kind === "NON_MEME" || row.kind === "TEMPLATELESS") {
      pred.className = "card-pred special";
      pred.textContent = row.kind;
    } else {
      pred.className = "card-pred";
      pred.textContent = row.prediction;
    }
    body.append(name, pred);

    if (row.confidence !== null && row.confidence !== undefined) {
      const conf = document.createElement("div");
      conf.className = "conf";
      const track = document.createElement("div");
      track.className = "conf-track";
      const fill = document.createElement("span");
      fill.style.width = (Math.max(0, Math.min(1, row.confidence)) * 100) + "%";
      track.append(fill);
      const value = document.createElement("span");
      value.className = "conf-value";
      value.textContent = row.confidence.toFixed(3);
      conf.append(track, value);
      body.append(conf);
    }

    top.append(key, body);
    card.append(top);

    // Examples only when the model named a specific template.
    if (row.kind === "TEMPLATE") {
      if (row.examples.length) {
        const strip = document.createElement("div");
        strip.className = "examples";
        row.examples.forEach((url) => {
          const img = document.createElement("img");
          img.loading = "lazy";
          img.alt = "Example of " + row.prediction;
          img.src = url;
          strip.append(img);
        });
        card.append(strip);
      } else {
        const none = document.createElement("div");
        none.className = "examples-none";
        none.textContent = "no example images on file for this template";
        card.append(none);
      }
    }

    card.addEventListener("click", () => toggleModel(row.model));
    dom.cards.append(card);
  });
}

function renderSelection() {
  for (const card of dom.cards.children) {
    card.classList.toggle("on", state.picked.has(card.dataset.model));
  }
  const locked = state.picked.size > 0;
  dom.truth.classList.toggle("off", locked);
  dom.truthSub.textContent = locked
    ? "Not needed - the true label comes from the model you checked."
    : "Your own judgment of the true label.";
  for (const kind of Object.keys(OPTS)) {
    const node = dom[OPTS[kind]];
    node.classList.toggle("on", !locked && state.truthKind === kind);
    node.disabled = locked;
  }
  const typing = !locked && state.truthKind === "TEMPLATE";
  dom.truthInput.hidden = !typing;
  dom.truthText.disabled = !typing;

  // Two models can only both be right if they said the same thing.
  const picked = state.item.rows.filter((r) => state.picked.has(r.model));
  const distinct = new Set(picked.map((r) => r.kind === "TEMPLATE" ? norm(r.prediction) : r.kind));
  dom.disagree.hidden = distinct.size <= 1;
  if (distinct.size > 1) {
    dom.disagree.textContent =
      "Those models predicted different labels, so they cannot all be right. Keep only the one that matches.";
  }

  dom.saveBtn.disabled = state.saving || !ready();
}

function ready() {
  if (state.picked.size > 0) return true;
  if (state.truthKind === "TEMPLATE") return dom.truthText.value.trim().length > 0;
  return state.truthKind !== null;
}

function renderProgress() {
  const done = state.session.tasks.filter((t) => t.done).length;
  const total = state.session.tasks.length;
  dom.barFill.style.width = total ? ((done / total) * 100) + "%" : "0";
  dom.progressText.textContent = done + " of " + total + " judged";
}

/* ------------------------------------------------------------- selection -- */

function toggleModel(model) {
  if (state.picked.has(model)) state.picked.delete(model);
  else state.picked.add(model);
  if (state.picked.size > 0) { state.truthKind = null; clearPreview(); }
  renderSelection();
}

function setTruth(kind, focus) {
  if (state.picked.size > 0) return;
  state.truthKind = state.truthKind === kind ? null : kind;
  if (state.truthKind !== "TEMPLATE") clearPreview();
  renderSelection();
  if (state.truthKind === "TEMPLATE" && focus !== false) {
    dom.truthText.focus();
    dom.truthText.select();
    refreshPreview();
  }
}

function clearPreview() {
  dom.truthPreview.innerHTML = "";
  dom.truthHint.textContent = "";
  dom.searchResults.innerHTML = "";
  state.chosenLabel = null;
  state.lastResults = [];
  if (state.searchTimer) { clearTimeout(state.searchTimer); state.searchTimer = null; }
}

function pickSearchResult(label) {
  // Clicking a hit commits it: the box becomes that exact name, so what gets saved is a
  // real template rather than whatever half-word the annotator had typed so far.
  state.chosenLabel = label;
  dom.truthText.value = label;
  dom.truthHint.textContent = "Will be saved as " + label + ".";
  renderSearch(state.lastResults);
  renderSelection();
}

function renderSearch(results) {
  dom.searchResults.innerHTML = "";
  const value = dom.truthText.value.trim();
  for (const hit of results) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "sr" + (state.chosenLabel === hit.label ? " on" : "");
    const shots = document.createElement("span");
    shots.className = "sr-shots";
    hit.examples.slice(0, 3).forEach((url) => {
      const img = document.createElement("img");
      img.loading = "lazy"; img.alt = hit.label; img.src = url;
      shots.append(img);
    });
    const name = document.createElement("span");
    name.className = "sr-name";
    name.textContent = hit.label;
    row.append(shots, name);
    row.addEventListener("click", () => pickSearchResult(hit.label));
    dom.searchResults.append(row);
  }
  // Always offer the escape hatch: none of the suggestions is right, keep my wording.
  if (value) {
    const own = document.createElement("button");
    own.type = "button";
    own.className = "sr sr-own" + (state.chosenLabel === null ? " on" : "");
    own.innerHTML = "None of these - save as <strong></strong>";
    own.querySelector("strong").textContent = value;
    own.addEventListener("click", () => {
      state.chosenLabel = null;
      dom.truthHint.textContent = "Will be saved as typed: " + value;
      renderSearch(state.lastResults);
      renderSelection();
    });
    dom.searchResults.append(own);
  }
}

async function runSearch() {
  const value = dom.truthText.value.trim();
  if (state.truthKind !== "TEMPLATE" || !value) return clearPreview();
  const seq = ++state.searchSeq;
  try {
    const data = await api("/api/search?q=" + encodeURIComponent(value) + "&k=5");
    if (seq !== state.searchSeq) return;          // a later keystroke already won
    state.lastResults = data.results || [];
    if (!state.lastResults.length) {
      dom.truthHint.textContent = "No close template. It will be saved as typed.";
    } else if (state.chosenLabel === null) {
      dom.truthHint.textContent = "Closest templates - click one, or keep your own wording.";
    }
    renderSearch(state.lastResults);
  } catch (err) {
    dom.searchResults.innerHTML = "";
  }
}

function refreshPreview() {
  // Debounced, so a burst of typing costs one query instead of one per character.
  if (state.chosenLabel && dom.truthText.value.trim() !== state.chosenLabel) state.chosenLabel = null;
  if (state.searchTimer) clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(runSearch, 220);
}

async function refreshPreviewExact() {
  const value = dom.truthText.value.trim();
  if (state.truthKind !== "TEMPLATE" || !value) return clearPreview();
  try {
    const data = await api("/api/examples?label=" + encodeURIComponent(value));
    dom.truthPreview.innerHTML = "";
    if (data.known) {
      dom.truthHint.textContent = "Matches " + data.label + ".";
      data.examples.forEach((url) => {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.alt = "Example of " + data.label;
        img.src = url;
        dom.truthPreview.append(img);
      });
      if (!data.examples.length) dom.truthHint.textContent += " No example images on file.";
    } else {
      dom.truthHint.textContent = "Not a label seen in the predictions file. It will be saved as typed.";
    }
  } catch (err) {
    clearPreview();
  }
}

/* ------------------------------------------------------------ navigation -- */

async function load(pos) {
  if (pos < 0 || pos >= state.session.tasks.length) return;
  state.pos = pos;
  const task = state.session.tasks[pos];
  try {
    state.item = await api("/api/items/" + task.index);
  } catch (err) {
    if (err.status === 401) return showGate(state.session ? state.session.roster : null);
    throw err;
  }

  dom.imageId.textContent = state.item.image_id;
  dom.position.textContent = pos + 1;
  dom.image.hidden = !state.item.image_exists;
  dom.wellEmpty.hidden = state.item.image_exists;
  if (state.item.image_exists) {
    dom.image.src = state.item.image_url;
    dom.image.alt = "Meme " + state.item.image_id;
  }
  dom.well.classList.remove("actual");
  dom.zoomBtn.textContent = "Actual size";

  state.picked = new Set();
  state.truthKind = null;
  dom.truthText.value = "";
  dom.notes.value = "";
  clearPreview();

  const prior = state.item.annotation;
  if (prior) {
    state.picked = new Set(prior.correct);
    dom.notes.value = prior.notes || "";
    if (!prior.correct.length && prior.true_label_kind) {
      state.truthKind = prior.true_label_kind;
      if (prior.true_label_kind === "TEMPLATE") dom.truthText.value = prior.true_label || "";
    }
  }

  state.startedAt = performance.now();
  renderCards();
  renderSelection();
  renderProgress();
  if (state.truthKind === "TEMPLATE") refreshPreview();
  const next = state.session.tasks[pos + 1];
  if (next) { const im = new Image(); im.src = "/api/image/" + next.index; }
}

function nextUnjudged(from) {
  const tasks = state.session.tasks;
  for (let i = from; i < tasks.length; i++) if (!tasks[i].done) return i;
  for (let i = 0; i < from; i++) if (!tasks[i].done) return i;
  return -1;
}

function jump() {
  const target = nextUnjudged(state.pos + 1);
  if (target === -1) toast("Nothing left unjudged.");
  else load(target);
}

/* ---------------------------------------------------------------- saving -- */

async function save() {
  if (state.saving) return;
  if (!ready()) {
    toast(state.truthKind === "TEMPLATE"
      ? "Type the template name, or pick another option."
      : "Check a model, or say what the image actually is.");
    if (state.truthKind === "TEMPLATE") dom.truthText.focus();
    return;
  }
  state.saving = true;
  renderSelection();
  try {
    await api("/api/items/" + state.session.tasks[state.pos].index + "/annotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        correct: [...state.picked],
        truth_kind: state.picked.size ? null : state.truthKind,
        truth_label: dom.truthText.value.trim(),
        notes: dom.notes.value.trim(),
        seconds: (performance.now() - state.startedAt) / 1000,
      }),
    });
    state.session.tasks[state.pos].done = true;
    renderProgress();
  } catch (err) {
    state.saving = false;
    if (err.status === 401) { showGate(state.session.roster); return; }
    toast("Not saved: " + err.message);
    renderSelection();
    return;
  }
  state.saving = false;
  const next = state.pos + 1;
  if (next < state.session.tasks.length) return load(next);
  const remaining = nextUnjudged(0);
  if (remaining === -1) { toast("All your images are judged. Download the CSV from the bar below."); renderSelection(); }
  else { toast("End of your set - jumping back to what's left."); load(remaining); }
}

function toggleZoom() {
  const actual = dom.well.classList.toggle("actual");
  dom.zoomBtn.textContent = actual ? "Fit to window" : "Actual size";
}

/* ----------------------------------------------------------- interaction -- */

document.addEventListener("keydown", (event) => {
  if (dom.gate.classList.contains("show")) {
    if (event.key === "Enter") { event.preventDefault(); signIn(); }
    return;
  }
  if (dom.help.open) { if (event.key === "Escape") dom.help.close(); return; }
  if (event.metaKey || event.ctrlKey || event.altKey) return;

  const target = event.target;
  if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT")) {
    if (event.key === "Escape") target.blur();
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); target.blur(); save(); }
    return;
  }

  const digit = Number(event.key);
  if (Number.isInteger(digit) && digit >= 1 && digit <= state.item.rows.length) {
    event.preventDefault();
    toggleModel(state.item.rows[digit - 1].model);
    return;
  }

  switch (event.key) {
    case "n": event.preventDefault(); setTruth("NON_MEME"); break;
    case "t": event.preventDefault(); setTruth("TEMPLATELESS"); break;
    case "o": event.preventDefault(); setTruth("TEMPLATE"); break;
    case "u": event.preventDefault(); setTruth("UNSURE"); break;
    case "Enter": event.preventDefault(); save(); break;
    case "ArrowLeft": event.preventDefault(); load(state.pos - 1); break;
    case "ArrowRight": event.preventDefault(); load(state.pos + 1); break;
    case "j": event.preventDefault(); jump(); break;
    case "f": event.preventDefault(); toggleZoom(); break;
    case "?": event.preventDefault(); dom.help.showModal(); break;
  }
});

dom.gateGo.addEventListener("click", signIn);
dom.optNonMeme.addEventListener("click", () => setTruth("NON_MEME"));
dom.optTemplateless.addEventListener("click", () => setTruth("TEMPLATELESS"));
dom.optTemplate.addEventListener("click", () => setTruth("TEMPLATE"));
dom.optUnsure.addEventListener("click", () => setTruth("UNSURE"));
dom.truthText.addEventListener("input", () => {
  renderSelection();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, 220);
});
dom.saveBtn.addEventListener("click", () => save());
dom.prevBtn.addEventListener("click", () => load(state.pos - 1));
dom.skipBtn.addEventListener("click", jump);
dom.zoomBtn.addEventListener("click", toggleZoom);
dom.helpBtn.addEventListener("click", () => dom.help.showModal());
dom.helpClose.addEventListener("click", () => dom.help.close());
dom.gateContinue.addEventListener("click", () => enterApp());

dom.gateRestart.addEventListener("click", async () => {
  if (!confirm("Start a fresh pass? Your existing judgments are kept in a renamed file.")) return;
  try {
    await api("/api/restart", { method: "POST" });
  } catch (err) {
    dom.gateErr.textContent = err.message;
    dom.gateErr.hidden = false;
    return;
  }
  await enterApp();
});

dom.switchBtn.addEventListener("click", async () => {
  await fetch("/api/signout", { method: "POST" });
  showGate(state.session ? state.session.roster : null);
});

/* ------------------------------------------------------------------ boot -- */

async function boot() {
  let session;
  try {
    session = await api("/api/session");
  } catch (err) {
    if (err.status === 401) {
      const info = await fetch("/api/whoami").then((r) => r.json()).catch(() => ({}));
      state.session = { roster: info.roster || [] };
      return showGate(info.roster || []);
    }
    document.body.textContent = "Could not reach the server: " + err.message;
    return;
  }
  state.session = session;
  state.signedIn = true;
  dom.total.textContent = session.tasks.length;
  dom.whoText.textContent = session.annotator;
  dom.modeText.textContent = session.models.length + " models";
  if (!session.tasks.length) {
    document.body.textContent = "No images are assigned to " + session.annotator + ".";
    return;
  }
  const first = nextUnjudged(0);
  await load(first === -1 ? 0 : first);
}

boot();
</script>
</body>
</html>
"""


# ===========================================================================
# Data
# ===========================================================================

@dataclass
class Prediction:
    model: str
    prediction: str
    confidence: float | None = None
    examples: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return label_kind(self.prediction)


@dataclass
class Item:
    index: int
    image_id: str
    image_path: Path
    preds: dict[str, Prediction] = field(default_factory=dict)


@dataclass
class Annotation:
    image_id: str
    correct: list[str]
    decision: str                       # judged | none_correct | unsure
    true_label: str = ""
    true_label_kind: str = ""
    true_label_source: str = ""
    notes: str = ""
    seconds: float = 0.0
    timestamp: str = ""
    display_order: list[str] = field(default_factory=list)


def clean(value) -> str:
    return str(value or "").strip()


def to_float(raw) -> float | None:
    raw = clean(raw)
    if raw == "" or raw.lower() in {"na", "n/a", "none", "nan", "null"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def detect_wide_models(fieldnames: list[str]) -> dict[str, tuple[str, str | None]]:
    lowered = {f.lower(): f for f in fieldnames}
    models: dict[str, tuple[str, str | None]] = {}
    for lower, original in lowered.items():
        for suffix in PRED_SUFFIXES:
            if lower.endswith(suffix) and len(lower) > len(suffix):
                model = original[: -len(suffix)]
                conf = None
                for csuffix in CONF_SUFFIXES:
                    if (model + csuffix).lower() in lowered:
                        conf = lowered[(model + csuffix).lower()]
                        break
                models[model] = (original, conf)
                break
    return models


def load_items(path: Path, images_root: Path, wanted: list[str] | None) -> tuple[list[Item], list[str]]:
    if not path.exists():
        sys.exit(f"predictions file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if "image_id" not in fields:
        sys.exit(f"{path} needs an image_id column. Columns found: {', '.join(fields)}")

    is_long = {"model", "prediction"} <= set(fields)
    wide_map = {} if is_long else detect_wide_models(fields)
    if not is_long and not wide_map:
        sys.exit(
            f"{path} is neither long nor wide format.\n"
            f"  long: image_id,image_path,model,prediction,confidence\n"
            f"  wide: image_id,image_path,<model>_pred,<model>_conf\n"
            f"  columns found: {', '.join(fields)}"
        )

    by_id: dict[str, Item] = {}
    found: list[str] = []

    def ensure(image_id: str, raw_path: str) -> Item:
        if image_id not in by_id:
            candidate = Path(raw_path) if raw_path else Path(f"{image_id}.jpg")
            if not candidate.is_absolute():
                candidate = images_root / candidate
            by_id[image_id] = Item(index=len(by_id), image_id=image_id, image_path=candidate)
        return by_id[image_id]

    for row in rows:
        image_id = clean(row.get("image_id"))
        if not image_id:
            continue
        item = ensure(image_id, clean(row.get("image_path")))

        if is_long:
            model = clean(row.get("model"))
            if not model or (wanted and model not in wanted):
                continue
            if model not in found:
                found.append(model)
            raw_examples = clean(row.get("template_examples")) or clean(row.get("template_ref"))
            item.preds[model] = Prediction(
                model=model,
                prediction=clean(row.get("prediction")),
                confidence=to_float(row.get("confidence")),
                examples=[p for p in (x.strip() for x in raw_examples.split("|")) if p],
            )
        else:
            for model, (pred_col, conf_col) in wide_map.items():
                if wanted and model not in wanted:
                    continue
                if model not in found:
                    found.append(model)
                item.preds[model] = Prediction(
                    model=model,
                    prediction=clean(row.get(pred_col)),
                    confidence=to_float(row.get(conf_col)) if conf_col else None,
                )

    models = [m for m in (wanted or sorted(found)) if m in found]
    if not models:
        sys.exit("no models matched. Found: " + (", ".join(sorted(found)) or "none"))

    items = list(by_id.values())
    for i, item in enumerate(items):
        item.index = i
    return items, models


def load_annotations(out_csv: Path, models: list[str]) -> dict[str, Annotation]:
    if not out_csv.exists():
        return {}
    result: dict[str, Annotation] = {}
    with out_csv.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            image_id = clean(row.get("image_id"))
            if not image_id:
                continue
            result[image_id] = Annotation(
                image_id=image_id,
                correct=[m for m in models if clean(row.get(f"{m}__correct")) == "1"],
                decision=clean(row.get("decision")) or "judged",
                true_label=clean(row.get("true_label")),
                true_label_kind=clean(row.get("true_label_kind")),
                true_label_source=clean(row.get("true_label_source")),
                notes=clean(row.get("notes")),
                seconds=to_float(row.get("seconds_on_image")) or 0.0,
                timestamp=clean(row.get("timestamp_utc")),
                display_order=[p for p in clean(row.get("display_order")).split("|") if p],
            )
    return result


def build_example_index(templates_root: Path | None, per_label: int) -> tuple[dict[str, list[Path]], dict[str, str]]:
    """normalized label -> example image paths, and normalized label -> display name."""
    examples: dict[str, list[Path]] = {}
    display: dict[str, str] = {}
    if not templates_root or not templates_root.is_dir():
        return examples, display

    # Layout 1: a folder per template, templates/Pepe_the_Frog/*.jpg
    for child in sorted(templates_root.iterdir()):
        if child.is_dir():
            shots = sorted(p for p in child.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if shots:
                key = normalize(child.name)
                examples.setdefault(key, []).extend(shots[:per_label])
                display.setdefault(key, child.name)

    # Layout 2: flat files, Pepe_the_Frog.jpg / Pepe_the_Frog_1.jpg / Pepe-the-frog-2.png
    for path in sorted(templates_root.iterdir()):
        if path.is_dir() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = re.sub(r"[ _-]*\d+$", "", path.stem)
        key = normalize(stem)
        if not key:
            continue
        bucket = examples.setdefault(key, [])
        if len(bucket) < per_label:
            bucket.append(path)
        display.setdefault(key, stem)

    return examples, display


# ===========================================================================
# Corpus - loaded once, read-only afterwards, shared by every annotator
# ===========================================================================

class Corpus:
    def __init__(self, args: argparse.Namespace):
        self.images_root = Path(args.images_root).resolve()
        self.templates_root = Path(args.templates_root).resolve() if args.templates_root else None
        self.out_dir = Path(args.out_dir).resolve()
        self.seed = args.seed
        self.per_label = max(1, args.examples)

        wanted = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
        self.items, self.models = load_items(Path(args.predictions).resolve(), self.images_root, wanted)
        if args.shuffle:
            random.Random(self.seed).shuffle(self.items)
            for i, item in enumerate(self.items):
                item.index = i

        self.by_id = {item.image_id: item for item in self.items}
        self.examples, self.display_name = build_example_index(self.templates_root, self.per_label)

        # Fold explicit per-row example paths into the same index.
        for item in self.items:
            for pred in item.preds.values():
                if pred.kind != KIND_TEMPLATE or not pred.examples:
                    continue
                key = normalize(pred.prediction)
                if key in self.examples:
                    continue
                resolved = []
                for raw in pred.examples[: self.per_label]:
                    candidate = Path(raw)
                    if not candidate.is_absolute() and self.templates_root:
                        candidate = self.templates_root / candidate
                    if candidate.exists():
                        resolved.append(candidate)
                if resolved:
                    self.examples[key] = resolved
                    self.display_name.setdefault(key, pred.prediction)

        vocab: dict[str, str] = dict(self.display_name)
        for item in self.items:
            for pred in item.preds.values():
                if pred.kind == KIND_TEMPLATE:
                    vocab.setdefault(normalize(pred.prediction), pred.prediction)
        self.templates = sorted(vocab.values(), key=str.lower)
        self.known_keys = set(vocab)
        self.canonical = dict(vocab)
        self.search = TemplateSearch(self.templates)

        # Roster and work assignment.
        self.roster = [n.strip() for n in args.annotators.split(",") if n.strip()] if args.annotators else []
        self.open_roster = not self.roster
        self.overlap = max(0, args.overlap)
        self.assignments = self._assign()

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _assign(self) -> dict[str, list[int]]:
        """Which image indices each rostered annotator sees.

        With no roster, or no --overlap, everyone sees everything. With both,
        the first --overlap images go to everyone (that shared block is what you
        compute inter-annotator agreement on) and the rest is dealt round-robin
        so no two people duplicate work.
        """
        if not self.roster or not self.overlap:
            return {slugify(n): [i.index for i in self.items] for n in self.roster}
        shared = [i.index for i in self.items[: self.overlap]]
        rest = [i.index for i in self.items[self.overlap:]]
        out = {slugify(n): list(shared) for n in self.roster}
        for position, index in enumerate(rest):
            out[slugify(self.roster[position % len(self.roster)])].append(index)
        return out

    def indices_for(self, slug: str) -> list[int]:
        if slug in self.assignments:
            return self.assignments[slug]
        return [i.index for i in self.items]      # unrostered walk-in sees everything

    def display_order(self, image_id: str, slug: str) -> list[str]:
        """Card order for one image: the real model list, shuffled.

        Seeded on the image id (and the corpus seed), so the order is stable across
        reloads and repeat saves and is the same for every annotator looking at that
        image, but differs from one image to the next -- so a model's position on
        screen can't be used to infer its identity across the set.
        """
        order = list(self.models)
        random.Random(f"{self.seed}:{image_id}").shuffle(order)
        return order

    def example_urls(self, label: str) -> list[str]:
        key = normalize(label)
        count = len(self.examples.get(key, [])[: self.per_label])
        return [f"/api/example/{key}/{i}" for i in range(count)]

    def search_payload(self, text: str, limit: int = 5) -> dict:
        hits = []
        for label, score in self.search.query(text, limit):
            hits.append({
                "label": label,
                "score": score,
                "examples": self.example_urls(label),
            })
        return {"query": text, "results": hits}

    def examples_payload(self, label: str) -> dict:
        key = normalize(label)
        return {
            "label": self.display_name.get(key, label),
            "known": key in self.known_keys,
            "examples": self.example_urls(label),
        }

    def fieldnames(self) -> list[str]:
        names = ["image_id", "annotator", "timestamp_utc", "decision", "n_models_correct"]
        names += [f"{m}__correct" for m in self.models]
        names += [f"{m}__prediction" for m in self.models]
        names += [f"{m}__confidence" for m in self.models]
        names += ["true_label", "true_label_kind", "true_label_source"]
        return names + ["notes", "display_order", "seconds_on_image"]


# ===========================================================================
# Annotator - one per person: own file, own dict, own lock
# ===========================================================================

class Annotator:
    def __init__(self, corpus: Corpus, name: str):
        self.corpus = corpus
        self.name = name
        self.slug = slugify(name)
        self.out_csv = corpus.out_dir / f"{self.slug}.csv"
        self.indices = corpus.indices_for(self.slug)
        self.annotations = load_annotations(self.out_csv, corpus.models)
        self.lock = threading.Lock()

    # -- payloads ---------------------------------------------------------- #

    def item_payload(self, index: int) -> dict:
        c = self.corpus
        item = c.items[index]
        rows = []
        for slot, model in enumerate(c.display_order(item.image_id, self.slug)):
            pred = item.preds.get(model)
            kind = pred.kind if pred else KIND_TEMPLATELESS
            rows.append({
                "model": model,                     # internal key only, never rendered
                "label": f"Model {slot + 1}",        # anonymized, position-based display name
                "prediction": pred.prediction if pred else "",
                "kind": kind,
                "missing": pred is None,
                # Always None: confidence values expose which prediction came from a
                # model that doesn't report one (e.g. the LLM), which would de-anonymize it.
                "confidence": None,
                "examples": c.example_urls(pred.prediction) if pred and kind == KIND_TEMPLATE else [],
            })
        ann = self.annotations.get(item.image_id)
        return {
            "index": index,
            "image_id": item.image_id,
            "image_url": f"/api/image/{index}",
            "image_exists": item.image_path.exists(),
            "rows": rows,
            "annotation": ({
                "correct": ann.correct,
                "decision": ann.decision,
                "true_label": ann.true_label,
                "true_label_kind": ann.true_label_kind,
                "notes": ann.notes,
            } if ann else None),
        }

    def session_payload(self) -> dict:
        c = self.corpus
        return {
            "annotator": self.name,
            "roster": c.roster,
            "models": c.models,
            "templates": c.templates,
            "tasks": [{"index": i, "image_id": c.items[i].image_id,
                       "done": c.items[i].image_id in self.annotations}
                      for i in self.indices],
        }

    # -- output ------------------------------------------------------------ #

    def row_for(self, ann: Annotation) -> dict:
        c = self.corpus
        item = c.by_id[ann.image_id]
        row = {
            "image_id": ann.image_id,
            "annotator": self.name,
            "timestamp_utc": ann.timestamp,
            "decision": ann.decision,
            "n_models_correct": len(ann.correct),
            "true_label": ann.true_label,
            "true_label_kind": ann.true_label_kind,
            "true_label_source": ann.true_label_source,
            "notes": ann.notes,
            "display_order": "|".join(ann.display_order),
            "seconds_on_image": f"{ann.seconds:.1f}",
        }
        for model in c.models:
            pred = item.preds.get(model)
            row[f"{model}__correct"] = 1 if model in ann.correct else 0
            row[f"{model}__prediction"] = pred.prediction if pred else ""
            row[f"{model}__confidence"] = ("" if not pred or pred.confidence is None
                                           else f"{pred.confidence:.6g}")
        return row

    def flush(self) -> None:
        """Atomic rewrite of this annotator's file only."""
        c = self.corpus
        fd, tmp = tempfile.mkstemp(dir=str(c.out_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=c.fieldnames())
                writer.writeheader()
                for index in self.indices:
                    ann = self.annotations.get(c.items[index].image_id)
                    if ann:
                        writer.writerow(self.row_for(ann))
            os.chmod(tmp, 0o644)
            os.replace(tmp, self.out_csv)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def archive(self) -> str | None:
        """Set this annotator's work aside and begin an empty pass.

        The old file is renamed, never deleted -- a mis-click on "start over" should not
        cost somebody an afternoon of judgments.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        moved = None
        if self.out_csv.exists():
            target = self.out_csv.with_name(f"{self.slug}.archived_{stamp}.csv")
            os.replace(self.out_csv, target)
            moved = target.name
        self.annotations = {}
        self.registry_log("restart")
        return moved

    def registry_log(self, event: str) -> None:
        registry = getattr(Handler, "registry", None)
        if registry is not None:
            registry.log_session(self, event)

    def save(self, index: int, payload: dict) -> dict:
        c = self.corpus
        item = c.items[index]
        order = c.display_order(item.image_id, self.slug)
        chosen = set(payload.get("correct", []))
        correct = [m for m in order if m in chosen and m in c.models]

        if correct:
            # A checked model's prediction IS the true label.
            winner = correct[0]
            pred = item.preds.get(winner)
            kind = pred.kind if pred else KIND_TEMPLATELESS
            true_label = kind if kind in (KIND_NON_MEME, KIND_TEMPLATELESS) else pred.prediction
            decision, source = "judged", f"model:{winner}"
        else:
            kind = clean(payload.get("truth_kind")) or KIND_UNSURE
            if kind == KIND_TEMPLATE:
                true_label = clean(payload.get("truth_label"))
                if not true_label:
                    return {"error": "no template name was given"}
                # Canonical spelling, so one template never lands in the CSV
                # under several spellings.
                true_label = c.canonical.get(normalize(true_label), true_label)
            elif kind == KIND_UNSURE:
                true_label = ""
            else:
                true_label = kind
            decision = "unsure" if kind == KIND_UNSURE else "none_correct"
            source = "annotator"

        with self.lock:
            self.annotations[item.image_id] = Annotation(
                image_id=item.image_id,
                correct=correct,
                decision=decision,
                true_label=true_label,
                true_label_kind=kind,
                true_label_source=source,
                notes=str(payload.get("notes", ""))[:2000],
                seconds=float(payload.get("seconds", 0) or 0),
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                display_order=order,
            )
            self.flush()
        return {"ok": True, "done": len(self.annotations), "true_label": true_label, "kind": kind}


class Registry:
    """Hands out Annotator objects, one per person, created on first sign-in."""

    def __init__(self, corpus: Corpus, gate: bool):
        self.corpus = corpus
        self.gate = gate
        self.people: dict[str, Annotator] = {}
        self.lock = threading.Lock()

    def get(self, slug: str) -> Annotator | None:
        return self.people.get(slug)

    def log_session(self, who: "Annotator", event: str) -> None:
        """Append one line to annotations/sessions.csv: who, when, how far along.

        Lets you see who has been working and for how long without opening anyone's
        annotations, and gives the sign-in screen the numbers it reports back.
        """
        path = self.corpus.out_dir / SESSION_LOG
        new = not path.exists()
        with self.lock:
            with path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if new:
                    writer.writerow(["timestamp_utc", "annotator", "event", "run_id",
                                     "judged", "assigned"])
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    who.name, event, RUN_ID,
                    len(who.annotations), len(who.indices),
                ])

    def last_session(self, name: str) -> dict | None:
        """The most recent sign-in by this person from an *earlier* run of the server.

        Read from sessions.csv rather than from their annotations, so someone who signed
        in and judged nothing is still told they have been here before -- otherwise a
        restart looks identical to a first visit and there is no way to tell whether the
        work is still there.
        """
        path = self.corpus.out_dir / SESSION_LOG
        if not path.exists():
            return None
        slug = slugify(name)
        latest = None
        try:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if slugify(row.get("annotator", "")) != slug:
                        continue
                    if row.get("run_id") == RUN_ID:      # the session happening right now
                        continue
                    latest = row
        except OSError:
            return None
        if latest is None:
            return None
        return {
            "timestamp": latest.get("timestamp_utc", ""),
            "judged": int(latest.get("judged") or 0),
            "event": latest.get("event", "sign_in"),
        }

    def sign_in(self, name: str) -> Annotator:
        slug = slugify(name)
        if not slug:
            raise ValueError("please enter a name")
        if self.corpus.roster and slug not in {slugify(n) for n in self.corpus.roster}:
            raise ValueError(f"'{name}' is not on the roster: {', '.join(self.corpus.roster)}")
        with self.lock:
            if slug not in self.people:
                canonical = next((n for n in self.corpus.roster if slugify(n) == slug), name)
                self.people[slug] = Annotator(self.corpus, canonical)
                print(f"  + {canonical} signed in ({len(self.people[slug].annotations)} already done, "
                      f"{len(self.people[slug].indices)} assigned)")
            who = self.people[slug]
        self.log_session(who, "sign_in")
        return who


# ===========================================================================
# HTTP
# ===========================================================================

class Handler(BaseHTTPRequestHandler):
    registry: Registry
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        line = fmt % args
        if "/api/image/" not in line and "/api/example/" not in line:
            sys.stderr.write(line + "\n")

    # -- plumbing ---------------------------------------------------------- #

    def _send(self, body: bytes, ctype: str, status: int = 200, cache: str = "no-store", cookie: str = ""):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200, cookie: str = ""):
        self._send(json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8", status, cookie=cookie)

    def _file(self, path: Path | None, cache: str = "public, max-age=3600"):
        if path is None or not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(path.read_bytes(), ctype, cache=cache)

    def _who(self) -> Annotator | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(raw)
        except Exception:
            return None
        morsel = cookie.get("annotator")
        if not morsel:
            return None
        slug, _, run = morsel.value.partition("|")
        if run != RUN_ID:            # cookie from an earlier run of the server
            return None
        return self.registry.get(slug)

    def _index(self, raw: str) -> int | None:
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if 0 <= value < len(self.registry.corpus.items) else None

    # -- routes ------------------------------------------------------------ #

    def do_GET(self):
        c = self.registry.corpus
        url = urlparse(self.path)
        parts = [p for p in unquote(url.path).split("/") if p]

        if not parts:
            return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if parts == ["api", "whoami"]:
            return self._json({"roster": c.roster, "open": c.open_roster})
        if parts == ["api", "search"]:
            query = (parse_qs(urlparse(self.path).query).get("q") or [""])[0]
            try:
                limit = int((parse_qs(urlparse(self.path).query).get("k") or ["5"])[0])
            except ValueError:
                limit = 5
            return self._json(c.search_payload(query, max(1, min(limit, 20))))
        if parts == ["api", "examples"]:
            label = (parse_qs(url.query).get("label") or [""])[0]
            return self._json(c.examples_payload(label))

        # Everything below is per-annotator.
        me = self._who()
        if me is None:
            return self._json({"error": "sign in first"}, 401)

        if parts == ["api", "session"]:
            return self._json(me.session_payload())
        if parts == ["api", "export"]:
            if not me.out_csv.exists():
                me.flush()
            return self._file(me.out_csv, cache="no-store")
        if len(parts) == 3 and parts[:2] == ["api", "items"]:
            i = self._index(parts[2])
            return self._json(me.item_payload(i)) if i is not None else self._json({"error": "bad index"}, 404)
        if len(parts) == 3 and parts[:2] == ["api", "image"]:
            i = self._index(parts[2])
            return self._file(c.items[i].image_path) if i is not None else self._json({"error": "bad index"}, 404)
        if len(parts) == 4 and parts[:2] == ["api", "example"]:
            try:
                position = int(parts[3])
            except ValueError:
                return self._json({"error": "bad position"}, 404)
            shots = c.examples.get(parts[2], [])
            return self._file(shots[position] if 0 <= position < len(shots) else None)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        parts = [p for p in unquote(urlparse(self.path).path).split("/") if p]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if parts == ["api", "signin"]:
            try:
                # looked up before sign_in, which would otherwise log this visit first
                prior = self.registry.last_session(str(payload.get("name", "")))
                me = self.registry.sign_in(str(payload.get("name", "")))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            cookie = f"annotator={me.slug}|{RUN_ID}; Path=/; Max-Age=2592000; SameSite=Lax"
            return self._json({
                "ok": True,
                "annotator": me.name,
                "judged": len(me.annotations),
                "assigned": len(me.indices),
                "resumed": bool(me.annotations),
                "last_session": prior,
            }, cookie=cookie)

        if parts == ["api", "restart"]:
            me = self._who()
            if me is None:
                return self._json({"error": "sign in first"}, 401)
            archived = me.archive()
            return self._json({"ok": True, "archived": archived, "judged": 0})
        if parts == ["api", "signout"]:
            return self._json({"ok": True}, cookie="annotator=; Path=/; Max-Age=0; SameSite=Lax")

        me = self._who()
        if me is None:
            return self._json({"error": "sign in first"}, 401)

        if len(parts) == 4 and parts[:2] == ["api", "items"] and parts[3] == "annotation":
            i = self._index(parts[2])
            if i is None:
                return self._json({"error": "bad index"}, 404)
            result = me.save(i, payload)
            return self._json(result, 400 if "error" in result else 200)
        return self._json({"error": "not found"}, 404)


# ===========================================================================
# Merge and agreement
# ===========================================================================

def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters over the same items."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    expected = sum((count_a[k] / n) * (count_b[k] / n) for k in set(count_a) | set(count_b))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def run_collect(specs: list[str], out_path: Path) -> None:
    """Fold the method scripts' open_set_predictions.csv files into one long-format predictions.csv.

    Each spec is MODEL=PATH, where PATH is a run directory (the timestamped folder a
    method wrote when launched with --open-set-dir) or the CSV inside it. sample_id
    becomes image_id so rows join back to Social_Media/random_sampled/sample_manifest.csv.
    """
    header = ["image_id", "image_path", "model", "prediction", "confidence"]
    out_rows: list[dict[str, str]] = []
    covered: dict[str, set[str]] = {}
    print()
    for spec in specs:
        if "=" not in spec:
            sys.exit(f"--collect expects MODEL=PATH, got {spec!r}")
        model, raw = (part.strip() for part in spec.split("=", 1))
        src = Path(raw).expanduser()
        if src.is_dir():
            src = src / "open_set_predictions.csv"
        if not src.exists():
            sys.exit(f"{model}: no such file {src}")
        with src.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
        needed = {"sample_id", "image_path", "pred_label"}
        if not needed <= fields:
            sys.exit(f"{src} is not an open_set_predictions.csv (needs {', '.join(sorted(needed))}); "
                     f"columns found: {', '.join(sorted(fields))}")

        kinds: Counter = Counter()
        ids: set[str] = set()
        for row in rows:
            image_id = clean(row.get("sample_id"))
            if not image_id:
                continue
            prediction = clean(row.get("pred_label"))
            kind = label_kind(prediction)
            confidence = None
            meme_prob = clean(row.get("meme_probability"))
            if meme_prob and kind == KIND_NON_MEME:
                confidence = 1.0 - float(meme_prob)        # two-stage gate: confidence in the reject
            else:
                for col in OPEN_SET_CONF_COLUMNS:
                    if clean(row.get(col)):
                        confidence = float(row[col])
                        break
            ids.add(image_id)
            kinds[kind] += 1
            out_rows.append({
                "image_id": image_id,
                "image_path": clean(row.get("image_path")),
                "model": model,
                "prediction": prediction,
                "confidence": "" if confidence is None else f"{confidence:.6f}",
            })
        covered[model] = ids
        summary = ", ".join(f"{k.lower()} {v}" for k, v in sorted(kinds.items()))
        print(f"  {model:28s} {len(ids):5d} images   {summary}")

    if not out_rows:
        sys.exit("nothing collected")
    every = set().union(*covered.values())
    for model, ids in covered.items():
        missing = len(every - ids)
        if missing:
            print(f"  warning: {model} has no prediction for {missing} image(s) that other models cover")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {out_path}: {len(every)} images x {len(covered)} models = {len(out_rows)} rows")


def run_merge(indir: Path, out_path: Path | None) -> None:
    files = sorted(p for p in indir.glob("*.csv") if not p.name.startswith("."))
    if not files:
        sys.exit(f"no CSV files found in {indir}")

    rows: list[dict] = []
    fieldnames: list[str] = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                row.setdefault("annotator", path.stem)
                if not clean(row.get("annotator")):
                    row["annotator"] = path.stem
                rows.append(row)

    models = sorted({c[:-9] for c in fieldnames if c.endswith("__correct")})
    people = sorted({clean(r.get("annotator")) for r in rows})
    judged = [r for r in rows if clean(r.get("decision")) != "unsure"]

    print(f"\n  {len(rows)} rows from {len(files)} files: {', '.join(people)}")
    print(f"  {len(rows) - len(judged)} marked unsure and excluded from the figures below\n")

    print("  per-model accuracy on each annotator's own set")
    header = "    " + "annotator".ljust(16) + "".join(m[:22].rjust(24) for m in models)
    print(header)
    for person in people:
        mine = [r for r in judged if clean(r.get("annotator")) == person]
        cells = ""
        for model in models:
            vals = [1 if clean(r.get(f"{model}__correct")) == "1" else 0 for r in mine]
            cells += (f"{sum(vals) / len(vals):.3f}" if vals else "-").rjust(24)
        print("    " + person.ljust(16) + cells + f"   (n={len(mine)})")

    # Agreement on images more than one person judged.
    by_image: dict[str, dict[str, str]] = defaultdict(dict)
    for row in judged:
        by_image[clean(row.get("image_id"))][clean(row.get("annotator"))] = clean(row.get("true_label"))
    overlap = {k: v for k, v in by_image.items() if len(v) > 1}

    print(f"\n  {len(overlap)} images judged by more than one annotator")
    if overlap:
        print("  pairwise agreement on true_label")
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                pairs = [(v[a], v[b]) for v in overlap.values() if a in v and b in v]
                if not pairs:
                    continue
                raw = sum(1 for x, y in pairs if x == y) / len(pairs)
                kappa = cohen_kappa(pairs)
                print(f"    {a} vs {b}:  n={len(pairs)}  raw={raw:.3f}  kappa={kappa:.3f}")

    if out_path:
        for row in rows:
            labels = list(by_image.get(clean(row.get("image_id")), {}).values())
            if labels:
                winner, count = Counter(labels).most_common(1)[0]
                tied = sum(1 for _, c in Counter(labels).items() if c == count) > 1
                row["consensus_label"] = "" if tied else winner
                row["n_annotators"] = len(labels)
                row["n_agree"] = count
            else:
                row["consensus_label"], row["n_annotators"], row["n_agree"] = "", 0, 0
        extra = ["consensus_label", "n_annotators", "n_agree"]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames + extra, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  merged -> {out_path}")
    print()


# ===========================================================================
# Demo data (placeholder images are plain SVG, so no Pillow needed)
# ===========================================================================

DEMO_TEMPLATES = ["Pepe_the_Frog", "Distracted_Boyfriend", "Drake_Hotline_Bling", "Two_Buttons",
                  "Change_My_Mind", "Expanding_Brain", "Woman_Yelling_At_Cat", "This_Is_Fine"]
DEMO_MODELS = ["rNN_pHash", "two_stage_densenet121", "pca_hdbscan_siglip2_dinov2",
               "gemini_2.5_flash_lite_3shot"]
DEMO_COLORS = ["#e46054", "#4c84d6", "#56b082", "#d6a844", "#966cc8", "#4bb3bd"]


def write_svg(path: Path, label: str, w: int, h: int, rng: random.Random) -> None:
    color = rng.choice(DEMO_COLORS)
    shapes = "".join(
        f'<circle cx="{rng.randrange(w)}" cy="{rng.randrange(max(1, int(h * .6)))}" '
        f'r="{rng.randrange(14, 44)}" fill="none" stroke="#fff" stroke-width="3"/>' for _ in range(6)
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" fill="#f4f4f2"/>'
        f'<rect width="{w}" height="{int(h * .62)}" fill="{color}"/>{shapes}'
        f'<text x="14" y="{int(h * .75)}" font-family="sans-serif" font-size="14" fill="#18181a">{label}</text>'
        f"</svg>",
        encoding="utf-8",
    )


def make_demo(n: int, outdir: Path, seed: int = 7) -> tuple[Path, Path, Path]:
    rng = random.Random(seed)
    images, templates = outdir / "images", outdir / "templates"
    images.mkdir(parents=True, exist_ok=True)

    for name in DEMO_TEMPLATES:
        folder = templates / name
        folder.mkdir(parents=True, exist_ok=True)
        for k in range(1, 4):
            write_svg(folder / f"{k}.svg", f"{name.replace('_', ' ')} #{k}", 300, 230, rng)

    classes = DEMO_TEMPLATES + ["NON_MEME", "TEMPLATELESS"]
    weights = [6] * len(DEMO_TEMPLATES) + [9, 9]
    rows = []
    for i in range(n):
        image_id = f"img_{i:04d}"
        truth = rng.choices(classes, weights=weights)[0]
        w, h = rng.choice([(560, 460), (470, 620), (640, 420)])
        write_svg(images / f"{image_id}.svg", f"{image_id} | {truth}", w, h, rng)
        for model in DEMO_MODELS:
            pred = truth if rng.random() < 0.5 else rng.choice([c for c in classes if c != truth])
            conf = "" if model.startswith("pca_hdbscan") and rng.random() < 0.3 else f"{rng.uniform(.3, .99):.3f}"
            rows.append({"image_id": image_id, "image_path": f"{image_id}.svg", "model": model,
                         "prediction": pred, "confidence": conf})

    predictions = outdir / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "image_path", "model", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    return predictions, images, templates


# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="predictions.csv", help="long or wide predictions CSV")
    ap.add_argument("--images-root", default=".", help="directory relative image_path values resolve against")
    ap.add_argument("--templates-root", default=None,
                    help="reference templates: a folder per label, or files named <label>_1.jpg")
    ap.add_argument("--out-dir", default="annotations",
                    help="directory of per-annotator CSVs (default: annotations/)")
    ap.add_argument("--models", default=None, help="comma-separated subset of models, in display order")
    ap.add_argument("--examples", type=int, default=3, help="example images shown per template (default 3)")
    ap.add_argument("--annotator", default=None,
                    help="single-annotator mode: skip the sign-in screen and use this name")
    ap.add_argument("--annotators", default=None,
                    help="comma-separated roster; shows a sign-in picker and gives each person their own file")
    ap.add_argument("--overlap", type=int, default=0,
                    help="with --annotators: first N images go to everyone (for agreement), rest is split")
    ap.add_argument("--shuffle", action="store_true", help="shuffle image order")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let others on the network connect")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--demo", type=int, metavar="N", help="generate N placeholder images into ./demo")
    ap.add_argument("--merge", metavar="DIR", default=None,
                    help="combine per-annotator CSVs in DIR, print agreement, then exit")
    ap.add_argument("--merge-out", default=None, help="with --merge: write the combined CSV here")
    ap.add_argument("--collect", nargs="+", metavar="MODEL=PATH", default=None,
                    help="build a long-format predictions CSV from the method scripts' "
                         "open_set_predictions.csv files (PATH = run dir or the CSV), then exit")
    ap.add_argument("--collect-out", default="predictions.csv", help="with --collect: where to write it")
    args = ap.parse_args()

    if args.collect:
        return run_collect(args.collect, Path(args.collect_out))

    if args.merge:
        return run_merge(Path(args.merge), Path(args.merge_out) if args.merge_out else None)

    if args.demo:
        predictions, images, templates = make_demo(args.demo, Path("demo"))
        args.predictions, args.images_root, args.templates_root = str(predictions), str(images), str(templates)
        if args.out_dir == "annotations":
            args.out_dir = "demo/annotations"
        print(f"  demo        {args.demo} placeholder images in ./demo\n")

    corpus = Corpus(args)
    gate = bool(corpus.roster) or not args.annotator
    registry = Registry(corpus, gate)
    Handler.registry = registry

    if not gate:
        registry.sign_in(args.annotator)   # single-annotator mode: no sign-in screen

    url = f"http://{args.host}:{args.port}"
    print(f"  images      {len(corpus.items)}")
    print(f"  models      {', '.join(corpus.models)}")
    print(f"  templates   {len(corpus.templates)} known, {len(corpus.examples)} with example images")
    if corpus.roster:
        counts = ", ".join(f"{n} {len(corpus.indices_for(slugify(n)))}" for n in corpus.roster)
        print(f"  annotators  {counts}"
              + (f"   (first {corpus.overlap} shared)" if corpus.overlap else "   (everyone sees everything)"))
    else:
        print(f"  annotators  {'sign-in, open roster' if gate else args.annotator}")
    print(f"  output      {corpus.out_dir}{os.sep}<annotator>.csv")
    print(f"  serving     {url}\n")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. annotations saved in", corpus.out_dir)


if __name__ == "__main__":
    main()