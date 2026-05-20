---
title: "From Symbols to Meaning: How Language AI Really Works"
description: "A history of AI from symbolic systems to transformers, with a debunking of the stochastic parrot critique"
pubDate: "May 20 2026"
heroImage: "/personal_blog/FS2M.jpg"
---
 
# From Symbols to Meaning: How Language AI Really Works
*Author: Fernando Benites*

AI was supposed to reason like a logician. Instead, it learned to navigate meaning like a reader moving through a library — finding related ideas not by following rules, but by sensing proximity. This article traces that journey, explains what large language models actually do to words, and honestly confronts the question: are they just very eloquent parrots?
 
<div class="not-prose my-8 rounded-xl border border-base-300 bg-base-100 p-6 shadow-sm">
  <div class="flex items-baseline justify-between mb-5 pb-3 border-b border-base-300/60">
    <h3 class="text-xs font-bold uppercase tracking-[0.2em] text-base-content/60 m-0">Table of contents</h3>
    <span class="text-xs text-base-content/40">9 sections</span>
  </div>
  <ol class="grid gap-x-6 gap-y-4 sm:grid-cols-2 list-none p-0 m-0">
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">1</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">The expectation: AI as a reasoning machine</div>
        <ul class="mt-1.5 space-y-0.5 text-xs text-base-content/65 list-none p-0 m-0">
          <li class="m-0">1.1 Symbolic AI and the knowledge-engineering dream</li>
          <li class="m-0">1.2 Why the dream collapsed</li>
          <li class="m-0">1.3 Connectionism and the first neural-network winter</li>
        </ul>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">2</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">The statistical turn: meaning as location</div>
        <ul class="mt-1.5 space-y-0.5 text-xs text-base-content/65 list-none p-0 m-0">
          <li class="m-0">2.1 N-grams: language as statistics</li>
          <li class="m-0">2.2 Word embeddings: words as coordinates</li>
        </ul>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">3</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">Attention: meaning that moves</div>
        <ul class="mt-1.5 space-y-0.5 text-xs text-base-content/65 list-none p-0 m-0">
          <li class="m-0">3.1 The bottleneck problem</li>
          <li class="m-0">3.2 Attention as semantic navigation</li>
          <li class="m-0">3.3 What attention actually does in a transformer</li>
        </ul>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">4</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">Emergence: when scale becomes something else</div>
        <ul class="mt-1.5 space-y-0.5 text-xs text-base-content/65 list-none p-0 m-0">
          <li class="m-0">4.1 GPT-3 and few-shot learning</li>
          <li class="m-0">4.2 Phase transitions in capability</li>
        </ul>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">5</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">Thinking tokens: teaching models to slow down</div>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">6</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">RAG: connecting memory to the world</div>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">7</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">Debunking the stochastic parrot</div>
        <ul class="mt-1.5 space-y-0.5 text-xs text-base-content/65 list-none p-0 m-0">
          <li class="m-0">7.1 Where the parrot critique lands hardest</li>
          <li class="m-0">7.2 The parrot thesis, fairly stated</li>
          <li class="m-0">7.3 The evidence against pure parroting</li>
          <li class="m-0">7.4 Prediction, planning, and the bounded role of randomness</li>
          <li class="m-0">7.5 When the parrot does appear: hallucination reconsidered</li>
          <li class="m-0">7.6 The middle ground</li>
        </ul>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">8</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">Implications for educators and practitioners</div>
      </div>
    </li>
    <li class="flex gap-3 m-0">
      <span class="font-mono text-2xl font-light text-primary/70 leading-none mt-0.5 select-none">9</span>
      <div class="flex-1 min-w-0">
        <div class="font-semibold text-sm leading-snug">References</div>
      </div>
    </li>
  </ol>
</div>
## 1  The expectation: AI as a reasoning machine
 
### 1.1 Symbolic AI and the knowledge-engineering dream
 
When Alan Turing asked in 1950 whether machines could think [1], the implicit model was the logician: an entity that manipulates symbols, applies rules, and arrives at correct conclusions by necessity. This expectation shaped the dominant research programme in artificial intelligence for decades.
 
From the mid-1950s through the 1980s, **symbolic AI** — sometimes called Good Old-Fashioned AI, or GOFAI — held that intelligence was computation over explicit representations. Build a large database of facts and rules, then let an inference engine derive new facts. Systems like MYCIN (1976) for medical diagnosis demonstrated genuine expertise in narrow domains [2]. The approach had deep intuitive appeal: it resembled how we imagine ourselves to think.
 
> **Classroom example — what symbolic AI looked like** A symbolic system for diagnosing a fever might
> contain rules like: *IF temperature > 38.5°C AND onset < 3 days AND no recent vaccination THEN
> suspect infection.* A doctor's expertise was painstakingly extracted, written as rules, and stored
> in a database. The machine followed the rules; it did not learn them.
 
### 1.2 Why the dream collapsed: the brittleness problem
 
The limits appeared as soon as the domains grew larger. Knowledge engineering — the laborious process of extracting and encoding expert knowledge — did not scale. The world contains too many exceptions, too much context-dependence, and too much tacit knowledge that experts cannot articulate. Philosopher Michael Polanyi captured the core problem: "we know more than we can tell" [3].
 
Ask a symbolic system about a fever in a child who has just returned from the tropics, and it fails — unless someone thought to add that rule. The first AI winter (1974–1980), triggered by the Lighthill Report's critical assessment of the field [18] and the Mansfield Amendment in the United States, saw funding for general AI research collapse. A modest revival followed with the expert-systems boom of the 1980s, but the second AI winter arrived around 1987–1993, when commercial expert-systems companies failed to deliver on their promises and the LISP-machine market collapsed. The question after both winters was no longer whether machines could reason in the symbolic sense, but whether that was even the right target.
 
### 1.3 The parallel story: connectionism and the first neural-network winter
 
The symbolic-AI history is only half the story. Running in parallel — and at the time, in direct competition for funding and intellectual prestige — was the **connectionist** tradition: the idea that intelligence might emerge from networks of simple, neuron-like units rather than from explicit rules. Frank Rosenblatt's *perceptron*, introduced in 1958, was the first practical learning algorithm of this kind. The press greeted it with extraordinary headlines: the *New York Times* reported that the perceptron would soon "walk, talk, see, write, reproduce itself and be conscious of its existence."
 
The hype outran the technology. By the mid-1960s, neural-network research was already slowing — limited by the computers of the era, by the lack of any training algorithm for networks with more than one layer of weights, and by the rising influence of the symbolic school. In 1969, Marvin Minsky and Seymour Papert published *Perceptrons* [21], a mathematically rigorous critique of single-layer networks. Their central result was that single-layer perceptrons could not learn even simple non-linearly-separable functions — the XOR function being the canonical example. They conjectured, on the basis of intuition rather than proof, that multi-layer extensions would face similar limits.
 
<details class="collapse collapse-arrow bg-base-200 border border-base-300 my-6">
<summary class="collapse-title font-semibold">🔍 <span class="text-xs uppercase tracking-wider opacity-60">Deep dive</span> — The XOR problem: what blocked early neural networks</summary>
<div class="collapse-content">

Imagine plotting four points: (0,0), (0,1), (1,0), (1,1). The OR function returns 1 unless both inputs are 0 — you can draw a straight line separating the "0" point from the three "1" points. AND is similar. But XOR returns 1 only when exactly one input is 1 — and there is no straight line that separates the two "1" points from the two "0" points; they lie on opposite diagonals. A single-layer perceptron, which can only carve up the input space with straight lines, simply cannot represent this. Multi-layer networks with non-linear activation functions can — but no one had a workable algorithm to train them in 1969.

</div>
</details>
 
The book did not single-handedly kill neural-network research, but it crystallised a perception that connectionism was a dead end and helped redirect funding decisively toward symbolic AI. What followed was a *first neural-network winter* in mainstream visibility — but the work itself did not stop. It went underground.
 
<details class="collapse collapse-arrow bg-base-200 border border-base-300 my-6">
<summary class="collapse-title font-semibold">📚 <span class="text-xs uppercase tracking-wider opacity-60">Historical context</span> — The underground decade: Linnainmaa, Werbos, Grossberg, Fukushima</summary>
<div class="collapse-content">

Through the 1970s and early 1980s, a scattered community continued building. In 1970, the Finnish mathematician Seppo Linnainmaa published, in his master's thesis, the modern form of reverse-mode automatic differentiation — the mathematical machinery that backpropagation requires — though without reference to neural networks at all [23]. In 1974, Paul Werbos's Harvard PhD thesis applied this technique specifically to training multi-layer networks [24]. He could not publish it on neural-network grounds for years; symbolic AI was in vogue, and his work on the topic only reached print around 1982. Stephen Grossberg, working largely outside the AI mainstream, developed adaptive resonance theory from 1976 onward, addressing how networks could learn continuously without catastrophically forgetting prior knowledge — the "stability-plasticity dilemma" — culminating in the 1987 ART1 architecture with Gail Carpenter [25]. Teuvo Kohonen developed self-organizing maps in the early 1980s — networks that learn topology-preserving representations of high-dimensional data without supervision. In Japan, Kunihiko Fukushima's 1980 Neocognitron [26] introduced a multilayer convolutional architecture for visual pattern recognition; it is the direct ancestor of every convolutional neural network in use today.

</div>
</details>
 
The thaw arrived in two stages. In 1982, the theoretical physicist John Hopfield published a short paper showing that a simple recurrent network of binary neurons could serve as a content-addressable associative memory, with stored patterns acting as energy minima [27]. The Hopfield network was elegant, easy to analyse, and — crucially — published by someone with the scientific credibility to be heard outside the AI community. It revived interest in connectionist methods within physics and cognitive science. Then in 1986, Rumelhart, Hinton and Williams published the paper [22] that brought backpropagation into the mainstream of AI: a clear, accessible demonstration that multi-layer networks could learn complex tasks, including XOR. The Parallel Distributed Processing volumes that accompanied this work made the case to a broad audience that connectionism was alive, productive, and serious.
 
Minsky and Papert's intuition about the limits of multi-layer networks turned out to be wrong, but the field had already lost the better part of a generation. The connectionist programme would only fully recover with the deep-learning revolution of the 2010s, when GPU hardware finally made it possible to train the kinds of large networks that the underground work of the 1970s and 80s had foreshadowed.
 
<details class="collapse collapse-arrow bg-base-200 border border-base-300 my-6">
<summary class="collapse-title font-semibold">🔍 <span class="text-xs uppercase tracking-wider opacity-60">Deep dive</span> — The Universal Approximation Theorem: Minsky's conjecture, mathematically refuted</summary>
<div class="collapse-content">

The formal refutation of the Minsky–Papert conjecture arrived in 1989 as well, in the form of the **Universal Approximation Theorem**. Cybenko proved that any continuous function on a compact subset of ℝ^n can be approximated to arbitrary precision by a feedforward network with a single hidden layer of finite width, given a sigmoid activation function [28]. Hornik, Stinchcombe and White independently proved a more general version the same year [29]; Hornik (1991) extended the result further, showing that the multilayer architecture itself — not the specific choice of activation function — is what gives neural networks their universal approximation capacity [30].

The practical implication is decisive: the limits Minsky and Papert had proved for single-layer perceptrons did not generalise to multi-layer networks, and their conjecture that they would was mathematically false. A neural network of sufficient size can represent *any* continuous function. The theorem does not say such a network is easy to train, or that it will generalise well from limited data, or that it is the most efficient representation — those are separate problems, and remain hard. But the question of whether the architecture is fundamentally limited in what it can express was settled. The answer is no.

</div>
</details>
 
> **Why this matters for the rest of the story** The two AI traditions — symbolic and connectionist —
> had radically different bets about what mattered for intelligence. The symbolic school bet on
> explicit representations and inference; the connectionist school bet on learned distributed
> representations and emergent behaviour from large networks. The symbolic bet dominated from roughly
> 1970 to the mid-2000s. Everything that follows in this article — embeddings, attention,
> transformers, emergent capabilities at scale — is the connectionist bet finally paying off. The
> story of modern AI is not just the rise of deep learning; it is the slow, fifty-year vindication of
> a tradition that had been largely written off.
 
<div id="sem-timeline-widget" style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:20px 22px;margin:1.8em 0;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;overflow:hidden;">
  <div style="color:#f0883e;font-size:10px;letter-spacing:2.5px;margin-bottom:14px;">INTERACTIVE — TIMELINE: SYMBOLIC vs CONNECTIONIST AI</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px;font-size:11px;">
    <span style="color:#7ab0f0;">● connectionist</span>
    <span style="color:#f0883e;">● symbolic / GOFAI</span>
    <span style="color:#8b949e;">● AI winters</span>
    <span style="color:#5dba8f;">● modern (statistical → LLM)</span>
  </div>
  <svg id="tl-svg" viewBox="0 0 760 200" style="width:100%;display:block;"></svg>
  <div id="tl-detail" style="background:#161b22;border:1px solid #30363d;border-radius:4px;padding:12px 14px;margin-top:12px;min-height:56px;font-size:12px;color:#c9d1d9;line-height:1.55;font-family:inherit;">
    <span style="color:#6e7681;">Hover an event to see details. Click to pin.</span>
  </div>
</div>
<script>
(function(){
var TLD=[
  {y:1950,cat:'sym',label:'Turing test',detail:'Turing, "Computing Machinery and Intelligence" — sets the implicit logician model of intelligence [1].'},
  {y:1958,cat:'con',label:'Perceptron',detail:'Rosenblatt — the first practical connectionist learning algorithm; press promises a machine that will "walk, talk, see, write, reproduce itself".'},
  {y0:1956,y1:1969,cat:'sym',label:'Symbolic AI ascendant',detail:'McCarthy, Newell & Simon; LISP, the General Problem Solver — symbolic AI dominates the field.'},
  {y:1969,cat:'win',label:'Perceptrons critique',detail:'Minsky & Papert — single-layer perceptrons cannot learn XOR; mistakenly extrapolated to multi-layer networks [21].'},
  {y0:1970,y1:1974,cat:'con',label:'Backprop, underground',detail:'Linnainmaa formalises reverse-mode autodiff [23]; Werbos applies it to neural nets in his Harvard PhD [24].'},
  {y0:1974,y1:1980,cat:'win',label:'First AI winter',detail:'Lighthill Report; Mansfield Amendment — funding for general AI research collapses [18].'},
  {y:1976,cat:'con',label:'Adaptive resonance theory',detail:'Grossberg — stability-plasticity dilemma; foundations of ART [25].'},
  {y0:1976,y1:1986,cat:'sym',label:'Expert systems boom',detail:'MYCIN, DENDRAL — narrow-domain symbolic systems briefly commercialised [2].'},
  {y:1980,cat:'con',label:'Neocognitron',detail:'Fukushima — convolutional architecture for vision; direct ancestor of every modern CNN [26].'},
  {y:1982,cat:'con',label:'Hopfield network',detail:'A physicist publishes neural-network research in PNAS; connectionism regains scientific credibility [27].'},
  {y:1986,cat:'con',label:'Backprop, mainstream',detail:'Rumelhart, Hinton & Williams — multi-layer networks shown to learn XOR and complex tasks [22]; PDP volumes follow.'},
  {y0:1989,y1:1991,cat:'con',label:'Universal Approximation Theorem',detail:'Cybenko, Hornik et al. — Minsky-Papert intuition formally refuted [28][29][30].'},
  {y0:1987,y1:1993,cat:'win',label:'Second AI winter',detail:'Expert-systems market collapses; LISP-machine companies fail.'},
  {y0:1993,y1:2012,cat:'mod',label:'Statistical methods era',detail:'n-grams, SVMs, shallow neural nets dominate machine learning.'},
  {y0:2013,y1:2017,cat:'mod',label:'Word2Vec, LSTMs',detail:'Mikolov et al. — meaning as geometry; LSTMs handle sequences; deep learning takes language [4][19].'},
  {y:2017,cat:'mod',label:'Transformer',detail:'Vaswani et al., "Attention Is All You Need" — meaning becomes context-sensitive [5].'},
  {y:2020,cat:'mod',label:'GPT-3',detail:'175B parameters — few-shot prompting and emergent abilities appear at scale [7].'},
  {y0:2022,y1:2026,cat:'mod',label:'Chain-of-thought, RAG, reasoning',detail:'LLMs as infrastructure; test-time compute (thinking tokens) emerges as a new scaling axis.'}
];
var CAT={
  con:{color:'#7ab0f0',fill:'#1c3a5a',lane:40,name:'connectionist'},
  sym:{color:'#f0883e',fill:'#5a3215',lane:70,name:'symbolic'},
  win:{color:'#8b949e',fill:'#2a2e32',lane:100,name:'winter'},
  mod:{color:'#5dba8f',fill:'#163a2e',lane:130,name:'modern'}
};
var Y0=1950,Y1=2026,X0=80,X1=730;
function px(yr){return X0+(yr-Y0)*(X1-X0)/(Y1-Y0);}
var pinned=null;
function setDetail(d){
  var p=document.getElementById('tl-detail');if(!p)return;
  if(!d){p.innerHTML='<span style="color:#6e7681;">Hover an event to see details. Click to pin.</span>';return;}
  var year=d.y!=null?d.y:d.y0+'–'+d.y1;
  var c=CAT[d.cat].color;
  p.innerHTML='<span style="color:'+c+';font-weight:bold;font-size:13px;">'+year+'</span> <span style="color:#6e7681;">·</span> <span style="color:#c9d1d9;font-weight:bold;">'+d.label+'</span><br><span style="color:#8b949e;">'+d.detail+'</span>';
}
function render(){
  var svg=document.getElementById('tl-svg'),ns='http://www.w3.org/2000/svg';
  if(!svg)return;svg.innerHTML='';
  function el(tag,attrs){var e=document.createElementNS(ns,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);return e;}
  Object.keys(CAT).forEach(function(k){
    var c=CAT[k];
    var lbl=el('text',{x:X0-6,y:c.lane+3,'text-anchor':'end',fill:c.color,'font-size':'8','font-family':'monospace',opacity:'0.6'});
    lbl.textContent=c.name;svg.appendChild(lbl);
    svg.appendChild(el('line',{x1:X0,y1:c.lane,x2:X1,y2:c.lane,stroke:c.color,'stroke-width':'0.5',opacity:'0.12','stroke-dasharray':'2,3'}));
  });
  svg.appendChild(el('line',{x1:X0,y1:165,x2:X1,y2:165,stroke:'#30363d','stroke-width':'1'}));
  for(var yr=1950;yr<=2020;yr+=10){
    var x=px(yr);
    svg.appendChild(el('line',{x1:x,y1:162,x2:x,y2:168,stroke:'#484f58','stroke-width':'1'}));
    var t=el('text',{x:x,y:183,'text-anchor':'middle',fill:'#8b949e','font-size':'10','font-family':'monospace'});
    t.textContent=yr;svg.appendChild(t);
  }
  TLD.forEach(function(d){
    var c=CAT[d.cat];
    var g=el('g',{style:'cursor:pointer;'});
    if(d.y0!=null){
      var xa=px(d.y0),xb=px(d.y1);
      g.appendChild(el('rect',{x:xa,y:c.lane-7,width:xb-xa,height:14,rx:'3',ry:'3',fill:c.fill,stroke:c.color,'stroke-width':'1.2'}));
    } else {
      var x=px(d.y);
      g.appendChild(el('circle',{cx:x,cy:c.lane,r:'5.5',fill:c.color,stroke:'#0d1117','stroke-width':'1.8'}));
    }
    g.addEventListener('mouseenter',function(){setDetail(d);});
    g.addEventListener('mouseleave',function(){setDetail(pinned);});
    g.addEventListener('click',function(e){e.stopPropagation();pinned=(pinned===d?null:d);setDetail(pinned||d);});
    svg.appendChild(g);
  });
  svg.addEventListener('click',function(){if(pinned){pinned=null;setDetail(null);}});
}
function init(){render();}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{setTimeout(init,0);}
})();
</script>
 
## 2  The statistical turn: meaning as location
 
### 2.1 N-grams: language as statistics
 
The statistical alternative asked a fundamentally different question: instead of encoding what language *means*, can we learn what language *does* by observing enormous amounts of it? A language model assigns a probability to every possible sequence of words. The simplest version, the **n-gram model**, estimates the probability of each word given the preceding few words, and requires nothing beyond counting.
 
This worked surprisingly well for tasks like speech recognition — in turn, the cliff between nlp-experts with a math/physics/ML background and pure linguists increases, culminating in an attribution to Frederick Jelinek of the sentence, "Every time I fire a linguist, the performance of the speech recogniser improves." But n-gram models had hard limits: they could not capture dependencies across more than a few words, and they treated every word as an atom with no internal relationship to any other word, as would any linguist point out. Furthermore, they were used in conjunction with the bag of words model: each word is independent of each other, usually being modelled by a dimension in a high-dimensional space, thus words like father and son are orthogonal and do not have any relation.
 
### 2.2 Word embeddings: words as coordinates
 
The breakthrough that set the stage for modern AI came from a deceptively simple idea: what if each word were not an atom, but a *point in space*? Bengio et al. (2003) showed that a neural network could learn a continuous vector representation — an embedding — for each word, placed so that words used in similar contexts end up nearby [4]. Meaning became geometry.
 
Mikolov et al. (2013) [19] made this tractable at scale and produced the result that made linguists stop and pay attention: `king − man + woman ≈ queen`. Arithmetic on meaning. The model had never been told what "royalty" or "gender" were — those relationships emerged from the geometry of the embedding space, learned entirely from patterns in text.
 
> **Intuition — the library analogy** Imagine a vast library where books are not sorted
> alphabetically, but by *meaning*. Books about oceans are shelved near books about rivers, which are
> near books about rain, which are near books about weather. "Warm" and "hot" sit nearby; "warm" and
> "hammer" sit far apart. A word embedding is exactly this: a location in a meaning-space learned from
> the pattern of which words tend to appear in the same neighbourhood as which others.
 
This was a conceptual departure from symbolic AI. Meaning was no longer stored in a database of propositions — it was implicit in the geometry of a high-dimensional space, learned from data. A word's location in that space encoded something about what it meant, without anyone ever defining it.

<div id="sem-ops-widget" style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:20px 22px;margin:1.8em 0;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;overflow:hidden;">
  <div style="color:#f0883e;font-size:10px;letter-spacing:2.5px;margin-bottom:14px;">INTERACTIVE — SEMANTIC OPERATIONS IN EMBEDDING SPACE</div>
  <div style="display:flex;gap:0;margin-bottom:14px;border-bottom:1px solid #30363d;">
    <button id="so-tab-analogy" onclick="soTab('analogy')" style="background:transparent;border:none;border-bottom:2px solid #58a6ff;color:#58a6ff;padding:6px 16px;font-family:inherit;font-size:12px;cursor:pointer;margin-bottom:-1px;">Word Analogy</button>
    <button id="so-tab-capitals" onclick="soTab('capitals')" style="background:transparent;border:none;border-bottom:2px solid transparent;color:#8b949e;padding:6px 16px;font-family:inherit;font-size:12px;cursor:pointer;margin-bottom:-1px;">Capitals</button>
    <button id="so-tab-size" onclick="soTab('size')" style="background:transparent;border:none;border-bottom:2px solid transparent;color:#8b949e;padding:6px 16px;font-family:inherit;font-size:12px;cursor:pointer;margin-bottom:-1px;">Comparative</button>
  </div>
  <div id="so-analogy-panel">
    <div style="color:#8b949e;font-size:11px;margin-bottom:10px;">Each pair shares the same offset vector. The model learned these geometric relationships from text alone.</div>
    <svg id="so-ana-svg" viewBox="0 0 760 320" style="width:100%;display:block;max-height:340px;"></svg>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px;">
      <span style="color:#c9d1d9;font-size:13px;">Complete the analogy:</span>
      <select id="so-ana-sel" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:5px 8px;font-family:inherit;font-size:12px;cursor:pointer;" onchange="soAnaRender()">
        <option value="0">king − man + woman = ?</option>
        <option value="1">uncle − man + woman = ?</option>
        <option value="2">actor − man + woman = ?</option>
        <option value="3">son − man + woman = ?</option>
      </select>
      <span id="so-ana-ans" style="color:#39d353;font-size:13px;font-weight:bold;min-width:60px;"></span>
    </div>
  </div>
  <div id="so-capitals-panel" style="display:none;">
    <div style="color:#8b949e;font-size:11px;margin-bottom:10px;">All country→capital pairs share the same direction and magnitude in embedding space — a learnt geometric rule.</div>
    <svg id="so-cap-svg" viewBox="0 0 760 290" style="width:100%;display:block;max-height:290px;"></svg>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px;">
      <span style="color:#c9d1d9;font-size:13px;">Capital of:</span>
      <select id="so-cap-sel" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:5px 8px;font-family:inherit;font-size:12px;cursor:pointer;" onchange="soCapRender()">
        <option value="0">France</option>
        <option value="1">Germany</option>
        <option value="2">Italy</option>
        <option value="3">Spain</option>
        <option value="4">Japan</option>
      </select>
      <span style="color:#8b949e;font-size:13px;">→</span>
      <span id="so-cap-ans" style="color:#39d353;font-size:13px;font-weight:bold;min-width:80px;"></span>
    </div>
  </div>
  <div id="so-size-panel" style="display:none;">
    <div style="color:#8b949e;font-size:11px;margin-bottom:10px;">Physical size is encoded geometrically. The model can answer comparative questions by measuring distances along the size axis.</div>
    <svg id="so-size-svg" viewBox="0 0 760 160" style="width:100%;display:block;max-height:160px;"></svg>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px;">
      <span style="color:#c9d1d9;font-size:13px;">A</span>
      <select id="so-sz-a" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:5px 8px;font-family:inherit;font-size:12px;cursor:pointer;" onchange="soSizeRender()">
        <option value="0">mouse</option><option value="1">rat</option><option value="2">cat</option>
        <option value="3">dog</option><option value="4">horse</option><option value="5">elephant</option>
      </select>
      <span style="color:#c9d1d9;font-size:13px;">is</span>
      <span id="so-sz-ans" style="color:#f0883e;font-size:13px;font-weight:bold;min-width:80px;">?</span>
      <span style="color:#c9d1d9;font-size:13px;">than a</span>
      <select id="so-sz-b" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:5px 8px;font-family:inherit;font-size:12px;cursor:pointer;" onchange="soSizeRender()">
        <option value="0">mouse</option><option value="1">rat</option><option value="2">cat</option>
        <option value="3">dog</option><option value="4" selected>horse</option><option value="5">elephant</option>
      </select>
    </div>
  </div>
  <div style="color:#6e7681;font-size:11px;line-height:1.6;margin-top:14px;">These operations emerge from the geometry of the embedding space — the model was never explicitly taught them.</div>
</div>
<script>
(function(){
var ns='http://www.w3.org/2000/svg';
function el(tag,attrs){var e=document.createElementNS(ns,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);return e;}
function arrow(svg,x1,y1,x2,y2,color,label){
  var id='arr-'+Math.random().toString(36).slice(2);
  var mk=el('marker',{id:id,markerWidth:'8',markerHeight:'8',refX:'6',refY:'3',orient:'auto'});
  var path=el('path',{d:'M0,0 L0,6 L8,3 z',fill:color});mk.appendChild(path);
  var defs=svg.querySelector('defs')||svg.insertBefore(el('defs',{}),svg.firstChild);
  defs.appendChild(mk);
  var line=el('line',{x1:x1,y1:y1,x2:x2,y2:y2,stroke:color,'stroke-width':'1.6','marker-end':'url(#'+id+')'});
  svg.appendChild(line);
  if(label){
    var mx=(x1+x2)/2,my=(y1+y2)/2;
    var t=el('text',{x:mx,y:my-6,'text-anchor':'middle',fill:color,'font-size':'9','font-family':'monospace',opacity:'0.85'});
    t.textContent=label;svg.appendChild(t);
  }
}
function dot(svg,x,y,r,fill,stroke,label,lside){
  svg.appendChild(el('circle',{cx:x,cy:y,r:r,fill:fill,stroke:stroke,'stroke-width':'1.8'}));
  if(label){
    var t=el('text',{x:lside==='left'?x-r-5:x+r+5,y:y+4,
      'text-anchor':lside==='left'?'end':'start',fill:'#c9d1d9','font-size':'11','font-family':'monospace'});
    t.textContent=label;svg.appendChild(t);
  }
}
/* ── ANALOGY ── */
var ANA=[
  {src:'king',tgt:'queen',ans:'queen'},
  {src:'uncle',tgt:'aunt',ans:'aunt'},
  {src:'actor',tgt:'actress',ans:'actress'},
  {src:'son',tgt:'daughter',ans:'daughter'}
];
window.soAnaRender=function(){
  var svg=document.getElementById('so-ana-svg');if(!svg)return;svg.innerHTML='';
  var idx=parseInt(document.getElementById('so-ana-sel').value)||0;
  var a=ANA[idx];
  // fixed coords: woman/tgt on top, man/src on bottom — gender axis vertical, royalty axis horizontal
  var mx=160,my=240,wx=160,wy=80,sx=560,sy=240,tx=560,ty=80;
  // axis labels
  var axY=el('text',{x:22,y:160,'text-anchor':'middle',fill:'#484f58','font-size':'9','font-family':'monospace',transform:'rotate(-90,22,160)'});
  axY.textContent='gender (male → female)';svg.appendChild(axY);
  var axX=el('text',{x:380,y:305,'text-anchor':'middle',fill:'#484f58','font-size':'9','font-family':'monospace'});
  axX.textContent='royalty (commoner → royal)';svg.appendChild(axX);
  // gender arrows (vertical, going UP from male to female)
  arrow(svg,mx,my-14,wx,wy+14,'#bc8cff',null);
  arrow(svg,sx,sy-14,tx,ty+14,'#bc8cff',null);
  // royalty arrows (horizontal, going RIGHT from commoner to royal); label on top arrow for clarity
  arrow(svg,wx+14,wy,tx-14,ty,'#58a6ff','royalty');
  arrow(svg,mx+14,my,sx-14,sy,'#58a6ff',null);
  // dots
  dot(svg,mx,my,9,'#1f6feb','#58a6ff','man','left');
  dot(svg,wx,wy,9,'#6e2fb0','#bc8cff','woman','left');
  dot(svg,sx,sy,9,'#1f6feb','#58a6ff',a.src,'right');
  // answer dot with pulsing ring
  dot(svg,tx,ty,9,'#0d5c2e','#39d353',a.tgt,'right');
  var pulse=el('circle',{cx:tx,cy:ty,r:'14',fill:'none',stroke:'#39d353','stroke-width':'1',opacity:'0.5'});
  svg.appendChild(pulse);
  // ? label shown before answer
  document.getElementById('so-ana-ans').textContent='→ '+a.ans;
};
/* ── CAPITALS ── */
var CAPS=[
  {country:'France',capital:'Paris',   cx:110,cy:110,px:370,py:108},
  {country:'Germany',capital:'Berlin', cx:90, cy:185,px:350,py:182},
  {country:'Italy',capital:'Rome',     cx:165,cy:230,px:425,py:228},
  {country:'Spain',capital:'Madrid',   cx:145,cy:152,px:405,py:150},
  {country:'Japan',capital:'Tokyo',    cx:200,cy:75, px:460,py:73}
];
window.soCapRender=function(){
  var svg=document.getElementById('so-cap-svg');if(!svg)return;svg.innerHTML='';
  var sel=parseInt(document.getElementById('so-cap-sel').value)||0;
  // draw all pairs dimmed
  CAPS.forEach(function(c,i){
    var active=i===sel;
    var color=active?'#f0883e':'#30363d';
    var tcol=active?'#c9d1d9':'#484f58';
    arrow(svg,c.cx+14,c.cy,c.px-14,c.py,color,'');
    dot(svg,c.cx,c.cy,active?9:6,active?'#2a1800':'#161b22',color,c.country,'left');
    dot(svg,c.px,c.py,active?9:6,active?'#0d2200':'#161b22',active?'#39d353':color,c.capital,'right');
  });
  document.getElementById('so-cap-ans').textContent=CAPS[sel].capital;
};
/* ── SIZE ── */
var ANIMALS=['mouse','rat','cat','dog','horse','elephant'];
var APOS=[80,155,237,325,455,640];
window.soSizeRender=function(){
  var svg=document.getElementById('so-size-svg');if(!svg)return;svg.innerHTML='';
  var ai=parseInt(document.getElementById('so-sz-a').value)||0;
  var bi=parseInt(document.getElementById('so-sz-b').value)||4;
  // axis line
  svg.appendChild(el('line',{x1:60,y1:80,x2:700,y2:80,stroke:'#30363d','stroke-width':'1.5'}));
  var axLbl=el('text',{x:380,y:145,'text-anchor':'middle',fill:'#484f58','font-size':'9','font-family':'monospace'});
  axLbl.textContent='← smaller                physical size axis                larger →';svg.appendChild(axLbl);
  ANIMALS.forEach(function(a,i){
    var active=i===ai||i===bi;
    var col=i===ai?'#58a6ff':i===bi?'#f0883e':'#30363d';
    svg.appendChild(el('circle',{cx:APOS[i],cy:80,r:active?9:5,fill:active?col:'#161b22',stroke:col,'stroke-width':'1.5'}));
    var t=el('text',{x:APOS[i],y:active?62:65,'text-anchor':'middle',fill:col,
      'font-size':active?'11':'9','font-family':'monospace','font-weight':active?'bold':'normal'});
    t.textContent=a;svg.appendChild(t);
  });
  // arrow between selected pair
  if(ai!==bi){
    var x1=APOS[Math.min(ai,bi)],x2=APOS[Math.max(ai,bi)];
    arrow(svg,x1+12,80,x2-12,80,ai<bi?'#58a6ff':'#f0883e',null);
  }
  var ans=ai<bi?'smaller':(ai>bi?'bigger':'the same size as');
  document.getElementById('so-sz-ans').textContent=ans;
};
/* ── TABS ── */
window.soTab=function(t){
  ['analogy','capitals','size'].forEach(function(x){
    var panel=document.getElementById('so-'+x+'-panel');
    var tab=document.getElementById('so-tab-'+x);
    var active=x===t;
    if(panel)panel.style.display=active?'block':'none';
    if(tab){tab.style.borderBottomColor=active?'#58a6ff':'transparent';tab.style.color=active?'#58a6ff':'#8b949e';}
  });
  if(t==='analogy')soAnaRender();
  else if(t==='capitals')soCapRender();
  else if(t==='size')soSizeRender();
};
function init(){
  soAnaRender();
  document.getElementById('so-sz-b').value=5;
  soSizeRender();
}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{setTimeout(init,0);}
})();
</script>

But static embeddings had one critical flaw: every word had exactly one location. "Bank" — financial institution? — always mapped to the same point in space, regardless of whether the surrounding sentence talked about loans or rivers. Meaning, in the real world, depends on context. Fixing this required something more dynamic.
 
## 3  Attention: meaning that moves
 
### 3.1 The bottleneck problem
 
By 2014, the leading architecture for machine translation used an encoder-decoder design with recurrent neural networks (LSTMs). The encoder read a sentence word by word, compressing it into a single fixed-size vector — like summarising a paragraph into one sentence. The decoder then generated the translation from that summary. The flaw was obvious to anyone who thought about it: forcing everything in a long, complex sentence into a single vector discards information. Long-range dependencies — the meaning at the end of a sentence that depends on context from its beginning — degraded over many sequential processing steps.
 
Bahdanau, Cho and Bengio (2015) proposed the key fix: instead of using one compressed summary, allow the decoder at each output step to look back at the *entire* input and decide which parts are relevant now [6]. This was soft attention — a learned, dynamic spotlight on the input.
 
### 3.2 Attention as semantic navigation
 
Here is the central reframing: **attention is not just a mechanism — it is what allows meaning to be context-sensitive**. In a static word embedding, every word has one fixed point in the semantic space. Attention allows that point to *move* depending on context.
 
Consider the word "bank" again. In a static embedding it sits at one location — ambiguously between financial-institution-space and river-space. With attention, the word "bank" in the sentence "The river bank was flooded" can shift its representation toward the river-related region of semantic space, because the model has learned to let "bank" be strongly influenced by nearby words like "river" and "flooded." In the sentence "The financial bank was insolvent," the same word shifts toward the financial-institution region, pulled by "financial" and "insolvent."
 
Each layer of a transformer performs this contextual repositioning. A word does not arrive at its final representation in one step — it is iteratively refined over many layers, each one updating every word's position in semantic space based on the full context of the sentence. By the final layer, "bank" in the river sentence and "bank" in the financial sentence have distinct representations, even though they started from the same point.
 
> **The key insight about attention** Think of each word as having a fuzzy, provisional meaning.
> Attention is the process by which that meaning is sharpened and contextualised — pulled toward
> nearby words that constrain its interpretation. The final representation of a word is not what the
> word means in the dictionary; it is what this particular word means in this particular sentence,
> given everything else that surrounds it.
 
This is why attention was such a fundamental step beyond static embeddings. Embeddings gave words locations. Attention gave those locations the ability to respond to their neighbourhood.

<div id="sem-da-widget" style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:20px 22px;margin:1.8em 0;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;overflow:hidden;">
  <div style="color:#f0883e;font-size:10px;letter-spacing:2.5px;margin-bottom:14px;">INTERACTIVE — ATTENTION AS SEMANTIC SHIFTING</div>
  <svg id="da-svg" viewBox="0 0 760 430" style="width:100%;display:block;background:transparent;"></svg>
  <div id="da-echo" style="color:#8b949e;font-size:12px;font-style:italic;min-height:18px;margin:4px 0 12px;padding-left:2px;"></div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
    <span style="color:#c9d1d9;font-size:13px;">Choose a word:</span>
    <button id="da-btn-bank" onclick="daSetWord('bank')" style="background:#1f6feb;color:#fff;border:1px solid #1f6feb;border-radius:4px;padding:4px 13px;font-family:inherit;font-size:12px;cursor:pointer;transition:all .15s;">bank</button>
    <button id="da-btn-bright" onclick="daSetWord('bright')" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:4px 13px;font-family:inherit;font-size:12px;cursor:pointer;transition:all .15s;">bright</button>
    <button id="da-btn-head" onclick="daSetWord('head')" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:4px 13px;font-family:inherit;font-size:12px;cursor:pointer;transition:all .15s;">head</button>
  </div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:14px;">
    <span style="color:#c9d1d9;font-size:13px;">Context:</span>
    <select id="da-ctx-sel" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:5px 8px;font-family:inherit;font-size:12px;min-width:290px;cursor:pointer;"></select>
  </div>
  <div style="color:#6e7681;font-size:11px;line-height:1.6;">Dots represent words from the model's semantic space. <span style="color:#8b949e;">The highlighted word's position shifts toward the cluster matching its contextual meaning — this is what attention does.</span></div>
</div>
<script>
(function(){
var DA={
  bank:{
    clusters:[
      {id:'finance',name:'finance',color:'#7ea8c4',fill:'rgba(58,100,155,0.14)',cx:185,cy:188,r:92,
       words:['equity','ledger','asset','bond','credit','deposit','loan','interest']},
      {id:'water',name:'water',color:'#5dba8f',fill:'rgba(50,170,130,0.14)',cx:550,cy:200,r:92,
       words:['canal','tide','current','mud','river','flood','stream','shore']},
      {id:'memory',name:'memory',color:'#c4a84a',fill:'rgba(170,150,50,0.14)',cx:358,cy:325,r:78,
       words:['stack','cache','register','buffer','memory','data','storage']}
    ],
    ctxs:[
      {t:'the river bank was flooded',cl:'water'},
      {t:'deposit money at the bank',cl:'finance'},
      {t:'memory bank in the CPU',cl:'memory'},
      {t:'the bank holds all my savings',cl:'finance'}
    ]
  },
  bright:{
    clusters:[
      {id:'light',name:'light',color:'#d4b545',fill:'rgba(200,175,50,0.14)',cx:185,cy:188,r:92,
       words:['luminous','radiant','glowing','shining','sunny','gleaming','dazzling','lit']},
      {id:'smart',name:'intelligence',color:'#7ab0f0',fill:'rgba(60,130,220,0.14)',cx:550,cy:200,r:92,
       words:['clever','smart','sharp','gifted','quick','astute','able','talented']},
      {id:'color',name:'color',color:'#d07090',fill:'rgba(200,80,120,0.14)',cx:358,cy:325,r:78,
       words:['vivid','vibrant','intense','bold','colorful','rich','saturated']}
    ],
    ctxs:[
      {t:'the bright sunlight hurt his eyes',cl:'light'},
      {t:'she is the brightest student in class',cl:'smart'},
      {t:'bright colors filled the canvas',cl:'color'},
      {t:'a bright flash lit up the sky',cl:'light'}
    ]
  },
  head:{
    clusters:[
      {id:'anatomy',name:'anatomy',color:'#b87ad0',fill:'rgba(160,80,200,0.14)',cx:185,cy:188,r:92,
       words:['skull','neck','face','crown','scalp','temple','chin','forehead']},
      {id:'leadership',name:'leadership',color:'#e09050',fill:'rgba(200,130,50,0.14)',cx:550,cy:200,r:92,
       words:['boss','chief','director','lead','manager','president','officer','exec']},
      {id:'position',name:'position',color:'#50c090',fill:'rgba(50,175,130,0.14)',cx:358,cy:325,r:78,
       words:['front','top','forefront','apex','tip','start','first','lead']}
    ],
    ctxs:[
      {t:'she hit her head on the doorframe',cl:'anatomy'},
      {t:'he is the head of the department',cl:'leadership'},
      {t:'at the head of the queue',cl:'position'},
      {t:'head of state addressed the nation',cl:'leadership'}
    ]
  }
};
var daW='bank',dX=363,dY=236,tX=363,tY=236;
function layout(cx,cy,r,ws){
  var p=[],n=ws.length,inn=Math.min(4,n),out=n-inn;
  for(var i=0;i<inn;i++){var a=(i/inn)*2*Math.PI+0.6;p.push({x:cx+r*0.42*Math.cos(a),y:cy+r*0.42*Math.sin(a)+2,w:ws[i]});}
  for(var i=0;i<out;i++){var a=(i/out)*2*Math.PI-0.25;p.push({x:cx+r*0.73*Math.cos(a),y:cy+r*0.73*Math.sin(a),w:ws[inn+i]});}
  return p;
}
function render(){
  var d=DA[daW],svg=document.getElementById('da-svg'),ns='http://www.w3.org/2000/svg';
  if(!svg)return;svg.innerHTML='';
  d.clusters.forEach(function(cl){
    var g=document.createElementNS(ns,'circle');
    g.setAttribute('cx',cl.cx);g.setAttribute('cy',cl.cy);g.setAttribute('r',cl.r+16);
    g.setAttribute('fill',cl.fill.replace('0.14','0.04'));g.setAttribute('stroke','none');svg.appendChild(g);
    var c=document.createElementNS(ns,'circle');
    c.setAttribute('cx',cl.cx);c.setAttribute('cy',cl.cy);c.setAttribute('r',cl.r);
    c.setAttribute('fill',cl.fill);c.setAttribute('stroke',cl.color);c.setAttribute('stroke-width','1.2');svg.appendChild(c);
    var nm=document.createElementNS(ns,'text');
    nm.setAttribute('x',cl.cx);nm.setAttribute('y',cl.cy-cl.r-10);nm.setAttribute('text-anchor','middle');
    nm.setAttribute('fill',cl.color);nm.setAttribute('font-size','11.5');nm.setAttribute('font-family','monospace');
    nm.setAttribute('letter-spacing','1');nm.textContent=cl.name;svg.appendChild(nm);
    layout(cl.cx,cl.cy,cl.r,cl.words).forEach(function(pt){
      var t=document.createElementNS(ns,'text');
      t.setAttribute('x',pt.x);t.setAttribute('y',pt.y);t.setAttribute('text-anchor','middle');
      t.setAttribute('fill',cl.color);t.setAttribute('font-size','9.5');t.setAttribute('font-family','monospace');
      t.setAttribute('opacity','0.72');t.textContent=pt.w;svg.appendChild(t);
    });
  });
  var dot=document.createElementNS(ns,'circle');
  dot.setAttribute('id','da-dot');dot.setAttribute('cx',dX);dot.setAttribute('cy',dY);
  dot.setAttribute('r','11');dot.setAttribute('fill','#152744');
  dot.setAttribute('stroke','#4090e0');dot.setAttribute('stroke-width','2');svg.appendChild(dot);
  var lbl=document.createElementNS(ns,'text');
  lbl.setAttribute('id','da-lbl');lbl.setAttribute('x',dX);lbl.setAttribute('y',dY+26);
  lbl.setAttribute('text-anchor','middle');lbl.setAttribute('fill','#58a6ff');
  lbl.setAttribute('font-size','12');lbl.setAttribute('font-family','monospace');
  lbl.setAttribute('font-weight','bold');lbl.textContent=daW;svg.appendChild(lbl);
}
function tick(){
  dX+=(tX-dX)*0.09;dY+=(tY-dY)*0.09;
  var dot=document.getElementById('da-dot'),lbl=document.getElementById('da-lbl');
  if(dot){dot.setAttribute('cx',dX);dot.setAttribute('cy',dY);}
  if(lbl){lbl.setAttribute('x',dX);lbl.setAttribute('y',dY+26);}
  requestAnimationFrame(tick);
}
function updCtx(){
  var d=DA[daW],idx=parseInt(document.getElementById('da-ctx-sel').value)||0;
  var ctx=d.ctxs[idx],cl=d.clusters.find(function(c){return c.id===ctx.cl;});
  var echo=document.getElementById('da-echo');
  if(echo)echo.textContent='"'+ctx.t+'"';
  if(cl){tX=cl.cx;tY=cl.cy;}
}
window.daSetWord=function(w){
  daW=w;dX=363;dY=236;tX=363;tY=236;
  ['bank','bright','head'].forEach(function(x){
    var b=document.getElementById('da-btn-'+x);if(!b)return;
    b.style.background=x===w?'#1f6feb':'#161b22';
    b.style.borderColor=x===w?'#1f6feb':'#30363d';
    b.style.color=x===w?'#fff':'#c9d1d9';
  });
  var sel=document.getElementById('da-ctx-sel'),d=DA[w];
  if(sel)sel.innerHTML=d.ctxs.map(function(c,i){return '<option value="'+i+'">'+c.t+'</option>';}).join('');
  render();setTimeout(updCtx,60);
};
function init(){
  window.daSetWord('bank');tick();
  var sel=document.getElementById('da-ctx-sel');
  if(sel)sel.addEventListener('change',updCtx);
}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{setTimeout(init,0);}
})();
</script>

### 3.3 What attention actually does in a transformer
 
Vaswani et al. (2017) — the "Attention Is All You Need" paper [5] — removed recurrence entirely. Every layer is an attention operation. The architecture processes all words simultaneously, and every word attends to every other word in the same step. This has two consequences.
 
First, long-range dependencies are trivial: a word at position 1 and a word at position 100 are just as directly connected as adjacent words. Distance no longer degrades the signal. Second, because there is no sequential processing, the entire computation is parallelisable on modern GPU hardware. This single property — parallelisability — unlocked training at scales previously unthinkable, and scale turned out to change everything.
 
The transformer uses **multi-head attention**: running many parallel attention operations simultaneously, each potentially learning to track a different kind of relationship. One head might learn to track grammatical agreement; another might track co-reference (who "she" refers to); another might track semantic similarity. The final representation is a combination of all of them. Meaning is not one thing — it is an aggregation of many simultaneous relationships.
 
> **Worked example — "The trophy didn't fit in the suitcase because it was too big"** What does "it"
> refer to — the trophy or the suitcase? Humans resolve this instantly: "it" refers to the trophy,
> because the trophy is too big to fit. A static word embedding cannot answer this — "it" has one
> representation regardless of context. A transformer with attention can learn to resolve this because
> it sees the entire sentence simultaneously. The model has learned, from countless examples, that
> when "fit...in" describes a containment relationship, the thing that is "too big" is the one that
> failed to fit, not the container. The representation of "it" shifts accordingly. Winograd schemas —
> sentences designed specifically to test this kind of reasoning — were considered very hard for AI;
> transformers handle most of them with high accuracy.
 
<details class="collapse collapse-arrow bg-base-200 border border-base-300 my-6">
<summary class="collapse-title font-semibold">🎥 <span class="text-xs uppercase tracking-wider opacity-60">Watch</span> — 3Blue1Brown visualises attention beautifully</summary>
<div class="collapse-content">

If you want to see this geometry move, Grant Sanderson (3Blue1Brown) animates every step of the transformer in his Neural Networks series. The chapter ["But what is a GPT? Visual intro to transformers"](https://www.youtube.com/watch?v=eMlx5fFNoYc) builds attention from first principles — query, key, and value as literal vectors moving through space — and pairs unusually well with the semantic-navigation framing above.

</div>
</details>

## 4  Emergence: when scale becomes something else
 
### 4.1 GPT-3 and few-shot learning
 
In May 2020, OpenAI published "Language Models are Few-Shot Learners" [7], introducing GPT-3: a transformer with 175 billion parameters — ten times larger than anything previously published. The central finding was not that a bigger model performed better on existing benchmarks. It was that GPT-3 demonstrated a qualitatively new *mode* of interaction: **few-shot prompting**.
 
Instead of fine-tuning the model with labelled examples, a user could simply write a few demonstrations in plain text — "Translate to French: Hello → Bonjour, Goodbye → Au revoir, Thank you →" — and the model would complete the pattern correctly, for essentially any task, without any gradient updates. This emerged naturally from pretraining on enough text; no one programmed it.
 
> **Example — zero-shot vs few-shot vs fine-tuning** *Fine-tuning* (old approach): show the model
> 10,000 labelled examples of positive and negative movie reviews, update its weights, deploy a
> sentiment classifier. Requires data, compute, and a separate model per task. *Few-shot prompting*
> (GPT-3): write in the input — "This film was brilliant: Positive. This film was dreadful: Negative.
> This film was a revelation:" — and the model responds "Positive." No training, no weight updates, no
> labelled data. The model infers the task from the examples in the prompt.
 
### 4.2 Phase transitions in capability
 
Wei et al. (2022) documented this systematically under the concept of **emergent abilities** [8]: capabilities that are absent in smaller models and appear sharply as model scale increases past a threshold. The paper documented over 100 such abilities: multi-step arithmetic, chain-of-thought reasoning, analogical reasoning — tasks that earlier models could not perform at any level, regardless of fine-tuning.
 
The discontinuity is philosophically significant. A model that cannot solve three-digit addition at any scale below 50 billion parameters, then suddenly can at 100 billion, is not just improving quantitatively — it is exhibiting a phase transition. Like water becoming ice, qualitatively new structure emerges from a quantitative change.
 
Seen through the semantic-space lens: at sufficient scale, the model's high-dimensional space becomes rich enough to contain implicit representations of concepts like "arithmetic" or "logical inference" — not as explicit rules, but as geometric regularities that the model can exploit when prompted in the right way. This is not the same as a symbolic reasoning engine, but it is not mere pattern-matching either.
 
<details class="collapse collapse-arrow bg-base-200 border border-base-300 my-6">
<summary class="collapse-title font-semibold">🔍 <span class="text-xs uppercase tracking-wider opacity-60">Deep dive</span> — Is emergence real, or a metric artefact? The Schaeffer critique</summary>
<div class="collapse-content">

Schaeffer et al. (2023) argued that apparent discontinuities can be artefacts of the metrics used — with nonlinear metrics, smooth underlying capability curves look discontinuous. This debate is ongoing and pedagogically important: emergence suggests qualitative novelty; smooth extrapolation suggests continuity with earlier systems. The truth is likely mixed: some capabilities are genuinely threshold-dependent; others are metric artefacts.

</div>
</details>
 
## 5  Thinking tokens: teaching models to slow down
 
Early language models answered questions in a single forward pass: input goes in, output comes out, one word at a time with no explicit intermediate reasoning. This is fast, fluent, and often correct — but it fails on tasks that require careful, multi-step thinking. The model essentially has to "know" the answer before it starts writing.
 
Wei et al. (2022) showed that prompting a model to produce intermediate reasoning steps dramatically improved performance on multi-step problems [9]: instead of asking "What is 23 × 47?", you write "Let's think step by step." The model generates: "23 × 40 = 920. 23 × 7 = 161. 920 + 161 = 1081." Each step becomes part of the model's context, and the model's subsequent predictions are conditioned on the full chain of reasoning — including its own intermediate conclusions.
 
> **Why "let's think step by step" actually works** In a transformer, the only "working memory" is the
> context window — the text that has been produced so far. When the model writes intermediate
> reasoning steps, those steps literally become part of the input for subsequent predictions. Writing
> "920 + 161" puts those numbers in the context; the next prediction is conditioned on them. The model
> is not reasoning in some hidden internal space — it is externalising its computation into text, and
> then reading that text as input for the next step.
 
OpenAI's o1 models (2024) operationalised this at training time using reinforcement learning: the model was trained to generate extended reasoning traces before answering, learning over time which reasoning strategies lead to correct outcomes [10]. These "thinking tokens" can run to thousands of words of private scratchpad computation. The key insight is a separation between *train-time scale* (model size) and *test-time compute* (how much thinking to do per answer) — a qualitatively new scaling axis.
 
From the semantic-space perspective: thinking tokens are intermediate waypoints. The model navigates through semantic space in multiple steps — each intermediate conclusion landing at a location that constrains where the next conclusion should be — rather than trying to jump from question to answer in one leap across a potentially very large distance in meaning-space.
 
## 6  RAG: connecting memory to the world
 
A language model's knowledge is encoded in its weights — the billions of numerical parameters adjusted during training. This is **parametric memory**: implicit, compressed, and frozen at training time. It has two structural weaknesses. It cannot be updated without retraining. And because knowledge is distributed across billions of parameters with no explicit index, the model cannot reliably attribute its answers to specific sources — a core driver of hallucination.
 
Lewis et al. (2020) proposed **Retrieval-Augmented Generation (RAG)** [11] as a complementary architecture: combine the language model with an external document store. At inference time, a query retrieves relevant documents, which are placed into the model's context window alongside the question. The model generates its answer conditioned on both its parametric knowledge and the retrieved evidence.
 
> **Example — RAG in a practical setting** A law firm deploys an AI assistant over its case files.
> Without RAG, the model would try to answer questions about specific cases from its training data —
> which doesn't include confidential case files, and which may be out of date. With RAG: the user asks
> "What was the outcome of the Schmidt vs. Hofmann case?", a retrieval step fetches the relevant case
> documents, these are added to the model's context, and the model generates an answer grounded in the
> actual documents. The firm's lawyers can check the retrieved documents directly.
 
RAG substantially reduces hallucination on knowledge-intensive tasks because the answer can be grounded in retrieved, verifiable text. It also allows the knowledge base to be updated without touching the model weights — significant in domains where information changes rapidly. The distinction between parametric memory (what the model "knows") and non-parametric memory (what it can look up) is one of the most pedagogically useful concepts in applied AI today.
 
> **Note for educators** RAG is a powerful teaching example because it makes explicit what is
> otherwise opaque: the distinction between a model generating from its own internal representations
> versus being grounded by external sources. Students can inspect the retrieved documents, trace why
> an answer went wrong (wrong retrieval? model ignored the evidence?), and reason about interventions.
> It concretises the abstract idea of "where does the model's knowledge come from."
 
## 7  Debunking the stochastic parrot
 
### 7.1 Where the parrot critique lands hardest
 
Before defending modern language models against the parrot critique, it is worth asking honestly: where on the history we have just traced does the critique actually land? The accusation is that the model "haphazardly stitches together sequences of linguistic forms ... without any reference to meaning." Read carefully, this is not a description of GPT-4. It is a remarkably precise description of the methods that came *before*.
 
Consider an **n-gram model**. It assigns probabilities to word sequences based on raw co-occurrence counts. It has no representation of meaning at all — words are opaque symbols whose only property is how often they appear next to other opaque symbols. When an n-gram model generates text, it is literally doing what Bender's quote describes: sampling sequences according to probabilistic information about how they combine, with no reference whatsoever to anything outside the corpus. The parrot accusation is not just apt here; it is a textbook description of the mechanism.
 
A **symbolic AI system** looks superficially different — it manipulates explicit rules — but the parrot characterisation applies almost as well, and for an analogous reason. The system has no meaning of its own; it has the meanings that its human authors transcribed into it. MYCIN does not *understand* infection; it executes rules that someone who understood infection wrote down. Ask it about a case its authors did not anticipate and it fails, because the meaning it manipulates is borrowed, not its own. This is a different kind of parroting — parroting a knowledge engineer rather than a corpus — but it is parroting nonetheless. The system reproduces its training input, just by a different route.
 
Even **static word embeddings** like Word2Vec do not fully escape this. They build a richer representation than n-grams — the geometry of the embedding space encodes semantic relationships — but they do so by averaging over all the contexts in which a word has been seen. The result is a single, fixed location per word, deaf to the specific sentence in which the word appears. "Bank" sits at one point in space whether you are talking about loans or rivers. The representation reflects training-data statistics but does not respond to the meaning of the current input. It is meaning frozen at training time, not meaning produced by the model.
 
> **The spectrum of meaning across methods** The history we have traced is, in part, a story of each
> generation of methods reducing the gap between "what the system manipulates" and "what the words
> mean." Symbolic AI manipulated borrowed meanings. N-grams manipulated nothing but symbols. Word
> embeddings introduced a geometric proxy for meaning. Attention made that geometry context-sensitive.
> Scale gave the geometry enough resolution to support generalisation. Each step weakened the parrot
> critique. The question for modern LLMs is whether the geometry has become rich enough, and the
> context-sensitivity sharp enough, that the critique no longer lands — or whether it merely lands
> less hard.
 
This reframing matters because the parrot critique was launched against transformer language models specifically, but its sharpest form actually applies most clearly to the methods that preceded them. The interesting empirical question is not whether *some* machine learning is parrot-like — for many older methods, this is uncontroversially true — but whether transformers at scale have crossed a threshold that earlier methods did not. That is the question we examine next.
 
### 7.2 The parrot thesis, fairly stated
 
In 2021, Bender, Gebru, McMillan-Major and Mitchell published "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" [12] — one of the most influential and most contested papers in AI ethics. Separating the technical claim from the paper's legitimate concerns about environmental cost, data bias, and labour exploitation, the core argument is this:
 
> **The parrot claim (Bender et al., 2021)** "A language model is a system for haphazardly stitching
> together sequences of linguistic forms it has observed in its vast training data, according to
> probabilistic information about how they combine, but without any reference to meaning." The model
> manipulates *form* — sequences of tokens — without *meaning* — any connection to the world. It is a
> stochastic parrot: statistically impressive, semantically empty.
 
This argument has intellectual roots in the symbol grounding problem from cognitive science [13]: the claim that symbols acquire meaning only through embodied, causal contact with the world — through seeing, touching, doing — not through relations to other symbols. A dictionary defines words using other words; nothing ever makes contact with the actual world. Under this view, no matter how sophisticated the statistics, a model trained only on text is trapped inside a hall of mirrors.
 
This is a coherent philosophical position and should be taken seriously. But as an empirical claim about what large transformer models actually do, it has become increasingly difficult to sustain. The evidence is worth examining carefully.
 
### 7.3 The evidence against pure parroting
 
**Generalisation beyond training distribution.** A pure parrot can only reproduce patterns it has observed. But large language models demonstrably generalise to inputs that are, by construction, outside their training data. GPT-4 achieved above the 90th percentile on the Uniform Bar Examination and 93% accuracy on the MATH benchmark of high-school olympiad problems [14]. The specific questions had not been seen in training. Handling novel instances of these tasks requires some form of structural generalisation — not just retrieval.
 
> **Example — the Winograd schema test** "The city councillors refused the demonstrators a permit
> because they feared violence. Who feared violence?" The answer ("the councillors") requires
> understanding that it is plausible for authorities to fear violence from demonstrators, not the
> reverse — knowledge about the world, not about word sequences. GPT-4 handles these correctly at
> near-human rates. A pure statistical parrot, associating "they" with the nearest plural noun, would
> systematically fail.
 
**Emergent internal world models.** Li et al. (2022) trained a language model purely on sequences of Othello game moves — no game states, just move notations [15]. They then probed the model's internal activations and found linearly decodable representations of the game board: the model had built an internal map of which squares were occupied and by whom, with no supervision on board states. This is a direct counterexample to the claim that LLMs represent only linguistic form. From text-like inputs, a structural understanding of the underlying system emerged.
 
**Mechanistic interpretability evidence.** Anthropic's interpretability research on Claude identified computational circuits — specific subgraphs of the attention and feedforward layers — that implement identifiable operations: fact lookup, name-binding between subject and predicate, multi-step planning in which intermediate conclusions are internally represented before being expressed [14]. The model is not haphazardly stitching; specific computations correspond to specific internal representations.
 
**Mathematical reasoning.** A 2024 workshop documented frontier LLMs generating coherent proofs of novel mathematical problems [14]. Mathematical proof is perhaps the hardest test for the parrot: the answer cannot be retrieved from memory, because the problem is new; it must be derived. Generating valid proofs requires — at minimum — something functionally equivalent to understanding axioms and logical inference.
 
**Context-directed extrapolation.** Madabushi et al. (2025) propose a precise framing: LLMs perform "context-directed extrapolation from training data priors" [16] — a mechanism that substantially exceeds statistical pattern repetition. The model infers from context which part of its learned structure is relevant, then extrapolates beyond the specific examples seen. This is not human reasoning, but it is not parroting either.
 
### 7.4 Prediction, planning, and the bounded role of randomness
 
There is one technical fact in the parrot argument that does survive scrutiny: a language model genuinely does produce its output one token at a time, and each token is, mechanically, sampled from a probability distribution over the vocabulary. In that narrow sense, the model is "picking the most likely next word" — exactly what a stochastic parrot would do. Why, then, does this not reduce the whole enterprise to elaborate autocomplete?
 
Two facts answer this — one about planning, one about geometry.
 
**The model plans further ahead than the next token.** Anthropic's 2025 interpretability study "On the Biology of a Large Language Model" [20] probed Claude 3.5 Haiku while it wrote a rhyming couplet. Given "He saw a carrot and had to grab it," the model produced "His hunger was like a starving rabbit." The natural assumption is that the model writes word by word and only at the end scrambles to find a rhyme. The internal traces showed something quite different: *before* generating any word of the second line, the model had already activated representations of candidate end-words that rhyme with "grab it" — "rabbit" among them — and then composed the entire line to land at the planned word. When the researchers intervened to suppress the "rabbit" representation and inject "habit" instead, the model adapted and produced a sensible line ending in "habit." This is not next-token greed; this is goal-directed generation, with the next-token mechanism serving the longer plan.
 
> **What this means in plain language** Imagine writing a sentence with a target word in mind for the
> end. You do not write each word in isolation — you write each word so that the sentence as a whole
> arrives at the target naturally. The model does something analogous. The token-by-token prediction
> is the *output mechanism*; the model's internal representations are already conditioned on where the
> sentence is going. Predicting the next token is the surface; the structure that produces it spans
> many tokens at once.
 
**The semantic-space framing dissolves much of the randomness concern.** Even when the model does sample stochastically from the next-token distribution, the consequences of that sampling are bounded. Here is why. The model's hidden state at each generation step encodes a *direction* in semantic space — a region of meaning where the output should land. The next-token distribution is, in effect, a probability cloud over words that occupy that region. The randomness chooses *which* word from the region, but the region itself is determined by everything that has come before. Whether the model outputs "happy," "joyful," or "elated" at a particular position usually does not change the meaning of the sentence — these words occupy roughly the same neighbourhood in semantic space. The choice is locally arbitrary but globally directed.
 
This is the key insight: the stochastic parrot critique pictures the model as essentially drawing words from a hat weighted by training-data frequency. The reality is closer to drawing from a hat that has been *placed* in a specific location in semantic space by the entire preceding context. The randomness operates at the level of word-choice within a meaning-region; the meaning-region is selected non-randomly, by the model's deep representation of what should be said next. This is why the same prompt to the same model produces outputs that differ in wording but generally agree in substance.
 
> **The reconciliation** The parrot critique is technically correct that prediction is word-by-word
> and probabilistic. It is empirically wrong that this makes the model semantically empty. The
> probabilistic step is the final, local choice within a region that the model's deeper
> representations have already navigated to. Planning operates over many tokens at once; the next-
> token mechanism merely executes that plan one word at a time. The randomness is a thin layer of
> variation on top of a deeply structured semantic process.
 
### 7.5 When the parrot does appear: hallucination reconsidered
 
If LLMs are not stochastic parrots in general, there is a precise regime in which they behave *as if* they were. **Hallucination** — the generation of fluent, plausible text that is factually incorrect — is best understood as exactly this: the model defaulting to high-probability completions in its training distribution when its parametric memory is absent or ambiguous on the queried fact.
 
Ask a large language model to name the population of a small Swiss municipality it has never encountered, and it will generate a number that sounds plausible — drawn from the distribution of how population figures are expressed — without any grounding in the actual fact. In this moment, the stochastic parrot characterisation is accurate. The model is producing statistically coherent text without reference to meaning.
 
Connecting this to the previous subsection: hallucination is what happens when the semantic-space direction is *under-determined*. When the model has been pulled by the context into a region of semantic space where its representations are sparse — because the relevant facts were not in training, or were ambiguous — the next-token distribution is correspondingly diffuse. The local randomness is no longer cushioned by a strong directional signal. In that gap, the parrot reappears. The semantic-space buffer is real, but it is only as strong as the model's representational density in the region being queried.
 
> **Hallucination as localised parrot behaviour** The parrot characterisation is most accurate not as
> a description of transformer models in general, but as a description of a specific failure mode: the
> model falling back on distributional priors when its parametric memory is insufficient. This is why
> RAG and chain-of-thought are effective mitigations — RAG supplies the missing factual grounding;
> chain-of-thought forces the model to make its reasoning explicit and therefore checkable. The parrot
> appears specifically in the gap between what the model was trained on and what it is being asked to
> retrieve.
 
<details class="collapse collapse-arrow bg-base-200 border border-base-300 my-6">
<summary class="collapse-title font-semibold">🔍 <span class="text-xs uppercase tracking-wider opacity-60">Deep dive</span> — Three types of hallucination, three different fixes</summary>
<div class="collapse-content">

Three types of hallucination correspond to three distinct failure modes [17]. *Fact-conflicting* hallucinations arise when generated claims contradict world knowledge — the closest to pure parrot behaviour. *Context-conflicting* hallucinations occur when the model loses consistency within a long conversation. *Input-conflicting* hallucinations arise when the model's output deviates from what the query explicitly specified. Each has a different cause and a different mitigation.

</div>
</details>
 
### 7.6 The middle ground
 
The most defensible current position is neither "pure parrot" nor "general intelligence." LLMs exhibit predictable, controllable capabilities that substantially exceed statistical pattern repetition — they generalise, they build internal representations, they reason when given the right scaffolding — but they do not exhibit robust general cognition of the human kind, and they have specific, systematic failure modes.
 
| Claim | Pure parrot | LLM (evidence-based) |
|---|---|---|
| Only reproduces training patterns | Claimed | Generalises OOD on exams, proofs, novel problems [14] |
| No internal world model | Claimed | Othello LM builds board representations [15] |
| Cannot multi-step reason | Claimed | Partial — requires CoT scaffolding; fails on some formal tasks |
| Semantically empty operations | Claimed | Partial — circuits implement fact lookup and inference [14] |
| Greedy next-token prediction with no plan | Claimed | Multi-token planning observed in rhyming-poetry circuits [20] |
| Hallucination as default | Claimed | Partial — arises when parametric memory is absent [17] |
 
This matters practically. The parrot framing underestimates LLMs and leads to misplaced complacency ("it's just autocomplete — it can't do anything important"). The AGI framing overestimates them and leads to either misplaced fear or uncritical deference. The honest middle ground is: these systems perform sophisticated semantic operations over learned representations, they generalise in ways that matter, and they fail in predictable ways that we can understand and partially mitigate.
 
## 8  Implications for educators and practitioners
 
**The semantic space framing is teachable.** The single most powerful reframe for a non-technical audience is to describe language models as navigators of a meaning-space, not rule-followers. Words are locations; context shifts those locations; attention is the mechanism of shifting; the model generates by navigating toward locations that are consistent with everything it has read so far. This is accessible, accurate, and immediately illuminates both what the model does well (nuanced contextual understanding) and what it does badly (facts it was never near in training).
 
**Hallucination is not random noise — and therefore it is addressable.** Understanding hallucination as the regime where the model falls back on distributional priors — the parrot regime — gives practitioners a framework for intervention. Retrieve grounding facts (RAG). Force explicit reasoning (chain-of-thought). Verify outputs against sources. Each of these targets a specific mechanism. Telling users "it sometimes makes things up" is less useful than explaining *when* and *why*, and what can be done about it.
 
**Chain-of-thought as a classroom activity.** Ask students to give the same complex question to a language model twice: once directly, and once with "think step by step" appended. Compare the outputs. The difference in quality on reasoning tasks is often dramatic and immediately convincing. Inspect the steps: which are correct? Where does the reasoning break down? This is a live, hands-on demonstration of the difference between parametric retrieval and active reasoning.
 
**The Winograd schema as a discussion anchor.** Sentences like the city council example above are designed specifically to test whether a system has world knowledge beyond word statistics. Give a class a set of Winograd schemas; ask them to predict which ones a language model will get right and which it will fail on. Then test it. The results often surprise students in both directions — failures on seemingly simple sentences, successes on apparently hard ones — and the discussion of why is rich.
 
**Scale and emergence require epistemic humility.** The emergence of qualitatively new capabilities at scale is genuinely surprising and not yet fully understood. Honest AI education acknowledges both the remarkable capabilities and the genuine uncertainty about their mechanisms and limits. Avoid the dismissive ("it's just statistics") and the credulous ("it reasons like a person"). The interesting and accurate position — that these systems do something genuinely novel that we are still learning to characterise — is also the most intellectually honest one.
 
This article was prepared by the ICE Industrial-AI team at the Institute for Computational Engineering (ICE), Eastern Switzerland University of Applied Sciences (OST). We welcome feedback and collaboration: [ice@ost.ch](mailto:ice@ost.ch)
 
## 9  References
 
- [1] Turing, A.M. (1950). Computing Machinery and Intelligence. *Mind*, 59(236), 433–460.
- [2] Buchanan, B.G. & Shortliffe, E.H. (Eds.) (1984). *Rule-Based Expert Systems: The MYCIN Experiments*. Addison-Wesley.
- [3] Polanyi, M. (1966). *The Tacit Dimension*. Doubleday.
- [4] Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. *JMLR*, 3, 1137–1155.
- [5] Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 30*. arXiv:1706.03762
- [6] Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*. arXiv:1409.0473
- [7] Brown, T. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS 33*. arXiv:2005.14165
- [8] Wei, J. et al. (2022). Emergent Abilities of Large Language Models. *TMLR*. arXiv:2206.07682
- [9] Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 35*. arXiv:2201.11903
- [10] OpenAI (2024). Learning to Reason with LLMs. https://openai.com/index/learning-to-reason-with-llms/
- [11] Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 33*. arXiv:2005.11401
- [12] Bender, E.M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots. *FAccT '21*, 610–623. https://doi.org/10.1145/3442188.3445922
- [13] Bender, E.M. & Koller, A. (2020). Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data. *ACL 2020*, 5185–5198.
- [14] Wikipedia contributors (2026). Stochastic parrot — evidence against. https://en.wikipedia.org/wiki/Stochastic_parrot (sources: GPT-4 Technical Report, OpenAI 2023; Anthropic mechanistic interpretability research 2024–2025; Berkeley frontier model workshop 2024)
- [15] Li, K. et al. (2022). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. arXiv:2210.13382
- [16] Madabushi, H.T., Torgbi, M., & Bonial, C. (2025). Neither Stochastic Parroting nor AGI. arXiv:2505.23323
- [17] Ji, Z. et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12). arXiv:2202.03629
- [18] Lighthill, J. (1973). Artificial Intelligence: A General Survey. In *Artificial Intelligence: A Paper Symposium*, Science Research Council, UK.
- [19] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*. arXiv:1301.3781
- [20] Lindsey, J. et al. (Anthropic, 2025). On the Biology of a Large Language Model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- [21] Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
- [22] Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
- [23] Linnainmaa, S. (1970). The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors. Master's thesis, University of Helsinki. (First publication of reverse-mode automatic differentiation.)
- [24] Werbos, P.J. (1974). Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences. PhD thesis, Harvard University.
- [25] Carpenter, G.A. & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*, 37, 54–115. (Foundations of ART from Grossberg 1976 onward.)
- [26] Fukushima, K. (1980). Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. *Biological Cybernetics*, 36(4), 193–202.
- [27] Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.
- [28] Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- [29] Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.
- [30] Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.