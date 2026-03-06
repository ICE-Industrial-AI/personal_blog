---
title: "Parallele & hochdimensionale Bayes-Optimierung"
description: "Dieser Artikel bietet einen detaillierten Überblick über einen fortschrittliche Bereich der Bayes'schen Optimierung (BO): die parallele (Batch-)Optimierung, die sich auf die q-Expected Improvement (q-EI) Akquisitionsfunktion konzentriert. Der Artikel stellt die fundamentalen Referenzen für jedes Thema bereit, analysiert die Entwicklung des Fachgebiets und präsentiert einen umfassenden Überblick über den aktuellen Stand der Technik."
pubDate: "March 6 2026"
heroImage: "/personal_blog/aikn.webp"
badge: "Latest"
---

# Parallele und hochdimensionale Bayes-Optimierung

**Author:** *Christoph Würsch, Institute for Computational Engineering ICE* <br>
*Eastern Switzerland University of Applied Sciences OST* <br>
**Date:** 6.3.2026<br>


> Dieser Artikel bietet einen detaillierten Überblick über einen fortschrittliche Bereich der Bayes'schen Optimierung (BO): die parallele (Batch-)Optimierung, die sich auf die q-Expected Improvement (q-EI) Akquisitionsfunktion konzentriert. Der Artikel stellt die fundamentalen Referenzen für jedes Thema bereit, analysiert die Entwicklung des Fachgebiets und präsentiert einen umfassenden Überblick über den aktuellen Stand der Technik.




## Inhaltsverzeichnis

- [Mathematische Grundlagen der Bayes'schen Optimierung](#mathematische-grundlagen-der-bayesschen-optimierung)
- [Effizienz durch Parallelisierung (q-EI)](#effizienz-durch-parallelisierung-q-ei)
  - [Ansatz 1: Verbesserung der Berechnung (Monte Carlo & Numerische Stabilität)](#ansatz-1-verbesserung-der-berechnung-monte-carlo--numerische-stabilität)
  - [Ansatz 2: Verwendung anderer Batch-fähiger Metriken](#ansatz-2-verwendung-anderer-batch-fähiger-metriken)
  - [Ansatz 3: Stichprobenbasierte Methoden (Thompson Sampling)](#ansatz-3-stichprobenbasierte-methoden-thompson-sampling)
  - [Ansatz 4: Asynchrone Methoden](#ansatz-4-asynchrone-methoden)
- [SOTA-Unterraum-Methoden: Sparse und lokale Optimierung](#sota-unterraum-methoden-sparse-und-lokale-optimierung)
  - [Sparse Unterräume: SAASBO](#sparse-unterräume-saasbo)
  - [2. Lokale Unterräume: TuRBO](#2-lokale-unterräume-turbo)
- [Welche Methode ist die beste?](#welche-methode-ist-die-beste)
- [Schlussfolgerung](#schlussfolgerung)
- [Literaturverzeichnis](#literaturverzeichnis)



## Mathematische Grundlagen der Bayes'schen Optimierung

Die Bayes'sche Optimierung (BO) ist ein sequentielles Verfahren zur Optimierung
von Black-Box-Funktionen $f(x)$, die teuer in der Auswertung sind (z.B. eine
Simulation). Das Ziel ist die Maximierung:
$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$
wobei $\mathcal{X} \subseteq \mathbb{R}^D$ der Designraum ist. BO besteht aus
zwei Kernkomponenten:

1. **Ein Surrogatmodell (Probabilistisch):** Ein statistisches
   Modell, das unseren Glauben (Belief) über $f(x)$ abbildet.
   Standardmässig wird ein **Gauss-Prozess (GP)** verwendet. Ein GP definiert eine Verteilung über
   Funktionen. Nach $t$ Beobachtungen $\mathcal{D}_t = \{(x_i, y_i)\}_{i=1}^t$
   (wobei $y_i = f(x_i) + \epsilon_i$) ist die Vorhersage (Posterior)
   für einen neuen Punkt $x$ ebenfalls gaussverteilt:
   $$
   f(x) | \mathcal{D}_t \sim \mathcal{N}(\mu_t(x), \sigma_t^2(x))
   $$
   wobei $\mu_t(x)$ der erwartete Funktionswert (Mittelwert) und
   $\sigma_t^2(x)$ die Unsicherheit (Varianz) an diesem Punkt ist.

2. **Eine Akquisitionsfunktion $a(x)$:** Eine Funktion, die den
   Nutzen der Evaluierung eines Punktes $x$ quantifiziert. Sie
   balanciert **Exploitation** (Ausnutzung: Suche, wo $\mu_t(x)$
   hoch ist) und **Exploration** (Erkundung: Suche, wo
   $\sigma_t^2(x)$ hoch ist). Der nächste zu simulierende Punkt $x_{t+1}$
   wird durch Maximierung der Akquisitionsfunktion gefunden:
   $$
   x_{t+1} = \arg\max_{x \in \mathcal{X}} a(x)
   $$

Die Grundlage der Bayes'schen Optimierung ist die Akquisitionsfunktion, die die Auswahl neuer Auswertungspunkte steuert. Die kanonische Akquisitionsfunktion ist das *sequentielle* *Expected Improvement* (EI). Dieser entscheidungstheoretische Ansatz wählt den nächsten Punkt $x$ zur Auswertung, indem er die erwartete Verbesserung gegenüber dem aktuell besten beobachteten Wert $f^*$ maximiert. In vielen realen Anwendungen, wie der Hyperparameter-Abstimmung, der Materialentwicklung oder der Medikamentenentdeckung, sind die Funktionsauswertungen zwar teuer (zeitaufwändig oder ressourcenintensiv), können aber parallel auf Multi-Core-Systemen oder Clustern durchgeführt werden. Die sequentielle Natur der Standard-BO wird in diesem Szenario zu einem Engpass, da Rechenressourcen ungenutzt bleiben, während auf den Abschluss einer einzelnen Auswertung gewartet wird. Dies schuf die zwingende Notwendigkeit, Batch-Akquisitionsfunktionen zu entwickeln, die einen Stapel (Batch) von $q$ Punkten gleichzeitig auswählen können.

Sei $f(x^+) = \max_{i=1...t} y_i$ der beste
bisher beobachtete Wert. Die Improvement-Funktion $I(x)$ ist definiert als
der Betrag, um den ein neuer Punkt $f(x)$ besser ist als $f(x^+)$:

$$
\boxed{
I(x) = \max(0, f(x) - f(x^+))
}
$$

Da $f(x)$ eine Zufallsvariable ist (gemäss dem GP-Posterior
$\mathcal{N}(\mu_t(x), \sigma_t^2(x))$), ist auch $I(x)$ eine Zufallsvariable.
EI ist der Erwartungswert dieser Verbesserung:
$$
a_{\rm EI}(x) = \mathbb{E}[I(x) | \mathcal{D}_t] = \mathbb{E}[\max(0, f(x) - f(x^+))]
$$
Dieser Erwartungswert hat eine geschlossene, analytische Form:

$$
\boxed{
a_{\rm EI}(x) = (\mu_t(x) - f(x^+)) \Phi(Z) + \sigma_t(x) \phi(Z)
}
$$

wobei:

- $Z = \frac{\mu_t(x) - f(x^+)}{\sigma_t(x)}$ (standardisierter Score)
- $\Phi(\cdot)$ die kumulative Verteilungsfunktion (CDF) der
  Standardnormalverteilung ist.
- $\phi(\cdot)$ die Wahrscheinlichkeitsdichtefunktion (PDF) der
  Standardnormalverteilung ist.

Das Standard-EI ist rein sequentiell. Für die parallele Auswertung von $q$
Punkten (Batch) benötigen wir eine Akquisitionsfunktion, die einen Batch
$\mathbf{X}_q = \{x_1, ..., x_q\}$ bewertet.

## Effizienz durch Parallelisierung (q-EI)

Die Entwicklung einer parallelen Version von EI war nicht trivial. Die naive Maximierung des sequentiellen EI $q$-mal hintereinander führt zu einer suboptimalen Auswahl, bei der alle $q$ Punkte im selben vielversprechenden Bereich landen. Die Herausforderung besteht darin, einen Batch von Punkten auszuwählen, der die *gemeinsame* Verbesserung maximiert und dabei die Interaktionen zwischen den Punkten im Batch probabilistisch berücksichtigt.

### Die Constant Liar Heuristik

Eine frühe und fundamentale heuristische Methode zur Generierung eines Batches war der *Constant Liar* (CL) Ansatz, der in der Arbeit von Ginsbourger et al. (2010) vorgestellt wurde. Der CL-Mechanismus funktioniert wie folgt: Um einen Batch der Grösse $q$ auszuwählen, wird der erste Punkt $x_1$ durch Maximierung des Standard-EI ausgewählt. Da der wahre Wert $f(x_1)$ noch nicht bekannt ist, lügt der Algorithmus und weist diesem Punkt einen heuristischen Wert zu (die Lüge) – typischerweise den Mittelwert der Gauss-Prozess-Prognose (GP) oder einen pessimistischen Wert (z.B. die untere Konfidenzgrenze). Das GP-Modell wird temporär mit diesem falschen Datenpunkt $(x_1, f_{lie})$ aktualisiert. Anschliessend wird der zweite Punkt $x_2$ durch Maximierung des EI auf Basis des aktualisierten Modells ausgewählt. Dieser Vorgang wird $q$-mal wiederholt.

Obwohl diese Heuristik schnell ist, ist sie nachweislich suboptimal und schneidet oft schlechter ab als eine echte sequentielle Auswertung. Ihr Hauptnachteil ist, dass sie die Interaktionen zwischen den Batch-Punkten nicht vollständig probabilistisch ausnutzt.

### Die Multi-Point EI Formulierung

Der wahre Goldstandard für Batch-EI ist das $q$-Expected Improvement (q-EI) Kriterium, auch als Multi-Point Expected Improvement (MPEI) bekannt. Dieses Kriterium zielt darauf ab, jenen Satz von $q$ Punkten $X_q = \{x_1, \ldots, x_q\}$ zu finden, der die *gemeinsame erwartete Verbesserung* über den aktuellen Bestwert $f^*$ maximiert. Die fundamentale Arbeit, die eine (eingeschränkt) effiziente analytische Formulierung hierfür lieferte, ist Chevalier und Ginsbourger (2013) [[chevalier2013](#ref-chevalier2013)], *Fast computation of the multi-points expected improvement*. Diese Arbeit zeigte, dass das hochdimensionale Integral des $q$-EI durch eine Zerlegung in eine Summe von $q$-dimensionalen kumulativen Verteilungsfunktionen (CDFs) von Gauss-Verteilungen berechnet werden kann. Diese $q$-dimensionalen CDFs können selbst mit den Algorithmen von Genz und Bretz (2009) [[genz2009](#ref-genz2009)] effizient approximiert werden. Eine spätere Arbeit von Marmin et al. (2015) [[marmin2015](#ref-marmin2015)] erweiterte dies um eine analytische Formel für den *Gradienten* des $q$-EI, um eine gradientenbasierte Optimierung der Akquisitionsfunktion zu ermöglichen.

Obwohl die Arbeit von Chevalier und Ginsbourger (2013) fundamental ist, löste sie das Problem nicht vollständig. Die direkte analytische Berechnung des $q$-EI ist für grosse $q$ rechnerisch unmöglich. Der Grund dafür ist, dass die **Anzahl der Aufrufe der $q$-dimensionalen Gauss-CDF *quadratisch* mit der Batch-Grösse $q$ wächst**.

Diese quadratische Skalierungsbarriere ist der *primäre Treiber* für das gesamte Forschungsfeld alternativer Batch-BO-Methoden. Sie zwang die Forscher, Ansätze zu finden, die entweder:

1. das $q$-EI-Integral effizienter approximieren (z.B. Monte Carlo),
2. eine andere, leichter zu berechnende Batch-Akquisitionsfunktion verwenden, oder
3. das synchrone Batch-Paradigma vollständig aufgeben.

### Ansatz 1: Verbesserung der Berechnung (Monte Carlo & Numerische Stabilität)

Der heute gängigste Ansatz zur Berechnung von $q$-EI vermeidet die analytische Formulierung vollständig und approximiert das $q$-EI-Integral mithilfe von **Monte-Carlo-Stichproben (MC)**. Das $q$-EI $\alpha(X)$ wird geschätzt, indem $N$ Stichproben $\xi_i$ aus der GP-Posterior-Verteilung an den $q$ Punkten gezogen werden. Für jede Stichprobe wird die Verbesserung berechnet, und die Ergebnisse werden gemittelt :

$$
\boxed{
\text{qEI}(X) \approx \frac{1}{N} \sum_{i=1}^N \max_{j=1, \ldots, q} \bigl\{ \max(\xi_{ij} - f^*, 0) \bigr\}
}
$$

Dieser MC-Ansatz ist die Grundlage für $q$-EI in modernen BO-Bibliotheken wie BoTorch.

Eine kritische SOTA-Entdeckung (NeurIPS 2023) zeigte jedoch, dass die klassische EI-Formulierung (sowohl analytisch als auch MC) unter schwerwiegenden *numerischen Pathologien* leidet, wie z.B. verschwindenden Gradienten bei der Optimierung der Akquisitionsfunktion. Die Autoren schlugen die **LogEI-Familie** vor, die die Akquisitionsfunktion im Log-Raum neu formuliert. Die SOTA-Variante **q-LogEI**  wendet diese numerischen Verbesserungen auf den MC-Schätzer an, was zu einer wesentlich höheren numerischen Stabilität und Optimierungsleistung führt, die oft die Leistung anderer SOTA-Methoden übertrifft.

### Ansatz 2: Verwendung anderer Batch-fähiger Metriken

- **q-Upper Confidence Bound (q-UCB):** Als direktes Analogon zu $q$-EI wählt $q$-UCB einen Batch von $q$ Punkten aus, der *gemeinsam* eine obere Konfidenzgrenze des Funktionswerts maximiert.
- Eine sehr aktuelle Studie aus dem Jahr 2025, die Batch-Akquisitionsfunktionen (qLogEI, qUCB und UCB/LP) auf schwierigen Benchmark-Problemen (Needle-in-a-Haystack und False Optimum) verglich, kam zu einem wichtigen Schluss: **q-UCB** zeigte die beste *Gesamtleistung* und erwies sich als zuverlässig und robust über alle Landschaftstypen hinweg. Die Studie empfiehlt q-UCB explizit als **Standardwahl, wenn die Funktionslandschaft a priori unbekannt ist**.
- **q-Expected Hypervolume Improvement (q-EHVI):** Für die *mehrzielige* BO (Multi-Objective BO, MOBO) ist die SOTA-parallele Akquisitionsfunktion das $q$-EHVI. Es erweitert die Hypervolumen-Verbesserungsmetrik auf die parallele $q$-Einstellung und ermöglicht die Berechnung exakter Gradienten für eine effiziente Optimierung.

### Ansatz 3: Stichprobenbasierte Methoden (Thompson Sampling)

Ein völlig anderer Ansatz, der deterministische Akquisitionsfunktionen vermeidet, ist *Thompson Sampling* (TS).

- **Fundamentale Referenz:** **Kandasamy et al. (2018)**, *Parallelised Bayesian Optimisation via Thompson Sampling* [[kandasamy2018](#ref-kandasamy2018)].
- **Mechanismus:** Um einen Batch der Grösse $q$ auszuwählen, zieht TS einfach $q$ *unabhängige* Funktionsrealisierungen (Stichproben) $g_i \sim \text{GP}(\cdot | \mathcal{D})$ aus der GP-Posterior-Verteilung. Anschliessend wird der Maximierer (Optimum) *jeder* dieser $q$ Stichproben gefunden. Diese $q$ Optima bilden den Batch.
- **Vorteile:** Die Vorteile von TS gegenüber $q$-EI sind immens :
  1. **Konzeptionelle Einfachheit:** Es ist konzeptionell viel einfacher als die komplexe $q$-EI-Formulierung.
  2. **Recheneffizienz:** Die Kosten skalieren *nicht* mit der Batch-Grösse $q$ (abgesehen von der Durchführung von $q$ unabhängigen Optimierungen).
  3. **Theoretische Garantien:** TS verfügt über starke No-Regret-Grenzen.
  4. **Asynchronität:** Es ist *natürlich* und trivial auf das asynchrone parallele Setting erweiterbar , was sein grösster praktischer Vorteil ist.

### Ansatz 4: Asynchrone Methoden

Synchrone Batch-Methoden wie $q$-EI oder $q$-UCB haben einen entscheidenden Nachteil in der Praxis: Sie sind ineffizient, wenn die Auswertungszeiten variieren. Sie zwingen alle $q$ Worker, auf den *langsamsten* Worker im Batch zu warten, bevor die nächste Runde beginnt, wodurch wertvolle Rechenressourcen ungenutzt bleiben.

- **Asynchrones Thompson Sampling (asyTS)**. Bei diesem Ansatz wartet ein Worker, der seine Aufgabe beendet, nicht auf andere. Er aktualisiert *sofort* das GP-Modell mit seinem neuen Datenpunkt, zieht eine *neue* Stichprobe aus dem aktualisierten Posterior und beginnt mit der nächsten Auswertung. Kandasamy et al. (2018) haben bewiesen, dass asyTS ein *geringeres Regret in Bezug auf die Wanduhrzeit* (wall-clock time) erreicht als synchrone oder sequentielle Versionen.
- Für nicht-TS-basierte Ansätze ist **PLAyBOOK (Penalising Locally for Asynchronous Bayesian Optimisation)** von **Alvi et al. (2019)** [[alvi2019](#ref-alvi2019)] eine SOTA-Methode. Sie verwendet eine lokale Bestrafungs-Heuristik (local penalization), um zu verhindern, dass asynchrone Worker Punkte auswählen, die zu nahe an anderen Punkten liegen, die *derzeit ausgewertet werden*.

Für praktische Anwendungen im Ingenieurwesen und in der Wissenschaft, bei denen die Simulationszeiten variabel sind, sind asynchrone Methoden (asyTS und PLAyBOOK) der *wahre* Stand der Technik und bieten eine weitaus höhere Ressourcenauslastung.

## SOTA-Unterraum-Methoden: Sparse und lokale Optimierung

Dies sind die modernen, dominanten Algorithmen, die die oben genannten fundamentalen Methoden in der Leistung übertreffen.

### Sparse Unterräume: SAASBO

- **Eriksson & Jankowiak (2021)** [[eriksson2021](#ref-eriksson2021)], *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces (SAASBO)*
- **Mechanismus:** SAASBO ist ein intelligenter Vanilla-Ansatz. Es verwendet ein Standard-GP-Modell mit ARD-Kernel (Automatic Relevance Determination), setzt aber einen speziellen **hierarchischen Sparsity-Prior** auf die inversen Längenskalen ($\rho_d$). Konkret wird ein Half-Cauchy-Prior verwendet: $\rho_d \sim \mathcal{HC}(\tau)$ und $\tau \sim \mathcal{HC}(\beta)$.
- **Funktionsweise:** Dieser Prior zwingt die meisten inversen Längenskalen $\rho_d$ dazu, nahe Null zu liegen (globale Schrumpfung durch $\tau$). Eine $\rho_d$ nahe Null bedeutet, dass die Funktion entlang der Dimension $d$ fast konstant ist, wodurch diese Dimension effektiv abgeschaltet wird. Der Algorithmus *lernt* somit automatisch einen *dünnbesetzten (sparse), achsenausgerichteten* Unterraum der relevanten Dimensionen. Die Inferenz (Anpassung) des Modells erfolgt typischerweise mit anspruchsvollen Methoden wie Hamiltonian Monte Carlo (HMC).
- **Leistung:** SAASBO erzielt eine hervorragende SOTA-Leistung bei realen Problemen, wie z.B. einer 124-dimensionalen Fahrzeugdesign-Optimierungsaufgabe , und ist oft schneller als Konkurrenzmethoden. Es stellt einen flexiblen und leistungsstarken Kompromiss dar.

### 2. Lokale Unterräume: TuRBO

- **Eriksson et al. (2019)** [[eriksson2019](#ref-eriksson2019)], *Scalable Global Optimization via Local Bayesian Optimization (TuRBO).*
- **Mechanismus:** TuRBO verwirft die Idee, *ein einziges* globales GP-Surrogatmodell für den gesamten hochdimensionalen Raum anzupassen. Stattdessen unterhält es eine *Sammlung* von *lokalen*, unabhängigen GP-Modellen, die jeweils nur innerhalb ihrer eigenen Trust Region (TR) agieren.
- **Funktionsweise:** Der Algorithmus führt eine Standard-BO innerhalb eines kleinen Hyperrechtecks (der TR) durch. Die Grösse der TR wird dynamisch angepasst: Bei einem Erfolg (Finden eines besseren Punktes) wird die TR erweitert; bei einem Misserfolg wird sie verkleinert. Eine globale Bandit-Strategie verwaltet die (potenziell parallelen) TRs und teilt ihnen Auswertungsbudgets zu.
- **Implikation:** TuRBO bewältigt hohe Dimensionen, indem es *nicht* versucht, die globale Funktion zu modellieren. Es konzentriert sich auf die *lokale Optimierung* in vielversprechenden Regionen, was rechnerisch und statistisch wesentlich einfacher ist. Es ist hochgradig effektiv und ein häufig genutzter SOTA-Benchmark.

## Welche Methode ist die beste?

Die vorangegangene Analyse der grundlegenden Arbeiten und der neueren State-of-the-Art-Literatur zeigt, dass es weder für die parallele noch für die hochdimensionale Bayes-Optimierung eine einzelne, in jedem Szenario optimale Methode gibt. Vielmehr hängt die Wahl des geeigneten Verfahrens wesentlich von der Problemstruktur, vom Rechenbudget und von der verfügbaren Parallelisierungsstrategie ab. Für klassische Batch-Szenarien ist insbesondere die Frage entscheidend, ob synchron oder asynchron ausgewertet wird; für hochdimensionale Probleme ist ausschlaggebend, ob tatsächlich eine niedrige effektive Struktur vorliegt oder ob die Schwierigkeiten vor allem aus ungeeigneten Priorannahmen des Surrogatmodells resultieren.

Im Bereich der parallelen Bayes-Optimierung bildet die Mehrpunkt-Expected Improvement Idee $q$-EI einen wichtigen konzeptionellen Ausgangspunkt. Die Arbeit von Chevalier und Ginsbourger [[chevalier2013](#ref-chevalier2013)] hat gezeigt, wie sich die zu erwartende Verbesserung für mehrere gleichzeitige Kandidaten formal fassen lässt. Diese Formulierung ist theoretisch elegant und für kleine Batchgrössen sehr aufschlussreich. In der Praxis stösst die klassische analytische Auswertung jedoch rasch an Grenzen, da die Berechnung mit wachsender Batchgrösse teuer und numerisch anspruchsvoll wird. Gerade bei grösseren Batches ist daher weniger die ursprüngliche analytische Form entscheidend als vielmehr die Frage, wie sich ihre Grundidee robust und skalierbar approximieren lässt.

Eine wesentliche Weiterentwicklung in diese Richtung liefert die LogEI-Familie von Ament et al. [[ament2023](#ref-ament2023)]. Der zentrale Beitrag dieser Arbeit besteht darin, dass sie die numerischen Pathologien der klassischen EI-Formulierung systematisch adressiert. Insbesondere in Regionen des Suchraums, in denen die herkömmliche EI numerisch praktisch verschwindet, bleibt die logarithmische Formulierung stabil und optimierbar. Für parallele Szenarien macht dies qLogEI zu einer besonders attraktiven Variante: Sie verbindet die bekannte Struktur von $q$-EI mit deutlich verbesserter numerischer Robustheit. Aus methodischer Sicht ist qLogEI daher eine der überzeugendsten Optionen, wenn man die Grundidee von Expected Improvement beibehalten, aber ihre praktischen Schwächen vermeiden möchte.

Daneben hat sich auch die Familie der Monte-Carlo-basierten Akquisitionsfunktionen als sehr einflussreich erwiesen. Wilson et al. [[wilson2018](#ref-wilson2018)] zeigen, dass sich Batch-Akquisitionsfunktionen wie q-UCB effizient mit stochastischen Schätzern behandeln lassen. q-UCB ist dabei vor allem deshalb interessant, weil es konzeptionell einfach, breit einsetzbar und in vielen Anwendungen robust ist. Während EI-artige Methoden oft stark von der lokalen Form der Prädiktionsverteilung abhängen, liefert UCB eine klar interpretierbare Exploration-Exploitation-Balance. In praktischen Batch-Szenarien ist q-UCB deshalb häufig eine sehr solide Standardwahl, insbesondere dann, wenn über die Struktur der Zielfunktion nur wenig bekannt ist und eine robuste Basismethode benötigt wird.

Noch deutlicher wird die Bedeutung der Problemstruktur, sobald die Auswertungen nicht mehr gleich lange dauern. In realen Anwendungen --- etwa bei Simulationen, Laborversuchen oder Hyperparameterstudien mit heterogenen Trainingszeiten --- sind synchrone Batches oft ineffizient, weil freie Worker auf die langsamste laufende Auswertung warten müssen. Genau hier setzen asynchrone Verfahren an. Kandasamy et al. [[kandasamy2018](#ref-kandasamy2018)] zeigen mit asynchronem Thompson Sampling, dass sich Bayes-Optimierung in parallelen Umgebungen mit sehr geringer algorithmischer Komplexität und zugleich hoher praktischer Effizienz betreiben lässt. Jeder frei werdende Worker erhält unmittelbar einen neuen Punkt, sodass Leerlauf minimiert wird. Ergänzend dazu schlagen Alvi et al. mit PLAyBOOK einen akquisitionsbasierten asynchronen Ansatz vor, der lokale Penalisationen verwendet, um Kollisionen zwischen gleichzeitig laufenden Auswertungen zu vermeiden [[alvi2019](#ref-alvi2019)]. Aus praktischer Sicht spricht daher vieles dafür, in realen Rechensystemen mit stark variablen Evaluationszeiten asynchrone Verfahren zu bevorzugen, da sie nicht nur recheneffizient, sondern häufig auch hinsichtlich der tatsächlichen Wall-Clock-Zeit überlegen sind.

Für hochdimensionale Bayes-Optimierung ergibt sich ein ähnlich differenziertes Bild. Lange Zeit wurde angenommen, dass Bayes-Optimierung in grossen Dimensionen nur dann funktionieren kann, wenn eine zusätzliche Niedrigdimensionalitätsannahme explizit eingebaut wird, etwa in Form eines aktiven Unterraums, additiver Struktur oder lokaler Restriktionen. Die Arbeit von Hvarfner et al. [[hvarfner2024](#ref-hvarfner2024)] relativiert diese Sichtweise jedoch grundlegend. Die Autoren zeigen, dass die schlechte Leistung von ``Vanilla-BO'' in hohen Dimensionen oft weniger auf die Dimension selbst als auf ungeeignete Priorannahmen --- insbesondere für die Längenskalen des Gaussprozessmodells --- zurückzuführen ist. Wird der Prior konsistent mit der Eingabedimension skaliert, verbessert sich die Leistungsfähigkeit standardmässiger GP-basierter Bayes-Optimierung drastisch. Diese Beobachtung ist methodisch bemerkenswert, weil sie nahelegt, dass viele spezialisierte HDBO-Verfahren in der Praxis zunächst gar nicht nötig sind. Ein sinnvoller erster Schritt besteht daher darin, ein Standard-GP-Modell mit dimensionsskalierter Priorisierung zu testen, bevor komplexere Architekturannahmen eingeführt werden.

Wenn sich dennoch zeigt, dass zusätzliche Strukturannahmen erforderlich sind, dann gehören SAASBO und TuRBO zu den wichtigsten allgemeinen Ansätzen. SAASBO von Eriksson und Jankowiak [[eriksson2021](#ref-eriksson2021)] ist besonders dann geeignet, wenn vermutet wird, dass nur ein kleiner Teil der Koordinaten für die Zielfunktion wirklich relevant ist. Durch einen sparsitätsfördernden hierarchischen Prior auf die Längenskalen kann das Modell automatisch lernen, welche Dimensionen aktiv sind und welche effektiv ignoriert werden können. Der Vorteil dieses Ansatzes liegt darin, dass keine harte dimensionsreduzierende Projektion vorgegeben werden muss; vielmehr entsteht die relevante Struktur direkt aus dem posterioren Inferenzprozess. Damit ist SAASBO vor allem für allgemeine kontinuierliche Probleme attraktiv, bei denen ein dünnbesetzter, achsenausgerichteter Unterraum plausibel ist.

TuRBO von Eriksson et al. [[eriksson2019](#ref-eriksson2019)] verfolgt demgegenüber eine andere, aber ebenso wirkungsvolle Strategie. Anstatt global ein einziges Surrogat über den gesamten hochdimensionalen Suchraum zu fitten, arbeitet TuRBO mit lokalen Trust Regions, innerhalb derer das Problem wesentlich besser modellierbar ist. Dieser Ansatz ist besonders überzeugend, wenn die Zielfunktion global sehr komplex, lokal aber glatt genug ist, damit ein Gaussprozess sinnvolle Vorhersagen treffen kann. TuRBO vermeidet dadurch die in hohen Dimensionen häufig auftretende Überexploration globaler Akquisitionsfunktionen und zählt inzwischen zu den etablierten Referenzmethoden für kontinuierliche HDBO-Probleme.

<!--
Anders gelagert sind hochdimensionale Optimierungsprobleme auf *strukturierten* Domänen, etwa bei Molekülen, Sequenzen oder komplexen Engineering-Designs. Hier wird häufig zunächst ein generatives Modell verwendet, um die Objekte in einen kontinuierlichen latenten Raum einzubetten, in dem dann Bayes-Optimierung durchgeführt wird. Ein klassisches Beispiel dafür ist der VAE-basierte Ansatz von Gómez-Bombarelli et al. [[gomezbombarelli2018](#ref-gomezbombarelli2018)]. Diese Idee ist sehr einflussreich geworden, weil sie Bayes-Optimierung auf diskrete oder kombinatorische Räume übertragbar macht. Allerdings zeigt die neuere Literatur deutlich, dass diese Vorgehensweise ein fundamentales Ausrichtungsproblem besitzt: Im latenten Raum wird häufig nicht exakt das optimiert, was im ursprünglichen Eingaberaum tatsächlich bewertet wird. Neuere Arbeiten bezeichnen dieses Phänomen explizit als *Value Discrepancy Problem* [[lee2025nfbo](#ref-lee2025nfbo)]. Der Kern des Problems liegt darin, dass Rekonstruktionsfehler und Verzerrungen des generativen Modells dazu führen können, dass vielversprechende Punkte im latenten Raum nach dem Dekodieren nicht die erwartete Qualität besitzen.

Gerade deshalb sind neuere Verfahren wie CoBO und NF-BO besonders interessant. CoBO von Lee et al. [[lee2023cobo](#ref-lee2023cobo)] versucht, den latenten Raum so zu lernen, dass geometrische Nähe im latenten Raum stärker mit der Ähnlichkeit der Zielwerte korreliert. Dies geschieht unter anderem durch zusätzliche Regularisierung, die die Repräsentation explizit auf das Optimierungsziel ausrichtet. NF-BO [[lee2025nfbo](#ref-lee2025nfbo)] geht noch einen Schritt weiter und ersetzt den VAE durch einen Normalizing Flow. Da Normalizing Flows invertierbare Abbildungen bereitstellen, wird der Rekonstruktionsfehler prinzipiell stark reduziert beziehungsweise in idealisierter Form eliminiert. Damit adressiert NF-BO direkt die Ursache der Value Discrepancy und stellt eine besonders vielversprechende Richtung für strukturierte Optimierungsprobleme dar.
-->

## Schlussfolgerung

Insgesamt lassen sich aus diesen Befunden mehrere praktische Schlussfolgerungen ableiten. Für synchrone Batch-Optimierung sind qLogEI und q-UCB derzeit besonders überzeugende Optionen: qLogEI aufgrund seiner numerischen Stabilität, q-UCB aufgrund seiner robusten und breit einsetzbaren Charakteristik. Sobald Evaluationszeiten heterogen sind, sind asynchrone Verfahren wie asyTS oder PLAyBOOK meist die naheliegendere Wahl, weil sie die vorhandenen Ressourcen wesentlich besser ausnutzen. Für hochdimensionale kontinuierliche Probleme sollte man nicht sofort zu komplexen Dimensionsreduktionsverfahren greifen, sondern zunächst ein Standard-BO-Verfahren mit sorgfältig skalierten Priors testen. Erst wenn dieses Basismodell nicht ausreicht, bieten SAASBO und TuRBO starke Alternativen, je nachdem, ob eher Sparsität oder lokale Modellierbarkeit zu erwarten ist. Für strukturierte Suchräume schliesslich bleibt die Optimierung im latenten Raum ein hochrelevantes Forschungsgebiet, wobei die Behandlung der Value Discrepancy derzeit eine der zentralen offenen Fragen darstellt. Genau an dieser Stelle setzen CoBO [[lee2023cobo](#ref-lee2023cobo)]  und NF-BO [[lee2025nfbo](#ref-lee2025nfbo)] an, die deshalb als besonders vielversprechende neuere Entwicklungen angesehen werden können.

---

## Literaturverzeichnis

<div id="ref-ginsbourger2008"></div>

- **[ginsbourger2008]**  
  D. Ginsbourger, R. Le Riche, und L. Carraro.  
  *A multipoint criterion for deterministic parallel global optimization based on Gaussian processes*.  
  Vorgestellt auf dem International Conference on Nonconvex Optimization and Its Applications (SIAM), 2008.

<div id="ref-balandat2020"></div>

- **[balandat2020]**  
  M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, und E. Bakshy.  
  *BoTorch: A Framework for Bayesian Optimization*.  
  In Advances in Neural Information Processing Systems (NeurIPS), 2020.

<div id="ref-gomez2018"></div>

- **[gomez2018]**  
  R. Gómez-Bombarelli, J. Wei, D. Duvenaud, J. M. Hernández-Lobato,  
  B. Sánchez-Lengeling, D. Sheberla, J. Aguilera-Iparraguirre, T. D. Hirzel,  
  R. P. Adams, und A. Aspuru-Guzik.  
  *Automatic chemical design using a data-driven continuous representation*.  
  In ACS Central Science, 4(2), S. 268–276, 2018.

<div id="ref-levesque2023"></div>

- **[levesque2023]**  
  J. Lévesque, G. P. A. G. van Zundert, S. T. R. Went, M. T. M. Emmerich,  
  und T. Bäck.  
  *Bayesian optimization of high-dimensional problems using a linear subspace*.  
  In IEEE Transactions on Evolutionary Computation, 27(1), S. 138–150, 2023.

<div id="ref-wang2016"></div>

- **[wang2016]**  
  Z. Wang, M. Zoghi, F. Hutter, D. Matheson, und N. de Freitas.  
  *Bayesian optimization in high dimensions via random embeddings*.  
  Vorgestellt auf der International Joint Conference on Artificial Intelligence (IJCAI), 2016.

<div id="ref-eriksson2019-a"></div>

- **[eriksson2019]**  
  D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, und M. Poloczek.  
  *Scalable global optimization via local Bayesian optimization*.  
  In Advances in Neural Information Processing Systems (NeurIPS), 2019.

<div id="ref-chevalier2013"></div>

- **[chevalier2013]**  
  C. Chevalier and D. Ginsbourger,  
  *Fast computation of the multi-points expected improvement with applications in batch selection*,  
  in *Learning and Intelligent Optimization*, Lecture Notes in Computer Science, vol. 7997, Springer, 2013, pp. 59--69.

<div id="ref-ament2023"></div>

- **[ament2023]**  
  S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy,  
  *Unexpected Improvements to Expected Improvement for Bayesian Optimization*,  
  in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

<div id="ref-wilson2018"></div>

- **[wilson2018]**  
  J. T. Wilson, M. Moriconi, F. Hutter, and M. P. Deisenroth,  
  *Maximizing acquisition functions for Bayesian optimization*,  
  in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

<div id="ref-kandasamy2018"></div>

- **[kandasamy2018]**  
  K. Kandasamy, A. Krishnamurthy, J. Schneider, and B. P{\'o}czos,  
  *Parallelised Bayesian Optimisation via Thompson Sampling*,  
  in *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 84, 2018, pp. 133--142.

<div id="ref-alvi2019"></div>

- **[alvi2019]**  
  A. S. Alvi, B. Ru, J. Calliess, S. J. Roberts, and M. A. Osborne,  
  *Asynchronous Batch Bayesian Optimisation with Improved Local Penalisation*,  
  in *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR, vol. 97, 2019, pp. 253--262.

<div id="ref-hvarfner2024"></div>

- **[hvarfner2024]**  
  C. Hvarfner, E. Orm Hellsten, and L. Nardi,  
  *Vanilla Bayesian Optimization Performs Great in High Dimensions*,  
  in *Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 235, 2024, pp. 502--510.

<div id="ref-eriksson2021"></div>

- **[eriksson2021]**  
  D. Eriksson and M. Jankowiak,  
  *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*,  
  in *Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR, vol. 161, 2021, pp. 493--503.

<div id="ref-eriksson2019"></div>

- **[eriksson2019]**  
  D. Eriksson, M. Pearce, J. R. Gardner, R. D. Turner, and M. Poloczek,  
  *Scalable Global Optimization via Local Bayesian Optimization*,  
  in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.

<div id="ref-gomezbombarelli2018"></div>

- **[gomezbombarelli2018]**  
  R. G{\'o}mez-Bombarelli, J. N. Wei, D. Duvenaud, J. M. Hern{\'a}ndez-Lobato, B. S{\'a}nchez-Lengeling, D. Sheberla, J. Aguilera-Iparraguirre, T. D. Hirano, R. P. Adams, and A. Aspuru-Guzik,  
  *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules*,  
  *ACS Central Science*, vol. 4, no. 2, pp. 268--276, 2018.

<div id="ref-lee2023cobo"></div>

- **[lee2023cobo]**  
  S. Lee, J. Chu, S. Kim, J. Ko, and H. J. Kim,  
  *Advancing Bayesian Optimization via Learning Correlated Latent Space*,  
  in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

<div id="ref-lee2025nfbo"></div>

- **[lee2025nfbo]**  
  S. Lee, J. Park, J. Chu, M. Yoon, and H. J. Kim,  
  *Latent Bayesian Optimization via Autoregressive Normalizing Flows*,  
  in *International Conference on Learning Representations (ICLR)*, 2025.

<div id="ref-genz2009"></div>

- **[genz2009]**  
  A. Genz and F. Bretz,  
  *Computation of Multivariate Normal and $t$ Probabilities*,  
  Lecture Notes in Statistics, vol. 195, Springer, Dordrecht, 2009.

<div id="ref-marmin2015"></div>

- **[marmin2015]**  
  S. Marmin, C. Chevalier, and D. Ginsbourger,  
  *Differentiating the multipoint Expected Improvement for optimal batch design*,  
  in *Learning and Intelligent Optimization*,  
  Lecture Notes in Computer Science, vol. 8994,  
  Springer, Cham, 2015, pp. 37--48.