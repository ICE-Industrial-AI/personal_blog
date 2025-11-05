---
title: "Methoden der Versuchsplanung: Optimal Design und Sobol-Sequenzen"
description: "Die Erstellung von Surrogatmodellen zielt darauf ab, rechenintensive Simulationen durch schnelle Näherungsmodelle zu ersetzen. Der Erfolg dieses Ansatzes hängt fundamental von der Qualität und Effizienz der verwendeten Trainingsdaten ab, die durch Methoden der Versuchsplanung (Design of Experiments, DoE) generiert werden. Raumfüllende Pläne (wie Latin Hypercube und Sobol-Sequenzen) sind essenziell, wenn das Surrogatmodell (z.B. ein Neuronales Netz oder ein Gauss-Prozess) eine Black-Box-Funktion ohne a-priori-Modellstruktur approximieren soll. Die Sobol-Sequenz sticht hierbei durch ihre überlegene Gleichmässigkeit (geringe Diskrepanz) und ihre Erweiterbarkeit hervor, was oft zu genaueren Modellen bei gleicher Anzahl von Simulationspunkten führt. Optimale Pläne (wie D- oder I-Optimalität) werden eingesetzt, wenn das Surrogatmodell selbst ein spezifisches statistisches Modell ist (z.B. eine Polynom-Antwortfläche). Diese Methoden optimieren die Position der Stützpunkte gezielt, um die Parameter dieses Modells mit minimaler Unsicherheit zu schätzen. Die korrekte Wahl der Sampling-Strategie ist somit ein fundamentaler Schritt, der die Effizienz und die Vorhersagegüte des finalen Surrogatmodells direkt bestimmt."
pubDate: "Nov 4 2025"
heroImage: "/personal_blog/aikn.webp"
badge: "Latest"
---



# Methoden der Versuchsplanung: Optimal Design und Sobol-Sequenzen
*Author: Christoph Würsch, ICE, Eastern Switzerland University of Applied Sciences, OST*

Dieser Blog behandelt fundamentale Methoden der Versuchsplanung (Design of Experiments, DoE) und stellt zwei primäre Ansätze gegenüber: die modellbasierte optimale Versuchsplanung und modellunabhängige, raumfüllende Sampling-Strategien.
- Der erste Teil des Dokuments führt in die **optimale Versuchsplanung** ein, die darauf abzielt, einen Versuchsplan $\mathbf{X}$ für ein spezifisches statistisches Modell (z.B. Polynomregression) zu optimieren. Der Fokus liegt auf der **Informationsmatrix** $\mathbf{M} = \mathbf{X}^T \mathbf{X}$ und den davon abgeleiteten Optimalitätskriterien (D-, A-, I- und G-Optimalität), die jeweils unterschiedliche Aspekte der Modellgüte, wie die Parametervarianz oder die Vorhersagevarianz, optimieren.
- Der zweite Teil vergleicht drei zentrale **raumfüllende Sampling-Methoden** für die Erstellung von Surrogatmodellen, wenn *kein* a-priori-Modell angenommen wird. Es werden das einfache **Monte-Carlo-Sampling (MC)**, das **Latin Hypercube Sampling (LHS)** und **Quasi-Monte-Carlo-Methoden (QMC)** am Beispiel der **Sobol-Sequenz** detailliert. Der Vergleich hebt die Unterschiede in Uniformität, Konvergenzrate ($O(N^{-1/2})$ für MC vs. $O(N^{-1}(\log N)^s)$ für QMC) und Erweiterbarkeit hervor, wobei die Sobol-Sequenz die beste mehrdimensionale Raumabdeckung und Flexibilität bietet.
- Der dritte Teil, "Sobol-Sequenzen 'from scratch'", bietet eine detaillierte technische Implementierung der Sobol-Sequenz. Es wird der **Bratley-und-Fox-Algorithmus**  erklärt, der iterativ Sobol-Punkte generiert. Dieser Ansatz basiert auf primitiven Polynomen über dem Körper $\mathbb{F}_2$ , der Vorkalkulation von **Richtungszahlen** (Direction Numbers) und der effizienten Anwendung von **bitweisen XOR-Operationen**, um eine hochgradig uniforme Punktmenge zu erzeugen.


## Inhaltsverzeichnis

1.  [Optimale Versuchsplanung (Optimal Design)](#1-optimale-versuchsplanung-optimal-design)
    * [Die Informationsmatrix](#11-die-informationsmatrix)
    * [Optimalitätskriterien](#12-optimalitätskriterien)
    * [Modellbasiert vs. Raumfüllend](#13-modellbasiert-vs-raumfüllend)
2.  [Sampling-Methoden für die Versuchsplanung (DoE)](#2-sampling-methoden-für-die-versuchsplanung-doe)
    * [Pseudo-Zufalls-Sampling (Monte Carlo, MC)](#21-pseudo-zufalls-sampling-monte-carlo-mc)
    * [Latin Hypercube Sampling (LHS)](#22-latin-hypercube-sampling-lhs)
    * [Quasi-Monte-Carlo (QMC) / Sobol-Sequenz](#23-quasi-monte-carlo-qmc--sobol-sequenz)
    * [Tabellarischer Vergleich](#24-tabellarischer-vergleich)
3.  [Sobol-Sequenzen ''from scratch''](#3-sobol-sequenzen-from-scratch)
    * [Primitive Polynome](#31-primitive-polynome)
    * [Richtungszahlen (Direction Numbers)](#32-richtungszahlen-direction-numbers)
4.  [Algorithmus (Bratley & Fox)](#4-algorithmus-bratley--fox)
    * [Stufe 1: Vorkalkulation der Richtungszahlen (V)](#41-stufe-1-vorkalkulation-der-richtungszahlen-v)
    * [Stufe 2: Generierung der Sequenzpunkte](#42-stufe-2-generierung-der-sequenzpunkte)
    * [Python Implementierung (Bratley & Fox)](#43-python-implementierung-bratley--fox)



<br>
<br>


# 1 Optimale Versuchsplanung (Optimal Design)

Im Gegensatz zu raumfüllenden Versuchsplänen (wie Latin Hypercube oder Sobol-Sequenzen), die modellunabhängig den gesamten Raum abtasten, ist die **optimale Versuchsplanung** (Optimal Design) **modellbasiert**. Das Ziel ist es, einen Versuchsplan $\mathbf{X}$ zu konstruieren, der ''optimal'' für die Anpassung (Fitting) eines **spezifischen statistischen Modells** ist, typischerweise eines Polynommodells (z.B. linear, quadratisch oder mit Interaktionen).

## 1.1 Die Informationsmatrix

Der Kern der optimalen Versuchsplanung ist die **Informationsmatrix** $\mathbf{M}$. Für ein Standard-Regressionsmodell $y = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$, wobei $\mathbf{X}$ die Versuchsplanmatrix (Designmatrix) ist, ist die Informationsmatrix definiert als:

$$
\mathbf{M} = \mathbf{X}^T \mathbf{X}
$$

Die Inverse dieser Matrix, $(\mathbf{X}^T \mathbf{X})^{-1}$, ist proportional zur Varianz-Kovarianz-Matrix der geschätzten Modellparameter $\hat{\boldsymbol{\beta}}$.

Ein ''optimaler'' Plan ist einer, der eine skalare Funktion dieser Matrix (oder ihrer Inversen) maximiert oder minimiert. Die verschiedenen Kriterien (D, A, I, G) definieren, welche Funktion optimiert wird.

## 1.2 Optimalitätskriterien

Es gibt mehrere gängige Kriterien, um die ''Güte'' eines Versuchsplans basierend auf der Informationsmatrix $\mathbf{M}$ zu bewerten.

### D-Optimalität

> Ein D-optimaler Plan **maximiert die Determinante** der Informationsmatrix:
> 
> $$
> \max \left( \text{det}(\mathbf{X}^T \mathbf{X}) \right)
> $$


Dies ist äquivalent zur Minimierung der Determinante der Kovarianzmatrix $(\mathbf{X}^T \mathbf{X})^{-1}$. Geometrisch bedeutet dies, dass das **Volumen des Konfidenzellipsoids** für die Modellparameter $\boldsymbol{\beta}$ minimiert wird. D-Optimalität ist das am häufigsten verwendete Kriterium.

### A-Optimalität

> Ein A-optimaler Plan **minimiert die Spur** (Summe der Diagonalelemente) der *inversen* Informationsmatrix:
>
> $$
> \min \left( \text{trace}(\mathbf{X}^T \mathbf{X})^{-1} \right)
> $$

Da die Diagonalelemente der Kovarianzmatrix $\text{Var}(\hat{\beta}_i)$ entsprechen, minimiert die A-Optimalität die **durchschnittliche Varianz** der Parameterschätzungen $\hat{\boldsymbol{\beta}}$.

### I-Optimalität (oder V-Optimalität)

> Ein I-optimaler Plan **minimiert die durchschnittliche Vorhersagevarianz** über den gesamten Design-Space $\mathcal{D}$:
>
> $$
> \min \left( \int_{\mathcal{D}} \text{Var}(\hat{y}(\mathbf{x})) \, d\mathbf{x} \right)
> $$

Dieses Kriterium ist ideal, wenn das Hauptziel die präzise Vorhersage (Interpolation) innerhalb des Versuchsraums ist, und nicht notwendigerweise die präziseste Schätzung der einzelnen Parameter $\beta_i$.

### G-Optimalität

> Ein G-optimaler Plan folgt einem Minimax-Ansatz. Er **minimiert die maximale Vorhersagevarianz** im Design-Space $\mathcal{D}$:
> 
> $$
> \min \left( \max_{\mathbf{x} \in \mathcal{D}} \text{Var}(\hat{y}(\mathbf{x})) \right)
> $$

Dieser Ansatz ist sehr robust, da er darauf abzielt, den ''Worst-Case'' (den Punkt im Raum mit der höchsten Vorhersageunsicherheit) zu kontrollieren und zu minimieren.

## 1.3 Modellbasiert vs. Raumfüllend

* **Optimale Pläne (D, A, G, I):** Werden verwendet, wenn ein spezifisches Regressionsmodell (z.B. $y \sim \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2$) bekannt ist und effizient gefittet werden soll. Sie sind ideal für die Parameter-Screening und die Optimierung von Antwortflächen (Response Surface Methodology).

* **Raumfüllende Pläne (LHS, Sobol):** Werden verwendet, wenn *kein* Modell a priori angenommen wird. Sie sind ideal für die allgemeine Exploration, die Erstellung von Surrogatmodellen (z.B. Gauss-Prozesse, Neuronale Netze) oder für Sensitivitätsanalysen.

<br>
<br>


# 2. Sampling-Methoden für die Versuchsplanung (DoE)

Die Erstellung eines **Surrogatmodells** (oder Metamodells) hat zum Ziel, eine rechenintensive ''Black-Box''-Funktion $f(\mathbf{X})$ durch ein rechengünstiges Näherungsmodell $\hat{f}(\mathbf{X})$ zu ersetzen. Diese Funktion $f$ ist oft das Ergebnis einer komplexen Simulation (z.B. FEM, CFD). Sei unser Eingaberaum (Design-Space) $\mathcal{D} \subset \R^s$, wobei $s$ die Anzahl der Dimensionen (Design-Parameter) ist. Für die Analyse normieren wir diesen Raum typischerweise auf den $s$-dimensionalen Einheits-Hyperwürfel $\mathcal{D} = [0, 1)^s$.

Ein **Versuchsplan** (Design of Experiments, DoE) ist eine Menge von $N$ Stützpunkten $\mathbf{X} = \{\mathbf{X}_1, \dots, \mathbf{X}_N\}$, an denen die teure Funktion $f$ ausgewertet wird. Das Ziel ist, diese $N$ Punkte so zu wählen, dass sie den Raum $\mathcal{D}$ ''optimal'' abdecken (engl. *space-filling*), um ein möglichst genaues Surrogatmodell $\hat{f}$ zu trainieren.

''Optimal'' bedeutet hierbei vor allem:

* **Uniformität:** Die Punkte sollen den Raum gleichmässig füllen, ohne Cluster (Ballungen) oder grosse Lücken (Lücken).
* **Orthogonalität:** Die Punkte sollten möglichst unkorreliert sein, um eine Verwechslung der Einflüsse verschiedener Parameter zu vermeiden.

Im Folgenden werden drei fundamentale Sampling-Strategien mathematisch und konzeptionell detailliert.

## 2.1 Pseudo-Zufalls-Sampling (Monte Carlo, MC)
Das Monte-Carlo-Sampling (MC) ist die fundamentalste stochastische Methode. Ein Versuchsplan $\mathbf{X}mat$ mit $N$ Punkten in $s$ Dimensionen wird generiert, indem $N \times s$ unabhängige Zufallszahlen $u_{i,j}$ aus der stetigen Gleichverteilung $\mathcal{U}(0, 1)$ gezogen werden.

Jede Koordinate jedes Punktes $\mathbf{X}_i = (x_{i,1}, \dots, x_{i,s})$ ist unabhängig von allen anderen:
$$
x_{i,j} \sim \mathcal{U}(0, 1) \quad \forall i \in \{1,\dots,N\}, j \in \{1,\dots,s\}
$$

Stellen Sie sich vor, Sie werfen eine Handvoll Sand (die $N$ Punkte) auf eine quadratische Platte (den 2D-Design-Space). Die Körner landen völlig zufällig. Es ist unvermeidlich, dass einige Bereiche dicht bedeckt sind (Cluster) und andere Bereiche fast leer bleiben (Lücken). Das zentrale Gütemass für die Gleichverteilung ist die **Diskrepanz** $D_N^*$. Sie misst die maximale Abweichung zwischen dem ''Volumen'' eines beliebigen Teilquaders (gemessen an der Anzahl der Punkte, die in ihn fallen) und seinem tatsächlichen geometrischen Volumen.

Für MC-Methoden ist der *erwartete* Fehler bei der numerischen Integration (ein verwandtes Problem) durch die Koksma-Hlawka-Ungleichung beschränkt. In der Praxis konvergiert der probabilistische Fehler der MC-Integration mit einer Rate von:
$$
\text{Fehler}_{\text{MC}} = O\left(\frac{1}{\sqrt{N}}\right)
$$
Diese Konvergenzrate ist (bemerkenswerterweise) **unabhängig von der Dimension $s$**, aber sie ist sehr langsam. Um den Fehler zu halbieren, muss die Anzahl der Punkte $N$ vervierfacht werden.

* **Vorteile:** Extrem einfach zu implementieren, probabilistisch unvoreingenommen (unbiased), trivial erweiterbar (neue Punkte können hinzugefügt werden, ohne die Statistik zu verletzen).
* **Nachteile:** Sehr ineffizient. Erzeugt unvermeidlich Cluster und Lücken. Für eine gute Raumabdeckung ist ein sehr grosses $N$ erforderlich.

## 2.2 Latin Hypercube Sampling (LHS)
LHS ist eine **stratifizierte** (geschichtete) Zufallsstichprobe, die das Clustering-Problem von MC in den 1D-Projektionen löst. Die Generierung eines LHS-Plans mit $N$ Punkten in $s$ Dimensionen folgt drei Schritten:

1.  **Stratifizierung:** Jede der $s$ Dimensionen wird in $N$ gleichwahrscheinliche Intervalle (Strata) unterteilt:
    $I_k = \left[\frac{k-1}{N}, \frac{k}{N}\right)$ für $k = 1, \dots, N$.
    
2.  **Permutation:** Für jede Dimension $j$ wird eine zufällige Permutation (Umsortierung) $\pi_j$ der Zahlen $\{1, \dots, N\}$ erstellt.
    
3.  **Generierung:** Der $i$-te Sample-Punkt $\mathbf{X}_i$ wird nun konstruiert. Die $j$-te Koordinate des $i$-ten Punktes ($x_{i,j}$) wird generiert, indem ein zufälliger Punkt aus dem $\pi_j(i)$-ten Intervall der $j$-ten Dimension gezogen wird.
    $$
    x_{i,j} = \frac{\pi_j(i) - 1 + U_{i,j}}{N} \quad \text{wobei } U_{i,j} \sim \mathcal{U}(0, 1)
    $$

Stellen Sie sich ein $N \times N$ Sudoku-Gitter (für 2D) oder $N$ Schachu-Türme auf einem $N \times N$ Brett vor. Das Ziel ist es, die Türme so zu platzieren, dass sich keine zwei Türme bedrohen. Das Ergebnis ist, dass in **jeder Zeile und jeder Spalte genau ein Turm** steht.

LHS stellt sicher, dass, wenn man den $s$-dimensionalen Raum auf eine beliebige einzelne Achse (Dimension) projiziert, **genau ein Punkt** in jedes der $N$ Intervalle fällt. LHS garantiert eine perfekte, gleichmässige Abdeckung in allen 1D-Projektionen. Dies ist ein enormer Vorteil gegenüber MC.

* **Vorteile:** Deutlich bessere Raumfüllung als MC. Erzwingt die Abtastung der Extrembereiche (der ''Ränder'' des Design-Space). Reduziert die Varianz bei der Schätzung von Modell-Outputs.
* **Nachteile:** Garantiert **keine** gute *mehrdimensionale* Uniformität. Die Punkte können sich immer noch auf Hyper-Diagonalen ansammeln, was zu unerwünschten Korrelationen im Versuchsplan führt. Die Erweiterbarkeit ist nicht gegeben; das Hinzufügen von $M$ neuen Punkten zu einem $N$-Punkte-Plan ergibt keinen validen $(N+M)$-Punkte-LHS-Plan.

## 2.3 Quasi-Monte-Carlo (QMC) / Sobol-Sequenz
QMC-Methoden verwenden **deterministische** Sequenzen mit **geringer Diskrepanz** (Low-Discrepancy Sequences, LDS), um die Nachteile von MC zu überwinden. Die Sobol-Sequenz ist eine der bekanntesten digitalen $(t, m, s)$-Sequenzen. Die Sobol-Sequenz ist keine Zufallsstichprobe, sondern eine hochgradig strukturierte, unendliche Sequenz von Punkten. Ihre Konstruktion basiert auf der Arithmetik im endlichen Körper $\mathbb{F}_2 = \{0, 1\}$ (wobei Addition $\equiv$ bitweises XOR $\oplus$ ist).

1.  **Richtungszahlen:** Für jede Dimension $j$ (bis $s$) wird ein Satz von *Richtungszahlen* $V_{j,k} \in (0, 1)$ aus primitiven Polynomen über $\mathbb{F}_2$ generiert.
    
2.  **Digitale Konstruktion:** Um den $n$-ten Punkt $\mathbf{X}_n = (x_{n,1}, \dots, x_{n,s})$ zu finden, wird der Index $n$ binär dargestellt: $n = (b_k \dots b_2 b_1)_2$.
    
3.  **XOR-Operation:** Die $j$-te Koordinate $x_{n,j}$ wird durch eine bitweise XOR-Summe der Richtungszahlen gebildet, die den '1'-Bits in $n$ entsprechen:
    $$
    x_{n,j} = (b_1 \cdot V_{j,1}) \oplus (b_2 \cdot V_{j,2}) \oplus \dots \oplus (b_k \cdot V_{j,k})
    $$
    (Der im Python-Code gezeigte Algorithmus ist eine *schnellere, iterative* Variante hiervon, die $x_n$ aus $x_{n-1}$ berechnet.)

Stellen Sie sich das systematische Kacheln eines Badezimmerbodens vor. Sie werfen die Kacheln nicht zufällig (MC). Sie legen sie auch nicht nur in sauberen Reihen (was zu Korrelationen führt). Sie verwenden ein komplexes, deterministisches, aber nicht-periodisch wirkendes Muster (ähnlich einer Penrose-Parkettierung), das darauf ausgelegt ist, **niemals Lücken** zu hinterlassen. Jeder neue Punkt (Kachel) wird gezielt in die grösste verbleibende Lücke gelegt.

QMC-Sequenzen sind explizit darauf ausgelegt, die Diskrepanz $D_N^*$ zu minimieren. Der Fehler bei der numerischen Integration konvergiert für QMC theoretisch mit:
$$
\text{Fehler}_{\text{QMC}} = O\left(\frac{(\log N)^s}{N}\right)
$$
Diese Konvergenz ist *deutlich* schneller als $O(N^{-1/2})$ von MC, solange die Dimension $s$ nicht zu gross ist (der Faktor $(\log N)^s$ ist der ''Fluch der Dimensionalität'' für QMC).

* **Vorteile:** Beste Uniformität und Raumfüllung im mehrdimensionalen Raum. Schnellste Konvergenzrate für Integration (und oft beste Surrogatmodell-Güte) bei niedrigen bis moderaten Dimensionen.
* **Erweiterbarkeit (Extensibility):** Ein herausragender Vorteil. Eine Sobol-Sequenz ist unendlich. Man kann $N=100$ Punkte berechnen, das Modell testen und bei Bedarf $N=100$ *weitere* Punkte hinzufügen. Die resultierende Menge von 200 Punkten behält die Eigenschaft der geringen Diskrepanz bei.
* **Nachteile:** Deterministisch. Könnte theoretisch mit einer periodischen Zielfunktion ''kollidieren''. Der theoretische Vorteil nimmt in sehr hohen Dimensionen ($s > 20 \text{-} 40$) ab.

## 2.4 Tabellarischer Vergleich
**Tabelle 1:** Mathematischer und praktischer Vergleich der Sampling-Methoden

| Eigenschaft | Monte Carlo (MC) | Latin Hypercube (LHS) | Sobol-Sequenz (QMC) |
| :--- | :--- | :--- | :--- |
| **Typ** | Pseudo-Zufällig | Stratifiziert-Zufällig | Deterministisch (Quasi-Zufall) |
| **Ziel** | Probabilistische Abdeckung | Gleichverteilung der 1D-Projektionen | Minimierung der $s$-dim. Diskrepanz |
| **Raumfüllung** | Gering (Cluster/Lücken) | Mittel (Keine 1D-Cluster) | **Sehr Hoch (Beste Uniformität)** |
| **Erweiterbarkeit** | **Ja** (trivial) | Nein (Neugenerierung) | **Ja** (Haupteigenschaft) |
| **Konvergenzrate** (Integrationsfehler) | $O(N^{-1/2})$ (langsam, $s$-unabhängig) | Besser als MC, variabel | $\approx O(N^{-1} (\log N)^s)$ **(sehr schnell bei kleinem $s$)** |
| **Ideal für** | Robuste Tests, sehr hohe Dimensionen ($s \gg 50$) | Standard-DoE, wenn $N$ feststeht, UQ | Surrogatmodell-Training, Sensitivitätsanalyse (Sobol-Ind.) |


<br>
<br>


# 3. Sobol-Sequenzen ''from scratch''
Eine *Sobol-Sequenz* ist eine sogenannte **Quasi-Zufalls-Sequenz** (auch *low-discrepancy sequence* oder Sequenz mit geringer Diskrepanz genannt). Im Gegensatz zu *Pseudo*-Zufallszahlen (PRNGs), wie sie in Standard-Monte-Carlo-Simulationen (MC) verwendet werden, zielt eine Quasi-Zufalls-Sequenz nicht darauf ab, statistischen Zufall zu imitieren. Das Hauptziel einer Sobol-Sequenz ist es, einen $s$-dimensionalen Raum (oft den Einheits-Hyperwürfel $[0, 1)^s$) so **gleichmässig wie möglich** abzudecken.

* **Pseudo-Zufall (Monte Carlo):** Erzeugt tendenziell Lücken (areas of under-sampling) und Cluster (areas of over-sampling), besonders bei geringer Punktzahl.
* **Quasi-Zufall (Sobol):** Ist **deterministisch** so konstruiert, dass jeder neue Punkt gezielt in die grösste existierende Lücke ''fällt''.

Diese Eigenschaft der gleichmässigen Abdeckung führt zu einer signifikant schnelleren Konvergenz bei numerischen Integrationen (Quasi-Monte-Carlo, QMC) und einer effizienteren Abtastung von Design-Spaces im *Design of Experiments* (DoE). Die Sobol-Sequenz ist eine **digitale Sequenz** (Basis 2). Ihre Konstruktion basiert auf der Arithmetik im endlichen Körper (Galois-Feld) $\mathbb{F}_2$, der nur aus den Elementen $\{0, 1\}$ besteht, wobei die Addition dem **bitweisen XOR** ($\oplus$) entspricht.

## 3.1 Primitive Polynome

Die Grundlage für jede Dimension $j$ der Sobol-Sequenz ist ein sorgfältig ausgewähltes **primitives Polynom** über $\mathbb{F}_2$. Ein solches Polynom vom Grad $d$ hat die Form:
$$
P(x) = x^d + a_1 x^{d-1} + a_2 x^{d-2} + \dots + a_{d-1} x + 1
$$
wobei alle Koeffizienten $a_k$ entweder 0 oder 1 sind ($a_k \in \mathbb{F}_2$). Diese Polynome sind die ''Generatoren'' für die Sequenz in dieser Dimension.

## 3.2 Richtungszahlen (Direction Numbers)

Aus jedem primitiven Polynom wird ein Satz von **Richtungszahlen** $V_{j,i}$ abgeleitet. Dies sind $L$-Bit-Integer (in der Python-Implementierung $L=32$), die als die ''magischen Konstanten'' des Algorithmus dienen.

Die Richtungszahlen werden über eine Rekurrenzrelation generiert, die direkt vom Polynom $P(x)$ abhängt:

1.  **Initialwerte:** Die ersten $d$ Richtungszahlen $m_1, \dots, m_d$ werden als ungerade ganze Zahlen $< 2^i$ gewählt. Diese sind die Konstanten, die im Python-Code als `SOBOL_INIT_M` gespeichert sind.
    
2.  **Rekurrenzrelation:** Für $i > d$ wird $m_i$ über eine XOR-basierte Rekurrenz berechnet, die die Koeffizienten $a_k$ des Polynoms nutzt:
    $$
    m_i = (2a_1 m_{i-1}) \oplus (2^2 a_2 m_{i-2}) \oplus \dots \oplus (2^{d-1} a_{d-1} m_{i-d+1}) \oplus (2^d m_{i-d}) \oplus m_{i-d}
    $$

In der Praxis (und im Python-Code) werden diese $m_i$ als $L$-Bit-Integer $V_{j,i}$ gespeichert, die bereits an die richtige Bit-Position verschoben (left-shifted) sind, um die Division durch $2^L$ am Ende zu vereinfachen.

# 4. Algorithmus (Bratley & Fox)
Der Algorithmus von Bratley \& Fox ist eine hocheffiziente Methode zur Erzeugung von Sobol-Punkten. Er ist deterministisch und basiert vollständig auf bitweisen Operationen (XOR, Bit-Shifts).

* **Stufe 1 (Setup):** Nutzt primitive Polynome und eine XOR-Rekurrenz, um die ''magischen'' Richtungszahlen $V$ für jede Dimension zu berechnen.
* **Stufe 2 (Iteration):** Nutzt den Index der rechtesten Null ($c$) von $n-1$, um $X_n$ aus $X_{n-1}$ und der einzelnen Richtungszahl $V_c$ zu berechnen.

Das Ergebnis ist eine Sequenz, die den $s$-dimensionalen Raum hervorragend gleichmässig abdeckt und sich ideal für QMC-Methoden und die Exploration von Design-Spaces eignet. Der folgende Python-Code verwendet **nicht** die naive Implementierung (die den Index $n$ direkt mit allen Richtungszahlen $V_{j,i}$ per XOR verknüpft). Stattdessen nutzt er einen **schnellen iterativen Algorithmus** (basierend auf Bratley und Fox, ACM Trans. Math. Softw. 659), der den $n$-ten Punkt $S_n$ sehr effizient aus dem vorherigen Punkt $S_{n-1}$ berechnet. Der Algorithmus besteht aus zwei Stufen.

## 4.1 Stufe 1: Vorkalkulation der Richtungszahlen (V)

Bevor die Sequenz generiert wird, berechnet der Code ein 2D-Array $\text{V[dimensions][MAX\_BITS]}$.
$$V[j][i] \equiv V_{j,i}$$
Dies ist die $i$-te $L$-Bit-Richtungszahl für die $j$-te Dimension.

1.  **Initialwerte ($i < d$):**
    Die ersten $d$ Werte (aus `SOBOL_INIT_M`) werden geladen und an die korrekte Bit-Position (ganz links) verschoben.
    $$
    V_{j,i} = m_i \ll (L - 1 - i) \quad \text{for } i = 0, \dots, d-1
    $$
    
2.  **Rekurrenz ($i \ge d$):**
    Die restlichen $V_{j,i}$ werden mit der bit-verschobenen Version der Rekurrenzrelation (siehe oben) berechnet. Der Python-Code implementiert dies als:
    
    $$V_{j,i} = (V_{j,i-d}) \oplus (V_{j,i-d} \gg d) \oplus \bigoplus_{k=1}^{d-1} (a_k \cdot V_{j,i-d+k})$$
    
    (Wobei $a_k \cdot V$ bedeutet: $V$, falls $a_k=1$, und $0$, falls $a_k=0$. Dies wird im Code durch die Bit-Maskierung von `temp_a` erreicht.)


![Stufe 1: Vorkalkulation der Richtungszahlen](/personal_blog/algo1_richtungszahlen.png)



## 4.2 Stufe 2: Generierung der Sequenzpunkte

Der Kern des schnellen Algorithmus ist die iterative Erzeugung des $n$-ten Punktes $S_n$.

Wir verwalten einen $s$-dimensionalen Vektor von $L$-Bit-Integern, $X_n$, der den ''Zustand'' der Sequenz darstellt. Der normalisierte Punkt $S_n$ ist einfach $S_n = X_n / 2^L$.

1.  **Initialisierung:**
    Der nullte Punkt $S_0$ ist der Ursprung.
    $$
    X_0 = (0, 0, \dots, 0)
    $$
    
2.  **Iteration (für $n = 1, 2, \dots, N$):**
    Um $X_n$ aus $X_{n-1}$ zu erhalten, wird eine **einzige** Richtungszahl pro Dimension verwendet.
    $$
    X_n = X_{n-1} \oplus V_c
    $$
    
3.  **Finden des Index $c$:**
    Der entscheidende Schritt ist die Bestimmung des Index $c$. Dieser Index $c$ ist definiert als:
    
    **Der Index des rechtesten Null-Bits (rightmost zero-bit) in der Binärdarstellung von $n-1$.**
    
    *Beispiele:*
    * $n=1$: $n-1 = 0$ (0b000). Rechteste 0 ist bei Index $c=0$.
    * $n=2$: $n-1 = 1$ (0b001). Rechteste 0 ist bei Index $c=1$.
    * $n=3$: $n-1 = 2$ (0b010). Rechteste 0 ist bei Index $c=0$.
    * $n=4$: $n-1 = 3$ (0b011). Rechteste 0 ist bei Index $c=2$.
    * $n=5$: $n-1 = 4$ (0b100). Rechteste 0 ist bei Index $c=0$.
    
    (Der Python-Code findet $c$ durch die `while (k & 1)`-Schleife.)
    
4.  **Anwendung des XOR:**
    Der $n$-te Punkt $X_n$ wird nun für **jede Dimension $j$** separat berechnet, wobei **derselbe Index $c$** verwendet wird:
    $$
    X_{n,j} = X_{n-1, j} \oplus V_{j,c}
    $$
    Der Unterschied zwischen den Dimensionen entsteht also, weil $V_{j,c}$ (das Array der Richtungszahlen) für jede Dimension $j$ einzigartig ist.
    
5.  **Normalisierung:**
    Der finale Punkt $S_n$ für die Ausgabe wird durch Normalisierung gewonnen:
    $$
    S_{n,j} = X_{n,j} \cdot (1.0 / 2^L)
    $$


![Stufe 2: Generierung der Sequenzpunkte](/personal_blog/algo2_sobol.png)






## 4.3 Python Implementierung (Bratley & Fox)

```python
import numpy as np
 
 # --- KONSTANTEN (Die ''Magischen Zahlen'' von Joe & Kuo, bis dim 6) ---
 # Diese Daten sind die ''Black Box'', die aus der mathematischen Theorie stammt.
 # new-joe-kuo-6-21201.txt
 
 # d = Grad des primitiven Polynoms
 SOBOL_DEGREES = [1, 2, 3, 3, 4, 4]
 # a = Koeffizienten des Polynoms (als einzelne Ganzzahl kodiert)
 SOBOL_COEFFS = [0, 1, 1, 3, 1, 3]
 # m = Die initialen Richtungszahlen (m_1, ..., m_d)
 SOBOL_INIT_M = [
 [1],
 [1, 3],
 [1, 3, 1],
 [1, 1, 1],
 [1, 1, 3, 3],
 [1, 3, 5, 13],
 ]
 # Maximale Anzahl an Bits (bestimmt die Praezision und Periodizitaet)
 MAX_BITS = 32
 # Normalisierungsfaktor (um von Integer auf [0, 1) zu kommen)
 NORMALIZER = 1.0 / (2**MAX_BITS)
 
 
 def sobol_sequence_from_scratch(num_points: int, dimensions: int) -> np.ndarray:
	 ''''''
	 Generiert eine Sobol-Sequenz ''from scratch'' unter expliziter
	 Verwendung von bitweisen Operationen und Richtungszahlen.
	 Basiert auf den Daten von S. Joe und F. Y. Kuo.
	 ''''''
	 
	 if dimensions > len(SOBOL_DEGREES):
		 raise ValueError(
		 f''Maximale Dimension {len(SOBOL_DEGREES)} ueberschritten. ''
		 ''Fuer mehr Dimensionen muessen die Konstanten erweitert werden.''
		 )
	 
	 # --- 1. VORBEREITUNG: GENERIERE DIE RICHTUNGSZAHLEN (V) ---
	 # V[j][i] ist die i-te Richtungszahl fuer die j-te Dimension.
	 # Wir berechnen sie aus den Konstanten (Polynomen).
	 
	 V = np.zeros((dimensions, MAX_BITS), dtype=np.uint64)
	 
	 for j in range(dimensions):
		 d = SOBOL_DEGREES[j]
		 a = SOBOL_COEFFS[j]
		 m = SOBOL_INIT_M[j]
	 
		 # 1a. Setze die initialen Werte (m_1 ... m_d)
		 # Wir muessen sie an die richtige Bit-Position schieben
		 for i in range(d):
			 # m_i wird zu V[j][i], nach links geshiftet
			 V[j, i] = np.uint64(m[i]) << (MAX_BITS - 1 - i)
		 
		 # 1b. Generiere die restlichen V's (i > d) ueber die Rekurrenzrelation
		 # V_i = a_1 V_{i-1} ^ a_2 V_{i-2} ^ ... ^ a_d V_{i-d} ^ (V_{i-d} >> d)
		 for i in range(d, MAX_BITS):
			 # V_{i-d} ^ (V_{i-d} >> d)
			 V[j, i] = V[j, i - d] ^ (V[j, i - d] >> d)
			 
			 # Wende die Koeffizienten 'a' an
			 temp_a = a
			 for k in range(d - 1):
				 # Wenn das k-te Bit von 'a' gesetzt ist, XOR anwenden
				 if (temp_a & 1):
					 V[j, i] = V[j, i] ^ V[j, i - (d - 1) + k]
					 temp_a >>= 1
	 
	 # --- 2. GENERIERUNG DER SEQUENZ ---
	 
	 # `X` speichert den aktuellen Punkt als Integer-Wert
	 X = np.zeros(dimensions, dtype=np.uint64)
	 
	 # `points` speichert die finale Sequenz (normalisierte Floats)
	 points = np.zeros((num_points, dimensions))
	 
	 # Der erste Punkt (i=0) ist immer der Ursprung [0, 0, ...]
	 # (bleibt bei points[0] als Nullen stehen)
	 
	 for i in range(1, num_points):
		 # 2a. Finde den Index 'c' des rechtesten Null-Bits in (i-1)
		 # Beispiel: i=1 -> (i-1)=0 (0b000) -> c=0
		 #          i=2 -> (i-1)=1 (0b001) -> c=1
		 #          i=3 -> (i-1)=2 (0b010) -> c=0
		 #          i=4 -> (i-1)=3 (0b011) -> c=2
		 
		 c = 0
		 k = i - 1  # (i-1) als Binaerzahl betrachten
		 while (k & 1):
		 	k >>= 1
		 	c += 1
		 	if c >= MAX_BITS:
		 		# Sollte bei normaler Nutzung nicht passieren
		 		print(f''Warnung: MAX_BITS ({MAX_BITS}) erreicht bei Punkt {i}'')
		 		continue
	 
	 # 2b. EXPLIZITE OPERATION: Generiere den neuen Punkt X_i
	 # X_i = X_{i-1} XOR V[c]
	 # (Wir wenden dies auf jede Dimension an)
	 for j in range(dimensions):
	 	X[j] = X[j] ^ V[j, c]
	 
	 # 2c. Speichere den normalisierten Punkt
	 # (Integer-Wert geteilt durch 2^MAX_BITS)
	 points[i, :] = X * NORMALIZER
	 
 return points
```

