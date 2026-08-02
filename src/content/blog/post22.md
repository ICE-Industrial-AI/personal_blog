---
title: "Dichtefunktionaltheorie (DFT): Das Arbeitspferd der modernen Materialwissenschaften und Quantenchemie"
description: "Die Dichtefunktionaltheorie (DFT) ist das Arbeitspferd der modernen Materialwissenschaften und Quantenchemie. Für Ingenieure, die neue Halbleiter, Batteriematerialien oder Katalysatoren entwickeln, ist ein grundlegendes Verständnis der DFT unerlässlich. Dieses Kapitel führt Schritt für Schritt in die theoretischen Grundlagen ein, beleuchtet die quantenmechanische Natur der Teilchen und zeigt, wie sich die Kohn-Sham-Gleichungen mathematisch aus dem Variationsprinzip ableiten lassen."
pubDate: "August 2 2026"
heroImage: "/personal_blog/aikn.webp"
---
 
*Author: Prof. Dr. Christoph Würsch, Institute for Computational Engineering ICE, Eastern Switzerland University of Applied Sciences OST*

[Download:](/personal_blog/DFT_Blog_WUCH.pdf)

## Abstract

Die Dichtefunktionaltheorie (DFT) ist das Arbeitspferd der modernen Materialwissenschaften und Quantenchemie. Für Ingenieure, die neue Halbleiter, Batteriematerialien oder Katalysatoren entwickeln, ist ein grundlegendes Verständnis der DFT unerlässlich. Dieses Kapitel führt Schritt für Schritt in die theoretischen Grundlagen ein, beleuchtet die quantenmechanische Natur der Teilchen und zeigt, wie sich die Kohn-Sham-Gleichungen mathematisch aus dem Variationsprinzip ableiten lassen.



## Inhaltsverzeichnis

1. [Das Vielteilchenproblem und der Hamilton-Operator](#1-das-vielteilchenproblem-und-der-hamilton-operator)
2. [Fermionen, Bosonen und das Pauli-Prinzip](#2-fermionen-bosonen-und-das-pauli-prinzip)
   - [2.1 Die Slater-Determinante](#21-die-slater-determinante)
3. [Der Kohn-Sham-Ansatz (1965)](#3-der-kohn-sham-ansatz-1965)
   - [3.1 Die Variationsableitung der Kohn-Sham-Gleichungen](#31-die-variationsableitung-der-kohn-sham-gleichungen)
4. [Self-Consistent-Field (SCF) Iteration](#4-self-consistent-field-scf-iteration)
   - [4.1 Approximationen für E_xc (Jacob's Ladder)](#41-approximationen-für-e_xc-jacobs-ladder)
5. [John Pople und der Durchbruch in der Chemie: Gaussian](#5-john-pople-und-der-durchbruch-in-der-chemie-gaussian)



## 1. Das Vielteilchenproblem und der Hamilton-Operator

In der klassischen Quantenmechanik wird der Zustand eines Systems durch die Vielteilchen-Schrödinger-Gleichung beschrieben. Für $N$ Elektronen in einem Molekül oder Festkörper hängt die Wellenfunktion $\Psi(\mathbf{r}_1, \mathbf{r}_2, \dots, \mathbf{r}_N)$ von $3N$ Raumkoordinaten ab.

Dank der **Born-Oppenheimer-Näherung** nehmen wir an, dass die schweren Atomkerne im Vergleich zu den leichten, schnellen Elektronen praktisch stillstehen. Der elektronische Hamilton-Operator (in atomaren Einheiten) lautet:

$$
\hat{H}_{\text{el}} = \hat{T} + \hat{V}_{\text{ext}} + \hat{V}_{\text{ee}}
$$

Wir können die drei Terme physikalisch und mathematisch aufschlüsseln:

**A) Die kinetische Energie der Elektronen ($\hat{T}$)**

$$
\hat{T} = -\frac{1}{2} \sum_{i=1}^N \nabla_i^2
$$

Der Laplace-Operator misst die Krümmung der Wellenfunktion. Je stärker gekrümmt (bzw. räumlich eingeschränkt) eine Wellenfunktion ist, desto höher ist die kinetische Energie des Elektrons. Dies spiegelt die Heisenbergsche Unschärferelation wider.

**B) Das externe Potential ($\hat{V}_{\text{ext}}$)**

$$
\hat{V}_{\text{ext}} = \sum_{i=1}^N v_{\text{ext}}(\mathbf{r}_i) = - \sum_{i=1}^N \sum_{A=1}^M \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|}
$$

Dies ist ein Einteilchen-Operator, der die elektrostatische Coulomb-Anziehung beschreibt, die jedes Elektron durch alle Atomkerne erfährt. Aus Sicht der DFT ist dies das „externe" Feld, das die Struktur des Systems definiert (z. B. ob es sich um ein Wassermolekül oder einen Kupferkristall handelt).

**C) Die Elektron-Elektron-Wechselwirkung ($\hat{V}_{\text{ee}}$)**

$$
\hat{V}_{\text{ee}} = \sum_{i < j} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}
$$

Dieser Zweiteilchen-Operator ist der Ursprung aller Komplexität im Vielteilchenproblem. Da die Position von Elektron $i$ durch die Position von Elektron $j$ beeinflusst wird, können die Differentialgleichungen nicht separiert werden.

Für reale Systeme wächst der Berechnungsaufwand dieser $3N$-dimensionalen Funktion exponentiell und wird selbst für Supercomputer unlösbar. Die brillante Idee von Walter Kohn war es, anstelle der komplexen Wellenfunktion die dreidimensionale **Elektronendichte** $n(\mathbf{r})$ als fundamentale Variable zu nutzen.



## 2. Fermionen, Bosonen und das Pauli-Prinzip

Bevor wir zur praktischen Lösung der DFT kommen, müssen wir die quantenmechanische Natur der Elektronen berücksichtigen. In der Quantenfeldtheorie wird das Verhalten von Teilchen durch ihren Eigendrehimpuls, den **Spin**, diktiert. Teilchen mit einem **halbzahligen Spin** ($s = 1/2, 3/2, \dots$) nennt man **Fermionen**. Dazu gehören Elektronen, Protonen und Neutronen. Die Wellenfunktion eines Systems aus identischen Fermionen muss **total antisymmetrisch** bezüglich des Austauschs der Koordinaten (Ort $\mathbf{r}$ und Spin $\sigma$) zweier beliebiger Teilchen sein. Wenn man die Koordinaten $\mathbf{x} = (\mathbf{r}, \sigma)$ von Teilchen $1$ und $2$ vertauscht, muss die Wellenfunktion das Vorzeichen wechseln:

$$
\Psi(\mathbf{x}_1, \mathbf{x}_2, \dots) = - \Psi(\mathbf{x}_2, \mathbf{x}_1, \dots)
$$

Das **Pauli-Ausschlussprinzip** ist eine direkte mathematische Konsequenz daraus. Befinden sich zwei Fermionen im exakt selben Zustand ($\mathbf{x}_1 = \mathbf{x}_2$), so gilt $\Psi = -\Psi$, woraus zwingend $\Psi = 0$ folgt. Dieser Zustand existiert physikalisch nicht. Dies zwingt Elektronen in höhere Energieschalen und verhindert, dass Festkörper in sich zusammenstürzen.

Teilchen mit einem **ganzzahligen Spin** ($s = 0, 1, 2, \dots$) nennt man **Bosonen** (z. B. Photonen oder Cooper-Paare in Supraleitern). Für sie muss die Wellenfunktion **total symmetrisch** sein:

$$
\Psi(\mathbf{x}_1, \mathbf{x}_2, \dots) = + \Psi(\mathbf{x}_2, \mathbf{x}_1, \dots)
$$

Da kein Vorzeichenwechsel stattfindet, entfällt das Pauli-Prinzip. Unendlich viele Bosonen können denselben Quantenzustand einnehmen, was zu makroskopischen Quantenphänomenen wie Lasern oder Bose-Einstein-Kondensaten führt.

### 2.1 Die Slater-Determinante

Um die geforderte Antisymmetrie für Elektronen mathematisch umzusetzen, nutzt man die **Slater-Determinante**. Sie ordnet die Spin-Orbitale $\chi_j$ in einer Determinantenform an:

$$
\Psi(\mathbf{x}_1, \dots, \mathbf{x}_N) = \frac{1}{\sqrt{N!}}
\begin{vmatrix}
\chi_1(\mathbf{x}_1) & \chi_2(\mathbf{x}_1) & \cdots & \chi_N(\mathbf{x}_1) \\
\chi_1(\mathbf{x}_2) & \chi_2(\mathbf{x}_2) & \cdots & \chi_N(\mathbf{x}_2) \\
\vdots & \vdots & \ddots & \vdots \\
\chi_1(\mathbf{x}_N) & \chi_2(\mathbf{x}_N) & \cdots & \chi_N(\mathbf{x}_N)
\end{vmatrix}
$$

Diese mathematische Struktur erzwingt die Physik:

- **Antisymmetrie:** Vertauscht man zwei Zeilen, wechselt die Determinante ihr Vorzeichen.
- **Pauli-Prinzip:** Sind zwei Elektronen im exakt gleichen Zustand (zwei Spalten identisch), wird die Determinante Null.



## 3. Der Kohn-Sham-Ansatz (1965)

Die gesamte DFT fusst auf zwei scheinbar einfachen, aber extrem mächtigen mathematischen Sätzen, die 1964 von Hohenberg und Kohn formuliert wurden:

**Theorem 1: Eineindeutigkeit**

Das externe Potential $v_{\text{ext}}(\mathbf{r})$ ist durch die Grundzustandsdichte $n(\mathbf{r})$ eindeutig bestimmt. Die gesamte Information steckt also bereits in $n(\mathbf{r})$.

**Theorem 2: Variationsprinzip**

Für das Energiefunktional $E[n]$ gilt, dass jede Testdichte $\tilde{n}(\mathbf{r})$, die die korrekte Teilchenzahl $N$ liefert, eine Energie erzeugt, die stets grösser oder gleich der exakten Grundzustandsenergie $E_0$ ist: $E[\tilde{n}] \ge E_0$.

<p align="center"><img src="/personal_blog/Walter-Kohn.png" alt="Walter Kohn (1923–2016) war ein österreichisch-US-amerikanischer theoretischer Physiker." width="520"></p>

*Abbildung 1: Walter Kohn (1923–2016) war ein österreichisch-US-amerikanischer theoretischer Physiker. Er wurde 1998 mit dem Nobelpreis für Chemie für die bahnbrechende Entwicklung der Dichtefunktionaltheorie (DFT) ausgezeichnet, welche die computergestützte Berechnung von Quantenstrukturen in der Chemie und Materialwissenschaft revolutionierte.*

> ### Walter Kohn – Ein Leben zwischen Flucht und Nobelpreis
>
> Walter Kohns (1923–2016) Lebensgeschichte ist ebenso bemerkenswert wie seine wissenschaftliche Leistung. Seine Reise zur Dichtefunktionaltheorie war von tiefen historischen Einschnitten geprägt: Kohn wuchs in einer jüdischen Familie im Wien der 1930er Jahre auf. Nach dem „Anschluss" Österreichs 1938 entging er dem Holocaust nur knapp: Im Alter von 15 Jahren wurde er durch einen rettenden *Kindertransport* nach England gebracht. Seine Eltern sah er nie wieder; sie wurden später im Konzentrationslager Auschwitz ermordet. 1940 wurde Kohn in Grossbritannien aufgrund seines Passes als „enemy alien" (feindlicher Ausländer) eingestuft und in ein britisches Internierungslager nach Kanada verschifft. Trotz der widrigen Umstände im Lager konnte er mit Unterstützung gefangener Wissenschaftler lernen und später an der Universität Toronto studieren, wo er seine Bachelor- und Masterabschlüsse in angewandter Mathematik erwarb.
>
> Nach dem Zweiten Weltkrieg ging er in die USA, promovierte an der Harvard University (unter dem Nobelpreisträger Julian Schwinger) und machte sich einen Namen in der Festkörperphysik. Er lehrte an der Carnegie Mellon University, der UC San Diego und später an der UC Santa Barbara, wo er Gründungsdirektor des renommierten Instituts für Theoretische Physik (KITP) wurde.
>
> Der revolutionäre Durchbruch zur Dichtefunktionaltheorie geschah fernab seiner US-Heimatuniversität. Während eines Sabbaticals (1963–1964) an der *École Normale Supérieure* in Paris traf Kohn auf den Postdoc Pierre Hohenberg. Bei gemeinsamen Diskussionen in Frankreich erkannten sie, dass die Elektronendichte ausreicht, um den Grundzustand eines Systems vollständig zu beschreiben – das Hohenberg-Kohn-Theorem war geboren. 1965 folgte, zurück in den USA, gemeinsam mit Lu Jeu Sham der praktische Kohn-Sham-Ansatz.
>
> Während Festkörperphysiker die DFT relativ schnell adaptierten (da das homogene Elektronengasmodell gut zu Metallen passte), stiess sie bei Quantenchemikern jahrzehntelang auf vehemente Ablehnung. Die Chemiker bevorzugten exakte wellenfunktionsbasierte Methoden (wie Hartree-Fock) und hielten frühe DFT-Näherungen für ungenaue „schmutzige Physik", da sie chemische Bindungen und Reaktionsbarrieren oft falsch vorhersagten.
>
> Dies änderte sich erst in den späten 1980er und frühen 1990er Jahren durch die Entwicklung besserer hybrider Funktionale (wie B3LYP). Den endgültigen Durchbruch in der Chemie brachte der britische Chemiker John Pople: Er integrierte die DFT-Algorithmen in ***Gaussian***, das weltweit am meisten genutzte Softwarepaket für Quantenchemie. Plötzlich konnten Chemiker hochkomplexe Moleküle effizient und präzise auf Knopfdruck berechnen. Für diese Revolutionierung der modernen Chemie teilten sich Walter Kohn (als Physiker) und John Pople 1998 gemeinsam den Nobelpreis für Chemie.

Das Funktional für die Hohenberg-Kohn-Theoreme ist exakt, aber in seiner Form unbekannt – insbesondere der Teil für die kinetische Energie lässt sich nicht präzise direkt aus der Dichte ableiten. Kohn und Sham lösten dieses Problem 1965 durch einen brillanten konstruktiven Trick: Sie führten ein *fiktives System nicht-wechselwirkender Elektronen* ein, welches exakt dieselbe Grundzustandsdichte $n(\mathbf{r})$ besitzt wie das reale, wechselwirkende System.

Da diese Elektronen nicht miteinander wechselwirken, lässt sich ihre Wellenfunktion exakt als eine einzelne Slater-Determinante aus Einteilchen-Orbitalen – den sogenannten Kohn-Sham-Orbitalen $\psi_i(\mathbf{r})$ – konstruieren.

Der direkte mathematische Zusammenhang zwischen diesen Orbitalen und der Elektronendichte lautet:

$$
n(\mathbf{r}) = \sum_{i=1}^N |\psi_i(\mathbf{r})|^2
$$

Das bedeutet, die Gesamtdichte ist einfach die Summe der Wahrscheinlichkeitsdichten aller besetzten Einteilchenzustände.

Das Energiefunktional wird in diesem Ansatz zerlegt in bekannte klassische/quantenmechanische Terme und einen Restterm:

$$
E[\{\psi_i\}] = T_{\text{s}}[n] + \int n(\mathbf{r}) v_{\text{ext}}(\mathbf{r}) \, d\mathbf{r} + E_{\text{H}}[n] + E_{\text{xc}}[n]
$$

Die einzelnen Terme haben folgende präzise physikalische und mathematische Bedeutung:

**1. Kinetische Energie des Referenzsystems ($T_{\text{s}}$)**

$$
T_{\text{s}}[n] = -\frac{1}{2} \sum_{i=1}^N \int \psi_i^*(\mathbf{r}) \nabla^2 \psi_i(\mathbf{r}) \, d\mathbf{r}
$$

Dies ist die exakte kinetische Energie des *fiktiven* Systems aus nicht-wechselwirkenden Elektronen. Frühere Modelle (wie das Thomas-Fermi-Modell) scheiterten daran, dass sie versuchten, die kinetische Energie direkt als Funktional der Dichte $n(\mathbf{r})$ auszudrücken, was physikalisch extrem ungenau ist. Durch den Umweg über die Orbitale $\psi_i$ deckt $T_{\text{s}}$ bereits über 99 % der wahren kinetischen Energie des realen Systems hochpräzise ab.

**2. Energie im externen Potential**

$$
V_{\text{ext}}[n] = \int n(\mathbf{r}) v_{\text{ext}}(\mathbf{r}) \, d\mathbf{r}
$$

Dieser Term beschreibt die klassische elektrostatische Anziehung zwischen der negativ geladenen Elektronenwolke $n(\mathbf{r})$ und den positiv geladenen, feststehenden Atomkernen, welche das externe Potential $v_{\text{ext}}(\mathbf{r})$ aufbauen.

**3. Hartree-Energie ($E_{\text{H}}$)**

$$
E_{\text{H}}[n] = \frac{1}{2} \iint \frac{n(\mathbf{r}) n(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\mathbf{r} \, d\mathbf{r}'
$$

Dies ist die klassische elektrostatische Coulomb-Abstossung der Elektronendichte mit sich selbst. Der Faktor $\frac{1}{2}$ korrigiert die doppelte Zählung der Wechselwirkungspaare. *Vorsicht:* Klassisch gesehen spürt hier ein Elektron an einem Ort $\mathbf{r}$ die Abstossung der *gesamten* Dichte – und damit fälschlicherweise auch einen Bruchteil seiner *eigenen* Dichte. Dieser physikalische Fehler wird als „Selbstwechselwirkungsfehler" (Self-Interaction Error) bezeichnet.

**4. Austausch-Korrelations-Energie ($E_{\text{xc}}$)**

Das **Austausch-Korrelations-Funktional** ist der einzige Term, dessen exakte Form unbekannt ist. Es agiert als Auffangbecken für alle quantenmechanischen Vielteilcheneffekte, die in den ersten drei Termen ignoriert oder fehlerhaft abgebildet wurden:

$$
E_{\text{xc}}[n] = \underbrace{(T[n] - T_{\text{s}}[n])}_{\text{Korrelation der kin. Energie}} + \underbrace{(E_{\text{ee}}[n] - E_{\text{H}}[n])}_{\text{Nicht-klassische Wechselwirkung}}
$$

Dieser Term muss zwei zentrale Aufgaben erfüllen: Zum einen muss er den quantenmechanischen Austausch (Pauli-Prinzip) und die Coulomb-Korrelation (die Elektronen weichen einander dynamisch aus) beschreiben. Zum anderen muss er exakt den unphysikalischen Selbstwechselwirkungsfehler aus dem Hartree-Term $E_{\text{H}}$ wieder herauskürzen. Die Entwicklung guter Näherungen für $E_{\text{xc}}$ ist bis heute das zentrale Forschungsgebiet der DFT.

### 3.1 Die Variationsableitung der Kohn-Sham-Gleichungen

Um die Gleichungen für die Orbitale zu finden, minimieren wir das Energiefunktional $E[\{\psi_i\}]$ bezüglich der komplex-konjugierten Orbitale $\psi_i^*(\mathbf{r})$ unter der Nebenbedingung, dass die Orbitale orthonormiert bleiben ($\int \psi_i^* \psi_j \, d\mathbf{r} = \delta_{ij}$). Wir definieren das Lagrange-Funktional $\mathcal{L}$:

$$
\mathcal{L} = E[\{\psi_i\}] - \sum_{i,j} \lambda_{ij} \left( \int \psi_i^*(\mathbf{r}) \psi_j(\mathbf{r}) \, d\mathbf{r} - \delta_{ij} \right)
$$

Wir bilden die funktionale Ableitung $\frac{\delta \mathcal{L}}{\delta \psi_i^*(\mathbf{r})} = 0$:

1. **Kinetische Energie:** Die Variation von $T_{\text{s}} = -\frac{1}{2} \sum_j \int \psi_j^* \nabla^2 \psi_j \, d\mathbf{r}$ liefert $-\frac{1}{2}\nabla^2 \psi_i(\mathbf{r})$.

2. **Dichteabhängige Terme ($v_{\text{ext}}, E_{\text{H}}, E_{\text{xc}}$):** Da diese Terme explizite Dichtefunktionale sind, verwenden wir die Kettenregel:

$$
\frac{\delta F[n]}{\delta \psi_i^*(\mathbf{r})} = \frac{\delta F[n]}{\delta n(\mathbf{r})} \frac{\delta n(\mathbf{r})}{\delta \psi_i^*(\mathbf{r})} = \frac{\delta F[n]}{\delta n(\mathbf{r})} \psi_i(\mathbf{r})
$$

   Die Summe der Ableitungen nach der Dichte definieren wir als das **effektive Potential** $v_{\text{eff}}(\mathbf{r})$:

$$
v_{\text{eff}}(\mathbf{r}) = v_{\text{ext}}(\mathbf{r}) + \int \frac{n(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\mathbf{r}' + \frac{\delta E_{\text{xc}}[n]}{\delta n(\mathbf{r})}
$$

3. **Lagrange-Term:** Die Variation der Orthogonalitätsbedingung liefert $\sum_j \lambda_{ij} \psi_j(\mathbf{r})$.

Nach einer geeigneten unitären Transformation zur Diagonalisierung der Matrix $\lambda_{ij}$ ($\lambda_{ij} = \varepsilon_i \delta_{ij}$) ergeben sich die **Kohn-Sham-Gleichungen**:

$$
\left( -\frac{1}{2}\nabla^2 + v_{\text{eff}}(\mathbf{r}) \right) \psi_i(\mathbf{r}) = \varepsilon_i \psi_i(\mathbf{r})
$$



## 4. Self-Consistent-Field (SCF) Iteration

Ein zentrales Problem der Kohn-Sham-Gleichungen ist, dass sie nichtlinear gekoppelt sind: Um die Gleichung zu lösen und die Orbitale $\psi_i(\mathbf{r})$ zu erhalten, benötigt man das effektive Potential $v_{\text{eff}}(\mathbf{r})$. Dieses Potential hängt jedoch von der Elektronendichte $n(\mathbf{r})$ ab, welche wiederum erst aus den Orbitalen $\psi_i(\mathbf{r})$ berechnet werden muss.

Daher löst man das System iterativ über einen sogenannten **Self-Consistent-Field (SCF)** Zyklus:

1. **Initialisierung:** Man rät eine Startdichte $n^{(0)}(\mathbf{r})$ (z. B. durch Überlagerung atomarer Dichten).
2. **Potentialberechnung:** Mit der aktuellen Dichte wird das effektive Potential $v_{\text{eff}}(\mathbf{r})$ konstruiert.
3. **Lösen der Eigenwertgleichung:** Die Kohn-Sham-Gleichungen werden diagonalisiert, um neue Orbitale $\psi_i(\mathbf{r})$ und Energien $\varepsilon_i$ zu erhalten.
4. **Dichteaktualisierung:** Eine neue Elektronendichte $n^{(\text{neu})}(\mathbf{r}) = \sum_{i} |\psi_i(\mathbf{r})|^2$ wird berechnet.
5. **Konvergenzprüfung:** Unterscheidet sich $n^{(\text{neu})}$ von der vorherigen Dichte nur noch minimal (innerhalb einer Toleranzschwelle), ist die Lösung „selbstkonsistent" und der Algorithmus endet. Falls nicht, wird eine neue Dichte gemischt (z. B. $n^{(k+1)} = \alpha n^{(\text{neu})} + (1-\alpha) n^{(k)}$) und man springt zurück zu Schritt 2.

<p align="center"><img src="/personal_blog/SCF.png" alt="Ablaufdiagramm des iterativen Self-Consistent-Field (SCF) Zyklus." width="410"></p>

*Abbildung 2: Ablaufdiagramm des iterativen Self-Consistent-Field (SCF) Zyklus zur Lösung der Kohn-Sham-Gleichungen.*

### 4.1 Approximationen für E_xc (Jacob's Ladder)

Die Kohn-Sham-DFT wäre eine exakte Theorie, wenn wir die exakte mathematische Form des Austausch-Korrelations-Funktionals $E_{\text{xc}}[n]$ kennen würden. Da dies nicht der Fall ist, müssen Näherungen verwendet werden. Diese Näherungen werden oft nach der sogenannten *Jacob's Ladder* klassifiziert, bei der jede höhere Stufe physikalisch genauer, aber auch rechenintensiver wird:

| Stufe | Näherung (Acronym) | Beschreibung |
|:-----:|:--------------------|:-------------|
| 1 | **LDA** (Local Density Approx.) | $E_{\text{xc}}$ hängt an jedem Ort $\mathbf{r}$ nur von der lokalen Dichte $n(\mathbf{r})$ ab (basiert auf dem homogenen Elektronengas). |
| 2 | **GGA** (Generalized Gradient Approx.) | Berücksichtigt zusätzlich den Gradienten der Dichte $\nabla n(\mathbf{r})$. Typische Funktionale: PBE, BLYP. Standard in der Festkörperphysik. |
| 3 | **Meta-GGA** | Bezieht zusätzlich die zweite Ableitung (Laplace-Operator) oder die kinetische Energiedichte $\tau(\mathbf{r})$ der Orbitale mit ein (z. B. SCAN). |
| 4 | **Hybrid-Funktionale** | Mischt einen Teil des exakten quantenmechanischen Austauschs (Hartree-Fock-Austausch) bei. Typische Funktionale: B3LYP, PBE0. Sehr beliebt in der Quantenchemie für präzise Bandlücken. |

*Tabelle 1: Hierarchie der Dichtefunktionale (Jacob's Ladder nach John Perdew).*

Durch den eleganten mathematischen Trick des Kohn-Sham-Ansatzes wurde das unlösbare $3N$-dimensionale Vielteilchenproblem auf ein System von $N$ gekoppelten 3-dimensionalen Einteilchengleichungen reduziert, die SCF-iterativ gelöst werden können. Mit der Wahl der richtigen Näherung für $E_{\text{xc}}$ auf der Jacob's Ladder lassen sich Materialeigenschaften wie Bandlücken, Elastizitätsmodule und Reaktionsenthalpien heute effizient und zuverlässig am Computer simulieren.



## 5. John Pople und der Durchbruch in der Chemie: Gaussian

Während die Dichtefunktionaltheorie (DFT) von Walter Kohn in der Festkörperphysik relativ rasch an Bedeutung gewann, stiess sie in der theoretischen Chemie lange Zeit auf tiefe Skepsis. Diese Zurückhaltung wurde letztlich durch die Arbeit des britischen Mathematikers und theoretischen Chemikers **Sir John A. Pople** (1925–2004) überwunden. Für diese Leistung teilte sich Pople 1998 den Nobelpreis für Chemie zu gleichen Teilen mit Walter Kohn.

<p align="center"><img src="/personal_blog/John_A_Pople.png" alt="Sir John Anthony Pople (1925–2004)." width="450"></p>

*Abbildung 3: Sir John Anthony Pople (1925–2004) war ein bahnbrechender britischer Mathematiker und theoretischer Chemiker, der 1998 mit dem Nobelpreis für Chemie ausgezeichnet wurde. Er erhielt den Preis gemeinsam mit Walter Kohn für seine bahnbrechende Entwicklung von computergestützten Methoden in der Quantenchemie. Durch seine Arbeit verwandelte sich die theoretische Chemie von einer mathematischen Nische in ein mächtiges Alltagswerkzeug für die gesamte chemische Forschung.*

John Pople war der unbestrittene Pionier der *ab initio* Wellenfunktionsmethoden. Er entwickelte systematische Hierarchien (wie die Møller-Plesset-Störungstheorie und Coupled-Cluster-Ansätze), mit denen sich die exakte Lösung der Schrödinger-Gleichung durch zunehmenden Rechenaufwand schrittweise und kontrolliert annähern liess.

Aus Poples Sicht besass die frühe DFT von Kohn zwei entscheidende Makel:

1. Sie war keine *systematisch verbesserbare* Theorie. Wenn ein Näherungsfunktional für $E_{\text{xc}}$ versagte, gab es keinen strikten mathematischen Weg, den Fehler systematisch zu reduzieren (anders als bei der Erweiterung eines Basissatzes in der Wellenfunktionsmechanik).
2. Frühe Näherungen wie die LDA (Local Density Approximation) scheiterten kläglich bei der Vorhersage grundlegender chemischer Bindungsenergien und Reaktionsbarrieren.

Trotz dieser philosophischen Differenzen zu Kohns Ansatz war Pople ein ungemein pragmatischer Wissenschaftler. Als in den späten 1980er und frühen 1990er Jahren leistungsfähigere Gradienten-korrigierte Funktionale (GGA) und insbesondere Hybrid-Funktionale (wie das berühmte B3LYP) aufkamen, erkannte Pople sofort das immense Potenzial: Diese neuen DFT-Methoden lieferten Ergebnisse, die an hochgenaue Wellenfunktionsmethoden heranreichten, benötigten aber nur einen Bruchteil der Rechenzeit.

Poples grösstes Vermächtnis war die Entwicklung des Softwarepakets **Gaussian**, das er 1970 erstmals veröffentlichte. Sein erklärtes Ziel war es, komplexe quantenchemische Rechnungen nicht nur theoretischen Physikern, sondern allen praktizierenden Chemikern und Ingenieuren zugänglich zu machen – als verlässliche „Black-Box-Methode".

Anfang der 1990er Jahre entschied sich Pople für einen paradigmatischen Schritt: Er integrierte die DFT als voll nutzbare Methode in sein Programm (insbesondere ab der Version *Gaussian 92/DFT*). Pople und sein Team entwickelten hochstabile numerische Integrationsgitter (Grids) zur Auswertung der Funktionale und optimierten die analytische Berechnung der Energie-Gradienten, die für die automatische Geometrieoptimierung von Molekülen zwingend notwendig sind.

Durch Poples Implementierung wurde Kohns Theorie über Nacht für Tausende von Forschern anwendbar. Die Kombination aus Kohns theoretischer Brillanz und Poples algorithmischer Umsetzung löste einen beispiellosen Boom in der rechnergestützten Materialwissenschaft aus. Ingenieure und Chemiker konnten nun Reaktionspfade und Eigenschaften von Katalysatoren oder Polymeren an Standardcomputern berechnen, ohne tiefe Kenntnisse der zugrundeliegenden Quantenmechanik haben zu müssen.

Noch heute ist das von Pople ins Leben gerufene Programm *Gaussian* eines der dominierenden und am weitesten verbreiteten Werkzeuge in der akademischen und industriellen Forschung. Aktuelle Versionen sowie detaillierte Dokumentationen finden sich unter dem offiziellen Webauftritt: <https://gaussian.com/>

<p align="center"><img src="Gaussian_1970.jpg" alt="Gaussian Version 1970." width="450"></p>

*Abbildung 4: Gaussian Version 1970.*



## Anhang: PDF-Version

Die vollständige, gesetzte PDF-Version dieses Beitrags: [DFT_Blog_WUCH.pdf](DFT_Blog_WUCH.pdf)
