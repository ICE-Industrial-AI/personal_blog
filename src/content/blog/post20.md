---
title: "Relativistische Effekte passieren schon im Schneckentempo"
description: "Die Magnetkraft ist im Wesentlichen die elektrische Kraft, betrachtet aus einem anderen Bezugssystem. Der magnetische Teil der Lorentzkraft ist eine relativistische Manifestation des elektrischen Feldes. Das Versagen der Flussregel bei der unipolaren Induktion unterstreicht die Notwendigkeit, E- und B-Felder als Komponenten eines einzigen relativistischen Objekts zu behandeln: des Faraday-Tensors. Selbst bei den niedrigen Geschwindigkeiten einer Kupferscheibe ist Magnetismus im Grunde ein relativistisches Phänomen."
pubDate: "April 12 2026"
heroImage: "/personal_blog/aikn.webp"
badge: "Latest"
---

# Relativistische Effekte im Schneckentempo

**Author:** *Christoph Würsch, Institute for Computational Engineering ICE* <br>
*Eastern Switzerland University of Applied Sciences OST* <br>
**Date:** 12.4.2026<br>



> Eine tiefgreifende und verblüffende Erkenntnis ist, dass die **Magnetkraft im Wesentlichen die elektrische Kraft ist, betrachtet aus einem anderen Bezugssystem**. Der „magnetische“ Teil der Lorentzkraft ist eine relativistische Manifestation des elektrischen Feldes. Das Versagen der Flussregel bei der unipolaren Induktion unterstreicht die Notwendigkeit, $\vec{E}$ und $\vec{B}$ als Komponenten eines einzigen relativistischen Objekts zu behandeln: des Faraday-Tensors. Selbst bei den „niedrigen“ Geschwindigkeiten einer Kupferscheibe ist Magnetismus im Grunde ein relativistisches Phänomen.

[Download:](/personal_blog/LorentzForce.pdf)

---

### Inhaltsverzeichnis
1. [Die Lorentzkraft als relativistischer Effekt](#1-die-lorentzkraft-als-relativistischer-effekt)
2. [Unipolare Induktion (Die Faradaysche Scheibe)](#2-unipolare-induktion-die-faradaysche-scheibe)
3. [Das Paradoxon des bewegten Stabes](#3-das-paradoxon-des-bewegten-stabes)
4. [Der Lorentz-Boost in Matrixform](#4-der-lorentz-boost-in-matrixform)

---

### 1. Die Lorentzkraft als relativistischer Effekt
Das Faradaysche Induktionsgesetz wird oft als fundamentales Postulat des klassischen Elektromagnetismus präsentiert. Eine tiefere Analyse zeigt jedoch, dass die Trennung zwischen elektrischen und magnetischen Kräften vom Bezugssystem abhängt. Dieser Blog zeigt, wie die Lorentzkraft natürlich aus den Anforderungen der Speziellen Relativitätstheorie (SRT) hervorgeht und dabei Paradoxien löst, an denen die klassische Flussregel zu scheitern scheint.

Der Standardausdruck für die induzierte elektrische Spannung $U_{\text{ind}}$ (elektromotorische Kraft EMK) ist durch die Änderung des magnetischen Flusses $\Phi$ gegeben:

$$
\boxed{
U_{\text{ind}} = -\frac{d\Phi}{dt} = -\frac{d}{dt} \int_S \vec{B} \cdot d\vec{A}
}
$$

Während diese Formel für geschlossene Leiterschleifen, die ihre Form ändern oder zeitlich variierenden Feldern ausgesetzt sind, robust ist, stößt sie in Systemen, in denen die Geometrie der „Schleife“ zweideutig ist, auf konzeptionelle Schwierigkeiten.

---

### 2. Unipolare Induktion (Die Faradaysche Scheibe)
Ein klassisches Problem, das den Unterschied zwischen der „Flussregel“ und der Lorentzkraft verdeutlicht, ist die **Faradaysche Scheibe**, auch bekannt als Unipolar-Generator. Dieses Gerät besteht aus einer leitenden Scheibe, die in einem gleichmäßigen, stationären Magnetfeld $\vec{B}$ parallel zur Rotationsachse rotiert.

Betrachten wir eine leitende Kupferscheibe (Faraday-Rad), die in einem gleichmäßigen, konstanten Magnetfeld $\vec{B}$ parallel zu ihrer Rotationsachse rotiert.

1. Der magnetische Fluss $\Phi$ durch jeden Sektor der Scheibe bleibt konstant, da $\vec{B}$ homogen ist und sich die Fläche innerhalb des Stromkreises (Schleifkontakte zur Mitte) im traditionellen Sinne nicht ändert: $\frac{d\Phi}{dt} = 0$.
2. Dennoch wird eine Potenzialdifferenz $U_{\text{ind}}$ zwischen der Achse und dem Rand gemessen.

Das folgende Diagramm veranschaulicht den Aufbau. Beachten Sie, dass während die Scheibe rotiert, das Magnetfeld und der externe Stromkreis (Schleifkontakte) im Labor-Bezugssystem stationär bleiben.


![Unipolarinduktion bei einer rotierenden Kupferscheibe](/personal_blog/Unipolarinduktion.png)

> **Abbildung 1:** Draufsicht auf die Faradaysche Scheibe. Die blauen kreisförmigen Pfeile um die Kreuze stellen das Magnetfeld $\vec{B}$ dar, das in die Seite hinein gerichtet ist. Die Lorentzkraft $\vec{F}_L = q(\vec{v} \times \vec{B})$ treibt die Ladungen radial zum Rand.

Erinnern wir uns an die integrale Form des Faradayschen Gesetzes (die Flussregel):

$$
U_{\text{ind}} = -\frac{d\Phi}{dt} = -\frac{d}{dt} \int_S \vec{B} \cdot d\vec{A}
$$

In diesem Aufbau:
1. Das Magnetfeld $\vec{B}$ ist zeitlich konstant ($\partial \vec{B} / \partial t = 0$).
2. Die durch den Stromkreis definierte Fläche $S$ (Achse $\to$ Rand $\to$ externe Verkabelung) ändert weder ihre Form noch ihre Ausrichtung relativ zum $\vec{B}$-Feld, während die Scheibe rotiert.

Folglich ist $\frac{d\Phi}{dt} = 0$. Rein basierend auf der Flussregel würde man fälschlicherweise $U_{\text{ind}} = 0$ vorhersagen.

**Lösung über die Lorentzkraft:**
Die tatsächliche Induktion tritt auf, weil sich die Ladungsträger (Elektronen) mit der Scheibe mit der Geschwindigkeit $\vec{v}$ bewegen. In einem radialen Abstand $r$ vom Zentrum beträgt die Geschwindigkeit $\vec{v} = \vec{\omega} \times \vec{r}$.

Die Lorentzkraft auf eine Ladung $q$ ist:

$$
\vec{F}_L = q(\vec{v} \times \vec{B})
$$

Für eine Scheibe in der $xy$-Ebene und $\vec{B} = B\hat{z}$ ist die Geschwindigkeit $\vec{v} = \omega r \hat{\phi}$. Die resultierende Kraft ist:

$$
\vec{F}_L = q(\omega r \hat{\phi} \times B\hat{z}) = q\omega r B \hat{r}
$$

Diese radiale Kraft wirkt wie ein „nicht-konservatives“ elektrisches Feld $\vec{E}_{\text{eff}} = \vec{v} \times \vec{B}$. Die induzierte EMK ist das Linienintegral dieses Feldes von der Achse ($r=0$) zum Rand ($r=R$):

$$
U_{\text{ind}} = \int_0^R (\vec{v} \times \vec{B}) \cdot d\vec{r} = \int_0^R \omega r B \, dr = \frac{1}{2}\omega R^2 B
$$

> **Die relativistische Erkenntnis**
> Das Paradoxon der Faradayschen Scheibe entsteht nur, wenn wir darauf beharren, dass Induktion durch einen sich ändernden magnetischen Fluss verursacht werden muss. Die Relativitätstheorie lehrt uns, dass die Unterscheidung zwischen „Flussänderung“ und „Bewegung durch ein Feld“ vom Bezugssystem abhängt. Im Labor-Bezugssystem gibt es kein $\vec{E}$-Feld, nur $\vec{B}$, und die Kraft ist magnetisch. Würden wir uns auf ein Ladungselement in der Scheibe setzen, sähen wir eine stationäre Scheibe, aber ein *transformiertes* elektrisches Feld $\vec{E}'$, das für das Potenzial verantwortlich ist.



### 3. Das Paradoxon des bewegten Stabes
Betrachten wir einen leitenden Stab der Länge $L$, der sich mit der Geschwindigkeit $\vec{v} = v\hat{x}$ durch ein konstantes Magnetfeld $\vec{B} = B\hat{z}$ bewegt.

* **Labor-Bezugssystem ($S$):** Die Ladungen im Stab erfahren eine Lorentzkraft $\vec{F}_m = q(\vec{v} \times \vec{B}) = -qvB\hat{y}$. Dies führt zu einer Ladungstrennung und einer induzierten Spannung $U = vBL$.
* **Stab-Bezugssystem ($S'$):** In dem System, das sich mit dem Stab bewegt, ist der Stab stationär ($\vec{v}' = 0$). Da $\vec{B}$ konstant und homogen ist, gilt $d\Phi/dt = 0$. Würden wir die klassische Mechanik strikt anwenden, sollte keine Kraft auf die Ladungen wirken und keine Spannung auftreten.

Diese Diskrepanz deutet darauf hin, dass sich das Magnetfeld in $S$ als *elektrisches* Feld in $S'$ manifestieren muss.


![Bewegter Stab im homogenten Magnetfeld](/personal_blog/BewegterStab.png)
> **Abbildung 2:** Im Labor-System ist die Kraft magnetisch. Im Stab-System ist der Stab stationär, und die Kraft ist rein elektrisch aufgrund der Lorentz-Transformation des Faraday-Tensors.

Um das Stab-Paradoxon zu lösen, nutzen wir die Lorentz-Transformation des elektromagnetischen Feldtensors $F^{\mu\nu}$. Für einen Boost in $x$-Richtung mit $\beta = v/c$ und $\gamma = (1-\beta^2)^{-1/2}$ transformieren die Felder wie folgt:

$$
\vec{E}' = \gamma(\vec{E} + \vec{v} \times \vec{B}) - \frac{\gamma^2}{\gamma+1} \frac{\vec{v}}{c^2}(\vec{v} \cdot \vec{E})
$$

$$
\vec{B}' = \gamma(\vec{B} - \frac{1}{c^2}\vec{v} \times \vec{E}) - \frac{\gamma^2}{\gamma+1} \frac{\vec{v}}{c^2}(\vec{v} \cdot \vec{B})
$$

Im System des Stabes ($S'$), wo $\vec{E} = 0$ im Labor gilt, finden wir:

$$
\vec{E}' = \gamma(\vec{v} \times \vec{B})
$$

Bei niedrigen Geschwindigkeiten ($v \ll c$, $\gamma \approx 1$) „sieht“ der Stab ein elektrisches Feld $\vec{E}' = \vec{v} \times \vec{B}$. Die Kraft im Bezugssystem des Stabes ist rein elektrostatisch: $\vec{F}' = q\vec{E}'$.



### 4. Der Lorentz-Boost in Matrixform
Um den Ursprung dieser Kraft zu verstehen, müssen wir untersuchen, wie sich elektromagnetische Felder zwischen Inertialsystemen transformieren. Wir definieren einen Boost in $x$-Richtung mit der Geschwindigkeit $v$. Sei $\beta = v/c$ und $\gamma = (1-\beta^2)^{-1/2}$. Die Lorentz-Transformationsmatrix $\Lambda^\mu_{\ \nu}$ lautet:

$$
\Lambda^\mu_{\ \nu} = \begin{pmatrix} 
\gamma & -\beta\gamma & 0 & 0 \\ 
-\beta\gamma & \gamma & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1 
\end{pmatrix}
$$

In der Relativitätstheorie sind $\vec{E}$ und $\vec{B}$ keine unabhängigen Vektoren, sondern Komponenten des antisymmetrischen **Faraday-Tensors** $F^{\mu\nu}$:

$$
F^{\mu\nu} = \begin{pmatrix} 
0 & -E_x/c & -E_y/c & -E_z/c \\ 
E_x/c & 0 & -B_z & B_y \\ 
E_y/c & B_z & 0 & -B_x \\ 
E_z/c & -B_y & B_x & 0 
\end{pmatrix}
$$

Die Transformation in ein bewegtes System $S'$ ist durch die Tensorkontraktion $F' = \Lambda F \Lambda^T$ gegeben. Betrachten wir einen leitenden Stab im Labor-System ($S$), der sich mit $\vec{v} = v\hat{x}$ in einem konstanten Feld $\vec{B} = B\hat{z}$ bewegt. In $S$ gilt $\vec{E}=0$.

Die einzigen Nicht-Null-Komponenten von $F^{\mu\nu}$ sind $F^{12} = -B$ und $F^{21} = B$.

$$
F = \begin{pmatrix} 
0 & 0 & 0 & 0 \\ 
0 & 0 & -B & 0 \\ 
0 & B & 0 & 0 \\ 
0 & 0 & 0 & 0 
\end{pmatrix}
$$

Wir berechnen die neue elektrische Feldkomponente $E'_y$ über $F'^{02} = -E'_y/c$:

$$
\begin{aligned}
F'^{02} &= \Lambda^0_{\ \alpha} F^{\alpha \beta} \Lambda^2_{\ \beta} \\
F'^{02} &= \Lambda^0_{\ 1} F^{12} \Lambda^2_{\ 2} \quad (\text{da nur } \Lambda^2_{\ 2}=1 \text{ und } F^{02}=0) \\
-\frac{E'_y}{c} &= (-\beta\gamma) \cdot (-B) \cdot (1) \\
E'_y &= -\beta\gamma c B = -\gamma v B
\end{aligned}
$$

Im Grenzfall von Alltagsgeschwindigkeiten ($v \ll c$):
1. $\gamma = \frac{1}{\sqrt{1-(v/c)^2}} \approx 1$
2. Daher gilt $E'_y \approx -v B$

> **Fazit**
> Im Labor-System sehen wir eine magnetische Kraft $\vec{F} = q(\vec{v} \times \vec{B})$, welche die Ladungen an die Enden des Stabes drückt. Im System des Stabes ist der Stab stationär, aber die relativistische Transformation erzeugt ein **elektrisches Feld** $\vec{E}' = \vec{v} \times \vec{B}$, das exakt dieselbe Arbeit verrichtet. **Magnetismus ist schlichtweg die elektrische Kraft, betrachtet aus einem bewegten Bezugssystem.**