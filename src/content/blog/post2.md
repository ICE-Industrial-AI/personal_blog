---
title: "XAI für CNN: Attributionsmethoden zur Bildanalyse"
description: "Attributionsmethoden stellen ein wichtiges Instrument dar, um die Entscheidungsprozesse neuronaler Netze transparent zu machen. Sie ordnen jedem Eingabemerkmal einen Beitrag zur Vorhersage zu und ermöglichen somit eine Bewertung der Relevanz einzelner Merkmale. In diesem Beitrag werden gängige Attributionsmethoden für CNNs vorgestellt und mathematisch fundiert beschrieben. Besonderes Augenmerk liegt auf Gradienten-basierten Verfahren wie den Integrated Gradients, welche den Beitrag eines Pixels zur Vorhersage durch Integration entlang eines Pfades von einer Baseline zur Eingabe quantifizieren. Darüber hinaus werden Konzepte wie Layer-wise Relevance Propagation (LRP), DeepLIFT und neuere Visualisierungstechniken wie SUMMIT diskutiert. Ziel ist es, ein Verständnis für die methodischen Grundlagen und praktischen Herausforderungen der Erklärbarkeit tiefer Modelle zu vermitteln.."
pubDate: "Jul 14 2025"
heroImage: "/personal_blog/aikn.webp"
---
# XAI für CNN: Attributionsmethoden zur Bildanalyse

**Reader, Systemtechnik BSc, HS 2026**
Kurs Applied Neural Networks (ANN)  
*Author: Christoph Würsch*

## Abstract

Attributionsmethoden stellen ein wichtiges Instrument dar, um die Entscheidungsprozesse neuronaler Netze transparent zu machen. Sie ordnen jedem Eingabemerkmal einen Beitrag zur Vorhersage zu und ermöglichen somit eine Bewertung der Relevanz einzelner Merkmale. In diesem Beitrag werden gängige Attributionsmethoden für CNNs vorgestellt und mathematisch fundiert beschrieben. Besonderes Augenmerk liegt auf Gradienten-basierten Verfahren wie den Integrated Gradients, welche den Beitrag eines Pixels zur Vorhersage durch Integration entlang eines Pfades von einer Baseline zur Eingabe quantifizieren. Darüber hinaus werden Konzepte wie Layer-wise Relevance Propagation (LRP), DeepLIFT und neuere Visualisierungstechniken wie SUMMIT diskutiert. Ziel ist es, ein Verständnis für die methodischen Grundlagen und praktischen Herausforderungen der Erklärbarkeit tiefer Modelle zu vermitteln.

## 1. Einführung in die Attribution bei CNNs

Convolutional Neural Networks (CNNs) haben eine herausragende Leistungsfähigkeit in Aufgaben der Bilderkennung und -regression erzielt. Ihre komplexe, hierarchische Struktur macht sie jedoch zu "Black Boxes": Es ist oft unklar, auf welche Merkmale im Eingangsbild sich das Modell für seine Entscheidung stützt. Explainable AI (XAI) zielt darauf ab, diese Black Box zu öffnen und die Entscheidungsfindung von Modellen nachvollziehbar zu machen.

Ein zentraler Ansatz hierfür sind **Attributionsmethoden**. Die grundlegende Idee ist, die Vorhersage eines Modells auf seine Eingabemerkmale "zurückzuführen" (zu attribuieren). Für Bilddaten bedeutet dies, jedem Pixel des Eingangsbildes einen Relevanz- oder Wichtigkeitswert zuzuordnen. Das Ergebnis ist eine Heatmap, oft als **Saliency Map** bezeichnet, die visuell hervorhebt, welche Bildbereiche für die Ausgabe des Netzwerks (z.B. die Klassifizierung als "Hund") am einflussreichsten waren.

### 1.1 Mathematische Definition der Attribution

Sei $F: \mathbb{R}^d \to \mathbb{R}$ eine Funktion, die ein neuronales Netzwerk repräsentiert. Für ein Eingangsbild $x \in \mathbb{R}^d$, das als Vektor von $d$ Pixeln betrachtet wird, gibt $F(x)$ einen Skalar aus. Dieser Skalar kann der Logit-Wert für eine bestimmte Klasse bei einer Klassifikationsaufgabe oder der vorhergesagte Wert bei einer Regressionsaufgabe sein.

Eine **Attribution** ist eine Zuweisung eines Relevanzwertes $A_i(x)$ zu jedem Eingabemerkmal (Pixel) $x_i$. Das Ziel ist die Erstellung einer Attributionskarte (oder Vektor) $A(x) \in \mathbb{R}^d$. 

$$A(x) = (A_1(x), A_2(x), \ldots, A_d(x))$$

Diese Karte $A(x)$ soll die Wichtigkeit jedes Pixels $x_i$ für den finalen Output $F(x)$ quantifizieren.

### 1.2 Standard-Attribution: Sensitivity Maps

Der direkteste Weg, die "Sensitivität" des Outputs in Bezug auf eine kleine Änderung eines Input-Pixels zu messen, ist die Berechnung des Gradienten.

**Definition (Sensitivity Map):** Die Attribution eines Pixels $x_i$ wird als der partielle Ableitungswert der Output-Funktion $F$ nach diesem Pixel definiert.

$$A_i^{\text{Sens}}(x) = \frac{\partial F(x)}{\partial x_i}$$

Die gesamte Attributionskarte ist somit der Gradient des Outputs bezüglich des Inputs:

$$A^{\text{Sens}}(x) = \nabla_x F(x)$$

**Interpretation:** Der Wert $\frac{\partial F(x)}{\partial x_i}$ gibt an, wie stark sich der Output $F(x)$ ändert, wenn das Pixel $x_i$ infinitesimal klein verändert wird. Ein hoher absoluter Wert bedeutet eine hohe Relevanz des Pixels für die Entscheidung. Zur Visualisierung wird oft der Absolutbetrag oder das Quadrat des Gradienten verwendet.

### 1.3 Integrated Gradients (IG)

Ein Problem der einfachen Gradientenmethode ist die Sättigung. Wenn ein Neuron bereits stark aktiviert ist (z.B. durch eine ReLU-Aktivierungsfunktion), kann sein Gradient null sein, obwohl das Neuron entscheidend für das Ergebnis ist. Integrated Gradients (IG) löst dieses Problem, indem es die Gradienten entlang eines Pfades von einem Referenzbild (Baseline) $x'$ zum eigentlichen Bild $x$ integriert. Die Baseline $x'$ ist typischerweise ein informationsloses Bild, z.B. ein komplett schwarzes Bild.

**Definition (Integrated Gradients):** Die Attribution eines Pixels $x_i$ mittels IG ist definiert als:

$$A_i^{\text{IG}}(x) ::= (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x)}{\partial x_i}\Big|_{x' + \alpha(x - x')} d\alpha$$

#### Eigenschaften und Interpretation:

- **Pfadintegral:** Die Formel integriert die Gradienten entlang der geraden Linie im Merkmalsraum von der Baseline $x'$ zum Bild $x$.
- **Vollständigkeit (Completeness):** Eine wichtige Eigenschaft von IG ist, dass die Summe aller Attributionswerte der Differenz der Modellvorhersage zwischen dem Bild $x$ und der Baseline $x'$ entspricht:

$$\sum_{i=1}^{d} A_i^{\text{IG}}(x) = F(x) - F(x')$$

Dies macht die Attributionen "vollständig" und direkt interpretierbar als Beiträge zur Gesamtänderung des Outputs.

## 2. Gradienten-basierte Saliency-Methoden

Diese Methoden basieren alle auf der Rückpropagierung von Gradienten vom Output zum Input.

### 2.1 Saliency Maps (nach Simonyan et al., 2014)

Historisch gesehen ist dies eine der ersten und einfachsten Methoden. Sie ist in ihrer reinsten Form identisch mit der oben definierten Sensitivity Map.

**Algorithmus 1: Berechnung einer Saliency Map**
1. **Input:** Modell $F$, Eingangsbild $x$, Zielklasse $c$.
2. Führe einen Forward-Pass mit $x$ durch, um alle Aktivierungen zu berechnen.
3. Berechne den Score $S_c(x)$ für die Zielklasse $c$. Dies ist der Output $F(x)$.
4. Berechne den Gradienten des Scores bezüglich des Eingangsbildes:
   $$M(x) = \nabla_x S_c(x) = \frac{\partial S_c(x)}{\partial x}$$
5. **Visualisierung:**
   - Aggregiere die Gradienten über die Farbkanäle, z.B. durch den Maximalwert des Absolutbetrags für jedes Pixel: $m_{ij} = \max_k |M_{ijk}|$.
   - Normalisiere die resultierende 2D-Karte $m$ zur Darstellung als Heatmap.
6. **Output:** Saliency Map $m$.

### 2.2 SmoothGrad (Smilkov et al., 2017)

Standard-Gradientenkarten sind oft visuell verrauscht, was die Interpretation erschwert. SmoothGrad reduziert dieses Rauschen durch einen einfachen, aber effektiven Mittelungsprozess. Die Intuition ist, dass das wahre Relevanz-Signal bei leichten Störungen des Bildes stabil bleibt, während das Rauschen im Gradienten zufällig ist und sich bei Mittelung herauskürzt.

**Algorithmus 2: SmoothGrad**
1. **Input:** Modell $F$, Eingangsbild $x$, Anzahl der Samples $n$, Rauschlevel (Standardabweichung) $\sigma$.
2. Initialisiere eine leere Akkumulator-Karte $M_{\text{avg}} \leftarrow 0$.
3. **Für** $i = 1$ bis $n$:
   - Erzeuge einen zufälligen Rauschvektor $\epsilon_i \sim N(0, \sigma^2)$.
   - Erstelle ein gestörtes Bild: $x_{\text{noisy}} = x + \epsilon_i$.
   - Berechne die Gradienten-basierte Saliency Map für das gestörte Bild: $M_i = \nabla_x F(x_{\text{noisy}})$.
   - Addiere die Karte zum Akkumulator: $M_{\text{avg}} \leftarrow M_{\text{avg}} + M_i$.
4. Berechne den Durchschnitt: $M_{\text{smooth}} = \frac{1}{n} M_{\text{avg}}$.
5. **Output:** Geglättete Saliency Map $M_{\text{smooth}}$.

## 3. SUMMIT: Skalierbare Interpretierbarkeit durch Aktivierungs- und Attributions-Zusammenfassungen

Während Saliency Maps die Wichtigkeit von Pixeln für ein einzelnes Bild erklären, zielt SUMMIT (SUMmarization of Activations and Attributions) darauf ab, die interne Funktionsweise eines CNNs über einen gesamten Datensatz hinweg zu aggregieren und zu visualisieren. Der Kern von SUMMIT ist die Erstellung eines Attributionsgraphen.

### 3.1 Die Idee des Attributionsgraphen

Ein **Attributionsgraph** ist ein gerichteter azyklischer Graph (DAG), $G = (V, E)$, der die kausalen Einflüsse zwischen den internen "Konzepten", die von den Neuronen des Netzwerks gelernt wurden, darstellt.

- **Knoten (Nodes) $V$:** Jeder Knoten repräsentiert eine Gruppe von semantisch ähnlichen Neuronenaktivierungen innerhalb eines Layers. Ein Knoten steht also nicht für ein einzelnes Neuron, sondern für ein wiederkehrendes Muster oder "Konzept" (z.B. "Augen", "Felltextur").
- **Kanten (Edges) $E$:** Eine gerichtete Kante von einem Knoten $u$ in Layer $l_i$ zu einem Knoten $v$ in einem späteren Layer $l_j$ ($j > i$) quantifiziert, wie stark das von $u$ repräsentierte Konzept zur Aktivierung des von $v$ repräsentierten Konzepts beiträgt.

### 3.2 Attribution in SUMMIT

SUMMIT erweitert den Begriff der Attribution. Statt die Relevanz von Input-Pixeln für den finalen Output zu messen, misst SUMMIT die Relevanz der Aktivierung eines Neurons in einem früheren Layer für die Aktivierung eines Neurons in einem späteren Layer. Hierfür wird das Framework der Integrated Gradients (IG) verwendet.

**Mathematische Definition:** Sei $a_k^l(x)$ die Aktivierung des $k$-ten Neurons im Layer $l$ für das Eingangsbild $x$. Die Attribution der Aktivierung des Neurons $i$ in Layer $l_1$ auf die Aktivierung des Neurons $j$ in einem späteren Layer $l_2$ wird definiert als:

$$\text{Attribution}(a_i^{l_1}, a_j^{l_2}) = \int_{\alpha=0}^{1} \frac{\partial a_j^{l_2}(\text{path}(\alpha))}{\partial a_i^{l_1}} d\alpha$$

Hierbei ist der Integrationspfad im Aktivierungsraum des Layers $l_1$ definiert, typischerweise von einem Baseline-Aktivierungsvektor (z.B. Nullvektor) zum tatsächlichen Aktivierungsvektor des Layers $l_1$.

### 3.3 Konstruktion des Attributionsgraphen

#### Bestimmung der Knoten (Nodes)

Die Knoten werden durch Clustering von Neuronenaktivierungen über einen gesamten Datensatz (z.B. alle Bilder der Klasse "Katze") ermittelt.

1. **Aktivierungen sammeln:** Für einen gegebenen Layer $l$ und einen Datensatz $D$ werden alle Aktivierungsvektoren $\{a^l(x) | x \in D\}$ gesammelt.
2. **Dimensionalitätsreduktion:** Da die Anzahl der Neuronen pro Layer sehr hoch sein kann, wird typischerweise eine Dimensionsreduktion wie PCA (Principal Component Analysis) auf die gesammelten Aktivierungen angewendet.
3. **Clustering:** Auf den dimensionalitätsreduzierten Aktivierungen wird ein Clustering-Algorithmus (z.B. k-Means) ausgeführt.
4. **Knotenerstellung:** Jedes resultierende Cluster $C_k^l$ bildet einen Knoten im Graphen. Dieser Knoten repräsentiert eine Gruppe von Neuronen, die auf ähnliche Merkmale im Datensatz ansprechen.

#### Bestimmung der Kantengewichte (Edge Weights)

Das Gewicht einer Kante $w(u, v)$ von einem Knoten $u = C_k^{l_1}$ zu einem Knoten $v = C_m^{l_2}$ misst den aggregierten Einfluss.

1. **Paarweise Attribution:** Für jedes Bild im Datensatz wird die paarweise Attribution (mittels IG) zwischen allen Neuronen im Quell-Cluster $u$ und allen Neuronen im Ziel-Cluster $v$ berechnet.
2. **Aggregation:** Das Kantengewicht ist die Summe dieser Attributionen, gemittelt über den gesamten Datensatz.

**Formel:** Das Gewicht der Kante vom Knoten $u$ (in Layer $l_1$) zum Knoten $v$ (in Layer $l_2$) ist:

$$w(u, v) = \mathbb{E}_{x \in D}\left[\sum_{i \in u} \sum_{j \in v} \text{Attribution}(a_i^{l_1}(x), a_j^{l_2}(x))\right]$$

wobei $\mathbb{E}_{x \in D}[\cdot]$ den Erwartungswert (Durchschnitt) über alle Bilder $x$ im Datensatz $D$ bezeichnet. Ein hohes Kantengewicht bedeutet, dass das von Knoten $u$ repräsentierte niedrigstufige Merkmal ein starker kausaler Faktor für die Erkennung des von Knoten $v$ repräsentierten höherstufigen Merkmals ist.

## 4. DeepLIFT: Attribution durch Differenz-zur-Referenz

DeepLIFT (Deep Learning Important FeaTures) ist eine weitere einflussreiche, rückpropagierungsbasierte Attributionsmethode, die eine grundlegend andere Herangehensweise als rein gradientenbasierte Ansätze verfolgt. Anstatt die infinitesimale Sensitivität (den Gradienten) an einem einzigen Punkt zu messen, quantifiziert DeepLIFT die Wichtigkeit von Merkmalen, indem es die Aktivierungsänderung eines Neurons im Vergleich zu einem "Referenzzustand" betrachtet.

### 4.1 Die DeepLIFT-Philosophie: Differenz statt Gradient

Die Kernidee von DeepLIFT ist, die Ausgabe-Differenz eines Modells im Vergleich zu einer Referenz-Ausgabe durch die Eingabe-Differenz im Vergleich zu einer Referenz-Eingabe zu erklären. Die Referenz-Eingabe (oder Baseline) ist ein vom Benutzer gewählter, informativ neutraler Input, wie z.B. ein schwarzes Bild oder ein Vektor aus Nullen.

Dieser Ansatz löst direkt das Problem der Gradientensättigung. Selbst wenn der Gradient eines Neurons an einem bestimmten Punkt null ist (z.B. bei einer gesättigten ReLU- oder Sigmoid-Einheit), ist die Differenz seiner Aktivierung im Vergleich zur Referenzaktivierung in der Regel nicht null. Dadurch kann DeepLIFT auch dann einen relevanten Wichtigkeits-Score propagieren, wenn gradientenbasierte Methoden versagen würden. Die Methode ist rechnerisch effizient, da die Scores in einem einzigen Backward-Pass berechnet werden können.

DeepLIFT zerlegt die Vorhersage eines Netzwerks für eine bestimmte Eingabe, indem es die Beiträge aller Neuronen zu jedem Eingabemerkmal zurückpropagiert.

**Differenz-zur-Referenz ($\Delta t$):** Sei $t$ die Aktivierung eines Zielneurons für eine gegebene Eingabe und $t_0$ seine Aktivierung für die Referenz-Eingabe. Die "Differenz-zur-Referenz" $\Delta t$ ist definiert als:

$$\Delta t = t - t_0$$

**Beitragswerte ($C$) und die "Summation-to-Delta"-Eigenschaft:** DeepLIFT weist den Differenzen der Eingangsneuronen $\Delta x_i$ Beitragswerte $C_{\Delta x_i \Delta t}$ zu. Diese Werte quantifizieren den Anteil an der Gesamtdifferenz $\Delta t$, der auf die Differenz $\Delta x_i$ zurückzuführen ist. Diese Beitragswerte müssen eine fundamentale Eigenschaft erfüllen, die als Summation-to-Delta bezeichnet wird:

$$\sum_i C_{\Delta x_i \Delta t} = \Delta t$$

Diese Eigenschaft stellt sicher, dass die Summe der Beiträge der Eingabedifferenzen exakt die Zieldifferenz ergibt, wodurch eine vollständige und exakte Zerlegung der Ausgabe gewährleistet wird.

**Multiplikatoren ($m$):** Um die Beiträge effizient durch das Netzwerk zurückzupropagieren, führt DeepLIFT das Konzept der "Multiplikatoren" ein. Der Multiplikator $m_{\Delta x \Delta t}$ ist definiert als das Verhältnis des Beitrags zur Differenz:

$$m_{\Delta x \Delta t} = \frac{C_{\Delta x \Delta t}}{\Delta x}$$

Dieser Multiplikator verhält sich analog zu einer partiellen Ableitung, operiert aber auf finiten Differenzen ($\Delta$) statt auf infinitesimalen Änderungen ($d$). Diese Multiplikatoren gehorchen einer Kettenregel, die der für Gradienten ähnelt, was die Rückpropagierung ermöglicht: Wenn $t$ von Neuronen $y_j$ abhängt, die wiederum von Neuronen $x_i$ abhängen, gilt:

$$m_{\Delta x_i \Delta t} = \sum_j m_{\Delta x_i \Delta y_j} \cdot m_{\Delta y_j \Delta t}$$

### 4.2 Propagierungsregeln für Nichtlinearitäten

Die zentrale Herausforderung besteht darin, die Multiplikatoren für nichtlineare Aktivierungsfunktionen zu definieren. DeepLIFT schlägt hierfür verschiedene Regeln vor.

#### Die Linear-Regel

Für affine Transformationen (z.B. in Dense- oder Convolutional-Layern ohne die Aktivierungsfunktion), bei denen $t = \sum_i w_i x_i + b$, ist die Regel einfach. Da $\Delta t = \sum_i w_i \Delta x_i$, ist der Multiplikator einfach das Gewicht:

$$m_{\Delta x_i \Delta t} = w_i$$

#### Die Rescale-Regel

Für nichtlineare Aktivierungsfunktionen $t = f(x)$ mit einem einzigen Input $x$ (wie ReLU, Sigmoid, Tanh) approximiert die Rescale-Regel den Multiplikator als die Steigung der Sekante zwischen dem Referenzpunkt und dem tatsächlichen Aktivierungspunkt:

$$m_{\Delta x \Delta t} = \frac{\Delta t}{\Delta x} = \frac{f(x) - f(x_0)}{x - x_0}$$

Diese Regel ist die Standardimplementierung und löst das Sättigungsproblem, da $\Delta t$ auch dann ungleich null sein kann, wenn der lokale Gradient $\frac{dt}{dx}$ null ist.

#### Die RevealCancel-Regel

Die Rescale-Regel kann in bestimmten Szenarien irreführend sein, insbesondere wenn positive und negative Beiträge sich gegenseitig aufheben und so die Wichtigkeit von Merkmalen verschleiern. Die RevealCancel-Regel wurde entwickelt, um solche Abhängigkeiten aufzudecken, indem sie positive und negative Beiträge getrennt behandelt.

Dazu werden die Differenzen $\Delta x$ und $\Delta t$ in ihre positiven und negativen Anteile zerlegt ($\Delta x = \Delta x^+ + \Delta x^-$). Die Regel definiert dann separate Multiplikatoren für diese Anteile. Beispielsweise wird der Beitrag von $\Delta x^+$ zur positiven Differenz $\Delta t^+$ wie folgt definiert:

$$\Delta t^+ = \frac{1}{2}\left((f(x_0 + \Delta x^+) - f(x_0)) + (f(x_0 + \Delta x^- + \Delta x^+) - f(x_0 + \Delta x^-))\right)$$

Der Multiplikator ist dann $m_{\Delta x^+ \Delta t^+} = \frac{\Delta t^+}{\Delta x^+}$. Diese Formulierung berechnet den durchschnittlichen Effekt von $\Delta x^+$ einmal ohne und einmal mit dem Vorhandensein von $\Delta x^-$. Dies verhindert, dass sich gegenläufige Effekte gegenseitig auslöschen, und kann so verborgene Abhängigkeiten aufdecken. Während die RevealCancel-Regel theoretisch robuster ist, wird in der Praxis oft die einfachere und schnellere Rescale-Regel bevorzugt.

## 5. Übersicht der Attributionsmethoden

| Methode | Grundidee | Formel (vereinfacht) | Vorteile / Nachteile |
|---------|-----------|----------------------|---------------------|
| **Saliency Maps** (Simonyan et al.) | Berechnet die Relevanz eines Pixels als den Absolutwert des Gradienten der Klassen-Score-Funktion in Bezug auf dieses Pixel. Misst, wie stark sich die Ausgabe ändert, wenn sich ein Eingangspixel ändert. | $R(x) = \left\|\frac{\partial S_c(x)}{\partial x}\right\|$ | **Vorteile:** Einfach zu implementieren und zu verstehen; Schnell zu berechnen (nur ein Backpropagation-Pass). **Nachteile:** Gradienten können "gesättigt" sein und daher wichtige, aber nicht-lokale Informationen ignorieren; Oft visuell verrauscht; Unterscheidet nicht zwischen positiver und negativer Evidenz. |
| **Gradient × Input** (Shrikumar et al.) | Multipliziert den Gradienten elementweise mit dem Eingangsbild. Die Idee ist, dass nicht nur die Sensitivität (Gradient), sondern auch die Intensität des Merkmals selbst für die Relevanz wichtig ist. | $R(x) = x \odot \frac{\partial S_c(x)}{\partial x}$ | **Vorteile:** Berücksichtigt sowohl die Pixelintensität als auch den Gradienten; Kann schärfere und interpretierbarere Karten als Saliency Maps erzeugen. **Nachteile:** Erbt einige der Probleme von reinen Gradientenmethoden; Die Interpretation kann schwierig sein. |
| **Integrated Gradients** (Sundararajan et al.) | Integriert die Gradienten entlang eines geradlinigen Pfades von einer "Baseline" (z.B. ein schwarzes Bild) zum eigentlichen Eingangsbild. Dadurch werden die Probleme der Gradientensättigung umgangen. | $R_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial S_c(x' + \alpha(x-x'))}{\partial x_i} d\alpha$ | **Vorteile:** Erfüllt wichtige Axiome wie "Completeness"; Weniger anfällig für Gradientensättigung. **Nachteile:** Die Wahl der Baseline ist entscheidend; Rechenintensiver als einfache Gradientenmethoden. |
| **LRP** (Bach et al.) | Verteilt den Vorhersagewert des Netzwerks rückwärts durch die Schichten, bis hin zur Eingangsebene. Dabei gelten Erhaltungsregeln, sodass die Gesamt-Relevanz in jeder Schicht gleich bleibt. | Relevanz wird rückwärts propagiert: $R_j^{(l)} = \sum_k \frac{z_{jk}}{\sum_j z_{jk} + \epsilon} R_k^{(l+1)}$ | **Vorteile:** Basiert auf einem klaren theoretischen Prinzip (Relevanzerhaltung); Liefert oft saubere, gut interpretierbare Heatmaps; Kann zwischen positiver und negativer Relevanz unterscheiden. **Nachteile:** Implementierung ist komplexer; Die Wahl der Regel beeinflusst das Ergebnis. |
| **DeepLIFT** (Shrikumar et al.) | Vergleicht die Aktivierung jeder Neuron mit einer "Referenzaktivierung" (abgeleitet von einer Baseline). Propagiert "Kontributions-Scores" anstelle von Gradienten. | Basiert auf einer "Sum-to-Delta"-Regel: $\sum_i C_{\Delta x_i \Delta t} = \Delta t$ | **Vorteile:** Löst das Problem der diskontinuierlichen Gradienten bei ReLUs; Kann positive und negative Beiträge aufdecken; Erfüllt eine Form des "Completeness"-Axioms. **Nachteile:** Benötigt wie IG eine Baseline; Die konzeptionelle Komplexität ist höher. |
| **Grad-CAM** (Selvaraju et al.) | Verwendet die Gradienten, die in die letzte Konvolutionsebene fließen, um die Wichtigkeit jedes Feature-Map-Kanals für eine bestimmte Klasse zu berechnen. Erzeugt eine grobe Lokalisierungskarte. | $L_c^{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$ mit $\alpha_k^c = \frac{1}{Z}\sum_i\sum_j\frac{\partial S^c}{\partial A_{ij}^k}$ | **Vorteile:** Klassen-diskriminativ; Benötigt keine Modifikation der Architektur; Liefert oft gute, interpretierbare Lokalisierungen. **Nachteile:** Die Auflösung der Heatmap ist durch die Größe der letzten Feature-Map begrenzt; Ist keine vollständige Pixel-Attributionsmethode. |
| **Guided Backpropagation** (Springenberg et al.) | Kombiniert Vanilla Backpropagation und Deconvolutional Networks. Beim Rückpropagieren durch eine ReLU-Einheit werden nur positive Gradienten weitergegeben, und nur zu Neuronen, die im Forward Pass eine positive Aktivierung hatten. | Modifizierte ReLU-Rückpropagation: $R^{(l)} = (f^{(l)} > 0) \odot (R^{(l+1)} > 0) \odot R^{(l+1)}$ | **Vorteile:** Erzeugt sehr scharfe, hochauflösende und visuell ansprechende Visualisierungen. **Nachteile:** Ist nicht mehr direkt an die Entscheidung des Modells gekoppelt; Die Visualisierung kann irreführend sein. |

## Literatur

1. Smilkov, Daniel, Nikhil Thorat, Been Kim, Fernanda B. Viégas and Martin Wattenberg. SUMMIT: Scaling Deep Learning Interpretability by Visualizing
Activation and Attribution Summarizations. ArXiv, abs/1704.03313, 2017.
2. Smilkov, Daniel, Nikhil Thorat, Been Kim, Fernanda B. Viégas and Martin Wattenberg. SmoothGrad: removing noise by adding noise. ArXiv, abs/
1706.03825, 2017.
3. Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. Axiomatic Attribution for Deep Networks. Proceedings of the 34th International Conference on
Machine Learning, 2017.
4. Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency
Maps. ArXiv, abs/1312.6034, 2014.
5. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important Features Through Propagating Activation Differences. arXiv preprint ar-
Xiv:1704.02685.
6. Shrikumar, A., Greenside, P., & Kundaje, A. (2016). Not Just a Black Box: Learning Important Features Through Propagating Activation Differences.
arXiv preprint arXiv:1605.01713.
7. Captum Team. (n.d.). DeepLIFT Documentation. Captum.ai. Retrieved from https://captum.ai/api/deep_lift.html
8. Pysquad. (2024). DeepLIFT Explained: Python Techniques for AI Transparency. Medium.
9. Stabenau, M. et al. (2023). innsight: Get Deep Insights into Your Neural Network. R package version 0.1.0.
10. University of Waterloo. (2017). STAT946F17: Learning Important Features Through Propagating Activation Differences. StatWiki.
11. Ancona, M., Ceolini, E., Öztireli, C., & Gross, M. (2018). Towards better understanding of gradient-based attribution methods for Deep Neural Networks.
International Conference on Learning Representations (ICLR).
12. Tseng, G. (2018, November 16). How to explain deep learning models, part 2: SHAP and the path to better model interpretability. Personal Blog.
13. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important Features Through Propagating Activation Differences. Proceedings of the 34th
International Conference on Machine Learning (ICML).
14. Shrikumar, A. (2017). DeepLIFT: A method for explaining predictions of deep neural networks (15 min tutorial). [Video]. YouTube.
15. Tseng, G. (2018, November 16). How to explain deep learning models, part 2: SHAP and the path to better model interpretability. Personal Blog.
16. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important Features Through Propagating Activation Differences. Proceedings of the 34th
International Conference on Machine Learning (ICML).
