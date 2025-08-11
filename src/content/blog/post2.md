---
title: "XAI f\u00fcr CNN: Attributionsmethoden zur Bildanalyse"
description: "Attributionsmethoden stellen ein wichtiges Instrument dar, um die Entscheidungsprozesse neuronaler Netze transparent zu machen. Sie ordnen jedem Eingabemerkmal einen Beitrag zur Vorhersage zu und erm\u00f6glichen somit eine Bewertung der Relevanz einzelner Merkmale. In diesem Beitrag werden g\u00e4ngige Attributionsmethoden f\u00fcr CNNs vorgestellt und mathematisch fundiert beschrieben. Besonderes Augenmerk liegt auf Gradienten-basierten Verfahren wie den Integrated Gradients, welche den Beitrag eines Pixels zur Vorhersage durch Integration entlang eines Pfades von einer Baseline zur Eingabe quantifizieren. Dar\u00fcber hinaus werden Konzepte wie Layer-wise Relevance Propagation (LRP), DeepLIFT und neuere Visualisierungstechniken wie SUMMIT diskutiert. Ziel ist es, ein Verst\u00e4ndnis f\u00fcr die methodischen Grundlagen und praktischen Herausforderungen der Erkl\u00e4rbarkeit tiefer Modelle zu vermitteln.."
pubDate: "Jul 14 2025"
heroImage: "/personal_blog/aikn.webp"
---

# XAI f\u00fcr CNN: Attributionsmethoden zur Bildanalyse
*Author: Christoph Würsch*

Attributionsmethoden stellen ein wichtiges Instrument dar, um die Entscheidungsprozesse neuronaler Netze transparent zu machen. Sie ordnen jedem Eingabemerkmal einen Beitrag zur Vorhersage zu und erm\u00f6glichen somit eine Bewertung der Relevanz einzelner Merkmale. In diesem Beitrag werden g\u00e4ngige Attributionsmethoden f\u00fcr CNNs vorgestellt und mathematisch fundiert beschrieben. Besonderes Augenmerk liegt auf Gradienten-basierten Verfahren wie den Integrated Gradients, welche den Beitrag eines Pixels zur Vorhersage durch Integration entlang eines Pfades von einer Baseline zur Eingabe quantifizieren. Dar\u00fcber hinaus werden Konzepte wie Layer-wise Relevance Propagation (LRP), DeepLIFT und neuere Visualisierungstechniken wie SUMMIT diskutiert. Ziel ist es, ein Verst\u00e4ndnis f\u00fcr die methodischen Grundlagen und praktischen Herausforderungen der Erkl\u00e4rbarkeit tiefer Modelle zu vermitteln.

## Einf\u00fchrung in die Attribution bei CNNs

Convolutional Neural Networks (CNNs) haben eine herausragende Leistungsf\u00e4higkeit in Aufgaben der Bilderkennung und -regression erzielt. Ihre komplexe, hierarchische Struktur macht sie jedoch zu "Black Boxes": Es ist oft unklar, auf welche Merkmale im Eingangsbild sich das Modell f\u00fcr seine Entscheidung st\u00fctzt. Explainable AI (XAI) zielt darauf ab, diese Black Box zu \u00f6ffnen und die Entscheidungsfindung von Modellen nachvollziehbar zu machen.

Ein zentraler Ansatz hierf\u00fcr sind *Attributionsmethoden*. Die grundlegende Idee ist, die Vorhersage eines Modells auf seine Eingabemerkmale "zur\u00fcckzuf\u00fchren" (zu attribuieren). F\u00fcr Bilddaten bedeutet dies, jedem Pixel des Eingangsbildes einen Relevanz- oder Wichtigkeitswert zuzuordnen. Das Ergebnis ist eine Heatmap, oft als *Saliency Map* bezeichnet, die visuell hervorhebt, welche Bildbereiche f\u00fcr die Ausgabe des Netzwerks (z.B. die Klassifizierung als "Hund") am einflussreichsten waren.

### Mathematische Definition der Attribution

Sei $F: \mathbb{R}^d \to \mathbb{R}$ eine Funktion, die ein neuronales Netzwerk repr\u00e4sentiert. F\u00fcr ein Eingangsbild $x \in \mathbb{R}^d$, das als Vektor von $d$ Pixeln betrachtet wird, gibt $F(x)$ einen Skalar aus. Dieser Skalar kann der Logit-Wert f\u00fcr eine bestimmte Klasse bei einer Klassifikationsaufgabe oder der vorhergesagte Wert bei einer Regressionsaufgabe sein.

Eine *Attribution* ist eine Zuweisung eines Relevanzwertes $A_i(x)$ zu jedem Eingabemerkmal (Pixel) $x_i$. Das Ziel ist die Erstellung einer Attributionskarte (oder Vektor) $A(x) \in \mathbb{R}^d$.

$$
A(x) = (A_1(x), A_2(x), \dots, A_d(x))
$$

Diese Karte $A(x)$ soll die Wichtigkeit jedes Pixels $x_i$ f\u00fcr den finalen Output $F(x)$ quantifizieren.

### Standard-Attribution: Sensitivity Maps

Der direkteste Weg, die "Sensitivit\u00e4t" des Outputs in Bezug auf eine kleine \u00c4nderung eines Input-Pixels zu messen, ist die Berechnung des Gradienten.

#### Definition (Sensitivity Map):

Die Attribution eines Pixels $x_i$ wird als der partielle Ableitungswert der Output-Funktion $F$ nach diesem Pixel definiert.

$$
A_i^{\text{Sens}}(x) = \frac{\partial F(x)}{\partial x_i}
$$

Die gesamte Attributionskarte ist somit der Gradient des Outputs bez\u00fcglich des Inputs:

$$
A^{\text{Sens}}(x) = \nabla_x F(x)
$$

#### Interpretation:

Der Wert $\frac{\partial F(x)}{\partial x_i}$ gibt an, wie stark sich der Output $F(x)$ \u00e4ndert, wenn das Pixel $x_i$ infinitesimal klein ver\u00e4ndert wird. Ein hoher absoluter Wert bedeutet eine hohe Relevanz des Pixels f\u00fcr die Entscheidung. Zur Visualisierung wird oft der Absolutbetrag oder das Quadrat des Gradienten verwendet.

### Integrated Gradients (IG)

Ein Problem der einfachen Gradientenmethode ist die S\u00e4ttigung. Wenn ein Neuron bereits stark aktiviert ist (z.B. durch eine ReLU-Aktivierungsfunktion), kann sein Gradient null sein, obwohl das Neuron entscheidend f\u00fcr das Ergebnis ist. Integrated Gradients (IG) l\u00f6st dieses Problem, indem es die Gradienten entlang eines Pfades von einem Referenzbild (Baseline) $x'$ zum eigentlichen Bild $x$ integriert. Die Baseline $x'$ ist typischerweise ein informationsloses Bild, z.B. ein komplett schwarzes Bild.

#### Definition (Integrated Gradients):

Die Attribution eines Pixels $x_i$ mittels IG ist definiert als:

$$
A_i^{\text{IG}}(x) ::= (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x)}{\partial x_i}\Big \vert_{x' + \alpha(x - x')} \,d\alpha
$$

#### Eigenschaften und Interpretation:

- **Pfadintegral:** Die Formel integriert die Gradienten entlang der geraden Linie im Merkmalsraum von der Baseline $x'$ zum Bild $x$.
- **Vollst\u00e4ndigkeit (Completeness):** Eine wichtige Eigenschaft von IG ist, dass die Summe aller Attributionswerte der Differenz der Modellvorhersage zwischen dem Bild $x$ und der Baseline $x'$ entspricht:

$$
\sum_{i=1}^{d} A_i^{\text{IG}}(x) = F(x) - F(x')
$$

Dies macht die Attributionen "vollst\u00e4ndig" und direkt interpretierbar als Beitr\u00e4ge zur Gesamt\u00e4nderung des Outputs.

## Gradienten-basierte Saliency-Methoden

Diese Methoden basieren alle auf der R\u00fcckpropagierung von Gradienten vom Output zum Input.

### Saliency Maps (nach Simonyan et al., 2014)

Historisch gesehen ist dies eine der ersten und einfachsten Methoden. Sie ist in ihrer reinsten Form identisch mit der oben definierten Sensitivity Map.

#### Algorithmus: Berechnung einer Saliency Map

1. **Input:** Modell $F$, Eingangsbild $x$, Zielklasse $c$.
2. F\u00fchre einen Forward-Pass mit $x$ durch, um alle Aktivierungen zu berechnen.
3. Berechne den Score $S_c(x)$ f\u00fcr die Zielklasse $c$. Dies ist der Output $F(x)$.
4. Berechne den Gradienten des Scores bez\u00fcglich des Eingangsbildes:

   $$
   M(x) = \nabla_x S_c(x) = \frac{\partial S_c(x)}{\partial x}
   $$

5. **Visualisierung:**
   - Aggregiere die Gradienten \u00fcber die Farbkan\u00e4le, z.B. durch den Maximalwert des Absolutbetrags f\u00fcr jedes Pixel: $m_{ij} = \max_{k} |M_{ijk}|$.
   - Normalisiere die resultierende 2D-Karte $m$ zur Darstellung als Heatmap.

6. **Output:** Saliency Map $m$.

### SmoothGrad (Smilkov et al., 2017)

Standard-Gradientenkarten sind oft visuell verrauscht, was die Interpretation erschwert. SmoothGrad reduziert dieses Rauschen durch einen einfachen, aber effektiven Mittelungsprozess. Die Intuition ist, dass das wahre Relevanz-Signal bei leichten St\u00f6rungen des Bildes stabil bleibt, w\u00e4hrend das Rauschen im Gradienten zuf\u00e4llig ist und sich bei Mittelung herausk\u00fcrzt.

#### Algorithmus: SmoothGrad

1. **Input:** Modell $F$, Eingangsbild $x$, Anzahl der Samples $n$, Rauschlevel (Standardabweichung) $\sigma$.
2. Initialisiere eine leere Akkumulator-Karte $M_{avg} \leftarrow 0$.
3. **For** $i = 1$ to $n$:
   - Erzeuge einen zuf\u00e4lligen Rauschvektor $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.
   - Erstelle ein gest\u00f6rtes Bild: $x_{noisy} = x + \epsilon_i$.
   - Berechne die Gradienten-basierte Saliency Map f\u00fcr das gest\u00f6rte Bild: $M_i = \nabla_x F(x_{noisy})$.
   - Addiere die Karte zum Akkumulator: $M_{avg} \leftarrow M_{avg} + M_i$.
4. Berechne den Durchschnitt: $M_{smooth} = \frac{1}{n} M_{avg}$.
5. **Output:** Gegl\u00e4ttete Saliency Map $M_{smooth}$.
